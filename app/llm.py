"""Free text -> checked Proposal rows, via an LLM on Runware.

Pipeline:  call_runware (raw model text)  ->  parse_and_validate (pydantic parse + one check
pass per proposal, pure)  ->  propose (persists every row, valid or not, for the manager).

Runware: POST https://api.runware.ai/v1 with a JSON array of tasks. Text generation is the
``textInference`` task; the response is ``{"data": [{"taskType", "taskUUID", "text", ...}]}``
(or ``{"errors": [...]}``). Claude Sonnet on Runware only supports ``outputFormat: TEXT`` (no
JSON mode / jsonSchema), so JSON is demanded in the system prompt and enforced by pydantic here.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date as Date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlmodel import Session as DBSession, select

from app.models import Proposal, Session, Vehicle

LONDON = ZoneInfo("Europe/London")  # same as app.seed.LONDON (not imported: avoids pulling in app.db)

RUNWARE_URL = "https://api.runware.ai/v1"
RUNWARE_MODEL = os.environ.get("RUNWARE_MODEL", "anthropic-claude-sonnet-4-6")
RUNWARE_TIMEOUT_S = 60.0

ALLOWED_FIELDS = ("arrival_at", "departure_at", "arrival_soc", "target_soc")
TIME_FIELDS = ("arrival_at", "departure_at")
SOC_FIELDS = ("arrival_soc", "target_soc")


# ---------- schema the LLM must produce ----------

class ProposalLLM(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    vehicle: str  # as written by the model, e.g. "Van 03"
    field: Literal["arrival_at", "departure_at", "arrival_soc", "target_soc"]
    value: str  # "23:30" (London local) for times; "0.2" or "20%" for SoC
    quote: str  # exact span copied from the user's text


class ProposalBatch(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    proposals: list[ProposalLLM]


assert ProposalLLM.model_fields["field"].annotation.__args__ == ALLOWED_FIELDS


class RunwareError(RuntimeError):
    """Raised when the Runware call fails (network, timeout, HTTP status, or bad envelope)."""


# ---------- LLM call ----------

def _system_prompt(vehicle_names: list[str]) -> str:
    names = "\n".join(f"- {n}" for n in vehicle_names)
    return f"""You convert a London EV van depot manager's free-text note into proposed changes to tonight's charging sessions.

Known vehicles (use these exact names):
{names}

Each session has exactly these editable fields:
- arrival_at: when the van plugs in at the depot tonight. value = local London time, 24h "HH:MM" (e.g. "23:30").
- departure_at: when the van must leave (usually next morning). value = local London time, 24h "HH:MM" (e.g. "05:00").
- arrival_soc: battery state of charge on arrival. value = percentage string, e.g. "20%".
- target_soc: required state of charge at departure. value = percentage string, e.g. "100%" ("full" means "100%").

Rules:
- Output ONLY a JSON object, no prose and no code fences, exactly of the form:
  {{"proposals": [{{"vehicle": "<name>", "field": "<field>", "value": "<string>", "quote": "<string>"}}]}}
- Every key is required; no other keys. All values are JSON strings.
- One proposal per (vehicle, field) change. One sentence may yield several proposals.
- "quote" must be copied VERBATIM (exact characters, a contiguous span) from the user's note and must be the part that states this change. Never paraphrase or fix typos in the quote.
- Only propose changes the note actually states. Approximate values ("about 23:30", "maybe 20%") are fine: use the stated number. Do not invent values.
- If nothing in the note maps to these fields, output {{"proposals": []}}."""


def _api_key() -> str:
    key = os.environ.get("RUNWARE_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(Path.home() / ".env")
        except ImportError:  # pragma: no cover
            pass
        key = os.environ.get("RUNWARE_API_KEY")
    if not key:
        raise RunwareError("RUNWARE_API_KEY is not set")
    return key


def call_runware(text: str, vehicle_names: list[str], client: httpx.Client | None = None) -> str:
    """Send one textInference task to Runware; return the model's raw text output."""
    task_uuid = str(uuid.uuid4())
    body = [{
        "taskType": "textInference",
        "taskUUID": task_uuid,
        "model": RUNWARE_MODEL,
        "deliveryMethod": "sync",
        "outputFormat": "TEXT",
        "includeUsage": True,
        "messages": [{"role": "user", "content": text}],
        "settings": {"systemPrompt": _system_prompt(vehicle_names), "maxTokens": 1500},
    }]
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    own = client is None
    client = client or httpx.Client(timeout=RUNWARE_TIMEOUT_S)
    try:
        r = client.post(RUNWARE_URL, json=body, headers=headers, timeout=RUNWARE_TIMEOUT_S)
    except httpx.TimeoutException as e:
        raise RunwareError(f"Runware request timed out after {RUNWARE_TIMEOUT_S:.0f}s") from e
    except httpx.HTTPError as e:
        raise RunwareError(f"Runware request failed: {type(e).__name__}: {e}") from e
    finally:
        if own:
            client.close()

    try:
        payload = r.json()
    except ValueError:
        payload = None
    if r.status_code != 200 or not isinstance(payload, dict) or payload.get("errors"):
        detail = ""
        if isinstance(payload, dict) and payload.get("errors"):
            err = payload["errors"][0]
            detail = (err.get("message") or err.get("code") or json.dumps(err)[:200]) if isinstance(err, dict) else str(err)
        else:
            detail = r.text[:200]
        raise RunwareError(f"Runware HTTP {r.status_code}: {detail}")
    data = payload.get("data") or []
    item = next((d for d in data if isinstance(d, dict) and d.get("taskUUID") == task_uuid), data[0] if data else None)
    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
        raise RunwareError("Runware response had no text output")
    return item["text"]


# ---------- parsing / validation (pure) ----------

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_fences(s: str) -> str:
    m = _FENCE.search(s)
    return (m.group(1) if m else s).strip()


def _short_pydantic(e: ValidationError) -> str:
    errs = e.errors(include_url=False)
    if not errs:
        return str(e)[:200]
    first = errs[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = f"{loc}: {first['msg']}" if loc else first["msg"]
    if len(errs) > 1:
        msg += f" (+{len(errs) - 1} more)"
    return msg[:300]


def _norm_name(s: str) -> str:
    """'van 3' / 'VAN-03' / 'Van 03' -> 'van3'."""
    s = re.sub(r"\d+", lambda m: m.group().lstrip("0") or "0", s.casefold())
    return re.sub(r"[^0-9a-z]", "", s)


def _digits(s: str) -> str | None:
    nums = re.findall(r"\d+", s)
    return (nums[0].lstrip("0") or "0") if len(nums) == 1 else None


def _resolve_vehicle(name: str, vehicles: list[Vehicle]) -> tuple[Vehicle | None, str | None]:
    key = _norm_name(name)
    if not key:
        return None, "vehicle name is empty"
    hits = [v for v in vehicles if _norm_name(v.name) == key]
    if not hits and key.isdigit():  # bare number ("3", "#3") -> the one vehicle with that number
        hits = [v for v in vehicles if _digits(v.name) == key]
    if not hits:
        return None, f"unknown vehicle '{name}' (known: {', '.join(v.name for v in vehicles)})"
    if len(hits) > 1:
        return None, f"ambiguous vehicle '{name}' (matches {', '.join(v.name for v in hits)})"
    return hits[0], None


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def _parse_time(field: str, value: str, d: Date) -> datetime:
    m = _HHMM.match(value)
    if not m:
        raise ValueError(f"{field} value '{value}' is not a 24h HH:MM time")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        raise ValueError(f"{field} value '{value}' is not a valid time")
    # Session.date is the evening the session starts. Afternoon/evening times belong to that
    # date; early-morning times to the next day. (Same rule for both fields.)
    day = d if h >= 12 else d + timedelta(days=1)
    out = datetime.combine(day, time(h, mi), tzinfo=LONDON).astimezone(timezone.utc)
    if out.astimezone(LONDON).time() != time(h, mi):  # e.g. 01:30 on the spring-forward night
        raise ValueError(f"{field} value '{value}' does not exist on {day} (clock change)")
    return out


_NUM = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*(%?)\s*$")


def _parse_soc(field: str, value: str) -> float:
    m = _NUM.match(value)
    if not m:
        raise ValueError(f"{field} value '{value}' is not a number or percentage")
    x = float(m.group(1))
    if m.group(2) == "%" or x >= 2:  # "20%" or bare "20" -> percent; "0.2" / "1" -> fraction
        x /= 100
    elif x > 1:
        raise ValueError(f"{field} value '{value}' is ambiguous (fraction or percent?)")
    if not 0 <= x <= 1:
        raise ValueError(f"{field} value '{value}' is outside 0-100%")
    return round(x, 4)


def _ws(s: str) -> str:
    return " ".join(s.split()).casefold()


def _fmt(dt: datetime) -> str:
    return dt.astimezone(LONDON).strftime("%H:%M")


def _check(p: ProposalLLM, raw_text: str, vehicles: list[Vehicle],
           sessions: dict[int, Session], out: Proposal) -> str | None:
    """Fill `out` with normalised values; return the first error message, or None if valid."""
    # a. vehicle
    v, err = _resolve_vehicle(p.vehicle, vehicles)
    if err:
        return err
    out.vehicle_id = v.id
    # b. session
    s = sessions.get(v.id)
    if s is None:
        return f"{v.name} has no session for this date"
    # c. value
    try:
        if p.field in TIME_FIELDS:
            new_dt = _parse_time(p.field, p.value, s.date)
            out.value = new_dt.isoformat()
        else:
            new_soc = _parse_soc(p.field, p.value)
            out.value = repr(new_soc)
    except ValueError as e:
        return str(e)
    # d. consistency with the rest of the session (proposed value replaces the current one)
    if p.field in TIME_FIELDS:
        arr = new_dt if p.field == "arrival_at" else _utc(s.arrival_at)
        dep = new_dt if p.field == "departure_at" else _utc(s.departure_at)
        if not arr < dep:
            return (f"arrival {_fmt(arr)} would not be before departure {_fmt(dep)} "
                    f"for {v.name} (London time)")
    else:
        a = new_soc if p.field == "arrival_soc" else s.arrival_soc
        t = new_soc if p.field == "target_soc" else s.target_soc
        if not t > a:
            return f"target_soc {t:.0%} would not exceed arrival_soc {a:.0%} for {v.name}"
    # e. quote grounding: verbatim, and it must actually evidence this change
    if not p.quote.strip() or _ws(p.quote) not in _ws(raw_text):
        return f"quote '{p.quote}' does not appear in the original text"
    vnum = _digits(v.name)
    if vnum and vnum not in _nums(raw_text):
        return f"{v.name} is not mentioned in the original text"
    q, qnums = _ws(p.quote), _nums(p.quote)
    if p.field in TIME_FIELDS:
        h = new_dt.astimezone(LONDON).hour
        ok = str(h) in qnums or str(h % 12 or 12) in qnums or any(w in q for w in ("midnight", "noon"))
    else:
        pct = str(round(new_soc * 100))
        ok = pct in qnums or (new_soc == 1 and "full" in q) or (new_soc == 0 and "empty" in q)
    if not ok:
        return f"quote '{p.quote}' does not support {p.field} = {p.value}"
    return None


def _nums(s: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", s)}


def parse_and_validate(raw_text: str, llm_output: str, vehicles: list[Vehicle],
                       sessions: dict[int, Session], now: datetime | None = None) -> list[Proposal]:
    """Parse the LLM output with pydantic and check each proposal once. Returns UNSAVED rows.

    `sessions` maps vehicle_id -> that vehicle's Session for the date being planned.
    """
    now = _utc(now) if now else datetime.now(timezone.utc)
    try:
        batch = ProposalBatch.model_validate_json(_strip_fences(llm_output))
    except ValidationError as e:
        return [Proposal(raw_text=raw_text, status="invalid", created_at=now,
                         error=f"LLM output failed schema validation: {_short_pydantic(e)}")]

    rows: list[Proposal] = []
    for p in batch.proposals:
        row = Proposal(raw_text=raw_text, field=p.field, value=p.value, quote=p.quote,
                       status="pending", created_at=now)
        try:
            err = _check(p, raw_text, vehicles, sessions, row)
        except Exception as e:  # never let one bad proposal 500 the request
            err = f"could not validate proposal: {type(e).__name__}"
        if err:
            row.status, row.error = "invalid", err
        rows.append(row)
    return rows


# ---------- orchestration ----------

def propose(db: DBSession, raw_text: str, session_date: Date,
            client: httpx.Client | None = None) -> list[Proposal]:
    """Free text -> saved Proposal rows (pending or invalid) for `session_date`."""
    vehicles = list(db.exec(select(Vehicle).order_by(Vehicle.name)).all())
    sessions = {s.vehicle_id: s for s in db.exec(select(Session).where(Session.date == session_date))}
    try:
        out = call_runware(raw_text, [v.name for v in vehicles], client=client)
        rows = parse_and_validate(raw_text, out, vehicles, sessions)
    except Exception as e:  # RunwareError, or a malformed response shape
        msg = str(e) if isinstance(e, RunwareError) else f"unexpected response ({type(e).__name__})"
        rows = [Proposal(raw_text=raw_text, status="invalid", error=f"LLM call failed: {msg}",
                         created_at=datetime.now(timezone.utc))]
    for r in rows:
        db.add(r)
    db.commit()
    for r in rows:
        db.refresh(r)
    return rows
