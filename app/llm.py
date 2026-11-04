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
from dataclasses import dataclass, field as dc_field
from datetime import date as Date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlmodel import Session as DBSession, select

from app.timeutil import LONDON, to_utc
from app.models import Proposal, Session, Vehicle


RUNWARE_URL = "https://api.runware.ai/v1"
RUNWARE_MODEL = os.environ.get("RUNWARE_MODEL", "google:gemini@3.1-flash-lite")  # benchmarked: p95 1.6s, 100% acc, ~11x cheaper than Sonnet 4.6
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
- Do not propose a value the note says is unchanged or already in place ("as planned", "still 05:00").
- If nothing in the note changes tonight's sessions, output {{"proposals": []}}. An empty list is a normal, correct answer (e.g. for "all vans as planned" or an unrelated note)."""


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
        try:
            r = client.post(RUNWARE_URL, json=body, headers=headers, timeout=RUNWARE_TIMEOUT_S)
        except httpx.ConnectError:  # one retry on a transient connect/SSL failure
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


_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def _hhmm(field: str, value: str) -> time:
    m = _HHMM.match(value)
    if not m:
        raise ValueError(f"{field} value '{value}' is not a 24h HH:MM time")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        raise ValueError(f"{field} value '{value}' is not a valid time")
    return time(h, mi)


def _local(field: str, value: str, day: Date, t: time) -> datetime:
    """London `t` on `day`, as UTC; rejects times that do not exist on that day."""
    out = datetime.combine(day, t, tzinfo=LONDON).astimezone(timezone.utc)
    if out.astimezone(LONDON).time() != t:  # e.g. 01:30 on the spring-forward night
        raise ValueError(f"{field} value '{value}' does not exist on {day} (clock change)")
    return out


def _parse_time(field: str, value: str, d: Date) -> datetime:
    """Arrival rule. Session.date is the evening the session starts: afternoon/evening times
    belong to that date, early-morning times to the next day."""
    t = _hhmm(field, value)
    return _local(field, value, d if t.hour >= 12 else d + timedelta(days=1), t)


MAX_DWELL = timedelta(hours=24)


def _resolve_departure(value: str, d: Date, arrival: datetime) -> datetime | None:
    """Departure rule: the earliest of {d, d+1} at that London time that is strictly after
    `arrival` and less than 24h after it (so a late-shift "13:00" is the next afternoon).
    None if neither qualifies; raises ValueError for a bad value, or if the only candidate in
    range falls in a clock-change gap."""
    t = _hhmm("departure_at", value)
    gap: ValueError | None = None
    for day in (d, d + timedelta(days=1)):
        try:
            dt = _local("departure_at", value, day, t)
        except ValueError as e:
            gap = gap or e
            continue
        if arrival < dt and dt - arrival < MAX_DWELL:
            return dt
    if gap:
        raise gap
    return None


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


@dataclass
class _Running:
    """A van's session as it stands after the valid proposals seen so far in this note."""
    name: str
    arrival_at: datetime  # UTC
    departure_at: datetime  # UTC
    arrival_soc: float
    target_soc: float
    src: dict[str, int] = dc_field(default_factory=dict)  # field -> 1-based proposal number that set it

    @classmethod
    def of(cls, v: Vehicle, s: Session) -> "_Running":
        return cls(v.name, to_utc(s.arrival_at), to_utc(s.departure_at), s.arrival_soc, s.target_soc)

    def earlier(self, f: str, shown: str) -> str:
        """Suffix naming the earlier proposal in this note that set field `f`, if any."""
        n = self.src.get(f)
        return f"; conflicts with earlier proposal {n} in this note ({self.name} {f} = {shown})" if n else ""


def _check(p: ProposalLLM, raw_text: str, vehicles: list[Vehicle], sessions: dict[int, Session],
           running: dict[int, _Running], seen: dict[tuple[int, str], int], n: int,
           out: Proposal) -> tuple[str | None, object]:
    """Fill `out` with normalised values; return (first error or None, normalised new value).

    `running` holds each van's session after the earlier valid proposals in this batch; `seen`
    maps (vehicle_id, field) -> number of the first proposal in this batch that targeted it.
    """
    # a. vehicle
    v, err = _resolve_vehicle(p.vehicle, vehicles)
    if err:
        return err, None
    out.vehicle_id = v.id
    # b. session
    s = sessions.get(v.id)
    if s is None:
        return f"{v.name} has no session for this date", None
    r = running.setdefault(v.id, _Running.of(v, s))
    # c. one change per (van, field) per note
    first = seen.get((v.id, p.field))  # only VALID earlier proposals count (recorded on success)
    if first is not None:
        return f"duplicate: {v.name} {p.field} was already proposed earlier in this note (proposal {first})", None
    # d. value
    try:
        if p.field == "departure_at":
            new = _resolve_departure(p.value, s.date, r.arrival_at)
            if new is None:
                return (f"departure {_hhmm(p.field, p.value):%H:%M} is not within 24h after arrival "
                        f"{_fmt(r.arrival_at)} for {v.name} (London time)"
                        f"{r.earlier('arrival_at', _fmt(r.arrival_at))}"), None
            out.value = new.isoformat()
        elif p.field == "arrival_at":
            new = _parse_time(p.field, p.value, s.date)
            out.value = new.isoformat()
        else:
            new = _parse_soc(p.field, p.value)
            out.value = repr(new)
    except ValueError as e:
        return str(e), None
    # e. no-op: the value must differ from the (running) session's current one
    cur = getattr(r, p.field)
    if p.field in TIME_FIELDS:
        if new.replace(second=0, microsecond=0) == cur.replace(second=0, microsecond=0):
            return f"no change: {v.name} {p.field} is already {_fmt(cur)}", None
    elif abs(new - cur) <= 0.005 + 1e-9:
        return f"no change: {v.name} {p.field} is already {cur:.0%}", None
    # f. consistency with the rest of the (running) session: the proposed value replaces the current one
    if p.field in TIME_FIELDS:
        arr = new if p.field == "arrival_at" else r.arrival_at
        dep = new if p.field == "departure_at" else r.departure_at
        if not arr < dep:
            other = "departure_at" if p.field == "arrival_at" else "arrival_at"
            return (f"arrival {_fmt(arr)} would not be before departure {_fmt(dep)} "
                    f"for {v.name} (London time){r.earlier(other, _fmt(getattr(r, other)))}"), None
    else:
        a = new if p.field == "arrival_soc" else r.arrival_soc
        t = new if p.field == "target_soc" else r.target_soc
        if not t > a:
            other = "target_soc" if p.field == "arrival_soc" else "arrival_soc"
            return (f"target_soc {t:.0%} would not exceed arrival_soc {a:.0%} for {v.name}"
                    f"{r.earlier(other, f'{getattr(r, other):.0%}')}"), None
    # g. quote grounding: verbatim, and it must actually evidence this change
    if not p.quote.strip() or _ws(p.quote) not in _ws(raw_text):
        return f"quote '{p.quote}' does not appear in the original text", None
    vnum = _digits(v.name)
    if vnum and vnum not in _nums(raw_text):
        return f"{v.name} is not mentioned in the original text", None
    q, qnums = _ws(p.quote), _nums(p.quote)
    if p.field in TIME_FIELDS:
        h = new.astimezone(LONDON).hour
        ok = str(h) in qnums or str(h % 12 or 12) in qnums or any(w in q for w in ("midnight", "noon"))
    else:
        pct = str(round(new * 100))
        ok = pct in qnums or (new == 1 and "full" in q) or (new == 0 and "empty" in q)
    if not ok:
        return f"quote '{p.quote}' does not support {p.field} = {p.value}", None
    return None, new


def _nums(s: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", s)}


def parse_and_validate(raw_text: str, llm_output: str, vehicles: list[Vehicle],
                       sessions: dict[int, Session], now: datetime | None = None) -> list[Proposal]:
    """Parse the LLM output with pydantic and check each proposal once, in order. Returns UNSAVED rows.

    `sessions` maps vehicle_id -> that vehicle's Session for the date being planned.
    """
    now = to_utc(now) if now else datetime.now(timezone.utc)
    try:
        batch = ProposalBatch.model_validate_json(_strip_fences(llm_output))
    except ValidationError as e:
        return [Proposal(raw_text=raw_text, status="invalid", created_at=now,
                         error=f"LLM output failed schema validation: {_short_pydantic(e)}")]

    # Proposals are checked in order against a running copy of each van's session: every valid
    # proposal is applied to the copy, so later ones are judged against the note as a whole.
    running: dict[int, _Running] = {}
    seen: dict[tuple[int, str], int] = {}
    rows: list[Proposal] = []
    for n, p in enumerate(batch.proposals, start=1):
        row = Proposal(raw_text=raw_text, field=p.field, value=p.value, quote=p.quote,
                       status="pending", created_at=now)
        try:
            err, new = _check(p, raw_text, vehicles, sessions, running, seen, n, row)
        except Exception as e:  # never let one bad proposal 500 the request
            err = f"could not validate proposal: {type(e).__name__}"
        if err:
            row.status, row.error = "invalid", err
        else:
            r = running[row.vehicle_id]
            setattr(r, p.field, new)
            r.src[p.field] = n
            seen[(row.vehicle_id, p.field)] = n
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
