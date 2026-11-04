"""Tests for app.llm. No network, no real database."""
import json
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session as DBSession, SQLModel, create_engine, select

from app import llm
from app.llm import ProposalBatch, call_runware, parse_and_validate, propose
from app.models import Proposal, Session, Vehicle

LONDON = ZoneInfo("Europe/London")
D = date(2026, 11, 4)  # GMT (no DST offset), so local == UTC
NOW = datetime(2026, 11, 4, 17, 0, tzinfo=timezone.utc)

# mirrors app.seed.FLEET (copied, not imported: app.seed pulls in app.db's engine)
FLEET = [
    ("Van 01", "18:30", "06:00", 0.25, 0.90),
    ("Van 02", "19:00", "06:30", 0.30, 0.90),
    ("Van 03", "19:30", "07:00", 0.20, 0.85),
    ("Van 04", "20:00", "05:30", 0.15, 0.95),
    ("Van 05", "21:00", "07:30", 0.40, 0.90),
    ("Van 06", "22:00", "06:00", 0.10, 0.90),
    ("Van 07", "23:30", "05:00", 0.10, 1.00),
    ("Van 08", "18:00", "07:00", 0.35, 0.80),
]
TEXT = "Van 3 stuck in traffic, now arriving about 23:30 with maybe 20% left; Van 7 must be full by 5am"


def _utc(d: date, hhmm: str, next_day: bool = False) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.combine(d + timedelta(days=int(next_day)), time(h, m), tzinfo=LONDON).astimezone(timezone.utc)


def fleet(d: date = D) -> tuple[list[Vehicle], dict[int, Session]]:
    vehicles, sessions = [], {}
    for i, (name, arr, dep, soc0, soc1) in enumerate(FLEET, start=1):
        vehicles.append(Vehicle(id=i, site_id=1, name=name, battery_kwh=75, max_charge_kw=22))
        sessions[i] = Session(id=i, vehicle_id=i, date=d, arrival_at=_utc(d, arr),
                              departure_at=_utc(d, dep, next_day=True), arrival_soc=soc0, target_soc=soc1)
    return vehicles, sessions


def out(*proposals: dict) -> str:
    return json.dumps({"proposals": list(proposals)})


def P(vehicle, field, value, quote):
    return {"vehicle": vehicle, "field": field, "value": value, "quote": quote}


def run(llm_output: str, text: str = TEXT, d: date = D) -> list[Proposal]:
    vehicles, sessions = fleet(d)
    return parse_and_validate(text, llm_output, vehicles, sessions, now=NOW)


HAPPY = out(
    P("Van 03", "arrival_at", "23:30", "now arriving about 23:30"),
    P("Van 03", "arrival_soc", "20%", "maybe 20% left"),
    P("Van 07", "target_soc", "100%", "Van 7 must be full by 5am"),
    P("Van 07", "departure_at", "05:00", "Van 7 must be full by 5am"),
)


# ---------- happy paths ----------

def test_happy_path_time_and_soc():
    rows = run(HAPPY)
    assert [r.status for r in rows] == ["pending"] * 4, [r.error for r in rows]
    arr, soc, tgt, dep = rows
    assert (arr.vehicle_id, arr.field, arr.value) == (3, "arrival_at", "2026-11-04T23:30:00+00:00")
    assert (soc.vehicle_id, soc.field, soc.value) == (3, "arrival_soc", "0.2")
    assert (tgt.vehicle_id, tgt.value) == (7, "1.0")
    assert dep.value == "2026-11-05T05:00:00+00:00"  # departure 05:00 -> next morning
    for r in rows:
        assert r.raw_text == TEXT and r.created_at == NOW and r.error is None and r.id is None
    assert arr.quote == "now arriving about 23:30"


def test_bst_conversion_and_morning_arrival():
    summer = date(2026, 7, 1)  # BST = UTC+1
    text = "Van 1 back at 00:30. Van 2 leaves 13:00 tomorrow"
    rows = run(out(P("Van 01", "arrival_at", "00:30", "Van 1 back at 00:30")), text, summer)
    assert rows[0].status == "pending", rows[0].error
    assert rows[0].value == "2026-07-01T23:30:00+00:00"  # 00:30 BST on 2 July


@pytest.mark.parametrize("raw,expected", [("20%", "0.2"), ("0.2", "0.2"), ("20", "0.2"), ("95 %", "0.95"),
                                          ("1", "1.0")])
def test_soc_formats(raw, expected):
    assert repr(llm._parse_soc("target_soc", raw)) == expected


def test_code_fenced_json():
    rows = run("```json\n" + HAPPY + "\n```")
    assert [r.status for r in rows] == ["pending"] * 4
    rows = run("Here you go:\n```\n" + HAPPY + "\n```\n")
    assert len(rows) == 4 and all(r.status == "pending" for r in rows)


def test_empty_proposals():
    assert run('{"proposals": []}') == []


# ---------- schema failures: one invalid row ----------

def _single_schema_failure(rows):
    assert len(rows) == 1
    r = rows[0]
    assert r.status == "invalid" and r.error.startswith("LLM output failed schema validation: ")
    assert r.raw_text == TEXT and r.created_at == NOW
    assert r.vehicle_id is None and r.field is None and r.value is None and r.quote is None
    return r


def test_malformed_json():
    r = _single_schema_failure(run('{"proposals": [{"vehicle": "Van 03", '))
    assert "json" in r.error.lower()


def test_prose_instead_of_json():
    _single_schema_failure(run("Sure! Van 3 arrives at 23:30."))


def test_extra_keys_rejected():
    bad = P("Van 03", "arrival_at", "23:30", "now arriving about 23:30") | {"confidence": 0.9}
    r = _single_schema_failure(run(out(bad)))
    assert "confidence" in r.error and "Extra inputs" in r.error
    _single_schema_failure(run(json.dumps({"proposals": [], "notes": "x"})))


def test_unknown_field_and_non_string_value_rejected():
    _single_schema_failure(run(out(P("Van 03", "battery_kwh", "60", "Van 3"))))
    _single_schema_failure(run(out(P("Van 03", "arrival_soc", 0.2, "maybe 20% left"))))  # strict: str only
    _single_schema_failure(run(json.dumps([P("Van 03", "arrival_soc", "20%", "maybe 20% left")])))


# ---------- per-proposal checks ----------

def test_unknown_vehicle():
    rows = run(out(P("Van 12", "arrival_at", "23:30", "now arriving about 23:30"),
                   P("Van 03", "arrival_soc", "20%", "maybe 20% left")))
    assert rows[0].status == "invalid" and "unknown vehicle 'Van 12'" in rows[0].error
    assert rows[0].vehicle_id is None and rows[0].field == "arrival_at"
    assert rows[1].status == "pending"  # one bad proposal does not sink the others


@pytest.mark.parametrize("name", ["van 3", "VAN 03", "Van3", "van-03", "3"])
def test_normalised_vehicle_name(name):
    rows = run(out(P(name, "arrival_at", "23:30", "now arriving about 23:30")))
    assert rows[0].status == "pending", rows[0].error
    assert rows[0].vehicle_id == 3


def test_ambiguous_vehicle():
    vehicles, sessions = fleet()
    vehicles.append(Vehicle(id=9, site_id=1, name="van 3", battery_kwh=60, max_charge_kw=11))
    rows = parse_and_validate(TEXT, out(P("Van 3", "arrival_at", "23:30", "arriving about 23:30")),
                              vehicles, sessions, now=NOW)
    assert rows[0].status == "invalid" and rows[0].error.startswith("ambiguous vehicle 'Van 3'")


def test_vehicle_without_session():
    vehicles, sessions = fleet()
    del sessions[3]
    rows = parse_and_validate(TEXT, out(P("Van 03", "arrival_at", "23:30", "arriving about 23:30")),
                              vehicles, sessions, now=NOW)
    assert rows[0].status == "invalid" and rows[0].error == "Van 03 has no session for this date"
    assert rows[0].vehicle_id == 3


@pytest.mark.parametrize("value", ["150%", "150", "-10%", "lots"])
def test_bad_soc(value):
    rows = run(out(P("Van 03", "arrival_soc", value, "maybe 20% left")))
    assert rows[0].status == "invalid" and "arrival_soc value" in rows[0].error
    assert rows[0].value == value  # raw model value kept for display


@pytest.mark.parametrize("value", ["11:30pm", "24:00", "23:75", "tonight"])
def test_bad_time(value):
    rows = run(out(P("Van 03", "arrival_at", value, "now arriving about 23:30")))
    assert rows[0].status == "invalid" and "arrival_at value" in rows[0].error


def test_arrival_after_departure():
    # Van 07 departs 05:00; an arrival of 06:00 (next morning) is after it
    text = "Van 7 now not back until 06:00"
    rows = run(out(P("Van 07", "arrival_at", "06:00", "not back until 06:00")), text)
    assert rows[0].status == "invalid" and "would not be before departure" in rows[0].error
    # departure moved before current arrival (Van 07 arrives 23:30, leaving 22:00 same evening)
    rows = run(out(P("Van 07", "departure_at", "22:00", "Van 7")), "Van 7 leaves at 22:00")
    assert rows[0].status == "invalid" and "would not be before departure" in rows[0].error


def test_target_not_above_arrival_soc():
    text = "Van 8 arriving with 85% charge"
    rows = run(out(P("Van 08", "arrival_soc", "85%", "arriving with 85% charge")), text)  # target 80%
    assert rows[0].status == "invalid" and "target_soc 80% would not exceed arrival_soc 85%" in rows[0].error


def test_fabricated_quote():
    rows = run(out(P("Van 03", "arrival_at", "23:30", "Van 3 will arrive at 23:30")))
    assert rows[0].status == "invalid" and "does not appear in the original text" in rows[0].error
    rows = run(out(P("Van 03", "arrival_at", "23:30", "   ")))
    assert rows[0].status == "invalid" and "quote" in rows[0].error


def test_quote_case_and_whitespace_insensitive():
    rows = run(out(P("Van 03", "arrival_at", "23:30", "NOW  arriving\nabout 23:30")))
    assert rows[0].status == "pending", rows[0].error


def test_schema_models_are_strict():
    assert ProposalBatch.model_config["extra"] == "forbid"
    assert llm.ProposalLLM.model_config["strict"] is True


# ---------- call_runware / propose with a mocked Runware ----------

def runware_ok(text: str, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        task = json.loads(request.content)[0]
        return httpx.Response(200, json={"data": [{
            "taskUUID": task["taskUUID"], "taskType": "textInference", "text": text,
            "finishReason": "stop",
            "usage": {"promptTokens": 600, "completionTokens": 120, "totalTokens": 720,
                      "promptTokensDetails": {"cachedTokens": 0, "cacheCreationTokens": 0}},
            "cost": 0.0034}]})
    return handler


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("RUNWARE_API_KEY", "test-key")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    vehicles, sessions = fleet()
    with DBSession(engine) as s:
        for v in vehicles:
            s.add(v)
        s.commit()
        for x in sessions.values():
            s.add(x)
        s.commit()
        yield s


def test_call_runware_request_shape(monkeypatch):
    monkeypatch.setenv("RUNWARE_API_KEY", "test-key")
    seen = []
    with httpx.Client(transport=httpx.MockTransport(runware_ok("```json\n{}\n```", seen))) as c:
        assert call_runware(TEXT, ["Van 01", "Van 02"], client=c) == "```json\n{}\n```"
    req = seen[0]
    assert str(req.url) == "https://api.runware.ai/v1"
    assert req.headers["authorization"] == "Bearer test-key"
    body = json.loads(req.content)
    assert isinstance(body, list) and len(body) == 1
    task = body[0]
    assert task["taskType"] == "textInference" and task["model"] == llm.RUNWARE_MODEL
    assert task["messages"] == [{"role": "user", "content": TEXT}]
    sp = task["settings"]["systemPrompt"]
    assert "Van 01" in sp and "Van 02" in sp and all(f in sp for f in llm.ALLOWED_FIELDS)


def test_propose_saves_rows(db):
    seen = []
    with httpx.Client(transport=httpx.MockTransport(runware_ok("```json\n" + HAPPY + "\n```", seen))) as c:
        rows = propose(db, TEXT, D, client=c)
    assert len(rows) == 4 and all(r.id is not None and r.status == "pending" for r in rows)
    stored = db.exec(select(Proposal).order_by(Proposal.id)).all()
    assert [(p.vehicle_id, p.field) for p in stored] == [
        (3, "arrival_at"), (3, "arrival_soc"), (7, "target_soc"), (7, "departure_at")]
    assert "Van 08" in json.loads(seen[0].content)[0]["settings"]["systemPrompt"]


def test_propose_http_failure(db):
    def boom(request):
        return httpx.Response(500, json={"errors": [{"code": "internalError", "message": "upstream exploded"}]})
    with httpx.Client(transport=httpx.MockTransport(boom)) as c:
        rows = propose(db, TEXT, D, client=c)
    assert len(rows) == 1 and rows[0].id is not None
    assert rows[0].status == "invalid"
    assert rows[0].error == "LLM call failed: Runware HTTP 500: upstream exploded"
    assert rows[0].raw_text == TEXT and rows[0].vehicle_id is None


def test_propose_timeout(db):
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)
    with httpx.Client(transport=httpx.MockTransport(slow)) as c:
        rows = propose(db, TEXT, D, client=c)
    assert rows[0].status == "invalid" and rows[0].error.startswith("LLM call failed: Runware request timed out")


def test_call_runware_raises_on_http_error(monkeypatch):
    monkeypatch.setenv("RUNWARE_API_KEY", "test-key")
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="unauthorized"))) as c:
        with pytest.raises(llm.RunwareError, match="HTTP 401"):
            call_runware(TEXT, ["Van 01"], client=c)


# ---------- regressions from the review ----------

def test_irrelevant_quote_is_rejected():
    """A real substring that says nothing about the change must not validate (hallucination / injection)."""
    [r] = run(out(P("Van 03", "target_soc", "100%", "Van 3")), text="Van 3 arrives 23:30")
    assert r.status == "invalid" and "does not support" in r.error


def test_vehicle_not_named_in_text_is_rejected():
    """The LLM cannot invent a change for a van the note never mentions."""
    [r] = run(out(P("Van 05", "arrival_soc", "20%", "20%")), text="Van 3 is back with 20% left")
    assert r.status == "invalid" and "not mentioned" in r.error


def test_huge_digit_vehicle_name_does_not_crash():
    """A pathological vehicle string becomes an invalid row instead of a 500."""
    [r] = run(out(P("Van " + "9" * 5000, "arrival_soc", "20%", "20%")))
    assert r.status == "invalid"


@pytest.mark.parametrize("value", ["1.5", "1.99"])
def test_ambiguous_bare_soc_rejected(value):
    """A bare SoC between 1 and 2 is neither a clear fraction nor a clear percent."""
    [r] = run(out(P("Van 03", "arrival_soc", value, "maybe 20% left")))
    assert r.status == "invalid" and "ambiguous" in r.error


def test_time_in_spring_forward_gap_rejected():
    """01:30 does not exist in London on 29 Mar 2026."""
    d = date(2026, 3, 28)
    [r] = run(out(P("Van 03", "arrival_at", "01:30", "arriving 01:30")), text="Van 3 arriving 01:30", d=d)
    assert r.status == "invalid" and "clock change" in r.error


def test_malformed_runware_payload_becomes_invalid_row():
    """An unexpected response shape is stored as one invalid proposal, not raised."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"errors": {"oops": 1}}))
    with DBSession(engine) as db, httpx.Client(transport=transport) as client:
        rows = propose(db, "Van 3 at 20%", D, client=client)
    assert len(rows) == 1 and rows[0].status == "invalid" and rows[0].error.startswith("LLM call failed")
