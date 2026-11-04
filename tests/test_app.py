"""End-to-end app flow on in-memory SQLite: no network, no Neon."""
import json
from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session as DBSession, SQLModel, create_engine, select

import app.ingest as ingest
import app.llm as llm
import app.main as main
import app.seed as seed_mod
from app.db import get_db
from app.models import Plan, PriceSlot, Proposal, Session, Vehicle

D = date(2026, 11, 4)


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(seed_mod, "engine", engine)
    monkeypatch.setattr(seed_mod, "init_db", lambda: None)
    monkeypatch.setattr(main, "tonight", lambda: D)

    def fake_refresh(db, start, end, client=None):
        rows = []
        for i, s in enumerate(ingest._slot_starts(start, end)):
            p = PriceSlot(start_at=s, price_p_per_kwh=30.0 if 4 <= i < 20 else 50.0, carbon_g_per_kwh=150.0,
                          source="octopus", source_url="test", fetched_at=datetime.now(timezone.utc))
            db.add(p)
            rows.append(p)
        db.commit()
        return rows
    monkeypatch.setattr(ingest, "refresh_prices", fake_refresh)

    def get_test_db():
        with DBSession(engine) as s:
            yield s
    main.app.dependency_overrides[get_db] = get_test_db
    yield TestClient(main.app), engine
    main.app.dependency_overrides.clear()


def test_full_flow(client, monkeypatch):
    c, engine = client
    assert c.get("/").status_code == 200  # auto-seeds tonight
    assert c.post("/prices/refresh").status_code == 200
    assert c.post("/plan").status_code == 200
    r = c.get("/")
    assert "Optimised cost" in r.text and "revision 1" in r.text
    with DBSession(engine) as s:
        plan = s.exec(select(Plan)).one()
        assert plan.cost_gbp <= plan.baseline_cost_gbp + 1e-6
    c.post(f"/plan/{plan.id}/publish")
    with DBSession(engine) as s:
        assert s.get(Plan, plan.id).status == "published"

    # LLM proposal → pending → accept → session updated
    canned = json.dumps({"proposals": [
        {"vehicle": "Van 03", "field": "arrival_soc", "value": "20%", "quote": "only 20% left"},
        {"vehicle": "Van 99", "field": "arrival_soc", "value": "20%", "quote": "only 20% left"},
    ]})
    monkeypatch.setattr(llm, "call_runware", lambda *a, **k: canned)
    c.post("/proposals", data={"text": "Van 03 is back with only 20% left"})
    with DBSession(engine) as s:
        props = s.exec(select(Proposal).order_by(Proposal.id)).all()
        assert [p.status for p in props] == ["pending", "invalid"]
        pid = props[0].id
    c.post(f"/proposals/{pid}/accept")
    with DBSession(engine) as s:
        v3 = s.exec(select(Vehicle).where(Vehicle.name == "Van 03")).one()
        sess = s.exec(select(Session).where(Session.vehicle_id == v3.id)).one()
        assert sess.arrival_soc == pytest.approx(0.2)
        assert s.get(Proposal, pid).status == "accepted"

    # re-plan creates revision 2 as a draft
    c.post("/plan")
    with DBSession(engine) as s:
        assert [p.revision for p in s.exec(select(Plan).order_by(Plan.revision))] == [1, 2]
