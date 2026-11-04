import math
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session as DBSession, col, func, select

from app.db import get_db
from app.models import Allocation, Plan, PriceSlot, Proposal, Session, Site, Vehicle, VanOutcome
from app.seed import LONDON, seed, tonight

app = FastAPI(title="Overnight Charging Planner")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
SLOT = timedelta(minutes=30)
ETA = 0.9


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


templates.env.filters["local"] = lambda dt, fmt="%H:%M": _utc(dt).astimezone(LONDON).strftime(fmt) if dt else ""


def _isolocal(v: str) -> str:
    try:
        return _utc(datetime.fromisoformat(v)).astimezone(LONDON).strftime("%a %H:%M")
    except ValueError:
        return v  # invalid rows keep the model's raw value


templates.env.filters["isolocal"] = _isolocal


# ---------- helpers ----------

def _window(d: Date) -> tuple[datetime, datetime, list[datetime]]:
    from app.ingest import tonight_window
    start, end = tonight_window(d)
    n = int((end - start) / SLOT)
    return start, end, [start + i * SLOT for i in range(n)]


def _sessions(db: DBSession, d: Date) -> list[tuple[Vehicle, Session]]:
    rows = db.exec(select(Vehicle, Session).join(Session, Session.vehicle_id == Vehicle.id)
                   .where(Session.date == d).order_by(Session.arrival_at)).all()
    if not rows:  # new evening: create tonight's demo sessions on first visit
        seed(d)
        rows = db.exec(select(Vehicle, Session).join(Session, Session.vehicle_id == Vehicle.id)
                       .where(Session.date == d).order_by(Session.arrival_at)).all()
    return list(rows)


def _prices(db: DBSession, d: Date, refresh: bool = False) -> list[PriceSlot]:
    start, end, slots = _window(d)
    rows = db.exec(select(PriceSlot).where(PriceSlot.start_at >= start, PriceSlot.start_at < end)
                   .order_by(PriceSlot.start_at)).all()
    if refresh or len(rows) < len(slots):
        from app.ingest import refresh_prices
        rows = refresh_prices(db, start, end)
    return list(rows)


def _latest_plan(db: DBSession, site_id: int, d: Date) -> Plan | None:
    return db.exec(select(Plan).where(Plan.site_id == site_id, Plan.date == d)
                   .order_by(col(Plan.revision).desc())).first()


def _redirect() -> RedirectResponse:
    return RedirectResponse("/", status_code=303)


# ---------- page ----------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: DBSession = Depends(get_db)):
    d = tonight()
    rows = _sessions(db, d)
    site = db.exec(select(Site)).first()
    _, _, slots = _window(d)
    prices = db.exec(select(PriceSlot).where(PriceSlot.start_at >= slots[0], PriceSlot.start_at < slots[-1] + SLOT)
                     .order_by(PriceSlot.start_at)).all()
    plan = _latest_plan(db, site.id, d) if site else None
    published = db.exec(select(Plan).where(Plan.date == d, Plan.status == "published")
                        .order_by(col(Plan.revision).desc())).first()
    grid, outcomes = {}, {}
    if plan:
        for a in db.exec(select(Allocation).where(Allocation.plan_id == plan.id)):
            grid[(a.vehicle_id, _utc(a.slot_start))] = a.kw
        outcomes = {o.vehicle_id: o for o in db.exec(select(VanOutcome).where(VanOutcome.plan_id == plan.id))}
    since = datetime.now(timezone.utc) - timedelta(hours=18)
    proposals = db.exec(select(Proposal).where(Proposal.created_at >= since)
                        .order_by(col(Proposal.created_at).desc()).limit(20)).all()
    names = {v.id: v.name for v, _ in rows}
    max_price = max((abs(p.price_p_per_kwh) for p in prices), default=1) or 1
    max_carbon = max((p.carbon_g_per_kwh or 0 for p in prices), default=1) or 1
    load = {s: sum(grid.get((v.id, s), 0) for v, _ in rows) for s in slots}
    return templates.TemplateResponse(request, "index.html", dict(
        site=site, date=d, rows=rows, slots=[_utc(s) for s in slots], prices=prices, plan=plan, published=published,
        grid=grid, outcomes=outcomes, proposals=proposals, names=names, load=load,
        max_price=max_price, max_carbon=max_carbon, eta=ETA,
        n_fallback=sum(p.source != "octopus" for p in prices),
    ))


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------- actions ----------

@app.post("/prices/refresh")
def refresh(db: DBSession = Depends(get_db)):
    _prices(db, tonight(), refresh=True)
    return _redirect()


@app.post("/plan")
def make_plan(db: DBSession = Depends(get_db)):
    from app.optimiser import VanIn, optimise
    d = tonight()
    site = db.exec(select(Site)).first()
    rows = _sessions(db, d)
    start, _, slots = _window(d)
    price_rows = _prices(db, d)
    by_start = {_utc(p.start_at): p for p in price_rows}
    prices = [by_start[s].price_p_per_kwh for s in slots]
    carb = [by_start[s].carbon_g_per_kwh for s in slots]
    known = [c for c in carb if c is not None]
    carbon = [c if c is not None else sum(known) / len(known) for c in carb] if known else None

    vans = []
    for v, s in rows:
        arr = math.ceil((_utc(s.arrival_at) - start) / SLOT)    # must be plugged in for the whole slot
        dep = math.floor((_utc(s.departure_at) - start) / SLOT)
        need = max(0.0, (s.target_soc - s.arrival_soc) * v.battery_kwh)
        vans.append(VanIn(id=v.id, arrival_slot=arr, departure_slot=dep, need_kwh=need, max_kw=v.max_charge_kw))
    labels = [s.astimezone(LONDON).strftime("%H:%M") for s in slots] + [(slots[-1] + SLOT).astimezone(LONDON).strftime("%H:%M")]
    out = optimise(vans, prices, site.site_limit_kw, eta=ETA, slot_h=0.5, carbon_g_per_kwh=carbon,
                   slot_labels=labels[:-1])

    rev = (db.exec(select(func.max(Plan.revision)).where(Plan.site_id == site.id, Plan.date == d)).one() or 0) + 1
    plan = Plan(site_id=site.id, date=d, revision=rev, status="draft", cost_gbp=out.cost_gbp,
                baseline_cost_gbp=out.baseline_cost_gbp, carbon_kg=out.carbon_kg,
                baseline_carbon_kg=out.baseline_carbon_kg, created_at=datetime.now(timezone.utc))
    db.add(plan)
    db.flush()
    for vr in out.vans:
        for t, kw in enumerate(vr.power_kw):
            if kw > 1e-6:
                db.add(Allocation(plan_id=plan.id, vehicle_id=vr.id, slot_start=slots[t], kw=round(kw, 3)))
        db.add(VanOutcome(plan_id=plan.id, vehicle_id=vr.id, need_kwh=vr.need_kwh,
                          delivered_kwh=vr.delivered_kwh, infeasible_reason=vr.reason))
    db.commit()
    return _redirect()


@app.post("/plan/{plan_id}/publish")
def publish(plan_id: int, db: DBSession = Depends(get_db)):
    plan = db.get(Plan, plan_id)
    if plan:
        for p in db.exec(select(Plan).where(Plan.date == plan.date, Plan.status == "published")):
            p.status = "superseded"
        plan.status = "published"
        db.commit()
    return _redirect()


@app.post("/proposals")
def create_proposals(text: str = Form(...), db: DBSession = Depends(get_db)):
    from app.llm import propose
    text = text.replace("\x00", "").strip()[:2000]
    if text:
        propose(db, text, tonight())
    return _redirect()


@app.post("/proposals/{pid}/{action}")
def decide(pid: int, action: str, db: DBSession = Depends(get_db)):
    p = db.get(Proposal, pid)
    if not p or p.status != "pending" or action not in ("accept", "reject"):
        return _redirect()
    if action == "reject":
        p.status = "rejected"
    else:
        s = db.exec(select(Session).where(Session.vehicle_id == p.vehicle_id, Session.date == tonight())).first()
        if not s:
            p.status, p.error = "invalid", "no session tonight for this vehicle"
        else:
            val = datetime.fromisoformat(p.value) if p.field.endswith("_at") else float(p.value)
            setattr(s, p.field, val)
            if _utc(s.arrival_at) >= _utc(s.departure_at) or s.target_soc <= s.arrival_soc:
                db.rollback()
                p = db.get(Proposal, pid)
                p.status, p.error = "invalid", "no longer consistent with tonight's session"
            else:
                p.status = "accepted"
    db.commit()
    return _redirect()
