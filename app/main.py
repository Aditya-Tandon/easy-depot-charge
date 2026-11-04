import json
import math
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session as DBSession, col, func, select

from app.db import get_db
from app.models import Allocation, Plan, PriceSlot, Proposal, Session, Site, Vehicle, VanOutcome
from app.seed import seed, tonight
from app.timeutil import SLOT, fmt_local, to_utc

app = FastAPI(title="Overnight Charging Planner")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
ETA = 0.9


def _isolocal(v: str) -> str:
    try:
        return fmt_local(datetime.fromisoformat(v), "%a %H:%M")
    except (TypeError, ValueError):
        return v  # invalid rows keep the model's raw value


def _gbp(x: float | None) -> str:
    return "–" if x is None else ("−£" if x < 0 else "£") + f"{abs(x):,.2f}"


templates.env.filters.update(local=fmt_local, isolocal=_isolocal, gbp=_gbp)


# ---------- helpers ----------

def _window(d: Date) -> tuple[datetime, datetime, list[datetime]]:
    from app.ingest import tonight_window
    start, end = tonight_window(d)
    n = int((end - start) / SLOT)
    return start, end, [start + i * SLOT for i in range(n)]


def _sessions(db: DBSession, d: Date) -> list[tuple[Vehicle, Session]]:
    q = (select(Vehicle, Session).join(Session, Session.vehicle_id == Vehicle.id)
         .where(Session.date == d).order_by(Session.arrival_at, Vehicle.name))
    rows = db.exec(q).all()
    if not rows:  # new evening: create tonight's demo sessions on first visit
        seed(d)
        rows = db.exec(q).all()
    return list(rows)


def _prices(db: DBSession, d: Date, refresh: bool = False) -> list[PriceSlot]:
    start, end, slots = _window(d)
    rows = db.exec(select(PriceSlot).where(PriceSlot.start_at >= start, PriceSlot.start_at < end)
                   .order_by(PriceSlot.start_at)).all()
    if refresh or len(rows) < len(slots):
        from app.ingest import refresh_prices
        rows = refresh_prices(db, start, end)
    return list(rows)


def _need(v: Vehicle, s: Session) -> float:
    return max(0.0, (s.target_soc - s.arrival_soc) * v.battery_kwh)


def _redirect(msg: str | None = None, anchor: str = "") -> RedirectResponse:
    url = "/" + (f"?msg={quote(msg)}" if msg else "") + (f"#{anchor}" if anchor else "")
    return RedirectResponse(url, status_code=303)


FIELD_LABEL = {"arrival_at": "Arrival", "departure_at": "Departure",
               "arrival_soc": "Charge on arrival", "target_soc": "Target charge"}


def _fmt_field(field: str | None, value) -> str:
    if value is None or field is None:
        return "–"
    try:
        if field.endswith("_at"):
            return fmt_local(value if isinstance(value, datetime) else datetime.fromisoformat(value), "%a %H:%M")
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return str(value)


# ---------- page ----------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, plan: int | None = None, msg: str | None = None, db: DBSession = Depends(get_db)):
    d = tonight()
    rows = _sessions(db, d)
    site = db.exec(select(Site)).first()
    _, _, slots = _window(d)
    price_rows = db.exec(select(PriceSlot).where(PriceSlot.start_at >= slots[0], PriceSlot.start_at < slots[-1] + SLOT)
                         .order_by(PriceSlot.start_at)).all()
    by_start = {to_utc(p.start_at): p for p in price_rows}
    prices = [by_start.get(s) for s in slots]

    plans = db.exec(select(Plan).where(Plan.site_id == site.id, Plan.date == d)
                    .order_by(col(Plan.revision).desc())).all()
    selected = next((p for p in plans if p.id == plan), plans[0] if plans else None)
    published = next((p for p in plans if p.status == "published"), None)

    grid: dict[tuple[int, datetime], float] = {}
    outcomes: dict[int, VanOutcome] = {}
    shortfalls: dict[int, int] = {}
    if plans:
        for o in db.exec(select(VanOutcome).where(col(VanOutcome.plan_id).in_([p.id for p in plans]))):
            if o.infeasible_reason:
                shortfalls[o.plan_id] = shortfalls.get(o.plan_id, 0) + 1
            if selected and o.plan_id == selected.id:
                outcomes[o.vehicle_id] = o
    if selected:
        for a in db.exec(select(Allocation).where(Allocation.plan_id == selected.id)):
            grid[(a.vehicle_id, to_utc(a.slot_start))] = a.kw

    # stale = tonight's sessions no longer match what the selected plan was built from
    stale = bool(selected) and (
        set(outcomes) != {v.id for v, _ in rows}
        or any(abs(outcomes[v.id].need_kwh - _need(v, s)) > 0.05 for v, s in rows if v.id in outcomes)
        or db.exec(select(func.count()).select_from(Proposal).where(
            Proposal.status == "accepted", Proposal.created_at > selected.created_at)).one() > 0
    )

    fleet = []
    for v, s in rows:
        cells = []
        for t in slots:
            kw = grid.get((v.id, t), 0.0)
            plugged = to_utc(s.arrival_at) <= t < to_utc(s.departure_at)
            level = min(4, max(1, math.ceil(4 * kw / v.max_charge_kw - 1e-9))) if kw > 0 else 0
            cells.append({"kw": kw, "plugged": plugged, "level": level})
        fleet.append({"v": v, "s": s, "need": _need(v, s), "o": outcomes.get(v.id), "cells": cells})

    since = datetime.now(timezone.utc) - timedelta(hours=18)
    proposals = db.exec(select(Proposal).where(Proposal.created_at >= since)
                        .order_by(col(Proposal.created_at).desc()).limit(30)).all()
    sess_by_vid = {v.id: s for v, s in rows}
    names = {v.id: v.name for v, _ in rows}
    prop_view = []
    for p in proposals:
        s = sess_by_vid.get(p.vehicle_id)
        prop_view.append({
            "p": p, "van": names.get(p.vehicle_id, "–"), "label": FIELD_LABEL.get(p.field or "", p.field or "–"),
            "before": _fmt_field(p.field, getattr(s, p.field)) if s is not None and p.field in FIELD_LABEL else "–",
            "after": _fmt_field(p.field, p.value) if p.status != "invalid" else (p.value or "–"),
        })
    pending = [x for x in prop_view if x["p"].status == "pending"]
    history = [x for x in prop_view if x["p"].status != "pending"]

    opt_load = [round(sum(grid.get((v.id, t), 0.0) for v, _ in rows), 2) for t in slots]
    base_load = json.loads(selected.baseline_load_json) if selected and selected.baseline_load_json else None
    chart = {
        "labels": [fmt_local(t) for t in slots],
        "price": [p.price_p_per_kwh if p else None for p in prices],
        "carbon": [p.carbon_g_per_kwh if p else None for p in prices],
        "fallback": [bool(p and p.source != "octopus") for p in prices],
        "opt": opt_load if selected else None,
        "base": base_load,
        "limit": site.site_limit_kw,
    }
    fetched = max((to_utc(p.fetched_at) for p in price_rows), default=None)
    n_short = sum(1 for o in outcomes.values() if o.infeasible_reason)
    return templates.TemplateResponse(request, "index.html", dict(
        site=site, date=d, slots=slots, fleet=fleet, plans=plans, plan=selected, published=published,
        shortfalls=shortfalls, stale=stale, pending=pending, history=history, chart=chart,
        n_prices=sum(p is not None for p in prices), n_fallback=sum(chart["fallback"]), fetched=fetched,
        n_short=n_short, eta=ETA, msg=msg, window=(fmt_local(slots[0]), fmt_local(slots[-1] + SLOT)),
    ))


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------- actions ----------

@app.post("/prices/refresh")
def refresh(db: DBSession = Depends(get_db)):
    rows = _prices(db, tonight(), refresh=True)
    n_oct = sum(p.source == "octopus" for p in rows)
    note = "" if n_oct == len(rows) else f" ({len(rows) - n_oct} on fallback prices: Agile not yet published)"
    return _redirect(f"Prices updated: {n_oct} of {len(rows)} half-hours from Octopus Agile{note}.")


@app.post("/plan")
def make_plan(db: DBSession = Depends(get_db)):
    from app.optimiser import VanIn, optimise
    d = tonight()
    site = db.exec(select(Site)).first()
    rows = _sessions(db, d)
    start, _, slots = _window(d)
    by_start = {to_utc(p.start_at): p for p in _prices(db, d)}
    prices = [by_start[s].price_p_per_kwh for s in slots]
    carb = [by_start[s].carbon_g_per_kwh for s in slots]
    known = [c for c in carb if c is not None]
    carbon = [c if c is not None else sum(known) / len(known) for c in carb] if known else None

    vans = []
    for v, s in rows:
        arr = math.ceil((to_utc(s.arrival_at) - start) / SLOT)    # must be plugged in for the whole slot
        dep = math.floor((to_utc(s.departure_at) - start) / SLOT)
        vans.append(VanIn(id=v.id, arrival_slot=arr, departure_slot=dep, need_kwh=_need(v, s), max_kw=v.max_charge_kw))
    out = optimise(vans, prices, site.site_limit_kw, eta=ETA, slot_h=0.5, carbon_g_per_kwh=carbon,
                   slot_labels=[fmt_local(s) for s in slots])
    base_load = [round(sum(b.power_kw[t] for b in out.baseline), 3) for t in range(len(slots))]

    rev = (db.exec(select(func.max(Plan.revision)).where(Plan.site_id == site.id, Plan.date == d)).one() or 0) + 1
    plan = Plan(site_id=site.id, date=d, revision=rev, status="draft", cost_gbp=out.cost_gbp,
                baseline_cost_gbp=out.baseline_cost_gbp, carbon_kg=out.carbon_kg,
                baseline_carbon_kg=out.baseline_carbon_kg, baseline_load_json=json.dumps(base_load),
                created_at=datetime.now(timezone.utc))
    db.add(plan)
    db.flush()
    for vr in out.vans:
        for t, kw in enumerate(vr.power_kw):
            if kw > 1e-6:
                db.add(Allocation(plan_id=plan.id, vehicle_id=vr.id, slot_start=slots[t], kw=round(kw, 3)))
        db.add(VanOutcome(plan_id=plan.id, vehicle_id=vr.id, need_kwh=vr.need_kwh,
                          delivered_kwh=vr.delivered_kwh, infeasible_reason=vr.reason))
    db.commit()
    short = sum(1 for vr in out.vans if vr.reason)
    tail = f" {short} van{'s' if short != 1 else ''} can't reach target, see below." if short else " All vans reach target."
    return _redirect(f"Draft revision {rev} ready: {_gbp(out.cost_gbp)}, saving {_gbp(out.savings_gbp)} "
                     f"vs charge-on-arrival.{tail}")


@app.post("/plan/{plan_id}/publish")
def publish(plan_id: int, db: DBSession = Depends(get_db)):
    plan = db.get(Plan, plan_id)
    if not plan:
        return _redirect("That plan no longer exists.")
    for p in db.exec(select(Plan).where(Plan.date == plan.date, Plan.status == "published")):
        p.status = "superseded"
    plan.status = "published"
    db.commit()
    return _redirect(f"Revision {plan.revision} published as tonight's charging plan.")


@app.post("/proposals")
def create_proposals(text: str = Form(...), db: DBSession = Depends(get_db)):
    from app.llm import propose
    text = text.replace("\x00", "").strip()[:2000]
    if not text:
        return _redirect("Type an update first.", "updates")
    rows = propose(db, text, tonight())
    if not rows:
        return _redirect("No changes to tonight's vans found in that note.", "updates")
    ok = sum(r.status == "pending" for r in rows)
    bad = len(rows) - ok
    parts = [f"{ok} change{'s' if ok != 1 else ''} to review"] + ([f"{bad} rejected by validation"] if bad else [])
    return _redirect(", ".join(parts) + ".", "updates")


@app.post("/proposals/{pid}/{action}")
def decide(pid: int, action: str, db: DBSession = Depends(get_db)):
    p = db.get(Proposal, pid)
    if not p or p.status != "pending" or action not in ("accept", "reject"):
        return _redirect("That change was already handled.", "updates")
    if action == "reject":
        p.status = "rejected"
        db.commit()
        return _redirect("Change rejected.", "updates")
    s = db.exec(select(Session).where(Session.vehicle_id == p.vehicle_id, Session.date == tonight())).first()
    if not s:
        p.status, p.error = "invalid", "no session tonight for this vehicle"
        db.commit()
        return _redirect("Couldn't apply: that van has no session tonight.", "updates")
    val = datetime.fromisoformat(p.value) if p.field.endswith("_at") else float(p.value)
    setattr(s, p.field, val)
    if to_utc(s.arrival_at) >= to_utc(s.departure_at) or s.target_soc <= s.arrival_soc:
        db.rollback()
        p = db.get(Proposal, pid)
        p.status, p.error = "invalid", "no longer consistent with tonight's session"
        db.commit()
        return _redirect("Couldn't apply: it conflicts with the van's current session.", "updates")
    p.status = "accepted"
    db.commit()
    return _redirect("Change applied. Re-optimise to update the plan.", "updates")
