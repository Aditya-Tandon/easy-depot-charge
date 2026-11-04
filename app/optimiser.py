"""Overnight charging optimiser. PURE: no DB, no network, no file I/O.

Contract (agreed before implementation; tests are written against this):

Time is discretised into N slots of `slot_h` hours (default 0.5). Slot t covers
[t*slot_h, (t+1)*slot_h). Slot indices are integers relative to the planning
window start.

A van is plugged in for slots arrival_slot <= t < departure_slot.
Energy drawn from the grid in slot t is p[v,t] * slot_h (kWh).
Energy reaching the battery is eta * p[v,t] * slot_h.
need_kwh is energy required IN THE BATTERY, so grid energy = need_kwh / eta.

Prices are in pence/kWh (grid side). Costs returned are in GBP.
Carbon (optional) is g/kWh (grid side). Carbon returned in kg.

Optimised plan: subject to
  0 <= p[v,t] <= max_kw            while plugged in, else p = 0
  sum_v p[v,t] <= site_limit_kw    for every t
  eta * sum_t p[v,t] * slot_h + unmet[v] >= need_kwh[v]
  unmet[v] >= 0                    (so the LP always solves; any shortfall
                                    is reported per van, never silently dropped)
optimise LEXICOGRAPHICALLY:
  (a) minimise total unmet kWh  sum_v unmet[v];
  (b) then minimise the worst fractional site shortfall
        max_v (unmet[v] - rate_short[v]) / need_kwh[v]
      where rate_short[v] = max(0, need - eta * max_kw * hours) is the part of the
      shortfall the van could never avoid (reasons 1/2; = need for no-window vans);
  (c) then minimise cost  sum_t price[t] * sum_v p[v,t] * slot_h.
Delivered energy is never more than need (no overcharging beyond need).

Baseline ("charge on arrival"): process slots in time order; in each slot,
vans that are plugged in and not yet full are served in arrival order
(ties: input order), each at min(max_kw, remaining_need/(eta*slot_h),
remaining site capacity). Same prices, same site limit.

Infeasibility reasons (VanResult.reason, None if fully met), checked in order:
  1. window: departure_slot <= arrival_slot, or window outside the horizon
     -> "no charging window: ..."
  2. rate:   eta * max_kw * hours_plugged_in < need_kwh
     -> "needs X kWh; max R kW x H h x eta E = Y kWh -> short Z kWh"
  3. site:   individually feasible but unmet > tol after optimisation
     -> "site limit L kW contended HH:MM-HH:MM: short Z kWh" (slot-index based
        label if no clock times given; see `slot_labels`)
  A rate-limited van (reason 2) that is ALSO squeezed by the site limit keeps its
  rate reason, and if its actual shortfall exceeds the rate shortfall by > 0.05 kWh
  gets "; site limit L kW contended HH:MM-HH:MM: further short W kWh" appended
  (W = actual shortfall - rate shortfall).
Vans failing 1 or 2 are still charged as much as possible (best effort).
Tolerance for "met": 1e-6 kWh (use 1e-3 when comparing in tests).

Clarifications (binding for implementation and tests):
  - All numbers inside reason strings are formatted with 1 decimal ({:.1f}),
    e.g. "short 40.8 kWh". Tests should assert on the reason's leading phrase
    ("no charging window", "needs ", "site limit") and the "short Z kWh" part.
  - The plug-in window is clipped to [0, N). Hours in reason 2 = clipped slots * slot_h.
  - Prices may be NEGATIVE (Agile does this). The optimiser must still never
    deliver more than need_kwh to any van (constraint, not just objective).
  - status: "empty" if no vans; "partial" if any van has a reason; else "optimal".
  - cost_gbp = sum_t price[t] * site_load_kw[t] * slot_h / 100.
    carbon_kg = sum_t carbon[t] * site_load_kw[t] * slot_h / 1000, None if no carbon.
    Same formulas for baseline_cost_gbp / baseline_carbon_kg.
  - PlanOut.vans and PlanOut.baseline are in the same order as the input vans.
  - Baseline vans get reason None if fully met; otherwise any non-empty reason.
  - Optimised cost is always <= baseline cost when every van is met in both.
  - Contended span: the slots in the van's (clipped) window where site load is at
    the limit (within tol), first..last; label is start of first slot to END of
    last slot ("18:30-19:30" for slots 1..2 of 18:00,18:30,19:00,19:30). If the
    last contended slot is the last label, the end is extrapolated by slot_h.
  - Fairness (b) is min-max, not full leximin: once the worst-off van is fixed,
    vans better off than it may still be split by cost. Rate-limited / no-window
    vans count only their site-attributable shortfall in (b), so an unavoidable
    shortfall never makes (b) vacuous for the rest of the fleet.
  - Implementation of the lexicographic order: stage 1 solves (a)+(c) with an
    unmet penalty that provably dominates any price/eta (so it is exactly (a)
    then (c)); if its unmet equals the sum of rate shortfalls, (b) is already 0
    and that plan is final (one solve). Otherwise stage 2 minimises (b) with
    total unmet fixed at the stage-1 optimum, and stage 3 minimises cost+penalty
    with (a) and (b) fixed. Each fixed optimum is relaxed by rel 1e-7 + abs 1e-6.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class VanIn:
    id: int
    arrival_slot: int        # first slot plugged in (inclusive)
    departure_slot: int      # first slot NOT plugged in (exclusive)
    need_kwh: float          # energy required in the battery
    max_kw: float            # max charger/vehicle rate (grid side)


@dataclass
class VanResult:
    id: int
    need_kwh: float
    delivered_kwh: float     # energy into battery (after eta)
    power_kw: list[float]    # length N, grid-side kW per slot
    reason: str | None = None


@dataclass
class PlanOut:
    status: str              # "optimal" | "partial" (some vans short) | "empty"
    vans: list[VanResult]
    cost_gbp: float
    carbon_kg: float | None
    site_load_kw: list[float]                    # length N
    baseline: list[VanResult] = field(default_factory=list)
    baseline_cost_gbp: float = 0.0
    baseline_carbon_kg: float | None = None

    @property
    def savings_gbp(self) -> float:
        return self.baseline_cost_gbp - self.cost_gbp


_TOL = 1e-6          # kWh tolerance for "met" / site limit binding
_ROUND = 9           # decimal places kept on power values


def _window(v: VanIn, n: int) -> tuple[int, int]:
    """Plug-in window clipped to [0, n). Empty if hi <= lo."""
    return max(0, v.arrival_slot), min(n, v.departure_slot)


def _f(x: float) -> str:
    return f"{x:.1f}"


def _rate_cap(v: VanIn, n: int, eta: float, slot_h: float) -> float:
    """Max battery kWh the van could take on its own (0 for an empty window)."""
    lo, hi = _window(v, n)
    if v.departure_slot <= v.arrival_slot or hi <= lo:
        return 0.0
    return eta * max(0.0, v.max_kw) * (hi - lo) * slot_h


def _pre_reason(v: VanIn, n: int, eta: float, slot_h: float) -> str | None:
    """Reasons 1 (window) and 2 (rate), independent of other vans."""
    need = v.need_kwh
    if need <= _TOL:
        return None
    lo, hi = _window(v, n)
    if v.departure_slot <= v.arrival_slot:
        return (f"no charging window: departure slot {v.departure_slot} <= "
                f"arrival slot {v.arrival_slot}; short {_f(need)} kWh")
    if hi <= lo:
        return (f"no charging window: slots {v.arrival_slot}-{v.departure_slot} "
                f"outside horizon 0-{n}; short {_f(need)} kWh")
    hours = (hi - lo) * slot_h
    max_kw = max(0.0, v.max_kw)
    cap = eta * max_kw * hours
    if cap < need - _TOL:
        return (f"needs {_f(need)} kWh; max {_f(max_kw)} kW x {_f(hours)} h x "
                f"eta {_f(eta)} = {_f(cap)} kWh -> short {_f(need - cap)} kWh")
    return None


def _label_span(first: int, last: int, labels: list[str] | None, slot_h: float) -> str:
    """Label for slots first..last inclusive: 'HH:MM-HH:MM' (end = end of last slot)."""
    if not labels or last >= len(labels):
        return f"slot {first}-{last}"
    start = labels[first]
    if last + 1 < len(labels):
        end = labels[last + 1]
    else:
        end = labels[last]
        try:  # extrapolate end-of-slot time if the label is HH:MM
            hh, mm = labels[last].split(":")
            mins = (int(hh) * 60 + int(mm) + round(slot_h * 60)) % (24 * 60)
            end = f"{mins // 60:02d}:{mins % 60:02d}"
        except ValueError:
            pass
    return f"{start}-{end}"


def _site_clause(v: VanIn, site_load: list[float], site_limit: float,
                 labels: list[str] | None, slot_h: float) -> str:
    """'site limit L kW contended HH:MM-HH:MM' for the van's contended slots."""
    lo, hi = _window(v, len(site_load))
    contended = [t for t in range(lo, hi) if site_load[t] >= site_limit - _TOL]
    first, last = (contended[0], contended[-1]) if contended else (lo, hi - 1)
    span = _label_span(first, last, labels, slot_h)
    return f"site limit {_f(site_limit)} kW contended {span}"


_FURTHER_MIN = 0.05  # kWh of extra (site) shortfall before a rate reason mentions it


def _results(vans, power, n, eta, slot_h, site_limit, labels):
    """Build VanResults + site load from a (V, n) power array."""
    power = np.where(power < 0, 0.0, power)
    power = np.round(power, _ROUND) + 0.0  # +0.0 drops -0.0
    site_load = [float(x) for x in np.round(power.sum(axis=0), _ROUND)] if len(vans) else [0.0] * n
    out = []
    for i, v in enumerate(vans):
        row = [float(x) for x in power[i]]
        delivered = eta * sum(row) * slot_h
        reason = _pre_reason(v, n, eta, slot_h)
        need = max(0.0, v.need_kwh)
        short = need - delivered
        if reason is None:
            if short > _TOL:
                clause = _site_clause(v, site_load, site_limit, labels, slot_h)
                reason = f"{clause}: short {_f(short)} kWh"
        elif reason.startswith("needs "):
            further = short - (need - _rate_cap(v, n, eta, slot_h))
            if further > _FURTHER_MIN:
                clause = _site_clause(v, site_load, site_limit, labels, slot_h)
                reason += f"; {clause}: further short {_f(further)} kWh"
        out.append(VanResult(id=v.id, need_kwh=v.need_kwh, delivered_kwh=delivered,
                             power_kw=row, reason=reason))
    return out, site_load


def _cost_carbon(site_load, prices, carbon, slot_h):
    cost = sum(p * l * slot_h for p, l in zip(prices, site_load)) / 100.0
    co2 = None
    if carbon is not None:
        co2 = sum(c * l * slot_h for c, l in zip(carbon, site_load)) / 1000.0
    return cost, co2


def _validate(prices, eta, slot_h, carbon=None, labels=None):
    n = len(prices)
    if eta <= 0 or slot_h <= 0:
        raise ValueError("eta and slot_h must be positive")
    if carbon is not None and len(carbon) != n:
        raise ValueError("carbon_g_per_kwh must have the same length as prices")
    if labels is not None and len(labels) < n:
        raise ValueError("slot_labels must have at least one label per slot")
    return n


def optimise(
    vans: list[VanIn],
    prices_p_per_kwh: list[float],
    site_limit_kw: float,
    eta: float = 0.9,
    slot_h: float = 0.5,
    carbon_g_per_kwh: list[float] | None = None,
    slot_labels: list[str] | None = None,   # e.g. ["18:00","18:30",...], used in reasons
) -> PlanOut:
    """Return optimised plan + charge-on-arrival baseline. See module docstring."""
    prices = [float(x) for x in prices_p_per_kwh]
    n = _validate(prices, eta, slot_h, carbon_g_per_kwh, slot_labels)
    site_limit = max(0.0, float(site_limit_kw))
    vans = list(vans)

    if not vans:
        zeros = [0.0] * n
        return PlanOut(status="empty", vans=[], cost_gbp=0.0,
                       carbon_kg=0.0 if carbon_g_per_kwh is not None else None,
                       site_load_kw=zeros, baseline=[], baseline_cost_gbp=0.0,
                       baseline_carbon_kg=0.0 if carbon_g_per_kwh is not None else None)

    # Decision variables: p[v,t] for each plugged-in (v,t) pair (vans with need > 0
    # and positive max_kw only), followed by unmet[v] for every van.
    V = len(vans)
    pv, pt = [], []
    for i, v in enumerate(vans):
        if v.need_kwh <= 0 or v.max_kw <= 0:
            continue
        lo, hi = _window(v, n)
        for t in range(lo, hi):
            pv.append(i)
            pt.append(t)
    P = len(pv)
    pv_a = np.asarray(pv, dtype=int)
    pt_a = np.asarray(pt, dtype=int)

    power = np.zeros((V, n))
    if P > 0:
        need = np.array([max(0.0, v.need_kwh) for v in vans])
        rate_short = np.array([max(0.0, need[i] - _rate_cap(v, n, eta, slot_h))
                               for i, v in enumerate(vans)])
        x = _solve_lexicographic(pv_a, pt_a, np.array([v.max_kw for v in vans], dtype=float),
                                 need, rate_short, np.asarray(prices), n, eta, slot_h,
                                 site_limit)
        ub = np.array([vans[i].max_kw for i in pv], dtype=float)
        power[pv_a, pt_a] = np.clip(x, 0.0, ub)

    results, site_load = _results(vans, power, n, eta, slot_h, site_limit, slot_labels)
    cost, co2 = _cost_carbon(site_load, prices, carbon_g_per_kwh, slot_h)

    base = baseline(vans, prices, site_limit_kw, eta=eta, slot_h=slot_h)
    base_load = [sum(r.power_kw[t] for r in base) for t in range(n)]
    base_cost, base_co2 = _cost_carbon(base_load, prices, carbon_g_per_kwh, slot_h)

    status = "partial" if any(r.reason for r in results) else "optimal"
    return PlanOut(status=status, vans=results, cost_gbp=cost, carbon_kg=co2,
                   site_load_kw=site_load, baseline=base,
                   baseline_cost_gbp=base_cost, baseline_carbon_kg=base_co2)


_REL, _ABS = 1e-7, 1e-6   # slack when fixing a previous stage's optimum


def _relax(opt: float) -> float:
    return opt + _REL * abs(opt) + _ABS


def _linprog(c, A, b, bounds):
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"charging LP failed: {res.message}")
    return res


def _solve_lexicographic(pv_a, pt_a, max_kw, need, rate_short, price_a, n, eta,
                         slot_h, site_limit):
    """Return p (length P) for the lexicographic objective in the module docstring.

    Variables: p[k] for each plugged-in (van, slot) pair k, then unmet[v] (V), then z.
    """
    P, V = len(pv_a), len(need)
    idx, uidx, zcol = np.arange(P), P + np.arange(V), P + V
    e = eta * slot_h
    # Site limit: sum_v p[v,t] <= L                         (n rows)
    # Need:      -e*sum_t p[v,t] - unmet[v] <= -need[v]     (V rows)
    # No over:    e*sum_t p[v,t] <= need[v]                 (V rows)
    rows = np.concatenate([pt_a, n + pv_a, n + V + pv_a, n + np.arange(V)])
    cols = np.concatenate([idx, idx, idx, uidx])
    vals = np.concatenate([np.ones(P), np.full(P, -e), np.full(P, e), -np.ones(V)])
    m0 = n + 2 * V
    b0 = np.concatenate([np.full(n, site_limit), -need, need])
    ub = np.concatenate([max_kw[pv_a], np.full(V, np.inf), [0.0]])  # z fixed to 0 in stage 1
    bounds = np.column_stack([np.zeros(P + V + 1), ub])

    # Stage 1: (a) then (c). The penalty per battery kWh unmet exceeds any
    # |price|/eta, so no cost saving can ever buy a kWh of unmet.
    penalty = (1e4 + 100.0 * float(np.max(np.abs(price_a)))) / eta
    c3 = np.concatenate([price_a[pt_a] * slot_h, np.full(V, penalty), [0.0]])
    A1 = coo_matrix((vals, (rows, cols)), shape=(m0, P + V + 1)).tocsr()
    res = _linprog(c3, A1, b0, bounds)
    unmet_opt = float(res.x[uidx].sum())
    if unmet_opt <= rate_short.sum() + _ABS:
        return res.x[:P]  # every van gets all it physically can: (b) = 0 already

    # Stage 2: (b) with (a) fixed.   sum_v unmet[v] <= U*  (1 row)
    #   unmet[v] - need[v] * z <= rate_short[v]  for need[v] > 0  (fairness rows)
    fv = np.flatnonzero(need > _TOL)
    rows2 = np.concatenate([rows, np.full(V, m0), m0 + 1 + np.arange(len(fv)),
                            m0 + 1 + np.arange(len(fv))])
    cols2 = np.concatenate([cols, uidx, uidx[fv], np.full(len(fv), zcol)])
    vals2 = np.concatenate([vals, np.ones(V), np.ones(len(fv)), -need[fv]])
    A2 = coo_matrix((vals2, (rows2, cols2)), shape=(m0 + 1 + len(fv), P + V + 1)).tocsr()
    b2 = np.concatenate([b0, [_relax(unmet_opt)], rate_short[fv]])
    bounds[zcol, 1] = np.inf
    c2 = np.zeros(P + V + 1)
    c2[zcol] = 1.0
    res = _linprog(c2, A2, b2, bounds)

    # Stage 3: (c) with (a) and (b) fixed (penalty kept so unmet is not shaved
    # up to the tolerance just to save cost).
    bounds[zcol, 1] = _relax(float(res.x[zcol]))
    res = _linprog(c3, A2, b2, bounds)
    return res.x[:P]


def baseline(
    vans: list[VanIn],
    prices_p_per_kwh: list[float],
    site_limit_kw: float,
    eta: float = 0.9,
    slot_h: float = 0.5,
) -> list[VanResult]:
    """Charge-on-arrival under the same site limit. See module docstring."""
    n = _validate(prices_p_per_kwh, eta, slot_h)
    site_limit = max(0.0, float(site_limit_kw))
    vans = list(vans)
    V = len(vans)
    power = np.zeros((V, n))
    remaining = [max(0.0, v.need_kwh) for v in vans]
    windows = [_window(v, n) for v in vans]
    order = sorted(range(V), key=lambda i: (vans[i].arrival_slot, i))
    for t in range(n):
        cap = site_limit
        for i in order:
            lo, hi = windows[i]
            if not (lo <= t < hi) or remaining[i] <= _TOL or cap <= 0:
                continue
            kw = min(max(0.0, vans[i].max_kw), remaining[i] / (eta * slot_h), cap)
            if kw <= 0:
                continue
            power[i, t] = kw
            cap -= kw
            remaining[i] -= eta * kw * slot_h
    results, _ = _results(vans, power, n, eta, slot_h, site_limit, None)
    return results
