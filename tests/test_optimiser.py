"""Blind contract tests for app.optimiser (written from the module docstring only).

Every expected number is derived by hand from the contract; the arithmetic is
shown in comments next to each assertion.

Conventions used throughout (from the contract):
  grid kWh in slot t   = p[t] * slot_h
  battery kWh in slot t = eta * p[t] * slot_h
  grid kWh to meet need = need_kwh / eta
  cost_gbp   = sum_t price[t] * site_load_kw[t] * slot_h / 100
  carbon_kg  = sum_t carbon[t] * site_load_kw[t] * slot_h / 1000
"""
from __future__ import annotations

import re

import pytest

from app.optimiser import PlanOut, VanIn, VanResult, baseline, optimise

TOL = 1e-3  # contract: "use 1e-3 when comparing in tests"


# --------------------------------------------------------------------------- helpers


def van(id: int, arrival: int, departure: int, need: float, max_kw: float) -> VanIn:
    """Short constructor so test set-ups fit on one line."""
    return VanIn(id=id, arrival_slot=arrival, departure_slot=departure,
                 need_kwh=need, max_kw=max_kw)


def approx(x: float):
    return pytest.approx(x, abs=TOL)


def short_kwh_in(reason: str) -> float:
    """Extract Z from the '... short Z kWh' part of a reason string."""
    m = re.search(r"short (-?[0-9]+\.[0-9]) kWh", reason)
    assert m, f"reason has no 'short Z kWh' part: {reason!r}"
    return float(m.group(1))


def grid_kwh(r: VanResult, slot_h: float) -> float:
    return sum(r.power_kw) * slot_h


def check_invariants(plan: PlanOut, vans: list[VanIn], prices: list[float],
                     site_limit_kw: float, eta: float, slot_h: float,
                     carbon: list[float] | None = None) -> None:
    """Properties every plan must satisfy, whatever the optimum looks like."""
    n = len(prices)
    assert len(plan.site_load_kw) == n
    assert [r.id for r in plan.vans] == [v.id for v in vans]
    assert [r.id for r in plan.baseline] == [v.id for v in vans]

    for results in (plan.vans, plan.baseline):
        for v, r in zip(vans, results):
            assert len(r.power_kw) == n
            lo, hi = max(v.arrival_slot, 0), min(v.departure_slot, n)
            for t, p in enumerate(r.power_kw):
                assert p >= -TOL, f"van {v.id} slot {t}: negative power {p}"
                assert p <= v.max_kw + TOL, f"van {v.id} slot {t}: {p} > max_kw"
                if not (lo <= t < hi):
                    assert p == approx(0.0), f"van {v.id} charging while unplugged at slot {t}"
            # delivered = eta * grid energy, and never above need
            assert r.delivered_kwh == approx(eta * grid_kwh(r, slot_h))
            assert r.delivered_kwh <= v.need_kwh + TOL
            assert r.need_kwh == approx(v.need_kwh)

        # site limit holds for the plan and for the baseline
        for t in range(n):
            assert sum(r.power_kw[t] for r in results) <= site_limit_kw + TOL

    # site_load_kw is the per-slot sum of the optimised van powers
    for t in range(n):
        assert plan.site_load_kw[t] == approx(sum(r.power_kw[t] for r in plan.vans))

    # cost / carbon formulas
    assert plan.cost_gbp == approx(
        sum(prices[t] * plan.site_load_kw[t] * slot_h for t in range(n)) / 100)
    base_load = [sum(r.power_kw[t] for r in plan.baseline) for t in range(n)]
    assert plan.baseline_cost_gbp == approx(
        sum(prices[t] * base_load[t] * slot_h for t in range(n)) / 100)
    if carbon is None:
        assert plan.carbon_kg is None
        assert plan.baseline_carbon_kg is None
    else:
        assert plan.carbon_kg == approx(
            sum(carbon[t] * plan.site_load_kw[t] * slot_h for t in range(n)) / 1000)
        assert plan.baseline_carbon_kg == approx(
            sum(carbon[t] * base_load[t] * slot_h for t in range(n)) / 1000)

    # status follows the reasons
    if not vans:
        assert plan.status == "empty"
    elif any(r.reason for r in plan.vans):
        assert plan.status == "partial"
    else:
        assert plan.status == "optimal"


# --------------------------------------------------------------------------- 1. trivial


def test_single_van_all_energy_in_cheapest_slot():
    """One van with plenty of rate puts all its energy in the single cheapest slot."""
    vans = [van(1, 0, 3, need=9.0, max_kw=50.0)]
    prices = [10.0, 1.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    # grid kWh = 9 / 0.9 = 10 kWh, all in slot 1 -> p = 10 / 0.5 = 20 kW
    assert r.power_kw == [approx(0.0), approx(20.0), approx(0.0)]
    assert grid_kwh(r, 0.5) == approx(10.0)
    assert r.delivered_kwh == approx(9.0)
    assert r.reason is None
    # cost = 1 p/kWh * 10 kWh = 10 p = 0.10 GBP
    assert plan.cost_gbp == approx(0.10)
    assert plan.status == "optimal"


def test_slot_length_one_hour():
    """With slot_h=1.0 the power equals grid kWh per slot (no 0.5 factor)."""
    vans = [van(1, 0, 3, need=9.0, max_kw=50.0)]
    prices = [10.0, 1.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=1.0)

    check_invariants(plan, vans, prices, 100.0, 0.9, 1.0)
    # grid kWh = 9 / 0.9 = 10 kWh in slot 1 -> p = 10 / 1.0 = 10 kW
    assert plan.vans[0].power_kw == [approx(0.0), approx(10.0), approx(0.0)]
    # cost = 1 p * 10 kWh = 0.10 GBP
    assert plan.cost_gbp == approx(0.10)


# --------------------------------------------------------------------------- 2. eta


@pytest.mark.parametrize(
    "eta, expected_grid_kwh, expected_cost_gbp",
    [
        (0.9, 20.0, 0.60),  # 18 / 0.9 = 20 kWh; 3 p * 20 kWh = 60 p = 0.60 GBP
        (1.0, 18.0, 0.54),  # 18 / 1.0 = 18 kWh; 3 p * 18 kWh = 54 p = 0.54 GBP
    ],
)
def test_grid_energy_is_need_over_eta(eta, expected_grid_kwh, expected_cost_gbp):
    """Grid energy drawn equals need/eta and delivered battery energy equals need."""
    vans = [van(1, 0, 2, need=18.0, max_kw=50.0)]
    prices = [3.0, 3.0]  # flat price: schedule not unique, so only totals are checked
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=eta, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, eta, 0.5)
    r = plan.vans[0]
    assert grid_kwh(r, 0.5) == approx(expected_grid_kwh)
    assert sum(plan.site_load_kw) * 0.5 == approx(expected_grid_kwh)
    assert r.delivered_kwh == approx(18.0)
    assert plan.cost_gbp == approx(expected_cost_gbp)


# --------------------------------------------------------------------------- 3. site limit


def test_site_limit_binds_in_cheapest_slot():
    """Two vans competing for the cheap slot are capped by the site limit, overflow goes to next-cheapest."""
    vans = [van(1, 0, 3, need=9.0, max_kw=20.0),
            van(2, 0, 3, need=9.0, max_kw=20.0)]
    prices = [10.0, 1.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=30.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 30.0, 0.9, 0.5)
    # Total grid need = 2 * 9/0.9 = 20 kWh.
    # Slot 1 (1 p) holds at most 30 kW * 0.5 h = 15 kWh  -> site load 30 kW
    # Remaining 5 kWh in slot 2 (5 p)                    -> site load 10 kW
    # (Per-van split is not unique; the site load is.)
    assert plan.site_load_kw == [approx(0.0), approx(30.0), approx(10.0)]
    for r in plan.vans:
        assert r.delivered_kwh == approx(9.0)
        assert r.reason is None
    # cost = (15 kWh * 1 p + 5 kWh * 5 p) / 100 = 40 p = 0.40 GBP
    assert plan.cost_gbp == approx(0.40)
    assert plan.status == "optimal"


# --------------------------------------------------------------------------- 4. per-van rate / window


def test_per_van_rate_limit_and_window():
    """Power never exceeds max_kw and is zero outside [arrival, departure), even if cheaper there."""
    # Slots 0, 4, 5 are very cheap but the van is only plugged in for slots 1..3.
    vans = [van(1, 1, 4, need=6.3, max_kw=7.0)]
    prices = [0.1, 10.0, 9.0, 8.0, 0.1, 0.1]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    # grid kWh = 6.3 / 0.9 = 7 kWh; one slot at max gives 7 kW * 0.5 h = 3.5 kWh
    # -> needs 2 full slots: the two cheapest in-window slots are 3 (8 p) and 2 (9 p)
    assert r.power_kw == [approx(0.0), approx(0.0), approx(7.0),
                          approx(7.0), approx(0.0), approx(0.0)]
    assert r.delivered_kwh == approx(6.3)
    # cost = (3.5 * 9 + 3.5 * 8) / 100 = (31.5 + 28) / 100 = 0.595 GBP
    assert plan.cost_gbp == approx(0.595)


# --------------------------------------------------------------------------- 5. reason 1: window


@pytest.mark.parametrize(
    "arrival, departure",
    [
        (2, 2),    # departure == arrival
        (3, 1),    # departure before arrival
        (10, 12),  # window entirely after the 4-slot horizon
    ],
)
def test_no_charging_window(arrival, departure):
    """A van with an empty (or out-of-horizon) window gets reason 'no charging window' and no power."""
    vans = [van(1, arrival, departure, need=10.0, max_kw=20.0)]
    prices = [5.0, 5.0, 5.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    assert r.reason is not None and r.reason.startswith("no charging window")
    assert r.power_kw == [approx(0.0)] * 4
    assert r.delivered_kwh == approx(0.0)
    assert plan.status == "partial"
    assert plan.cost_gbp == approx(0.0)


# --------------------------------------------------------------------------- 6. reason 2: rate


def test_rate_limited_van_reason_and_best_effort():
    """A van that cannot physically reach need is charged at max rate throughout and reports the shortfall."""
    vans = [van(1, 0, 2, need=20.0, max_kw=10.0)]
    prices = [5.0, 5.0, 5.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    # hours plugged in = 2 slots * 0.5 h = 1.0 h
    # max deliverable Y = 0.9 * 10 kW * 1.0 h = 9.0 kWh -> short Z = 20 - 9 = 11.0 kWh
    assert r.reason is not None and r.reason.startswith("needs ")
    assert "short 11.0 kWh" in r.reason
    # best effort: max rate in every plugged-in slot, nothing outside
    assert r.power_kw == [approx(10.0), approx(10.0), approx(0.0), approx(0.0)]
    assert r.delivered_kwh == approx(9.0)
    assert plan.status == "partial"
    # cost = 5 p * (10 kW * 1.0 h) = 50 p = 0.50 GBP
    assert plan.cost_gbp == approx(0.50)
    # baseline also can't meet it -> some non-empty reason
    assert plan.baseline[0].reason


def test_rate_check_uses_window_clipped_to_horizon():
    """Hours in the rate check count only slots inside [0, N) (window is clipped)."""
    # Window 2..10 on a 4-slot horizon clips to slots 2, 3 -> 1.0 h
    vans = [van(1, 2, 10, need=20.0, max_kw=10.0)]
    prices = [5.0, 5.0, 5.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    # Y = 0.9 * 10 kW * 1.0 h = 9.0 kWh -> short 20 - 9 = 11.0 kWh
    assert r.reason is not None and r.reason.startswith("needs ")
    assert "short 11.0 kWh" in r.reason
    assert r.power_kw == [approx(0.0), approx(0.0), approx(10.0), approx(10.0)]


def test_clipped_window_still_feasible():
    """A window that overhangs the horizon is clipped, and a need that fits the clipped window is met."""
    # Window 2..10 clips to slots 2, 3; need 4.5 kWh -> grid 5 kWh, capacity 10 kW * 1.0 h = 10 kWh
    vans = [van(1, 2, 10, need=4.5, max_kw=10.0)]
    prices = [1.0, 1.0, 7.0, 3.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    assert r.reason is None
    # 5 kWh fits exactly in slot 3 (3 p) at 10 kW * 0.5 h -> p = [0, 0, 0, 10]
    assert r.power_kw == [approx(0.0), approx(0.0), approx(0.0), approx(10.0)]
    # cost = 3 p * 5 kWh = 15 p = 0.15 GBP
    assert plan.cost_gbp == approx(0.15)


# --------------------------------------------------------------------------- 7. reason 3: site


def test_site_contention_reason_and_total_shortfall():
    """Vans individually feasible but jointly over the site limit get 'site limit' reasons with correct shortfall."""
    # Each van alone: 0.9 * 10 kW * 1.0 h = 9.0 kWh >= 8.0 kWh -> feasible on its own.
    vans = [van(1, 0, 2, need=8.0, max_kw=10.0),
            van(2, 0, 2, need=8.0, max_kw=10.0)]
    prices = [5.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=10.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 10.0, 0.9, 0.5)
    # Jointly: site can supply 10 kW * 1.0 h = 10 kWh grid -> 9.0 kWh battery in total.
    # Unmet is heavily penalised, so the site runs flat out in both slots.
    assert plan.site_load_kw == [approx(10.0), approx(10.0)]
    total_delivered = sum(r.delivered_kwh for r in plan.vans)
    assert total_delivered == approx(9.0)
    # total shortfall = 16 - 9 = 7.0 kWh (split between vans is not unique)
    assert sum(v.need_kwh - r.delivered_kwh for v, r in zip(vans, plan.vans)) == approx(7.0)

    short_vans = [r for r in plan.vans if r.reason]
    assert short_vans, "at least one van must be reported short"
    for v, r in zip(vans, plan.vans):
        unmet = v.need_kwh - r.delivered_kwh
        if unmet > TOL:
            assert r.reason is not None and r.reason.startswith("site limit")
            assert "10.0 kW" in r.reason  # L formatted {:.1f}
            assert short_kwh_in(r.reason) == pytest.approx(unmet, abs=0.051)
        else:
            assert r.reason is None
    assert plan.status == "partial"


def test_site_contention_reason_uses_slot_labels():
    """When slot_labels are given, the site-limit reason names the contended time using those labels."""
    vans = [van(1, 0, 2, need=8.0, max_kw=10.0),
            van(2, 0, 2, need=8.0, max_kw=10.0)]
    prices = [5.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=10.0, eta=0.9, slot_h=0.5,
                    slot_labels=["18:00", "18:30"])

    # Both slots are at the limit, so contention starts at slot 0 = "18:00".
    reasons = [r.reason for r in plan.vans if r.reason]
    assert reasons
    for reason in reasons:
        assert reason.startswith("site limit")
        assert "18:00" in reason


def test_reason_checks_are_ordered_window_before_rate():
    """An empty window wins over rate infeasibility (reason 1 is checked before reason 2)."""
    # need 100 kWh is also rate-infeasible, but the window is empty.
    vans = [van(1, 2, 2, need=100.0, max_kw=1.0)]
    plan = optimise(vans, [5.0, 5.0, 5.0], site_limit_kw=100.0, eta=0.9, slot_h=0.5)
    assert plan.vans[0].reason.startswith("no charging window")


# --------------------------------------------------------------------------- 8. negative prices


def test_negative_prices_never_overcharge():
    """With all-negative prices the optimiser still delivers exactly need, not more."""
    vans = [van(1, 0, 4, need=9.0, max_kw=50.0)]
    prices = [-10.0, -5.0, -20.0, -1.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    assert r.delivered_kwh == approx(9.0)
    # grid 10 kWh all in slot 2 (-20 p): p = 10 / 0.5 = 20 kW
    assert r.power_kw == [approx(0.0), approx(0.0), approx(20.0), approx(0.0)]
    # cost = -20 p * 10 kWh = -200 p = -2.00 GBP
    assert plan.cost_gbp == approx(-2.00)
    for b in plan.baseline:
        assert b.delivered_kwh <= 9.0 + TOL


def test_negative_prices_multi_van_no_overcharge():
    """Several vans under negative prices: no van ever receives more than its need."""
    vans = [van(1, 0, 4, need=5.0, max_kw=22.0),
            van(2, 1, 4, need=12.0, max_kw=11.0),
            van(3, 0, 2, need=0.0, max_kw=50.0)]
    prices = [-3.0, -8.0, -8.0, -1.0]
    plan = optimise(vans, prices, site_limit_kw=40.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 40.0, 0.9, 0.5)
    for v, r in zip(vans, plan.vans):
        assert r.delivered_kwh == approx(v.need_kwh)
    # van 3 has need 0 -> no power at all, even at negative prices
    assert plan.vans[2].power_kw == [approx(0.0)] * 4


# --------------------------------------------------------------------------- 9. baseline


def _staggered_case():
    # Input order is [A, B] but B arrives first -> B must be served first.
    vans = [van(1, 1, 6, need=9.0, max_kw=12.0),   # A: arrives slot 1
            van(2, 0, 6, need=9.0, max_kw=12.0)]   # B: arrives slot 0
    prices = [10.0, 20.0, 30.0, 1.0, 1.0, 1.0]
    return vans, prices, 16.0


def test_baseline_charge_on_arrival_exact_schedule():
    """Baseline serves earliest arrival first, each at min(max_kw, remaining need rate, site capacity)."""
    vans, prices, limit = _staggered_case()
    res = baseline(vans, prices, site_limit_kw=limit, eta=0.9, slot_h=0.5)
    # eta * slot_h = 0.45, so "remaining_need / 0.45" is the kW that finishes the van this slot.
    #
    # slot 0: only B plugged. B = min(12, 9/0.45=20, 16) = 12 kW -> +5.4 kWh, B left 3.6
    # slot 1: B first: min(12, 3.6/0.45=8, 16) = 8 kW -> B done; site left 16-8 = 8
    #         A: min(12, 9/0.45=20, 8) = 8 kW -> +3.6 kWh, A left 5.4
    # slot 2: A: min(12, 5.4/0.45=12, 16) = 12 kW -> A done
    a, b = res
    assert [r.id for r in res] == [1, 2]  # input order preserved
    assert a.power_kw == [approx(x) for x in [0.0, 8.0, 12.0, 0.0, 0.0, 0.0]]
    assert b.power_kw == [approx(x) for x in [12.0, 8.0, 0.0, 0.0, 0.0, 0.0]]
    assert a.delivered_kwh == approx(9.0)
    assert b.delivered_kwh == approx(9.0)
    assert a.reason is None and b.reason is None


def test_baseline_inside_plan_matches_and_cost():
    """PlanOut.baseline equals the standalone baseline, and baseline_cost_gbp follows the formula."""
    vans, prices, limit = _staggered_case()
    plan = optimise(vans, prices, site_limit_kw=limit, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, limit, 0.9, 0.5)
    a, b = plan.baseline
    assert a.power_kw == [approx(x) for x in [0.0, 8.0, 12.0, 0.0, 0.0, 0.0]]
    assert b.power_kw == [approx(x) for x in [12.0, 8.0, 0.0, 0.0, 0.0, 0.0]]
    # baseline site load = [12, 16, 12, 0, 0, 0]
    # cost = (10*12 + 20*16 + 30*12) * 0.5 / 100 = 800 * 0.5 / 100 = 4.00 GBP
    assert plan.baseline_cost_gbp == approx(4.00)


def test_baseline_ties_broken_by_input_order():
    """Vans arriving in the same slot are served in input order by the baseline."""
    vans = [van(7, 0, 2, need=4.5, max_kw=10.0),
            van(3, 0, 2, need=4.5, max_kw=10.0)]
    res = baseline(vans, [1.0, 1.0], site_limit_kw=10.0, eta=0.9, slot_h=0.5)
    # slot 0: van 7 first: min(10, 4.5/0.45=10, 10) = 10 kW -> done; site full, van 3 gets 0
    # slot 1: van 3: min(10, 10, 10) = 10 kW -> done
    assert res[0].id == 7 and res[1].id == 3
    assert res[0].power_kw == [approx(10.0), approx(0.0)]
    assert res[1].power_kw == [approx(0.0), approx(10.0)]


def test_baseline_unmet_van_has_reason():
    """A baseline van that ends short gets a non-empty reason; a met one gets None."""
    vans = [van(1, 0, 1, need=10.0, max_kw=10.0),   # max 0.9*10*0.5 = 4.5 kWh < 10
            van(2, 0, 2, need=4.5, max_kw=10.0)]
    res = baseline(vans, [1.0, 1.0], site_limit_kw=100.0, eta=0.9, slot_h=0.5)
    # van 1: 10 kW in slot 0 -> 4.5 kWh delivered, short 5.5
    assert res[0].power_kw == [approx(10.0), approx(0.0)]
    assert res[0].delivered_kwh == approx(4.5)
    assert res[0].reason
    # van 2: min(10, 4.5/0.45=10, 90) = 10 kW in slot 0 -> done
    assert res[1].power_kw == [approx(10.0), approx(0.0)]
    assert res[1].reason is None


# --------------------------------------------------------------------------- 10. optimised vs baseline


def test_optimised_beats_baseline_and_savings():
    """When every van is met, optimised cost <= baseline cost and savings = baseline - optimised."""
    vans, prices, limit = _staggered_case()
    plan = optimise(vans, prices, site_limit_kw=limit, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, limit, 0.9, 0.5)
    # Optimised: 20 kWh grid total; slots 3..5 at 1 p hold 16 kW * 0.5 h * 3 = 24 kWh
    # and each van can take 12 kW * 0.5 h * 3 = 18 kWh >= 10 kWh there.
    # -> cost = 20 kWh * 1 p = 20 p = 0.20 GBP
    assert plan.cost_gbp == approx(0.20)
    assert plan.baseline_cost_gbp == approx(4.00)
    assert plan.cost_gbp <= plan.baseline_cost_gbp + TOL
    # savings = 4.00 - 0.20 = 3.80 GBP
    assert plan.savings_gbp == approx(3.80)
    assert plan.savings_gbp == approx(plan.baseline_cost_gbp - plan.cost_gbp)
    assert plan.status == "optimal"


def test_optimised_never_worse_than_baseline_site_case():
    """Optimised cost <= baseline cost on the site-limit scenario (all vans met in both)."""
    vans = [van(1, 0, 3, need=9.0, max_kw=20.0),
            van(2, 0, 3, need=9.0, max_kw=20.0)]
    prices = [10.0, 1.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=30.0, eta=0.9, slot_h=0.5)
    # Baseline: slot 0 van 1 = min(20, 9/0.45=20, 30) = 20 kW (done); van 2 = min(20, 20, 10) = 10 kW
    #           slot 1 van 2 = min(20, (9-4.5)/0.45=10, 30) = 10 kW (done)
    # baseline load [30, 10, 0] -> cost = (10*30 + 1*10) * 0.5 / 100 = 155 / 100 = 1.55 GBP
    assert plan.baseline_cost_gbp == approx(1.55)
    assert plan.cost_gbp == approx(0.40)
    assert plan.savings_gbp == approx(1.15)


# --------------------------------------------------------------------------- 11. carbon


def test_carbon_computed_when_given():
    """carbon_kg and baseline_carbon_kg follow sum(carbon * load * slot_h) / 1000."""
    vans = [van(1, 0, 3, need=9.0, max_kw=50.0)]
    prices = [10.0, 1.0, 5.0]
    carbon = [100.0, 200.0, 300.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5,
                    carbon_g_per_kwh=carbon)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5, carbon=carbon)
    # Optimised: 10 kWh in slot 1 -> 200 g * 10 kWh / 1000 = 2.0 kg
    assert plan.carbon_kg == approx(2.0)
    # Baseline: slot 0 = min(50, 9/0.45=20, 100) = 20 kW -> 10 kWh at 100 g = 1.0 kg
    assert plan.baseline[0].power_kw == [approx(20.0), approx(0.0), approx(0.0)]
    assert plan.baseline_carbon_kg == approx(1.0)
    # Baseline cost = 10 p * 10 kWh = 1.00 GBP
    assert plan.baseline_cost_gbp == approx(1.00)


def test_carbon_none_when_not_given():
    """Without carbon intensities, carbon_kg and baseline_carbon_kg are None."""
    plan = optimise([van(1, 0, 3, need=9.0, max_kw=50.0)], [10.0, 1.0, 5.0],
                    site_limit_kw=100.0)
    assert plan.carbon_kg is None
    assert plan.baseline_carbon_kg is None


# --------------------------------------------------------------------------- 12. empty fleet


def test_empty_fleet():
    """No vans -> status 'empty', zero cost, zero site load of length N."""
    plan = optimise([], [10.0, 1.0, 5.0], site_limit_kw=100.0)
    assert plan.status == "empty"
    assert plan.vans == []
    assert plan.baseline == []
    assert plan.cost_gbp == approx(0.0)
    assert plan.baseline_cost_gbp == approx(0.0)
    assert plan.savings_gbp == approx(0.0)
    assert plan.site_load_kw == [approx(0.0)] * 3


# --------------------------------------------------------------------------- 13. ordering / shape


def test_output_order_and_lengths_follow_input():
    """vans and baseline come back in input order (not by id or arrival), each power_kw has length N."""
    vans = [van(7, 3, 6, need=4.0, max_kw=11.0),
            van(3, 0, 6, need=6.0, max_kw=11.0),
            van(5, 1, 4, need=2.0, max_kw=7.0)]
    prices = [9.0, 4.0, 6.0, 2.0, 8.0, 3.0]
    plan = optimise(vans, prices, site_limit_kw=50.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 50.0, 0.9, 0.5)
    assert [r.id for r in plan.vans] == [7, 3, 5]
    assert [r.id for r in plan.baseline] == [7, 3, 5]
    assert all(len(r.power_kw) == 6 for r in plan.vans + plan.baseline)
    assert len(plan.site_load_kw) == 6
    standalone = baseline(vans, prices, site_limit_kw=50.0, eta=0.9, slot_h=0.5)
    assert [r.id for r in standalone] == [7, 3, 5]


# --------------------------------------------------------------------------- 14. zero need


@pytest.mark.parametrize("prices", [[5.0, 1.0, 3.0], [-5.0, -1.0, -3.0]])
def test_zero_need_gets_no_power(prices):
    """A van with need_kwh = 0 gets no power (even at negative prices) and reason None."""
    vans = [van(1, 0, 3, need=0.0, max_kw=22.0)]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    assert r.power_kw == [approx(0.0)] * 3
    assert r.delivered_kwh == approx(0.0)
    assert r.reason is None
    assert plan.baseline[0].power_kw == [approx(0.0)] * 3
    assert plan.baseline[0].reason is None
    assert plan.status == "optimal"
    assert plan.cost_gbp == approx(0.0)


# =========================================================================== limitations
# ---- limitations ----
# Tests added after the first review: combined rate+site shortfall, fair splitting
# of scarce energy (lexicographic objective), and reviewer-flagged gaps.

import random  # noqa: E402
import time  # noqa: E402


def further_short_in(reason: str) -> float | None:
    """Extract W from '...: further short W kWh', or None if the clause is absent."""
    m = re.search(r"; site limit ([0-9]+\.[0-9]) kW contended (.+?): further short ([0-9]+\.[0-9]) kWh$",
                  reason)
    return float(m.group(3)) if m else None


# ---- L1. combined rate + site shortfall


def test_rate_and_site_limited_reports_further_short():
    """A van both rate-limited and site-squeezed appends 'further short W kWh' to its rate reason."""
    # Each van alone: cap = 0.9 * 10 kW * 1.0 h = 9.0 kWh < 20 -> rate short 11.0 kWh.
    # Site 10 kW < 10 + 10: total grid 10 kW * 1.0 h = 10 kWh -> 9.0 kWh battery for both.
    # So jointly the vans are short 40 - 9 = 31 kWh, of which 22 is rate and 9 is site.
    vans = [van(1, 0, 2, need=20.0, max_kw=10.0),
            van(2, 0, 2, need=20.0, max_kw=10.0)]
    prices = [5.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=10.0, eta=0.9, slot_h=0.5,
                    slot_labels=["18:00", "18:30"])

    check_invariants(plan, vans, prices, 10.0, 0.9, 0.5)
    assert sum(r.delivered_kwh for r in plan.vans) == approx(9.0)
    clauses = 0
    for v, r in zip(vans, plan.vans):
        assert r.reason.startswith("needs ")
        assert short_kwh_in(r.reason) == 11.0          # rate part unchanged
        w = further_short_in(r.reason)
        if w is not None:
            clauses += 1
            assert "; site limit 10.0 kW contended 18:00-19:00: further short" in r.reason
        # rate short (exactly 11.0) + further short (1-dp) == actual shortfall
        assert 11.0 + (w or 0.0) == pytest.approx(v.need_kwh - r.delivered_kwh, abs=0.051)
    assert clauses >= 1  # 9 kWh of site shortfall must be attributed to someone


def test_rate_limited_but_not_site_squeezed_has_no_further_clause():
    """A rate-limited van with ample site capacity keeps the plain rate reason (no 'further short')."""
    vans = [van(1, 0, 2, need=20.0, max_kw=10.0)]
    plan = optimise(vans, [5.0, 5.0], site_limit_kw=100.0, eta=0.9, slot_h=0.5)
    # delivered = 9.0 = cap -> actual short 11.0 == rate short -> no clause
    assert plan.vans[0].reason.endswith("short 11.0 kWh")
    assert "further" not in plan.vans[0].reason


# ---- L2. fair split of scarce energy (lexicographic objective)


def test_fair_split_identical_vans_one_slot():
    """Two identical vans in one slot with half the site capacity they need each get the same energy."""
    # Each needs 4.5 kWh -> 4.5 / (0.9 * 0.5) = 10 kW for the one slot; site gives 10 kW total.
    # Min-unmet: site full (10 kW -> 4.5 kWh battery). Fair: 5 kW each -> 2.25 kWh each.
    vans = [van(1, 0, 1, need=4.5, max_kw=10.0),
            van(2, 0, 1, need=4.5, max_kw=10.0)]
    prices = [5.0]
    plan = optimise(vans, prices, site_limit_kw=10.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 10.0, 0.9, 0.5)
    a, b = plan.vans
    assert a.delivered_kwh == approx(2.25)
    assert b.delivered_kwh == approx(2.25)
    assert a.power_kw == [approx(5.0)] and b.power_kw == [approx(5.0)]
    # both short 4.5 - 2.25 = 2.25 -> "short 2.2" or "2.3" ({:.1f} of 2.25); allow either
    for r in plan.vans:
        assert r.reason.startswith("site limit")
        assert short_kwh_in(r.reason) == pytest.approx(2.25, abs=0.051)


def test_fair_split_equal_fractional_shortfall_and_total_unchanged():
    """Vans with different needs end with equal fractional shortfall; total delivered stays at the min-unmet optimum."""
    # Both plugged in slots 0..1 (1.0 h), max 20 kW (rate-feasible: 0.9*20*1 = 18 >= 9).
    # Site 10 kW * 1.0 h = 10 kWh grid -> 9.0 kWh battery total = min-unmet optimum.
    # Need 9 + 4.5 = 13.5 -> unmet 4.5. Equal fraction f: 9f + 4.5f = 4.5 -> f = 1/3.
    # A delivers 9 * 2/3 = 6.0 kWh, B delivers 4.5 * 2/3 = 3.0 kWh.
    vans = [van(1, 0, 2, need=9.0, max_kw=20.0),
            van(2, 0, 2, need=4.5, max_kw=20.0)]
    prices = [5.0, 7.0]
    plan = optimise(vans, prices, site_limit_kw=10.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 10.0, 0.9, 0.5)
    a, b = plan.vans
    assert a.delivered_kwh + b.delivered_kwh == approx(9.0)
    assert a.delivered_kwh == approx(6.0)
    assert b.delivered_kwh == approx(3.0)
    assert (9.0 - a.delivered_kwh) / 9.0 == approx((4.5 - b.delivered_kwh) / 4.5)
    assert plan.site_load_kw == [approx(10.0), approx(10.0)]


def test_fairness_does_not_trade_total_energy():
    """Fairness never reduces total delivered: a van alone in its window still gets everything it can."""
    # slot_h 1, eta 1, site 10 kW. A: slot 0 only, need 10. B: slots 0..1, need 10.
    # Min unmet = 0: A takes slot 0 (10 kWh), B takes slot 1 (10 kWh). All met, no reasons.
    vans = [van(1, 0, 1, need=10.0, max_kw=10.0),
            van(2, 0, 2, need=10.0, max_kw=10.0)]
    prices = [1.0, 9.0]  # B would prefer cheap slot 0, but cost is the last priority
    plan = optimise(vans, prices, site_limit_kw=10.0, eta=1.0, slot_h=1.0)

    check_invariants(plan, vans, prices, 10.0, 1.0, 1.0)
    assert plan.vans[0].power_kw == [approx(10.0), approx(0.0)]
    assert plan.vans[1].power_kw == [approx(0.0), approx(10.0)]
    assert all(r.reason is None for r in plan.vans)
    # cost = (1 * 10 + 9 * 10) / 100 = 1.00 GBP
    assert plan.cost_gbp == approx(1.00)


def test_fairness_ignores_unavoidable_rate_shortfall():
    """A rate-limited van does not make the fairness stage vacuous for the others (fraction is site-attributable)."""
    # slot_h 1, eta 1, one slot, site 10 kW.
    # C: rate-limited, max 2 kW, need 10 -> cap 2, rate short 8 (80% short no matter what).
    # A, B: need 8 each, max 10. Site 10 = C 2 + 8 left for A and B (need 16).
    # Min unmet: site full -> total delivered 10, unmet 26 - 10 = 16 (8 of it is rate).
    # Site-attributable fractions: C (u_C - 8)/10, A u_A/8, B u_B/8.
    # Let C get c in [0, 2]; A = B = (10 - c)/2 (symmetric is best for max(A, B)).
    #   frac_C = (2 - c)/10 (decreasing in c), frac_A = (8 - (10 - c)/2)/8 = (6 + c)/16 (increasing).
    # At c = 0: max(0.2, 0.375) = 0.375; they would cross only at c < 0 -> optimum c = 0.
    # Hence A = B = 5.0, C = 0.0 (C's 2 kWh is a site shortfall on top of its 8 kWh rate shortfall).
    # (With the literal unmet/need metric C is >= 80% short whatever happens, so the
    #  fairness stage would leave A/B anywhere in [1.6, 8.4].)
    vans = [van(1, 0, 1, need=8.0, max_kw=10.0),
            van(2, 0, 1, need=8.0, max_kw=10.0),
            van(3, 0, 1, need=10.0, max_kw=2.0)]
    prices = [5.0]
    plan = optimise(vans, prices, site_limit_kw=10.0, eta=1.0, slot_h=1.0)

    check_invariants(plan, vans, prices, 10.0, 1.0, 1.0)
    assert plan.vans[0].delivered_kwh == approx(5.0)
    assert plan.vans[1].delivered_kwh == approx(5.0)
    assert plan.vans[2].delivered_kwh == approx(0.0)
    # C: rate short 10 - 2 = 8.0, further short (10 - 0) - 8 = 2.0
    assert plan.vans[2].reason.startswith("needs 10.0 kWh")
    assert further_short_in(plan.vans[2].reason) == 2.0


def _random_fleet(rng: random.Random, n_vans: int, n_slots: int) -> tuple[list[VanIn], list[float]]:
    vans = []
    for i in range(n_vans):
        a = rng.randrange(0, n_slots - 2)
        d = rng.randrange(a + 2, n_slots + 1)
        vans.append(van(i, a, d, need=rng.uniform(5, 60), max_kw=rng.choice([7.0, 11.0, 22.0])))
    prices = [rng.uniform(-5, 35) for _ in range(n_slots)]
    return vans, prices


@pytest.mark.parametrize("n_vans, n_slots, budget_s", [(8, 28, 0.25), (50, 48, 2.5)])
def test_solve_time(n_vans, n_slots, budget_s):
    """Lexicographic solve stays fast (targets 50 ms / 500 ms; asserted at 5x for CI noise)."""
    rng = random.Random(7)
    vans, prices = _random_fleet(rng, n_vans, n_slots)
    limit = 0.1 * sum(v.max_kw for v in vans)  # tight: forces all 3 LP stages (site-attributable shortfall)
    optimise(vans, prices, site_limit_kw=limit)  # warm-up
    t0 = time.perf_counter()
    plan = optimise(vans, prices, site_limit_kw=limit)
    dt = time.perf_counter() - t0
    check_invariants(plan, vans, prices, limit, 0.9, 0.5)
    assert dt < budget_s, f"{n_vans}x{n_slots} took {dt * 1000:.0f} ms"


# ---- L3. reviewer-flagged gaps


def _two_contending_vans(arrival: int, departure: int) -> list[VanIn]:
    # Each alone: 0.9 * 10 kW * 1.0 h = 9.0 >= 8 -> feasible; jointly the 10 kW site gives 9.0 < 16.
    return [van(1, arrival, departure, need=8.0, max_kw=10.0),
            van(2, arrival, departure, need=8.0, max_kw=10.0)]


def test_site_contention_label_exact_span():
    """Contention over slots 1..2 is labelled from the start of slot 1 to the start of slot 3."""
    labels = ["18:00", "18:30", "19:00", "19:30"]
    vans = _two_contending_vans(1, 3)
    plan = optimise(vans, [5.0] * 4, site_limit_kw=10.0, eta=0.9, slot_h=0.5, slot_labels=labels)
    # site load [0, 10, 10, 0] -> contended slots 1..2 -> "18:30-19:30"
    assert plan.site_load_kw == [approx(0.0), approx(10.0), approx(10.0), approx(0.0)]
    reasons = [r.reason for r in plan.vans if r.reason]
    assert reasons  # 7 kWh short in total (which van is short is not tested here)
    for reason in reasons:
        assert reason.startswith("site limit")
        assert "contended 18:30-19:30:" in reason


def test_site_contention_label_extrapolates_last_slot_end():
    """Contention ending at the last slot gets an end label extrapolated by slot_h."""
    labels = ["18:00", "18:30", "19:00", "19:30"]
    vans = _two_contending_vans(2, 4)
    plan = optimise(vans, [5.0] * 4, site_limit_kw=10.0, eta=0.9, slot_h=0.5, slot_labels=labels)
    # contended slots 2..3 -> "19:00" to end of slot 3 = 19:30 + 30 min = "20:00"
    reasons = [r.reason for r in plan.vans if r.reason]
    assert reasons
    for reason in reasons:
        assert "contended 19:00-20:00:" in reason


def test_site_contention_label_single_last_slot_one_hour_slots():
    """A single contended final slot with slot_h=1.0 extrapolates the end by 60 minutes."""
    # One slot (slot 3) of 1 h: each van alone 0.9 * 10 * 1 = 9 >= 8; jointly 9 < 16.
    vans = _two_contending_vans(3, 4)
    plan = optimise(vans, [5.0] * 4, site_limit_kw=10.0, eta=0.9, slot_h=1.0,
                    slot_labels=["18:00", "19:00", "20:00", "21:00"])
    reasons = [r.reason for r in plan.vans if r.reason]
    assert reasons
    for reason in reasons:
        assert "contended 21:00-22:00:" in reason


def test_mixed_fleet_feasible_vans_unaffected():
    """No-window + rate-limited vans don't stop feasible vans from being met (reason None), status partial."""
    vans = [van(1, 3, 3, need=10.0, max_kw=20.0),     # no window
            van(2, 0, 2, need=20.0, max_kw=10.0),     # rate: 0.9*10*1.0 = 9 < 20 -> short 11.0
            van(3, 0, 4, need=9.0, max_kw=11.0),      # 0.9*11*2 = 19.8 >= 9
            van(4, 1, 4, need=4.5, max_kw=7.0)]       # 0.9*7*1.5 = 9.45 >= 4.5
    prices = [8.0, 3.0, 1.0, 6.0]
    plan = optimise(vans, prices, site_limit_kw=50.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 50.0, 0.9, 0.5)
    assert plan.vans[0].reason.startswith("no charging window")
    assert plan.vans[1].reason.startswith("needs ") and "short 11.0 kWh" in plan.vans[1].reason
    assert "further" not in plan.vans[1].reason
    assert plan.vans[2].reason is None and plan.vans[2].delivered_kwh == approx(9.0)
    assert plan.vans[3].reason is None and plan.vans[3].delivered_kwh == approx(4.5)
    assert plan.status == "partial"


def test_need_exactly_at_rate_cap_is_met():
    """need == eta * max_kw * hours is feasible: reason None."""
    # cap = 1.0 * 10 kW * (2 * 0.5 h) = 10.0 kWh == need
    vans = [van(1, 0, 2, need=10.0, max_kw=10.0)]
    plan = optimise(vans, [5.0, 5.0], site_limit_kw=100.0, eta=1.0, slot_h=0.5)
    assert plan.vans[0].reason is None
    assert plan.vans[0].delivered_kwh == approx(10.0)
    assert plan.status == "optimal"


def test_need_just_above_rate_cap_is_short():
    """need slightly above the cap reports the small rate shortfall."""
    # cap 10.0; need 10.08 -> short 0.08 -> "{:.1f}" = "0.1" (not a rounding tie); needs "10.1"
    vans = [van(1, 0, 2, need=10.08, max_kw=10.0)]
    plan = optimise(vans, [5.0, 5.0], site_limit_kw=100.0, eta=1.0, slot_h=0.5)
    r = plan.vans[0]
    assert r.reason.startswith("needs 10.1 kWh")
    assert r.reason.endswith("short 0.1 kWh")
    assert r.delivered_kwh == approx(10.0)


def test_negative_arrival_clipped_and_rate_hours_from_clipped_window():
    """arrival_slot < 0 is clipped to 0; the rate reason's hours use the clipped window."""
    # Window -3..2 clips to slots 0, 1 -> 2 * 0.5 = 1.0 h; cap 0.9 * 10 * 1.0 = 9.0 -> short 11.0
    vans = [van(1, -3, 2, need=20.0, max_kw=10.0)]
    prices = [5.0, 5.0, 5.0, 5.0]
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=0.5)

    check_invariants(plan, vans, prices, 100.0, 0.9, 0.5)
    r = plan.vans[0]
    assert "x 1.0 h x" in r.reason
    assert "= 9.0 kWh -> short 11.0 kWh" in r.reason
    assert r.power_kw == [approx(10.0), approx(10.0), approx(0.0), approx(0.0)]


@pytest.mark.parametrize(
    "slot_h, departure, need, hours_txt, cap_txt, short_txt",
    [
        (1.0, 2, 30.0, "2.0", "18.0", "12.0"),   # 2 slots * 1 h; 0.9*10*2 = 18; 30-18 = 12
        (0.25, 4, 20.0, "1.0", "9.0", "11.0"),   # 4 slots * 0.25 h; 0.9*10*1 = 9; 20-9 = 11
    ],
)
def test_rate_reason_hours_for_other_slot_lengths(slot_h, departure, need, hours_txt, cap_txt, short_txt):
    """Rate reason hours = clipped slots * slot_h for slot_h other than 0.5."""
    vans = [van(1, 0, departure, need=need, max_kw=10.0)]
    prices = [5.0] * 6
    plan = optimise(vans, prices, site_limit_kw=100.0, eta=0.9, slot_h=slot_h)

    check_invariants(plan, vans, prices, 100.0, 0.9, slot_h)
    reason = plan.vans[0].reason
    assert f"x {hours_txt} h x" in reason
    assert f"= {cap_txt} kWh -> short {short_txt} kWh" in reason


def test_random_instances_properties():
    """Seeded random fleets: invariants hold; min-unmet beats baseline; cost <= baseline when baseline meets all."""
    rng = random.Random(20261104)
    checked_cost = 0
    for _ in range(50):
        n = rng.randint(2, 8)
        slot_h = rng.choice([0.25, 0.5, 1.0])
        eta = rng.choice([0.85, 0.9, 1.0])
        vans = []
        for i in range(rng.randint(1, 4)):
            a = rng.randint(-2, n - 1)
            d = a + rng.choice([-1, 0, 1, 2, 3, 4, 6, 8])  # includes empty windows
            vans.append(van(i, a, d, need=round(rng.uniform(0, 8), 2),
                            max_kw=rng.choice([0.0, 7.0, 11.0, 22.0, 22.0])))
        prices = [round(rng.uniform(-5, 35), 2) for _ in range(n)]
        limit = round(rng.uniform(5, 50), 1)
        plan = optimise(vans, prices, site_limit_kw=limit, eta=eta, slot_h=slot_h)

        check_invariants(plan, vans, prices, limit, eta, slot_h)
        total_opt = sum(r.delivered_kwh for r in plan.vans)
        total_base = sum(r.delivered_kwh for r in plan.baseline)
        assert total_opt >= total_base - TOL  # (a) min total unmet
        if all(r.reason is None for r in plan.baseline):
            checked_cost += 1
            assert all(r.reason is None for r in plan.vans)
            assert plan.cost_gbp <= plan.baseline_cost_gbp + 1e-6
    assert checked_cost >= 5  # the cost property was actually exercised
