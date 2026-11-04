"""Tests for app.ingest. No real network: every request goes through httpx.MockTransport."""
import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app import ingest
from app.ingest import fetch_slots, refresh_prices, tonight_window
from app.models import PriceSlot

UTC = timezone.utc
START = datetime(2026, 11, 4, 18, 0, tzinfo=UTC)
END = datetime(2026, 11, 5, 8, 0, tzinfo=UTC)
N_SLOTS = 28


def _z(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _product(code, direction="IMPORT", available_from="2024-10-01T00:00:00+01:00", available_to=None,
             is_business=False):
    return {
        "code": code, "direction": direction, "full_name": code, "display_name": code,
        "is_variable": True, "is_green": False, "is_tracker": False, "is_prepay": False,
        "is_business": is_business, "is_restricted": False, "term": 12,
        "available_from": available_from, "available_to": available_to,
        "links": [{"href": f"https://api.octopus.energy/v1/products/{code}/", "method": "GET", "rel": "self"}],
        "brand": "OCTOPUS_ENERGY",
    }


# Two pages, so discovery has to follow `next`; the Agile import product is on page 2.
PRODUCTS_PAGE_1 = {
    "count": 4,
    "next": "https://api.octopus.energy/v1/products/?page=2",
    "previous": None,
    "results": [
        _product("VAR-22-11-01"),
        _product("AGILE-OUTGOING-19-05-13", direction="EXPORT", available_from="2018-01-01T00:00:00Z"),
    ],
}
PRODUCTS_PAGE_2 = {
    "count": 4,
    "next": None,
    "previous": "https://api.octopus.energy/v1/products/",
    "results": [
        _product("AGILE-18-02-21", available_from="2018-02-21T00:00:00Z", available_to="2022-01-01T00:00:00Z"),
        _product("AGILE-24-10-01"),
    ],
}


def _rates(start: datetime, n: int, base: float = 20.0) -> dict:
    """Octopus standard-unit-rates shape: newest first, like the real API."""
    results = []
    for i in range(n):
        t = start + timedelta(minutes=30 * i)
        v = base + i
        results.append({"value_exc_vat": round(v / 1.05, 2), "value_inc_vat": v,
                        "valid_from": _z(t), "valid_to": _z(t + timedelta(minutes=30)),
                        "payment_method": None})
    results.reverse()
    return {"count": len(results), "next": None, "previous": None, "results": results}


def _carbon(start: datetime, n: int, base: int = 100) -> dict:
    """Carbon Intensity regional shape. Like the real API it also includes the slot ending at `from`."""
    rows = []
    for i in range(-1, n):
        t = start + timedelta(minutes=30 * i)
        rows.append({"from": t.strftime("%Y-%m-%dT%H:%MZ"),
                     "to": (t + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                     "intensity": {"forecast": base + i, "index": "moderate"},
                     "generationmix": [{"fuel": "gas", "perc": 40.0}, {"fuel": "wind", "perc": 60.0}]})
    return {"data": {"regionid": 13, "dnoregion": "UKPN London", "shortname": "London", "data": rows}}


def make_client(rates=None, carbon=None, products=(PRODUCTS_PAGE_1, PRODUCTS_PAGE_2), log=None):
    """MockTransport client routing the three endpoints. A value of None -> HTTP 500."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = request.url
        if log is not None:
            log.append(str(url))
        if url.host == "api.octopus.energy" and url.path == "/v1/products/":
            page = products[1] if url.params.get("page") == "2" else products[0]
            return httpx.Response(200, json=page)
        if url.host == "api.octopus.energy" and url.path.endswith("/standard-unit-rates/"):
            if rates is None:
                return httpx.Response(500)
            # json.dumps (unlike httpx's json=) lets a test send a bare NaN/Infinity token, as a real body could.
            return httpx.Response(200, content=json.dumps(rates).encode(),
                                  headers={"content-type": "application/json"})
        if url.host == "api.carbonintensity.org.uk":
            return httpx.Response(200, json=carbon) if carbon is not None else httpx.Response(500)
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- discovery

def test_discovery_picks_current_agile_import_product():
    log = []
    with make_client(log=log) as c:
        code = ingest.discover_agile_product(c, now=datetime(2026, 11, 4, 12, tzinfo=UTC))
    assert code == "AGILE-24-10-01"  # not the EXPORT one, not the expired one, not VAR
    assert any("page=2" in u for u in log)  # followed pagination


def test_fetch_uses_discovered_product_and_region_c_url():
    log = []
    with make_client(rates=_rates(START, N_SLOTS), carbon=_carbon(START, N_SLOTS), log=log) as c:
        slots = fetch_slots(START, END, c)
    expected = ("https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/"
                "E-1R-AGILE-24-10-01-C/standard-unit-rates/"
                "?period_from=2026-11-04T18:00:00Z&period_to=2026-11-05T08:00:00Z")
    assert {s.source_url for s in slots} == {expected}
    assert expected in log
    assert "https://api.carbonintensity.org.uk/regional/intensity/2026-11-04T18:00Z/2026-11-05T08:00Z/regionid/13" in log


# --------------------------------------------------------------------------- merge

def test_merge_price_and_carbon_by_slot():
    with make_client(rates=_rates(START, N_SLOTS), carbon=_carbon(START, N_SLOTS)) as c:
        slots = fetch_slots(START, END, c)
    assert len(slots) == N_SLOTS
    assert [s.start_at for s in slots] == [START + timedelta(minutes=30 * i) for i in range(N_SLOTS)]
    for i, s in enumerate(slots):
        assert s.id is None
        assert s.start_at.tzinfo is not None and s.start_at.utcoffset() == timedelta(0)
        assert s.source == "octopus"
        assert s.price_p_per_kwh == 20.0 + i  # rates arrive newest-first; matched by time not order
        assert s.carbon_g_per_kwh == 100 + i  # the extra pre-window carbon row is ignored
        assert s.fetched_at.tzinfo is not None


def test_multi_slot_rate_expands_and_dd_preferred():
    rates = {"count": 3, "next": None, "previous": None, "results": [
        {"value_exc_vat": 0, "value_inc_vat": 99.0, "valid_from": _z(START), "valid_to": _z(START + timedelta(hours=1)),
         "payment_method": "NON_DIRECT_DEBIT"},
        {"value_exc_vat": 0, "value_inc_vat": 15.0, "valid_from": _z(START), "valid_to": _z(START + timedelta(hours=1)),
         "payment_method": "DIRECT_DEBIT"},
        {"value_exc_vat": 0, "value_inc_vat": 10.0, "valid_from": _z(START + timedelta(hours=1)), "valid_to": None,
         "payment_method": None},
    ]}
    with make_client(rates=rates, carbon=None) as c:
        slots = fetch_slots(START, START + timedelta(hours=2), c)
    assert [s.price_p_per_kwh for s in slots] == [15.0, 15.0, 10.0, 10.0]
    assert all(s.source == "octopus" for s in slots)


# --------------------------------------------------------------------------- fallback

def test_partial_octopus_coverage_marks_fallback_slots():
    # Agile for tomorrow not yet published: only the first 12 slots (until 00:00 UTC) have prices.
    with make_client(rates=_rates(START, 12), carbon=_carbon(START, 20)) as c:
        slots = fetch_slots(START, END, c)
    assert len(slots) == N_SLOTS
    octo, fb = slots[:12], slots[12:]
    assert all(s.source == "octopus" for s in octo)
    assert all(s.source == "fallback" and s.source_url == "synthetic:agile-like-curve" for s in fb)
    assert all(s.price_p_per_kwh == ingest.fallback_price(s.start_at) for s in fb)
    # carbon present for the first 20, None afterwards
    assert all(s.carbon_g_per_kwh is not None for s in slots[:20])
    assert all(s.carbon_g_per_kwh is None for s in slots[20:])


def test_fallback_curve_shape():
    london_peak = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)  # 17:00 GMT
    night = datetime(2026, 11, 5, 2, 0, tzinfo=UTC)
    summer_peak = datetime(2026, 7, 1, 15, 0, tzinfo=UTC)  # 16:00 BST -> peak
    assert ingest.fallback_price(london_peak) > ingest.fallback_price(night)
    assert ingest.fallback_price(summer_peak) == ingest.fallback_price(london_peak)
    assert ingest.fallback_price(night) == min(ingest.fallback_price(START + timedelta(minutes=30 * i))
                                               for i in range(48))


def test_total_network_failure_all_fallback_no_exception():
    def boom(request):
        raise httpx.ConnectError("no network", request=request)
    with httpx.Client(transport=httpx.MockTransport(boom)) as c:
        slots = fetch_slots(START, END, c)
    assert len(slots) == N_SLOTS
    assert all(s.source == "fallback" for s in slots)
    assert all(s.carbon_g_per_kwh is None for s in slots)


def test_non_200_and_garbage_json_fall_back():
    def handler(request):
        if request.url.path == "/v1/products/":
            return httpx.Response(200, content=b"<html>not json</html>")
        return httpx.Response(503)
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        slots = fetch_slots(START, END, c)
    assert all(s.source == "fallback" and s.carbon_g_per_kwh is None for s in slots)


def test_rates_endpoint_500_but_carbon_ok():
    with make_client(rates=None, carbon=_carbon(START, N_SLOTS)) as c:
        slots = fetch_slots(START, END, c)
    assert all(s.source == "fallback" for s in slots)
    assert all(s.carbon_g_per_kwh is not None for s in slots)


# --------------------------------------------------------------------------- tonight_window

@pytest.mark.parametrize("d, start, end, n", [
    # normal GMT night
    (date(2026, 11, 4), datetime(2026, 11, 4, 18, tzinfo=UTC), datetime(2026, 11, 5, 8, tzinfo=UTC), 28),
    # BST -> GMT overnight (clocks go back 02:00 BST, 25 Oct 2026): 15h window
    (date(2026, 10, 24), datetime(2026, 10, 24, 17, tzinfo=UTC), datetime(2026, 10, 25, 8, tzinfo=UTC), 30),
    # GMT -> BST overnight (clocks go forward 01:00 GMT, 29 Mar 2026): 13h window
    (date(2026, 3, 28), datetime(2026, 3, 28, 18, tzinfo=UTC), datetime(2026, 3, 29, 7, tzinfo=UTC), 26),
    # normal BST night
    (date(2026, 7, 1), datetime(2026, 7, 1, 17, tzinfo=UTC), datetime(2026, 7, 2, 7, tzinfo=UTC), 28),
])
def test_tonight_window(d, start, end, n):
    s, e = tonight_window(d)
    assert (s, e) == (start, end)
    assert s.utcoffset() == timedelta(0) and e.utcoffset() == timedelta(0)
    assert (e - s) == timedelta(minutes=30 * n)


def test_fetch_slots_across_dst_change_has_30_slots():
    s, e = tonight_window(date(2026, 10, 24))
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as c:
        slots = fetch_slots(s, e, c)
    assert len(slots) == 30
    assert len({x.start_at for x in slots}) == 30


# --------------------------------------------------------------------------- refresh_prices

@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_refresh_prices_upsert_no_duplicates_and_updates(db):
    with make_client(rates=_rates(START, N_SLOTS, base=20.0), carbon=_carbon(START, N_SLOTS)) as c:
        first = refresh_prices(db, START, END, c)
    assert len(first) == N_SLOTS
    assert all(r.id is not None for r in first)
    ids = [r.id for r in first]

    with make_client(rates=_rates(START, N_SLOTS, base=50.0), carbon=_carbon(START, N_SLOTS, base=300)) as c:
        second = refresh_prices(db, START, END, c)

    rows = db.exec(select(PriceSlot)).all()
    assert len(rows) == N_SLOTS
    assert [r.id for r in second] == ids
    assert [r.price_p_per_kwh for r in second] == [50.0 + i for i in range(N_SLOTS)]
    assert [r.carbon_g_per_kwh for r in second] == [300 + i for i in range(N_SLOTS)]
    # returned rows are sorted and tz-aware UTC even though SQLite drops tzinfo
    assert [r.start_at for r in second] == [START + timedelta(minutes=30 * i) for i in range(N_SLOTS)]
    assert all(r.start_at.tzinfo is not None for r in second)
    assert not db.dirty  # tz normalisation must not leave pending writes


def test_refresh_fallback_replaced_by_octopus_but_not_vice_versa(db):
    with make_client(rates=_rates(START, 12), carbon=_carbon(START, N_SLOTS)) as c:
        first = refresh_prices(db, START, END, c)
    assert sum(r.source == "fallback" for r in first) == 16

    # Later in the day the full night is published -> fallback rows upgraded.
    with make_client(rates=_rates(START, N_SLOTS, base=40.0), carbon=_carbon(START, N_SLOTS)) as c:
        second = refresh_prices(db, START, END, c)
    assert all(r.source == "octopus" for r in second)
    assert second[-1].price_p_per_kwh == 40.0 + N_SLOTS - 1

    # Outage on a later refresh: real prices and carbon are kept, not overwritten by synthetic ones.
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as c:
        third = refresh_prices(db, START, END, c)
    assert len(db.exec(select(PriceSlot)).all()) == N_SLOTS
    assert all(r.source == "octopus" for r in third)
    assert [r.price_p_per_kwh for r in third] == [40.0 + i for i in range(N_SLOTS)]
    assert all(r.carbon_g_per_kwh is not None for r in third)


# --------------------------------------------------------------------------- limitations

def test_concurrent_refresh_conflict_is_retried_not_raised(db):
    """A row inserted by another session between our select and commit is retried, not raised."""
    other_slot = START + timedelta(minutes=30)
    real_exec, raced = db.exec, []

    def exec_then_race(stmt, *a, **kw):
        rows = list(real_exec(stmt, *a, **kw))
        if not raced:  # first select only: a concurrent refresh lands the same slot right now
            raced.append(True)
            with Session(db.get_bind()) as other:
                other.add(PriceSlot(start_at=other_slot, price_p_per_kwh=1.0, source="fallback",
                                    source_url="synthetic:agile-like-curve", fetched_at=START))
                other.commit()
        return rows

    db.exec = exec_then_race
    with make_client(rates=_rates(START, N_SLOTS), carbon=_carbon(START, N_SLOTS)) as c:
        saved = refresh_prices(db, START, END, c)
    del db.exec

    assert raced
    rows = db.exec(select(PriceSlot)).all()
    assert len(rows) == N_SLOTS and len({r.start_at for r in rows}) == N_SLOTS
    assert len(saved) == N_SLOTS and all(r.id is not None for r in saved)
    assert [r.start_at for r in saved] == [START + timedelta(minutes=30 * i) for i in range(N_SLOTS)]
    # the racing fallback row was upgraded in place with the real Octopus price
    assert saved[1].source == "octopus" and saved[1].price_p_per_kwh == 21.0


def test_one_bad_carbon_row_only_drops_that_slot():
    """A non-numeric forecast blanks only its own slot's carbon, not the whole fetch."""
    carbon = _carbon(START, N_SLOTS)
    carbon["data"]["data"][3]["intensity"]["forecast"] = "n/a"  # index 0 is the pre-window row -> slot 2
    with make_client(rates=_rates(START, N_SLOTS), carbon=carbon) as c:
        slots = fetch_slots(START, END, c)
    assert slots[2].carbon_g_per_kwh is None
    assert [s.carbon_g_per_kwh for i, s in enumerate(slots) if i != 2] == [100 + i for i in range(N_SLOTS) if i != 2]


@pytest.mark.parametrize("bad", ["nan", "inf", "-Infinity", float("nan"), float("inf"), "abc", None, True])
def test_non_finite_or_non_numeric_price_falls_back_for_that_slot(bad):
    """A nan/inf/non-numeric value_inc_vat is ignored, so only that slot falls back."""
    rates = _rates(START, N_SLOTS)
    rates["results"][-1 - 5]["value_inc_vat"] = bad  # newest-first list -> slot 5
    with make_client(rates=rates, carbon=None) as c:
        slots = fetch_slots(START, END, c)
    assert slots[5].source == "fallback" and slots[5].price_p_per_kwh == ingest.fallback_price(slots[5].start_at)
    assert all(s.source == "octopus" and s.price_p_per_kwh == 20.0 + i for i, s in enumerate(slots) if i != 5)


def test_unaligned_rate_only_applies_to_slots_it_fully_covers():
    """A rate whose valid_from/valid_to are off the half hour never prices a slot it only partly covers."""
    q = timedelta(minutes=15)
    rates = {"count": 2, "next": None, "previous": None, "results": [
        # 18:15-18:45 covers no whole slot -> 18:00 and 18:30 must fall back
        {"value_inc_vat": 77.0, "valid_from": _z(START + q), "valid_to": _z(START + 3 * q), "payment_method": None},
        # 19:15-20:30 fully covers 19:30 and 20:00 only (not 19:00, not 20:30)
        {"value_inc_vat": 5.0, "valid_from": _z(START + 5 * q), "valid_to": _z(START + 10 * q), "payment_method": None},
    ]}
    with make_client(rates=rates, carbon=None) as c:
        slots = fetch_slots(START, START + timedelta(hours=3), c)
    assert [s.source for s in slots] == ["fallback", "fallback", "fallback", "octopus", "octopus", "fallback"]
    assert [s.price_p_per_kwh for s in slots[3:5]] == [5.0, 5.0]
