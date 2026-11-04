"""Price + carbon ingestion for the London depot.

Sources
-------
* Octopus Agile unit rates, London region ("C"). The current Agile *import*
  product code is discovered from ``/v1/products/`` (paginated) rather than
  hardcoded, then rates come from
  ``/v1/products/{code}/electricity-tariffs/E-1R-{code}-C/standard-unit-rates/``.
  ``value_inc_vat`` (p/kWh) is used.
* Carbon Intensity API regional forecast for London (regionid 13):
  ``/regional/intensity/{from}/{to}/regionid/13`` -> ``intensity.forecast`` (gCO2/kWh).

Any half hour without an Octopus price gets a synthetic price from
``fallback_price`` (source="fallback"); missing carbon is ``None``. Network
errors, timeouts, non-200 responses and malformed JSON never propagate: they
just mean that source contributed nothing. A single bad row (non-numeric or
non-finite value, bad timestamp) is skipped on its own and doesn't void the rest.

A rate is only applied to half-hour slots it *fully* covers: a rate whose
valid_from/valid_to are off the half hour never prices a slot it only partly
covers (that slot falls back instead of getting a possibly wrong price).
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable, Optional

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import set_committed_value
from sqlmodel import Session, select

from app.timeutil import LONDON, SLOT, UTC, to_utc
from app.models import PriceSlot

log = logging.getLogger(__name__)

TIMEOUT = 10.0

OCTOPUS_BASE = "https://api.octopus.energy/v1"
OCTOPUS_PRODUCTS_URL = f"{OCTOPUS_BASE}/products/"
OCTOPUS_REGION = "C"  # London
CARBON_BASE = "https://api.carbonintensity.org.uk"
CARBON_REGION_ID = 13  # London
FALLBACK_SOURCE_URL = "synthetic:agile-like-curve"

# A pass over the paginated products list is bounded so a bad "next" link can't loop forever.
_MAX_PAGES = 20


# --------------------------------------------------------------------------- time helpers



def _parse_ts(s: str) -> datetime:
    return to_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))


def _floor_slot(dt: datetime) -> datetime:
    dt = to_utc(dt)
    return dt.replace(minute=(dt.minute // 30) * 30, second=0, microsecond=0)


def _slot_starts(start: datetime, end: datetime) -> list[datetime]:
    """Half-hour slot starts covering [start, end). start is floored to the half hour."""
    t, end = _floor_slot(start), to_utc(end)
    out = []
    while t < end:
        out.append(t)
        t += SLOT
    return out


def _finite_float(v: Any) -> Optional[float]:
    """float(v) if v is a finite number (or numeric string), else None. bools are rejected."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _iso_z(dt: datetime) -> str:
    return to_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def tonight_window(d: date) -> tuple[datetime, datetime]:
    """18:00 Europe/London on ``d`` to 08:00 Europe/London on ``d + 1``, as UTC (DST-safe)."""
    start = datetime.combine(d, time(18, 0), tzinfo=LONDON)
    end = datetime.combine(d + timedelta(days=1), time(8, 0), tzinfo=LONDON)
    return start.astimezone(UTC), end.astimezone(UTC)


# --------------------------------------------------------------------------- fallback curve

def fallback_price(slot_start: datetime) -> float:
    """Synthetic Agile-like unit rate (p/kWh inc VAT) by Europe/London local time of slot start.

    00:00-05:00  12.0  overnight trough (cheapest)
    05:00-07:00  18.0  early-morning ramp
    07:00-16:00  24.0  daytime
    16:00-19:00  38.0  evening peak (Agile's most expensive window)
    19:00-22:00  26.0  post-peak shoulder
    22:00-24:00  18.0  late evening
    """
    h = to_utc(slot_start).astimezone(LONDON).hour
    if h < 5:
        return 12.0
    if h < 7:
        return 18.0
    if h < 16:
        return 24.0
    if h < 19:
        return 38.0
    if h < 22:
        return 26.0
    return 18.0


# --------------------------------------------------------------------------- HTTP helpers

def _get_json(client: httpx.Client, url: str, **kw: Any) -> Optional[Any]:
    """GET -> parsed JSON, or None on any network error / non-200 / bad JSON."""
    try:
        r = client.get(url, timeout=TIMEOUT, **kw)
    except httpx.HTTPError as e:
        log.warning("GET %s failed: %s", url, e)
        return None
    if r.status_code != 200:
        log.warning("GET %s -> HTTP %s", url, r.status_code)
        return None
    try:
        return r.json()
    except ValueError as e:
        log.warning("GET %s returned invalid JSON: %s", url, e)
        return None


def _paginate(client: httpx.Client, url: str) -> Iterable[dict]:
    """Yield ``results`` items across Octopus-style paginated responses (follows ``next``)."""
    seen = set()
    for _ in range(_MAX_PAGES):
        if not url or url in seen:
            return
        seen.add(url)
        data = _get_json(client, url)
        if not isinstance(data, dict):
            return
        for item in data.get("results") or []:
            if isinstance(item, dict):
                yield item
        url = data.get("next")


# --------------------------------------------------------------------------- Octopus

def discover_agile_product(client: httpx.Client, now: Optional[datetime] = None) -> Optional[str]:
    """Return the code of the currently available Agile *import* product, or None.

    Filters: code starts with "AGILE", direction == "IMPORT", available_from <= now and
    (available_to is null or > now). Prefers non-business products, then the newest
    available_from.
    """
    now = to_utc(now or datetime.now(UTC))
    candidates = []
    for p in _paginate(client, OCTOPUS_PRODUCTS_URL):
        try:
            code = p.get("code") or ""
            if not code.upper().startswith("AGILE") or p.get("direction") != "IMPORT":
                continue
            frm = p.get("available_from")
            to = p.get("available_to")
            frm_dt = _parse_ts(frm) if frm else datetime.min.replace(tzinfo=UTC)
            if frm_dt > now or (to and _parse_ts(to) <= now):
                continue
            candidates.append((not p.get("is_business", False), frm_dt, code))
        except (ValueError, TypeError, AttributeError):
            continue
    if not candidates:
        return None
    return max(candidates)[2]


def octopus_rates_url(code: str, start: datetime, end: datetime) -> str:
    tariff = f"E-1R-{code}-{OCTOPUS_REGION}"
    return (f"{OCTOPUS_BASE}/products/{code}/electricity-tariffs/{tariff}/standard-unit-rates/"
            f"?period_from={_iso_z(start)}&period_to={_iso_z(end)}")


def _fetch_octopus(client: httpx.Client, slots: list[datetime]) -> tuple[dict[datetime, float], Optional[str]]:
    """Map slot start -> value_inc_vat for the given slots; plus the rates URL used."""
    if not slots:
        return {}, None
    code = discover_agile_product(client)
    if not code:
        log.warning("No current Agile import product found; using fallback prices")
        return {}, None
    url = octopus_rates_url(code, slots[0], slots[-1] + SLOT)
    wanted = set(slots)
    prices: dict[datetime, float] = {}
    preferred: dict[datetime, bool] = {}
    for rate in _paginate(client, url):
        try:
            value = _finite_float(rate["value_inc_vat"])
            frm = _parse_ts(rate["valid_from"])
            to = _parse_ts(rate["valid_to"]) if rate.get("valid_to") else slots[-1] + SLOT
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        if value is None:  # nan / inf / non-numeric: ignore, so the slot falls back
            continue
        # Some tariffs list separate DIRECT_DEBIT / NON_DIRECT_DEBIT rates; prefer DD (or untagged).
        pref = rate.get("payment_method") in (None, "DIRECT_DEBIT")
        # A rate may span several half hours: expand it onto each slot it *fully* covers
        # (an unaligned edge never prices a slot it only partly covers).
        t = _floor_slot(frm)
        if t < frm:
            t += SLOT
        while t + SLOT <= to and t <= slots[-1]:
            if t in wanted and (t not in prices or (pref and not preferred[t])):
                prices[t] = value
                preferred[t] = pref
            t += SLOT
    return prices, url


# --------------------------------------------------------------------------- Carbon Intensity

def carbon_url(start: datetime, end: datetime) -> str:
    fmt = "%Y-%m-%dT%H:%MZ"
    return (f"{CARBON_BASE}/regional/intensity/{to_utc(start).strftime(fmt)}/"
            f"{to_utc(end).strftime(fmt)}/regionid/{CARBON_REGION_ID}")


def _fetch_carbon(client: httpx.Client, slots: list[datetime]) -> dict[datetime, float]:
    """Map slot start -> regional forecast gCO2/kWh. Response shape:
    ``{"data": {"regionid": 13, ..., "data": [{"from", "to", "intensity": {"forecast", "index"}}]}}``
    (the outer "data" is occasionally a one-element list; both are handled).
    """
    if not slots:
        return {}
    payload = _get_json(client, carbon_url(slots[0], slots[-1] + SLOT))
    if not isinstance(payload, dict):
        return {}
    region = payload.get("data")
    if isinstance(region, list):
        region = region[0] if region else None
    if not isinstance(region, dict):
        return {}
    wanted = set(slots)
    out: dict[datetime, float] = {}
    for row in region.get("data") or []:
        try:
            t = _parse_ts(row["from"])
            forecast = _finite_float(row["intensity"]["forecast"])
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        # The API also returns the half hour ending at `from`; only keep requested slots.
        # A bad forecast (None, "n/a", nan) only leaves its own slot without carbon.
        if t in wanted and forecast is not None:
            out[t] = forecast
    return out


# --------------------------------------------------------------------------- public API

def fetch_slots(start: datetime, end: datetime, client: Optional[httpx.Client] = None) -> list[PriceSlot]:
    """Unsaved PriceSlot objects, one per half hour covering [start, end), sorted by start_at.

    Never raises for network problems: missing prices fall back to ``fallback_price``,
    missing carbon is None.
    """
    slots = _slot_starts(start, end)
    if not slots:
        return []
    own = client is None
    if own:
        client = httpx.Client(timeout=TIMEOUT, headers={"User-Agent": "algosoup-depot-planner/1.0"})
    try:
        try:
            prices, rates_url = _fetch_octopus(client, slots)
        except Exception:  # defensive: never let a source failure reach the caller
            log.exception("Octopus fetch failed")
            prices, rates_url = {}, None
        try:
            carbon = _fetch_carbon(client, slots)
        except Exception:
            log.exception("Carbon intensity fetch failed")
            carbon = {}
    finally:
        if own:
            client.close()

    fetched_at = datetime.now(UTC)
    out = []
    for t in slots:
        if t in prices:
            price, source, url = prices[t], "octopus", rates_url
        else:
            price, source, url = fallback_price(t), "fallback", FALLBACK_SOURCE_URL
        out.append(PriceSlot(start_at=t, price_p_per_kwh=price, carbon_g_per_kwh=carbon.get(t),
                             source=source, source_url=url, fetched_at=fetched_at))
    return out


def _normalise_row(row: PriceSlot) -> PriceSlot:
    """Make datetimes tz-aware UTC without marking the row dirty (SQLite drops tzinfo)."""
    for attr in ("start_at", "fetched_at"):
        v = getattr(row, attr)
        if v is not None and (v.tzinfo is None or v.utcoffset() != timedelta(0)):
            set_committed_value(row, attr, to_utc(v))
    return row


def _upsert(db: Session, fresh: list[PriceSlot]) -> list[PriceSlot]:
    """Select existing rows for the window, insert/update from ``fresh``, commit. Returns saved rows.

    New rows are inserted as copies, so ``fresh`` stays untouched if the commit fails and the
    caller can retry with it.
    """
    lo, hi = fresh[0].start_at, fresh[-1].start_at
    existing = {
        to_utc(r.start_at): r
        for r in db.exec(select(PriceSlot).where(PriceSlot.start_at >= lo, PriceSlot.start_at <= hi))
    }
    saved = []
    for s in fresh:
        row = existing.get(s.start_at)
        if row is None:
            new = PriceSlot(start_at=s.start_at, price_p_per_kwh=s.price_p_per_kwh,
                            carbon_g_per_kwh=s.carbon_g_per_kwh, source=s.source,
                            source_url=s.source_url, fetched_at=s.fetched_at)
            db.add(new)
            saved.append(new)
            continue
        if s.source == "fallback" and row.source == "octopus":
            if s.carbon_g_per_kwh is not None:
                row.carbon_g_per_kwh = s.carbon_g_per_kwh
                row.fetched_at = s.fetched_at
        else:
            row.price_p_per_kwh = s.price_p_per_kwh
            if s.carbon_g_per_kwh is not None:  # keep the last forecast rather than blanking it
                row.carbon_g_per_kwh = s.carbon_g_per_kwh
            row.source = s.source
            row.source_url = s.source_url
            row.fetched_at = s.fetched_at
        db.add(row)
        saved.append(row)
    db.commit()
    return saved


def refresh_prices(db: Session, start: datetime, end: datetime,
                   client: Optional[httpx.Client] = None) -> list[PriceSlot]:
    """fetch_slots + upsert by start_at, commit, and return the saved rows sorted by start_at.

    Existing rows are updated in place. One exception: a fallback value never overwrites an
    existing Octopus price (a transient outage shouldn't erase real prices), and a missing
    carbon forecast never blanks a stored one.

    Concurrency: the upsert is select-then-insert, so an overlapping refresh can insert the
    same start_at first and trip the unique index at commit. That IntegrityError is handled by
    rolling back and retrying the upsert once, re-reading existing rows (the racing rows are
    then updated in place). Note the rollback also discards any other uncommitted changes in
    ``db``. A second consecutive conflict is re-raised.
    """
    fresh = fetch_slots(start, end, client)
    if not fresh:
        return []
    try:
        saved = _upsert(db, fresh)
    except IntegrityError:
        log.warning("PriceSlot upsert hit a unique conflict (concurrent refresh?); retrying once")
        db.rollback()
        saved = _upsert(db, fresh)
    for row in saved:
        db.refresh(row)
        _normalise_row(row)
    return sorted(saved, key=lambda r: r.start_at)
