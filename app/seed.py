"""Seed one London depot, a small van fleet and tonight's sessions.

All datetimes are timezone-aware UTC (timestamptz). Run: python -m app.seed
"""
from datetime import date as Date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlmodel import Session as DBSession, select

from app.db import engine, init_db
from app.models import Session, Site, Vehicle

LONDON = ZoneInfo("Europe/London")

# name, battery_kwh, max_charge_kw, arrive (local, evening), depart (local, next morning), arrival_soc, target_soc
FLEET = [
    ("Van 01", 75, 22, "18:30", "06:00", 0.25, 0.90),
    ("Van 02", 75, 22, "19:00", "06:30", 0.30, 0.90),
    ("Van 03", 68, 11, "19:30", "07:00", 0.20, 0.85),
    ("Van 04", 90, 22, "20:00", "05:30", 0.15, 0.95),
    ("Van 05", 68, 11, "21:00", "07:30", 0.40, 0.90),
    ("Van 06", 75, 22, "22:00", "06:00", 0.10, 0.90),
    ("Van 07", 90, 22, "23:30", "05:00", 0.10, 1.00),  # tight window, likely short
    ("Van 08", 68, 7, "18:00", "07:00", 0.35, 0.80),
]
SITE_LIMIT_KW = 60.0


def tonight() -> Date:
    return datetime.now(LONDON).date()


def local_to_utc(d: Date, hhmm: str, next_day: bool = False) -> datetime:
    h, m = map(int, hhmm.split(":"))
    dt = datetime.combine(d + timedelta(days=1 if next_day else 0), time(h, m), tzinfo=LONDON)
    return dt.astimezone(timezone.utc)


def seed(d: Date | None = None) -> None:
    d = d or tonight()
    init_db()
    with DBSession(engine) as s:
        site = s.exec(select(Site)).first()
        if not site:
            site = Site(name="Park Royal Depot (London)", site_limit_kw=SITE_LIMIT_KW)
            s.add(site)
            s.commit()
            s.refresh(site)
        vans = {v.name: v for v in s.exec(select(Vehicle).where(Vehicle.site_id == site.id))}
        for name, batt, kw, *_ in FLEET:
            if name not in vans:
                v = Vehicle(site_id=site.id, name=name, battery_kwh=batt, max_charge_kw=kw)
                s.add(v)
                vans[name] = v
        s.commit()
        existing = {x.vehicle_id for x in s.exec(select(Session).where(Session.date == d))}
        for name, _b, _k, arr, dep, soc0, soc1 in FLEET:
            v = vans[name]
            s.refresh(v)
            if v.id in existing:
                continue
            s.add(Session(vehicle_id=v.id, date=d, arrival_at=local_to_utc(d, arr),
                          departure_at=local_to_utc(d, dep, next_day=True),
                          arrival_soc=soc0, target_soc=soc1))
        s.commit()


if __name__ == "__main__":
    seed()
    print("seeded", tonight())
