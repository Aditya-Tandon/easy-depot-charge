"""SQLModel tables. Single source of truth for the schema."""
from datetime import date as Date, datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Site(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    site_limit_kw: float


class Vehicle(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    site_id: int = Field(foreign_key="site.id", index=True)
    name: str
    battery_kwh: float
    max_charge_kw: float


class Session(SQLModel, table=True):
    """Tonight's arrival / departure / target for one vehicle."""
    id: Optional[int] = Field(default=None, primary_key=True)
    vehicle_id: int = Field(foreign_key="vehicle.id", index=True)
    date: Date = Field(index=True)  # the evening the session starts
    arrival_at: datetime
    departure_at: datetime
    arrival_soc: float  # 0..1
    target_soc: float  # 0..1


class PriceSlot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    start_at: datetime = Field(index=True, unique=True)  # tz-aware UTC, half-hour aligned
    price_p_per_kwh: float
    carbon_g_per_kwh: Optional[float] = None
    source: str  # "octopus" | "fallback"
    source_url: str
    fetched_at: datetime


class Plan(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    site_id: int = Field(foreign_key="site.id", index=True)
    date: Date = Field(index=True)
    revision: int
    status: str = "draft"  # "draft" | "published"
    cost_gbp: float
    baseline_cost_gbp: float
    carbon_kg: Optional[float] = None
    baseline_carbon_kg: Optional[float] = None
    baseline_load_json: Optional[str] = None  # JSON list: charge-on-arrival site kW per slot
    created_at: datetime


class Allocation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    plan_id: int = Field(foreign_key="plan.id", index=True)
    vehicle_id: int = Field(foreign_key="vehicle.id")
    slot_start: datetime
    kw: float


class VanOutcome(SQLModel, table=True):
    """Per-van summary for a plan (delivered energy, infeasibility reason)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    plan_id: int = Field(foreign_key="plan.id", index=True)
    vehicle_id: int = Field(foreign_key="vehicle.id")
    need_kwh: float
    delivered_kwh: float
    infeasible_reason: Optional[str] = None


class Proposal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    raw_text: str
    vehicle_id: Optional[int] = Field(default=None, foreign_key="vehicle.id")
    field: Optional[str] = None  # arrival_at | departure_at | arrival_soc | target_soc
    value: Optional[str] = None
    quote: Optional[str] = None
    status: str = "pending"  # pending | accepted | rejected | invalid
    error: Optional[str] = None
    created_at: datetime
