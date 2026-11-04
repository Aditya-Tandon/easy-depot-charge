import os
from pathlib import Path

from dotenv import load_dotenv
from sqlmodel import Session as DBSession, SQLModel, create_engine

load_dotenv(Path.home() / ".env")  # local dev; on Vercel env vars are injected
load_dotenv()


def _url() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite:///./local.db")
    # SQLAlchemy needs an explicit driver for psycopg3
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


engine = create_engine(_url(), pool_pre_ping=True, pool_size=2, max_overflow=0) if _url().startswith(
    "postgresql") else create_engine(_url())


def init_db() -> None:
    from app import models  # noqa: F401  (register tables)
    SQLModel.metadata.create_all(engine)


def get_db():
    with DBSession(engine) as s:
        yield s
