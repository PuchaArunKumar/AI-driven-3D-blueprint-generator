"""Database engine, session handling and the ORM model.

SQLite is the default so the application runs with no external services, but
every column type used here (including ``JSON``) is portable, so pointing
``DATABASE_URL`` at PostgreSQL is a configuration change rather than a
rewrite.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import JSON, Boolean, DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    """A design project and every artefact generated for it."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="Untitled design")
    prompt: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    design_spec: Mapped[dict | None] = mapped_column(JSON, default=None)
    images: Mapped[list] = mapped_column(JSON, default=list)
    exports: Mapped[dict] = mapped_column(JSON, default=dict)
    stats: Mapped[dict | None] = mapped_column(JSON, default=None)
    history: Mapped[list] = mapped_column(JSON, default=list)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    model_path: Mapped[str | None] = mapped_column(Text, default=None)
    model_format: Mapped[str | None] = mapped_column(String(16), default=None)
    thumbnail_path: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


@lru_cache(maxsize=1)
def get_engine():
    """Create the process-wide engine."""
    settings = get_settings()
    url = settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, echo=False, future=True, connect_args=connect_args)
    logger.info("Database engine ready: %s", url.split("///")[-1])
    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
