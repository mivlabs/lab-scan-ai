"""Хранилище прототипа: SQLite через SQLAlchemy. Для пилота — PostgreSQL, меняется URL."""

from __future__ import annotations

import os
from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, String, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200))
    birth_date: Mapped[date | None] = mapped_column(Date)
    observations: Mapped[list[ObservationRow]] = relationship(back_populates="patient")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(500))
    engine: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="review")  # processing | review | confirmed | failed
    error: Mapped[str | None] = mapped_column(String(500))  # почему распознавание не удалось
    raw: Mapped[dict] = mapped_column(JSON)
    normalized: Mapped[dict] = mapped_column(JSON)
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"))
    mis_ref: Mapped[str | None] = mapped_column(String(500))


class ObservationRow(Base):
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    code: Mapped[str] = mapped_column(String(20))
    loinc: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(200))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20))
    ref_low: Mapped[float | None] = mapped_column(Float)
    ref_high: Mapped[float | None] = mapped_column(Float)
    flag: Mapped[str | None] = mapped_column(String(2))
    collected_at: Mapped[date | None] = mapped_column(Date)
    patient: Mapped[Patient] = relationship(back_populates="observations")


class EngineRun(Base):
    """Тот же документ, прогнанный другим движком или моделью: для сравнения на одном бланке."""

    __tablename__ = "engine_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    engine: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="processing")  # processing | done | failed
    raw: Mapped[dict] = mapped_column(JSON, default=dict)
    normalized: Mapped[dict] = mapped_column(JSON, default=dict)
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Correction(Base):
    """Каждая правка человека: что предложила система и что записали. Материал для улучшения качества."""

    __tablename__ = "corrections"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    field: Mapped[str] = mapped_column(String(50))
    before: Mapped[str | None] = mapped_column(String(300))
    after: Mapped[str | None] = mapped_column(String(300))
    engine: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


def make_session_factory(url: str | None = None) -> sessionmaker[Session]:
    url = url or os.getenv("LABSCAN_DB", "sqlite:///data/labscan.db")
    if url.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(url.removeprefix("sqlite:///")) or ".", exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _add_missing_columns(engine) -> None:
    """create_all не меняет существующие таблицы; для прототипа достаточно добавить новые колонки.
    В пилоте — Alembic."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}'))


def _key(name: str) -> str:
    return " ".join(name.lower().replace("ё", "е").split())


def find_or_create_patient(s: Session, full_name: str, birth: date | None) -> Patient:
    """В прототипе сопоставляем по ФИО + дате рождения. В пилоте — по ИИН из МИС."""
    for p in s.scalars(select(Patient).where(Patient.birth_date == birth)):
        if _key(p.full_name) == _key(full_name):
            return p
    p = Patient(full_name=full_name.strip(), birth_date=birth)
    s.add(p)
    s.flush()
    return p
