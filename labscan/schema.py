"""Контракты данных: что отдаёт распознавание и что уходит в карту пациента."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class RawResult(BaseModel):
    """Строка бланка как она напечатана, до нормализации."""

    name: str = Field(description="Название показателя ровно как на бланке, например 'Гемоглобин (HGB)'")
    value: str = Field(description="Значение как напечатано, например '135' или '4,52'. Пустая строка, если не читается")
    unit: str | None = Field(default=None, description="Единица как напечатана, например 'г/л' или 'x10^9/л'")
    reference: str | None = Field(default=None, description="Референсный интервал как напечатан, например '120 - 160'")
    legible: bool = Field(default=True, description="false, если значение читается неуверенно")
    box: list[float] | None = Field(default=None, description="Где строка на странице: [x0, y0, x1, y1] в долях "
                                    "ширины и высоты от 0 до 1, левый верхний угол в (0, 0)")
    page: int | None = Field(default=None, description="Номер страницы с нуля; проставляется при склейке страниц")


class RawReport(BaseModel):
    """Весь бланк как он напечатан."""

    lab_name: str | None = Field(default=None, description="Название лаборатории")
    patient_name: str | None = Field(default=None, description="ФИО пациента как на бланке")
    birth_date: str | None = Field(default=None, description="Дата рождения как на бланке")
    collected_at: str | None = Field(default=None, description="Дата взятия биоматериала как на бланке")
    results: list[RawResult] = Field(default_factory=list)


class Status(StrEnum):
    ok = "ok"
    review = "review"  # человек должен посмотреть до записи в карту


class Flag(StrEnum):
    normal = "N"
    low = "L"
    high = "H"


class Observation(BaseModel):
    """Нормализованный показатель, готовый к записи в карту."""

    code: str | None
    loinc: str | None
    name: str
    value: float | None
    unit: str | None
    ref_low: float | None = None
    ref_high: float | None = None
    flag: Flag | None = None
    status: Status = Status.ok
    issues: list[str] = Field(default_factory=list)
    match_score: float = 0.0
    raw: RawResult


class NormalizedReport(BaseModel):
    lab_name: str | None
    patient_name: str | None
    birth_date: date | None
    collected_at: date | None
    observations: list[Observation]
    issues: list[str] = Field(default_factory=list)  # проблемы уровня бланка

    @property
    def needs_review(self) -> bool:
        return bool(self.issues) or any(o.status == Status.review for o in self.observations)
