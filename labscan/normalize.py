"""Нормализация и проверки: из «как напечатано» в «можно писать в карту».

Принцип: всё сомнительное получает статус review с понятной причиной,
а не молча записывается в карту пациента.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from .catalog import Analyte, convert, match_analyte, normalize_unit
from .schema import Flag, NormalizedReport, Observation, RawReport, RawResult, Status

MATCH_THRESHOLD = 0.85

_NUM = r"[-+]?\d+(?:[.,]\d+)?"


def parse_number(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(" ", "").replace(" ", "")
    m = re.fullmatch(rf"[<>≤≥]?({_NUM})", s)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def split_value_unit(value: str | None, unit: str | None) -> tuple[str | None, str | None]:
    """Модель иногда пишет «156 г/л» в value и оставляет unit пустым, или режет единицу пополам:
    «8.10 ×10¹²» в value и «/л» в unit (оба случая из eval). Число впереди однозначно; хвост
    становится единицей или склеивается с ней, если из них получается известная единица."""
    if not value:
        return value, unit
    m = re.fullmatch(rf"\s*([<>≤≥]?{_NUM})(?:\s*(\S.*?))?\s*", value)
    if not m or not m.group(2):
        return value, unit
    number, tail = m.group(1), m.group(2).strip()
    if not unit:
        return number, tail
    if normalize_unit(unit):
        return number, unit  # единица уже целая, хвост в value — её дубль
    for candidate in (tail + unit, f"{tail} {unit}"):
        if normalize_unit(candidate):
            return number, candidate
    return value, unit


def parse_reference(s: str | None) -> tuple[float | None, float | None]:
    if not s:
        return None, None
    t = s.strip().lower().replace(",", ".").replace("–", "-").replace("—", "-")
    m = re.search(rf"({_NUM})\s*-\s*({_NUM})", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(rf"(?:<|≤|до)\s*({_NUM})", t)
    if m:
        return 0.0, float(m.group(1))
    m = re.search(rf"(?:>|≥|от|более)\s*({_NUM})", t)
    if m:
        return float(m.group(1)), None
    return None, None


_DATE_FORMATS = ("%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    m = re.search(r"\d{1,4}[./-]\d{1,2}[./-]\d{2,4}", s)
    if not m:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(m.group(0), fmt).date()
        except ValueError:
            continue
    return None


def _round(a: Analyte, v: float | None) -> float | None:
    return None if v is None else round(v, max(a.decimals, 2))


def normalize_result(raw: RawResult) -> Observation:
    analyte, score = match_analyte(raw.name)
    issues: list[str] = []
    if analyte is None or score < MATCH_THRESHOLD:
        return Observation(
            code=analyte.code if analyte else None,
            loinc=analyte.loinc if analyte else None,
            name=analyte.name_ru if analyte else raw.name,
            value=parse_number(raw.value), unit=raw.unit,
            status=Status.review, match_score=score,
            issues=["показатель не найден в справочнике" if analyte is None
                    else f"неуверенное сопоставление названия ({score:.2f})"],
            raw=raw,
        )

    raw_value, raw_unit = split_value_unit(raw.value, raw.unit)
    value = parse_number(raw_value)
    unit = normalize_unit(raw_unit) if raw_unit else analyte.unit if analyte.unit == "%" else None
    if value is None:
        issues.append("значение не читается")
    if not raw.legible:
        issues.append("распознавание отметило значение как неуверенное")

    canon_value = None
    if value is not None:
        canon_value, err = convert(analyte, value, unit)
        if err:
            issues.append(err)

    ref_low, ref_high = parse_reference(raw.reference)
    if unit and unit != analyte.unit and unit in analyte.conversions:
        k = analyte.conversions[unit]
        ref_low = None if ref_low is None else ref_low * k
        ref_high = None if ref_high is None else ref_high * k

    flag = None
    if canon_value is not None:
        lo, hi = analyte.plausible
        if not lo <= canon_value <= hi:
            issues.append(f"значение вне физиологических пределов ({lo}–{hi} {analyte.unit}), вероятна ошибка чтения")
        if ref_low is not None and canon_value < ref_low:
            flag = Flag.low
        elif ref_high is not None and canon_value > ref_high:
            flag = Flag.high
        elif ref_low is not None or ref_high is not None:
            flag = Flag.normal
    if ref_low is not None and ref_high is not None and ref_low > ref_high:
        issues.append("референс перевёрнут, проверить")

    return Observation(
        code=analyte.code, loinc=analyte.loinc, name=analyte.name_ru,
        value=_round(analyte, canon_value), unit=analyte.unit,
        ref_low=_round(analyte, ref_low), ref_high=_round(analyte, ref_high), flag=flag,
        status=Status.review if issues else Status.ok, issues=issues, match_score=score, raw=raw,
    )


def normalize_report(raw: RawReport, today: date | None = None) -> NormalizedReport:
    today = today or date.today()
    issues: list[str] = []
    collected = parse_date(raw.collected_at)
    birth = parse_date(raw.birth_date)
    if raw.collected_at and collected is None:
        issues.append("дата взятия не распознана")
    if collected and collected > today:
        issues.append("дата взятия в будущем")
    if not raw.patient_name:
        issues.append("ФИО пациента не найдено")
    if not raw.results:
        issues.append("на бланке не найдено ни одного показателя")

    observations = [normalize_result(r) for r in raw.results]
    seen: dict[str, Observation] = {}
    for o in observations:
        if o.code and o.code in seen:
            for dup in (o, seen[o.code]):
                dup.status = Status.review
                if "показатель встречается дважды" not in dup.issues:
                    dup.issues.append("показатель встречается дважды")
        elif o.code:
            seen[o.code] = o

    return NormalizedReport(
        lab_name=raw.lab_name, patient_name=raw.patient_name.strip() if raw.patient_name else None,
        birth_date=birth, collected_at=collected, observations=observations, issues=issues,
    )
