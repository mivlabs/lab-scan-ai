"""Справочник лабораторных показателей.

Каждый показатель: внутренний код, LOINC, синонимы (как его пишут на бланках),
каноническая единица, коэффициенты перевода из других единиц и физиологические
пределы — значения за ними почти наверняка ошибка распознавания, а не пациент.

LOINC-коды подобраны для прототипа; перед продом сверить с loinc.org и
справочником МИС клиента.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Analyte:
    code: str
    loinc: str
    name_ru: str
    unit: str  # каноническая единица (ключ из UNIT_ALIASES)
    synonyms: tuple[str, ...]
    plausible: tuple[float, float]  # физиологические пределы в канонической единице
    ref: tuple[float, float]  # типичный референс взрослого, для генератора
    decimals: int = 1
    # единица -> множитель до канонической (value_canonical = value * factor)
    conversions: dict[str, float] = field(default_factory=dict)
    panel: str = "cbc"


ANALYTES: tuple[Analyte, ...] = (
    # Общий анализ крови
    Analyte("HGB", "718-7", "Гемоглобин", "g/L", ("гемоглобин", "hgb", "hb", "haemoglobin", "hemoglobin"),
            (30, 250), (120, 160), 0, {"g/dL": 10.0}),
    Analyte("RBC", "789-8", "Эритроциты", "10^12/L", ("эритроциты", "rbc", "red blood cells"),
            (1, 9), (3.8, 5.3), 2, {"10^6/uL": 1.0}),
    Analyte("WBC", "6690-2", "Лейкоциты", "10^9/L", ("лейкоциты", "wbc", "white blood cells"),
            (0.5, 100), (4.0, 9.0), 1, {"10^3/uL": 1.0}),
    Analyte("PLT", "777-3", "Тромбоциты", "10^9/L", ("тромбоциты", "plt", "platelets"),
            (5, 1500), (180, 320), 0, {"10^3/uL": 1.0}),
    Analyte("HCT", "4544-3", "Гематокрит", "%", ("гематокрит", "hct", "hematocrit"),
            (10, 70), (36, 48), 1),
    Analyte("MCV", "787-2", "Средний объём эритроцита", "fL",
            ("средний объем эритроцита", "средний объём эритроцита", "mcv"), (50, 130), (80, 100), 1),
    Analyte("NEUT%", "770-8", "Нейтрофилы", "%", ("нейтрофилы", "neut", "neutrophils", "нейтрофилы %", "neut%"),
            (1, 99), (47, 72), 1),
    Analyte("LYMPH%", "736-9", "Лимфоциты", "%", ("лимфоциты", "lymph", "lymphocytes", "лимфоциты %", "lymph%"),
            (1, 99), (19, 37), 1),
    Analyte("ESR", "4537-7", "СОЭ", "mm/h", ("соэ", "esr", "скорость оседания эритроцитов"),
            (0, 150), (2, 15), 0),
    # Биохимия
    Analyte("GLU", "2345-7", "Глюкоза", "mmol/L", ("глюкоза", "glu", "glucose"),
            (1, 40), (3.9, 6.1), 1, {"mg/dL": 1 / 18.016}, "bio"),
    Analyte("CHOL", "2093-3", "Холестерин общий", "mmol/L",
            ("холестерин общий", "холестерин", "chol", "cholesterol", "total cholesterol"),
            (1, 20), (3.0, 5.2), 2, {"mg/dL": 1 / 38.67}, "bio"),
    Analyte("HDL", "2085-9", "ЛПВП", "mmol/L", ("лпвп", "hdl", "холестерин лпвп", "hdl cholesterol"),
            (0.2, 5), (1.0, 2.2), 2, {"mg/dL": 1 / 38.67}, "bio"),
    Analyte("LDL", "2089-1", "ЛПНП", "mmol/L", ("лпнп", "ldl", "холестерин лпнп", "ldl cholesterol"),
            (0.2, 15), (1.5, 3.3), 2, {"mg/dL": 1 / 38.67}, "bio"),
    Analyte("TG", "2571-8", "Триглицериды", "mmol/L", ("триглицериды", "tg", "trig", "triglycerides"),
            (0.1, 30), (0.4, 1.7), 2, {"mg/dL": 1 / 88.57}, "bio"),
    Analyte("CREA", "2160-0", "Креатинин", "umol/L", ("креатинин", "crea", "creat", "creatinine"),
            (10, 2000), (62, 106), 0, {"mg/dL": 88.4}, "bio"),
    Analyte("UREA", "3091-6", "Мочевина", "mmol/L", ("мочевина", "urea"),
            (0.5, 60), (2.5, 8.3), 1, {}, "bio"),
    Analyte("ALT", "1742-6", "АЛТ", "U/L", ("алт", "аланинаминотрансфераза", "alt", "alat", "gpt"),
            (1, 5000), (5, 41), 0, {}, "bio"),
    Analyte("AST", "1920-8", "АСТ", "U/L", ("аст", "аспартатаминотрансфераза", "ast", "asat", "got"),
            (1, 5000), (5, 40), 0, {}, "bio"),
    Analyte("TBIL", "1975-2", "Билирубин общий", "umol/L",
            ("билирубин общий", "билирубин", "tbil", "bilirubin total", "total bilirubin"),
            (1, 700), (3.4, 20.5), 1, {"mg/dL": 17.1}, "bio"),
    Analyte("CRP", "1988-5", "С-реактивный белок", "mg/L",
            ("с-реактивный белок", "срб", "crp", "c-reactive protein"), (0, 500), (0, 5), 1, {}, "bio"),
    Analyte("HBA1C", "4548-4", "Гликированный гемоглобин", "%",
            ("гликированный гемоглобин", "hba1c", "гликозилированный гемоглобин"), (3, 20), (4.0, 6.0), 1, {}, "bio"),
    Analyte("TSH", "3016-3", "ТТГ", "mIU/L", ("ттг", "tsh", "тиреотропный гормон"),
            (0.005, 150), (0.4, 4.0), 2, {}, "bio"),
)

BY_CODE: dict[str, Analyte] = {a.code: a for a in ANALYTES}

# Как единицы пишут на бланках -> канонический ключ.
UNIT_ALIASES: dict[str, str] = {
    "г/л": "g/L", "g/l": "g/L",
    "г/дл": "g/dL", "g/dl": "g/dL",
    "ммоль/л": "mmol/L", "mmol/l": "mmol/L",
    "мг/дл": "mg/dL", "mg/dl": "mg/dL",
    "мкмоль/л": "umol/L", "umol/l": "umol/L", "µmol/l": "umol/L", "μmol/l": "umol/L",
    "ед/л": "U/L", "u/l": "U/L", "ме/л": "U/L", "iu/l": "U/L",
    "мг/л": "mg/L", "mg/l": "mg/L",
    "мм/ч": "mm/h", "mm/h": "mm/h", "мм/час": "mm/h", "mm/hr": "mm/h",
    "фл": "fL", "fl": "fL",
    "%": "%",
    "мме/л": "mIU/L", "miu/l": "mIU/L", "мкме/мл": "mIU/L", "uiu/ml": "mIU/L", "µiu/ml": "mIU/L",
    "10^9/л": "10^9/L", "10^9/l": "10^9/L", "10*9/л": "10^9/L", "10*9/l": "10^9/L", "х10^9/л": "10^9/L",
    "x10^9/l": "10^9/L", "×10^9/л": "10^9/L", "×10⁹/л": "10^9/L", "10⁹/л": "10^9/L", "тыс/мкл": "10^9/L",
    "10^3/ul": "10^3/uL", "10^3/µl": "10^3/uL", "x10^3/ul": "10^3/uL",
    "10^12/л": "10^12/L", "10^12/l": "10^12/L", "10*12/л": "10^12/L", "10*12/l": "10^12/L", "х10^12/л": "10^12/L",
    "x10^12/l": "10^12/L", "×10^12/л": "10^12/L", "×10¹²/л": "10^12/L", "10¹²/л": "10^12/L", "млн/мкл": "10^12/L",
    "10^6/ul": "10^6/uL", "10^6/µl": "10^6/uL", "x10^6/ul": "10^6/uL",
}

UNIT_LABELS_RU: dict[str, str] = {
    "g/L": "г/л", "g/dL": "г/дл", "mmol/L": "ммоль/л", "mg/dL": "мг/дл", "umol/L": "мкмоль/л",
    "U/L": "Ед/л", "mg/L": "мг/л", "mm/h": "мм/ч", "fL": "фл", "%": "%", "mIU/L": "мМЕ/л",
    "10^9/L": "×10⁹/л", "10^12/L": "×10¹²/л", "10^3/uL": "×10³/мкл", "10^6/uL": "×10⁶/мкл",
}


def _norm_text(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"\(.*?\)", " ", s)  # «Гемоглобин (HGB)» -> «гемоглобин»
    s = re.sub(r"[^\w%\-\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_unit(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower().replace(" ", "").replace(",", ".")
    key = key.replace("μ", "µ")
    if key in UNIT_ALIASES:
        return UNIT_ALIASES[key]
    # Уже канонический ключ в другом регистре
    for canon in set(UNIT_ALIASES.values()):
        if canon.lower() == key:
            return canon
    # Степени десяти в любом написании: «x10^9/л», «х10*9/л», «X10%3/uL», «Х109/л» (OCR съел «^»)
    m = re.fullmatch(r"[xх×*]?10\W?(⁹|¹²|³|⁶|12|9|3|6)/(л|l|мкл|ul|µl|n)", key)
    if m:
        power = {"⁹": "9", "¹²": "12", "³": "3", "⁶": "6"}.get(m.group(1), m.group(1))
        per_litre = m.group(2) in ("л", "l", "n")
        return {("9", True): "10^9/L", ("12", True): "10^12/L",
                ("3", False): "10^3/uL", ("6", False): "10^6/uL"}.get((power, per_litre))
    return None


_SYNONYM_INDEX: dict[str, str] = {}
for _a in ANALYTES:
    _SYNONYM_INDEX[_norm_text(_a.name_ru)] = _a.code
    _SYNONYM_INDEX[_a.code.lower()] = _a.code
    for _s in _a.synonyms:
        _SYNONYM_INDEX[_norm_text(_s)] = _a.code


def match_analyte(raw_name: str) -> tuple[Analyte | None, float]:
    """Сопоставить название с бланка показателю справочника. Возвращает (показатель, уверенность)."""
    import difflib

    name = _norm_text(raw_name)
    if not name:
        return None, 0.0
    if name in _SYNONYM_INDEX:
        return BY_CODE[_SYNONYM_INDEX[name]], 1.0
    # Аббревиатура в скобках: «Аланинаминотрансфераза (ALT)»
    for token in re.findall(r"\(([^)]+)\)", raw_name):
        t = _norm_text(token)
        if t in _SYNONYM_INDEX:
            return BY_CODE[_SYNONYM_INDEX[t]], 0.95
    # Любое слово/пара слов совпадает с синонимом целиком
    words = name.split()
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            chunk = " ".join(words[i:i + n])
            if len(chunk) >= 3 and chunk in _SYNONYM_INDEX:
                return BY_CODE[_SYNONYM_INDEX[chunk]], 0.9
    # Нечёткое совпадение — для опечаток OCR («Гемоглобип»)
    best = difflib.get_close_matches(name, list(_SYNONYM_INDEX), n=1, cutoff=0.8)
    if best:
        score = difflib.SequenceMatcher(None, name, best[0]).ratio()
        return BY_CODE[_SYNONYM_INDEX[best[0]]], round(score * 0.9, 3)
    return None, 0.0


def convert(analyte: Analyte, value: float, unit: str | None) -> tuple[float | None, str | None]:
    """Перевести значение в каноническую единицу. (None, причина) если не получается."""
    if unit is None:
        return None, "единица не распознана"
    if unit == analyte.unit:
        return value, None
    if unit in analyte.conversions:
        return value * analyte.conversions[unit], None
    return None, f"нет перевода {unit} → {analyte.unit}"
