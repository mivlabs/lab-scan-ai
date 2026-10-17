from datetime import date

import pytest

from labscan.catalog import match_analyte, normalize_unit
from labscan.normalize import normalize_report, normalize_result, parse_number, parse_reference
from labscan.schema import Flag, RawReport, RawResult, Status


@pytest.mark.parametrize("raw,code", [
    ("Гемоглобин", "HGB"), ("Гемоглобин (HGB)", "HGB"), ("HGB", "HGB"), ("Hb", "HGB"),
    ("Аланинаминотрансфераза (ALT)", "ALT"), ("АЛТ", "ALT"), ("Нейтрофилы, %", "NEUT%"),
    ("Холестерин общий", "CHOL"), ("Bilirubin, total", "TBIL"), ("СОЭ", "ESR"),
])
def test_match_analyte(raw, code):
    a, score = match_analyte(raw)
    assert a is not None and a.code == code and score >= 0.85


def test_ocr_typo_matched_but_sent_to_review():
    o = normalize_result(RawResult(name="Гемоглобип", value="135", unit="г/л"))
    assert o.code == "HGB" and o.status == Status.review


def test_unknown_analyte():
    a, score = match_analyte("Заказ №")
    assert a is None and score == 0


@pytest.mark.parametrize("raw,unit", [
    ("г/л", "g/L"), ("g/dL", "g/dL"), ("×10⁹/л", "10^9/L"), ("x10^3/uL", "10^3/uL"), ("Х109/л", "10^9/L"),
    ("X10*6/uL", "10^6/uL"), ("мкмоль/л", "umol/L"), ("Ед/л", "U/L"), ("мм/ч", "mm/h"), ("abc", None),
])
def test_normalize_unit(raw, unit):
    assert normalize_unit(raw) == unit


def test_parse_number_and_reference():
    assert parse_number("4,52") == 4.52
    assert parse_number("<5") == 5.0
    assert parse_number("4.5.2") is None
    assert parse_reference("120 - 160") == (120, 160)
    assert parse_reference("3,80–5,30") == (3.8, 5.3)
    assert parse_reference("до 5,0") == (0, 5.0)
    assert parse_reference("> 1.0") == (1.0, None)


def test_us_units_converted():
    o = normalize_result(RawResult(name="HGB", value="13.5", unit="g/dL", reference="12.0 - 16.0"))
    assert o.code == "HGB" and o.value == 135 and o.unit == "g/L"
    assert (o.ref_low, o.ref_high) == (120, 160) and o.flag == Flag.normal and o.status == Status.ok


def test_glucose_mgdl():
    o = normalize_result(RawResult(name="Glucose", value="126", unit="mg/dL", reference="70 - 99"))
    assert o.value == pytest.approx(6.99, abs=0.01) and o.flag == Flag.high


def test_implausible_value_goes_to_review():
    # OCR потерял запятую: 1350 г/л вместо 135
    o = normalize_result(RawResult(name="Гемоглобин", value="1350", unit="г/л", reference="120 - 160"))
    assert o.status == Status.review and any("физиологических" in i for i in o.issues)


def test_unknown_unit_goes_to_review():
    o = normalize_result(RawResult(name="Лейкоциты", value="6.1", unit="%10%n"))
    assert o.status == Status.review and o.value is None


def test_illegible_goes_to_review():
    o = normalize_result(RawResult(name="СОЭ", value="12", unit="мм/ч", legible=False))
    assert o.status == Status.review


def test_report_level_checks():
    raw = RawReport(patient_name=None, collected_at="01.01.2099", results=[
        RawResult(name="АЛТ", value="20", unit="Ед/л"), RawResult(name="ALT", value="21", unit="U/L")])
    n = normalize_report(raw, today=date(2026, 10, 17))
    assert "дата взятия в будущем" in n.issues and "ФИО пациента не найдено" in n.issues
    assert all(o.status == Status.review for o in n.observations)  # дубль показателя
    assert n.needs_review


def test_value_with_unit_in_one_field_is_split():
    """Eval на фото: модель написала «156 г/л» в value и оставила unit пустым."""
    from labscan.normalize import split_value_unit

    assert split_value_unit("156 г/л", None) == ("156", "г/л")
    assert split_value_unit("8.10 ×10¹²/л", None) == ("8.10", "×10¹²/л")
    assert split_value_unit("<5 мг/л", None) == ("<5", "мг/л")
    assert split_value_unit("156", None) == ("156", None)
    assert split_value_unit("156 г/л", "г/л") == ("156 г/л", "г/л")  # unit задан: value не трогаем
    o = normalize_result(RawResult(name="Гемоглобин (HGB)", value="156 г/л", unit=None, reference="(120 - 160)"))
    assert o.code == "HGB" and o.value == 156 and o.unit == "g/L" and o.status == "ok" and o.ref_high == 160
    o = normalize_result(RawResult(name="Гемоглобин", value="156 попугаев", unit=None))
    assert o.value is None and "единица не распознана" in o.issues and o.status == "review"
