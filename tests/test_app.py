import importlib

import pytest
from fastapi.testclient import TestClient

from labscan.extractors.base import Extraction
from labscan.schema import RawReport, RawResult


class StubExtractor:
    name = "stub"

    def extract(self, pages):
        return Extraction(RawReport(
            lab_name="ЦитоЛаб", patient_name="Иванова Мария Петровна", birth_date="01.02.1990",
            collected_at="10.10.2026", results=[
                RawResult(name="Гемоглобин", value="118", unit="г/л", reference="120 - 160", box=[0.1, 0.2, 0.5, 0.25]),
                RawResult(name="Глюкоза", value="52", unit="ммоль/л", reference="3,9 - 6,1"),  # неправдоподобно
            ]), "stub", 0.1)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LABSCAN_DATA", str(tmp_path))
    monkeypatch.setenv("LABSCAN_DB", f"sqlite:///{tmp_path}/t.db")
    import labscan.web.app as appmod

    appmod = importlib.reload(appmod)
    monkeypatch.setattr(appmod, "_extractor", StubExtractor())
    return TestClient(appmod.app)


def _png():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(buf, "PNG")
    return buf.getvalue()


def test_upload_review_confirm_flow(client):
    r = client.post("/upload", files={"file": ("scan.png", _png(), "image/png")}, follow_redirects=False)
    assert r.status_code == 303
    page = client.get(r.headers["location"]).text
    assert page.count('class="need"') == 1  # глюкоза 52 ушла на проверку
    assert "физиологических" in page

    # Врач исправляет глюкозу на 5,2 и подтверждает
    r = client.post("/documents/1/confirm", data={
        "patient_name": "Иванова Мария Петровна", "birth_date": "1990-02-01", "collected_at": "2026-10-10",
        "include_0": "on", "code_0": "HGB", "value_0": "118",
        "include_1": "on", "code_1": "GLU", "value_1": "5,2",
        "code_x0": "CRP", "value_x0": "3,1"}, follow_redirects=False)  # строку, пропущенную распознаванием, врач добавил сам
    assert r.status_code == 303 and r.headers["location"] == "/patients/1"
    card = client.get("/patients/1").text
    assert "Гемоглобин" in card and "118" in card and "5,2" in card and "С-реактивный белок" in card

    bundle = client.get("/api/documents/1/fhir").json()
    obs = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]
    hgb = next(o for o in obs if o["code"]["coding"][0]["code"] == "718-7")
    assert hgb["valueQuantity"]["value"] == 118 and hgb["interpretation"][0]["coding"][0]["code"] == "L"


def test_api_extract(client):
    r = client.post("/api/extract", files={"file": ("scan.png", _png(), "image/png")})
    body = r.json()
    assert r.status_code == 200 and body["engine"] == "stub"
    statuses = [o["status"] for o in body["report"]["observations"]]
    assert statuses == ["ok", "review"]


def test_rejects_wrong_format(client):
    r = client.post("/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_index_lists_samples_and_try_runs_extractor(client):
    page = client.get("/").text
    assert "Попробовать на примере" in page and "/try/" in page
    doc_id = next(line for line in page.splitlines() if 'action="/try/' in line).split('action="/try/')[1].split('"')[0]
    thumb = client.get(f"/samples/{doc_id}/thumb.png")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/png"
    r = client.post(f"/try/{doc_id}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/documents/1"
    assert client.post("/try/no_such_sample", follow_redirects=False).status_code == 404


def test_review_groups_rows_and_counts(client):
    r = client.post("/upload", files={"file": ("scan.png", _png(), "image/png")}, follow_redirects=False)
    page = client.get(r.headers["location"]).text
    assert "Требует внимания" in page and "Прочитано уверенно" in page
    assert 'name="value_0"' in page and 'name="value_1"' in page  # индексы строк не меняются после группировки
    assert 'class="need"' in page and 'class="row-ok"' in page
    assert 'data-box="0.1,0.2,0.5,0.25" data-page="0"' in page and "подсвечена на скане" in page


def test_chart_svg():
    from datetime import date
    from types import SimpleNamespace

    from labscan.catalog import BY_CODE
    from labscan.web.app import chart

    rows = [SimpleNamespace(value=v, ref_low=120.0, ref_high=160.0, flag=f, collected_at=date(2026, m, 1))
            for v, f, m in ((118.0, "L", 1), (131.0, "N", 3), (165.0, "H", 6))]
    svg = chart(rows, BY_CODE["HGB"])
    assert svg.startswith("<svg") and svg.count("<circle") == 3 and 'class="band"' in svg and "<polyline" in svg
    assert 'flag-H' in svg and "01.01.26" in svg and "01.06.26" in svg and ">165<" in svg
    one = chart(rows[:1], BY_CODE["HGB"])
    assert one.count("<circle") == 1 and "<polyline" not in one
    assert chart([], BY_CODE["HGB"]) == ""


class BrokenExtractor:
    name = "broken"

    def extract(self, pages):
        raise RuntimeError("OpenRouter HTTP 402: no credits")


def test_engine_failure_is_a_status_not_a_500(client, monkeypatch):
    import labscan.web.app as appmod

    monkeypatch.setattr(appmod, "_extractor", BrokenExtractor())
    r = client.post("/upload", files={"file": ("scan.png", _png(), "image/png")}, follow_redirects=False)
    assert r.status_code == 303
    page = client.get(r.headers["location"])
    assert page.status_code == 200 and "Распознавание не удалось" in page.text and "no credits" in page.text
    assert client.get("/api/documents/1/status").json()["status"] == "failed"
    assert "ошибка" in client.get("/").text
    # API для интеграций отвечает ошибкой сразу, без фоновой задачи
    assert client.post("/api/extract", files={"file": ("scan.png", _png(), "image/png")}).status_code == 502

    # Движок починили: «Повторить» запускает распознавание заново
    monkeypatch.setattr(appmod, "_extractor", StubExtractor())
    r = client.post("/documents/1/retry", follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/api/documents/1/status").json()["status"] == "review"
    assert 'class="need"' in client.get("/documents/1").text


def test_old_database_gets_new_columns(tmp_path):
    import sqlite3

    from labscan.db import make_session_factory

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, filename VARCHAR(300), path VARCHAR(500), "
                "engine VARCHAR(100), status VARCHAR(20), raw JSON, normalized JSON, seconds FLOAT, cost_usd FLOAT, "
                "created_at DATETIME, patient_id INTEGER, mis_ref VARCHAR(500))")
    con.commit()
    con.close()
    make_session_factory(f"sqlite:///{db.as_posix()}")
    cols = {row[1] for row in sqlite3.connect(db).execute("PRAGMA table_info(documents)")}
    assert "error" in cols


def test_compare_engines_on_one_document(client, monkeypatch):
    import labscan.web.app as appmod
    from labscan.extractors.base import Extraction
    from labscan.schema import RawReport, RawResult

    class OtherStub:
        name = "other"
        model = "m"

        def extract(self, pages):
            return Extraction(RawReport(patient_name="Иванова", results=[
                RawResult(name="Гемоглобин", value="121", unit="г/л", reference="120 - 160"),  # расходится со 118
                RawResult(name="Лейкоциты", value="6,1", unit="×10⁹/л")]), "other:m", 0.2, 0.001)  # глюкозы нет

    r = client.post("/upload", files={"file": ("scan.png", _png(), "image/png")}, follow_redirects=False)
    page = client.get("/documents/1/compare").text
    assert "Сравнение движков" in page and "stub" in page
    monkeypatch.setattr(appmod, "get_extractor", lambda name, model=None: OtherStub())
    r = client.post("/documents/1/compare", data={"engine": "openrouter", "model": "m"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/api/engine-runs/1/status").json()["status"] == "done"
    page = client.get("/documents/1/compare").text
    assert "other:m" in page and "Лейкоциты" in page and "— пропущен" in page
    s = appmod.SessionLocal()
    rows = appmod.compare_matrix([{"norm": s.get(appmod.Document, 1).normalized},
                                  {"norm": s.get(appmod.EngineRun, 1).normalized}])
    by = {row["code"]: row for row in rows}
    # 118 против 121; глюкоза только слева; лейкоциты только справа
    assert by["HGB"]["diff"] and by["GLU"]["diff"] and by["WBC"]["diff"]
    assert client.post("/documents/1/compare", data={"engine": "nope"}).status_code == 400


def test_compare_matrix_ignores_pending_columns():
    from labscan.web.app import compare_matrix

    obs = {"observations": [{"code": "HGB", "value": 118.0, "status": "ok", "flag": None, "issues": [],
                             "raw": {"name": "Гемоглобин", "value": "118", "unit": "г/л"}}]}
    rows = compare_matrix([{"norm": obs}, {"norm": None}])  # вторая колонка ещё распознаётся
    assert rows[0]["code"] == "HGB" and rows[0]["diff"] is False
    assert compare_matrix([{"norm": obs}, {"norm": {"observations": []}}])[0]["diff"] is True
