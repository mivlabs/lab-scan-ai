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
                RawResult(name="Гемоглобин", value="118", unit="г/л", reference="120 - 160"),
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
