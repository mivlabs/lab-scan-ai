"""Веб-прототип: загрузка бланка -> проверка человеком -> карта пациента -> выгрузка в МИС.

Запуск: uvicorn labscan.web.app:app --reload
"""

from __future__ import annotations

import io
import os
import uuid
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..catalog import ANALYTES, BY_CODE, UNIT_LABELS_RU
from ..db import Correction, Document, ObservationRow, Patient, find_or_create_patient, make_session_factory
from ..extractors import get_extractor
from ..fhir import to_bundle
from ..ingest import load_pages
from ..mis import OutboxMIS
from ..normalize import normalize_report, parse_date, parse_number
from ..schema import Flag

DATA = Path(os.getenv("LABSCAN_DATA", "data"))
UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MAX_BYTES = 20 * 1024 * 1024

HERE = Path(__file__).parent
app = FastAPI(title="Lab Scan AI")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def _fmt(value, code=None):
    if value is None:
        return ""
    a = BY_CODE.get(code)
    return f"{value:.{a.decimals}f}".replace(".", ",") if a else str(value)


templates.env.filters["fmt"] = _fmt
templates.env.filters["unit_ru"] = lambda u: UNIT_LABELS_RU.get(u, u or "")
EXTRA_ROWS = 3  # пустые строки для показателей, которые распознавание пропустило

SessionLocal = make_session_factory()
mis = OutboxMIS(DATA / "mis_outbox")
_extractor = None


def extractor():
    global _extractor
    if _extractor is None:
        _extractor = get_extractor()
    return _extractor


def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _process(data: bytes, filename: str, s: Session) -> Document:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Формат {ext} не поддерживается: нужен PDF, PNG или JPG")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Файл больше 20 МБ")
    path = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    ex = extractor()
    result = ex.extract(load_pages(data, filename))
    norm = normalize_report(result.report)
    doc = Document(filename=filename, path=str(path), engine=result.engine, raw=result.report.model_dump(),
                   normalized=norm.model_dump(mode="json"), seconds=result.seconds, cost_usd=result.cost_usd)
    s.add(doc)
    s.commit()
    return doc


# ---------- HTML ----------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, s: Session = Depends(db)):
    docs = s.scalars(select(Document).order_by(Document.id.desc()).limit(30)).all()
    patients = s.scalars(select(Patient).order_by(Patient.full_name)).all()
    return templates.TemplateResponse(request, "index.html", {
        "docs": docs, "patients": patients, "engine": type(extractor()).__name__})


@app.post("/upload")
async def upload(file: UploadFile = File(...), s: Session = Depends(db)):
    doc = _process(await file.read(), file.filename or "scan", s)
    return RedirectResponse(f"/documents/{doc.id}", status_code=303)


@app.get("/documents/{doc_id}", response_class=HTMLResponse)
def review(doc_id: int, request: Request, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    pages = len(load_pages(Path(doc.path).read_bytes(), doc.path))
    return templates.TemplateResponse(request, "review.html", {
        "doc": doc, "n": doc.normalized, "pages": range(pages), "analytes": ANALYTES, "extra": range(EXTRA_ROWS)})


@app.get("/documents/{doc_id}/page/{n}.png")
def page_image(doc_id: int, n: int, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    pages = load_pages(Path(doc.path).read_bytes(), doc.path)
    if not 0 <= n < len(pages):
        _404()
    buf = io.BytesIO()
    pages[n].save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/documents/{doc_id}/confirm")
async def confirm(doc_id: int, request: Request, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    if doc.status == "confirmed":
        return RedirectResponse(f"/patients/{doc.patient_id}", status_code=303)
    form = await request.form()
    n = doc.normalized

    def log(field, before, after):
        if str(before or "") != str(after or ""):
            s.add(Correction(document_id=doc.id, field=field, before=None if before is None else str(before),
                             after=None if after is None else str(after), engine=doc.engine))

    name = str(form.get("patient_name") or "").strip()
    birth = parse_date(str(form.get("birth_date") or ""))
    collected = parse_date(str(form.get("collected_at") or ""))
    if not name:
        raise HTTPException(400, "Укажите ФИО пациента")
    log("patient_name", n["patient_name"], name)
    log("birth_date", n["birth_date"], birth.isoformat() if birth else None)
    log("collected_at", n["collected_at"], collected.isoformat() if collected else None)
    patient = find_or_create_patient(s, name, birth)

    written = []
    rows = list(enumerate(n["observations"]))
    empty = {"code": None, "value": None, "ref_low": None, "ref_high": None}
    rows += [(f"x{k}", empty) for k in range(EXTRA_ROWS) if form.get(f"code_x{k}") and form.get(f"value_x{k}")]
    for i, o in rows:
        if isinstance(i, str):
            log(f"row{i}:added", None, form.get(f"code_{i}"))
        elif form.get(f"include_{i}") != "on":
            log(f"row{i}:excluded", o["code"], None)
            continue
        code = str(form.get(f"code_{i}") or "")
        value = parse_number(str(form.get(f"value_{i}") or ""))
        a = BY_CODE.get(code)
        if a is None or value is None:
            raise HTTPException(400, f"Строка {o.get('code') or i}: выберите показатель и укажите число")
        log(f"row{i}:code", o["code"], code)
        log(f"row{i}:value", o["value"], value)
        ref_low, ref_high = (o["ref_low"], o["ref_high"]) if code == o["code"] else (None, None)
        flag = (Flag.low if ref_low is not None and value < ref_low else
                Flag.high if ref_high is not None and value > ref_high else
                Flag.normal if ref_low is not None or ref_high is not None else None)
        row = ObservationRow(patient_id=patient.id, document_id=doc.id, code=code, loinc=a.loinc, name=a.name_ru,
                             value=value, unit=a.unit, ref_low=ref_low, ref_high=ref_high,
                             flag=flag.value if flag else None, collected_at=collected)
        s.add(row)
        written.append({"code": code, "loinc": a.loinc, "name": a.name_ru, "value": value, "unit": a.unit,
                        "ref_low": ref_low, "ref_high": ref_high, "flag": row.flag})

    doc.patient_id = patient.id
    doc.status = "confirmed"
    bundle = to_bundle(patient.full_name, patient.birth_date, collected, written, f"p{patient.id}")
    doc.mis_ref = mis.send(bundle, idempotency_key=f"doc-{doc.id}")
    s.commit()
    return RedirectResponse(f"/patients/{patient.id}", status_code=303)


@app.get("/patients/{pid}", response_class=HTMLResponse)
def patient_card(pid: int, request: Request, s: Session = Depends(db)):
    p = s.get(Patient, pid) or _404()
    rows = sorted(p.observations, key=lambda r: (r.collected_at or date.min, r.id))
    series: dict[str, list[ObservationRow]] = {}
    for r in rows:
        series.setdefault(r.code, []).append(r)
    order = [a.code for a in ANALYTES if a.code in series]
    return templates.TemplateResponse(request, "patient.html", {
        "p": p, "series": series, "order": order, "by_code": BY_CODE, "sparkline": sparkline})


# ---------- JSON API ----------

@app.post("/api/extract")
async def api_extract(file: UploadFile = File(...), s: Session = Depends(db)):
    """Для интеграций: файл -> нормализованные показатели со статусами проверки."""
    doc = _process(await file.read(), file.filename or "scan", s)
    return JSONResponse({"document_id": doc.id, "engine": doc.engine, "seconds": round(doc.seconds, 2),
                         "cost_usd": round(doc.cost_usd, 5), "report": doc.normalized})


@app.get("/api/documents/{doc_id}/fhir")
def api_fhir(doc_id: int, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    if doc.status != "confirmed" or not doc.mis_ref:
        raise HTTPException(409, "Документ ещё не подтверждён")
    return Response(Path(doc.mis_ref).read_bytes(), media_type="application/fhir+json")


@app.get("/healthz")
def healthz():
    return {"ok": True}


def _404():
    raise HTTPException(404, "Не найдено")


def sparkline(rows: list[ObservationRow], w: int = 160, h: int = 36) -> str:
    vals = [r.value for r in rows]
    if len(vals) < 2:
        return ""
    lows = [r.ref_low for r in rows if r.ref_low is not None]
    highs = [r.ref_high for r in rows if r.ref_high is not None]
    lo = min(vals + lows[-1:])
    hi = max(vals + highs[-1:])
    span = (hi - lo) or 1
    pts = [(4 + i * (w - 8) / (len(vals) - 1), h - 4 - (v - lo) / span * (h - 8)) for i, v in enumerate(vals)]
    band = ""
    if lows and highs:
        y1 = h - 4 - (highs[-1] - lo) / span * (h - 8)
        y2 = h - 4 - (lows[-1] - lo) / span * (h - 8)
        band = f'<rect x="0" y="{y1:.1f}" width="{w}" height="{max(y2 - y1, 1):.1f}" class="band"/>'
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5"/>' for x, y in pts)
    return f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{band}<polyline points="{path}"/>{dots}</svg>'
