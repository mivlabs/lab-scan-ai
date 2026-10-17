"""Веб-прототип: загрузка бланка -> проверка человеком -> карта пациента -> выгрузка в МИС.

Запуск: uvicorn labscan.web.app:app --reload
"""

from __future__ import annotations

import io
import json
import os
import uuid
from datetime import date
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..catalog import ANALYTES, BY_CODE, UNIT_LABELS_RU, Analyte
from ..db import Correction, Document, EngineRun, ObservationRow, Patient, find_or_create_patient, make_session_factory
from ..extractors import available_engines, get_extractor
from ..fhir import to_bundle
from ..ingest import load_pages
from ..mis import OutboxMIS
from ..normalize import normalize_report, parse_date, parse_number
from ..schema import Flag

DATA = Path(os.getenv("LABSCAN_DATA", "data"))
UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
SAMPLES = Path(os.getenv("LABSCAN_SAMPLES", "data/synth"))  # синтетические бланки для кнопки «попробовать»
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
templates.env.filters["date_ru"] = lambda d: d.strftime("%d.%m.%Y") if d else ""
EXTRA_ROWS = 3  # пустые строки для показателей, которые распознавание пропустило

# Оценка экономии времени на экране проверки. Это допущение, а не измерение: оно показано в подсказке.
MANUAL_SEC_PER_ROW = 20  # ввести строку руками: найти на бланке, набрать число, выбрать показатель
REVIEW_SEC_PER_ROW = 5  # сверить помеченную строку со сканом

SAMPLE_PICKS = (("citolab", "scan"), ("senim", "photo"), ("labone", "pdf"))
LAYOUT_RU = {"citolab": "ЦитоЛаб · таблица с сеткой", "senim": "Сенім · строки с точками",
             "labone": "Lab One · американские единицы"}
KIND_RU = {"scan": "скан", "photo": "фото с телефона", "pdf": "PDF, две страницы"}

SessionLocal = make_session_factory()
mis = OutboxMIS(DATA / "mis_outbox")
_extractor = None
_thumbs: dict[str, bytes] = {}


def extractor():
    global _extractor
    if _extractor is None:
        _extractor = get_extractor()
    return _extractor


def engine_label() -> str:
    ex = extractor()
    model = getattr(ex, "model", None)
    return f"{ex.name} · {model}" if model else ex.name


def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def sample_docs() -> list[dict]:
    """Три примера для кнопки «попробовать»: по одному на вёрстку и тип файла."""
    index = SAMPLES / "index.json"
    if not index.exists():
        return []
    ids = json.loads(index.read_text(encoding="utf-8"))
    out = []
    for layout, kind in SAMPLE_PICKS:
        doc_id = next((i for i in ids if i.endswith(f"_{layout}_{kind}")), None)
        if doc_id and (SAMPLES / f"{doc_id}.json").exists():
            truth = json.loads((SAMPLES / f"{doc_id}.json").read_text(encoding="utf-8"))
            out.append({"id": doc_id, "file": truth["file"], "layout": LAYOUT_RU[layout], "kind": KIND_RU[kind],
                        "rows": len(truth["results"])})
    return out


def _sample(doc_id: str) -> dict:
    return next((d for d in sample_docs() if d["id"] == doc_id), None) or _404()


def _create(data: bytes, filename: str, s: Session) -> Document:
    """Сохранить файл и завести документ в состоянии «распознаётся»."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Формат {ext} не поддерживается: нужен PDF, PNG или JPG")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Файл больше 20 МБ")
    path = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(data)
    doc = Document(filename=filename, path=str(path), engine=engine_label(), status="processing", raw={}, normalized={})
    s.add(doc)
    s.commit()
    return doc


def _recognize(doc_id: int) -> None:
    """Распознавание в фоне: ошибка движка становится статусом документа, а не ответом 500."""
    s = SessionLocal()
    try:
        doc = s.get(Document, doc_id)
        if doc is None or doc.status != "processing":
            return
        try:
            result = extractor().extract(load_pages(Path(doc.path).read_bytes(), doc.filename))
            norm = normalize_report(result.report)
            doc.engine, doc.raw, doc.normalized = result.engine, result.report.model_dump(), norm.model_dump(mode="json")
            doc.seconds, doc.cost_usd, doc.status, doc.error = result.seconds, result.cost_usd, "review", None
        except Exception as e:  # noqa: BLE001 — любая ошибка движка показывается пользователю с кнопкой «повторить»
            doc.status, doc.error = "failed", f"{type(e).__name__}: {e}"[:500]
        s.commit()
    finally:
        s.close()


def _enqueue(data: bytes, filename: str, s: Session, background: BackgroundTasks) -> Document:
    doc = _create(data, filename, s)
    background.add_task(_recognize, doc.id)
    return doc


def _process(data: bytes, filename: str, s: Session) -> Document:
    """Синхронный путь для API интеграций: ответ сразу с результатом."""
    doc = _create(data, filename, s)
    _recognize(doc.id)
    s.refresh(doc)
    if doc.status == "failed":
        raise HTTPException(502, f"Распознавание не удалось: {doc.error}")
    return doc


def review_count(normalized: dict) -> int:
    return len(normalized.get("issues") or []) + sum(o["status"] == "review" for o in normalized["observations"])


templates.env.globals["review_count"] = review_count


# ---------- HTML ----------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, s: Session = Depends(db)):
    docs = s.scalars(select(Document).order_by(Document.id.desc()).limit(30)).all()
    patients = s.scalars(select(Patient).order_by(Patient.full_name)).all()
    total = s.scalar(select(func.count()).select_from(Document)) or 0
    confirmed = s.scalar(select(func.count()).select_from(Document).where(Document.status == "confirmed")) or 0
    return templates.TemplateResponse(request, "index.html", {
        "docs": docs, "patients": patients, "engine": engine_label(), "samples": sample_docs(),
        "stats": {"total": total, "confirmed": confirmed, "waiting": total - confirmed}})


@app.post("/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...), s: Session = Depends(db)):
    doc = _enqueue(await file.read(), file.filename or "scan", s, background)
    return RedirectResponse(f"/documents/{doc.id}", status_code=303)


@app.post("/try/{doc_id}")
def try_sample(doc_id: str, background: BackgroundTasks, s: Session = Depends(db)):
    """Распознать готовый синтетический бланк: демо без поиска файла на диске."""
    sample = _sample(doc_id)
    doc = _enqueue((SAMPLES / sample["file"]).read_bytes(), sample["file"], s, background)
    return RedirectResponse(f"/documents/{doc.id}", status_code=303)


@app.post("/documents/{doc_id}/retry")
def retry(doc_id: int, background: BackgroundTasks, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    if doc.status == "failed":
        doc.status, doc.error = "processing", None
        s.commit()
        background.add_task(_recognize, doc.id)
    return RedirectResponse(f"/documents/{doc.id}", status_code=303)


@app.get("/api/documents/{doc_id}/status")
def api_status(doc_id: int, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    return {"status": doc.status, "engine": doc.engine, "seconds": round(doc.seconds, 2), "error": doc.error}


@app.get("/samples/{doc_id}/thumb.png")
def sample_thumb(doc_id: str):
    if doc_id not in _thumbs:
        sample = _sample(doc_id)
        page = load_pages((SAMPLES / sample["file"]).read_bytes(), sample["file"])[0]
        page.thumbnail((420, 420))
        buf = io.BytesIO()
        page.save(buf, "PNG", optimize=True)
        _thumbs[doc_id] = buf.getvalue()
    return Response(_thumbs[doc_id], media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/documents/{doc_id}", response_class=HTMLResponse)
def review(doc_id: int, request: Request, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    pages = len(load_pages(Path(doc.path).read_bytes(), doc.path))
    if doc.status in ("processing", "failed"):
        return templates.TemplateResponse(request, "processing.html", {"doc": doc, "pages": range(pages)})
    n = doc.normalized
    need = sum(o["status"] == "review" for o in n["observations"])
    total = len(n["observations"])
    stats = {"rows": total, "need": need, "ok": total - need,
             "manual_sec": total * MANUAL_SEC_PER_ROW, "review_sec": need * REVIEW_SEC_PER_ROW,
             "manual_per_row": MANUAL_SEC_PER_ROW, "review_per_row": REVIEW_SEC_PER_ROW}
    has_boxes = any(o["raw"].get("box") for o in n["observations"])
    return templates.TemplateResponse(request, "review.html", {
        "doc": doc, "n": n, "pages": range(pages), "analytes": ANALYTES, "extra": range(EXTRA_ROWS), "stats": stats,
        "has_boxes": has_boxes})


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
    panels = [(title, [a.code for a in ANALYTES if a.panel == key and a.code in series])
              for key, title in (("cbc", "Общий анализ крови"), ("bio", "Биохимия"))]
    docs = s.scalars(select(Document).where(Document.patient_id == p.id).order_by(Document.id.desc())).all()
    abnormal = sum(1 for code in series if series[code][-1].flag in ("H", "L"))
    return templates.TemplateResponse(request, "patient.html", {
        "p": p, "series": series, "panels": [(t, codes) for t, codes in panels if codes], "by_code": BY_CODE,
        "docs": docs, "chart": chart, "age": _age(p.birth_date), "abnormal": abnormal})


# ---------- сравнение движков на одном бланке ----------

def _run_engine(run_id: int) -> None:
    s = SessionLocal()
    try:
        run = s.get(EngineRun, run_id)
        doc = s.get(Document, run.document_id)
        try:
            name, _, model = run.engine.partition(":")
            result = get_extractor(name, model or None).extract(load_pages(Path(doc.path).read_bytes(), doc.filename))
            norm = normalize_report(result.report)
            run.engine, run.raw, run.normalized = result.engine, result.report.model_dump(), norm.model_dump(mode="json")
            run.seconds, run.cost_usd, run.status = result.seconds, result.cost_usd, "done"
        except Exception as e:  # noqa: BLE001 — ошибка движка показывается в колонке
            run.status, run.error = "failed", f"{type(e).__name__}: {e}"[:500]
        s.commit()
    finally:
        s.close()


def compare_matrix(columns: list[dict]) -> list[dict]:
    """Строки = показатели справочника (в порядке справочника) и нераспознанные названия; ячейки по колонкам.
    Строка помечена diff, если значения расходятся или показатель есть не во всех колонках."""
    per_col: list[dict] = []
    for c in columns:
        obs = (c.get("norm") or {}).get("observations") or []
        by: dict = {}
        for o in obs:
            key = o["code"] or f"?{o['raw']['name']}"
            by.setdefault(key, o)
        per_col.append(by)
    keys = [a.code for a in ANALYTES if any(a.code in by for by in per_col)]
    keys += sorted({k for by in per_col for k in by if k.startswith("?")})
    rows = []
    for key in keys:
        cells = [by.get(key) for by in per_col]
        values = [c["value"] for c in cells if c and c["value"] is not None]
        ref = values[0] if values else None
        same = all(abs(v - ref) <= max(0.011, abs(ref) * 0.002) for v in values) if ref is not None else True
        diff = any(c is None for c in cells) or not same
        rows.append({"code": key, "name": BY_CODE[key].name_ru if key in BY_CODE else key[1:],
                     "unit": BY_CODE[key].unit if key in BY_CODE else None, "cells": cells, "diff": diff})
    return rows


@app.get("/documents/{doc_id}/compare", response_class=HTMLResponse)
def compare(doc_id: int, request: Request, s: Session = Depends(db)):
    doc = s.get(Document, doc_id) or _404()
    runs = s.scalars(select(EngineRun).where(EngineRun.document_id == doc.id).order_by(EngineRun.id)).all()
    columns = [{"label": doc.engine, "norm": doc.normalized if doc.status not in ("processing", "failed") else None,
                "seconds": doc.seconds, "cost": doc.cost_usd, "status": doc.status, "main": True}]
    columns += [{"label": r.engine, "norm": r.normalized if r.status == "done" else None, "seconds": r.seconds,
                 "cost": r.cost_usd, "status": r.status, "error": r.error, "id": r.id} for r in runs]
    rows = compare_matrix(columns)
    pending = [r.id for r in runs if r.status == "processing"]
    return templates.TemplateResponse(request, "compare.html", {
        "doc": doc, "columns": columns, "rows": rows, "options": available_engines(), "pending": pending,
        "diff_count": sum(r["diff"] for r in rows)})


@app.post("/documents/{doc_id}/compare")
def compare_run(doc_id: int, request: Request, background: BackgroundTasks, s: Session = Depends(db),
                engine: str = Form(...), model: str = Form("")):
    doc = s.get(Document, doc_id) or _404()
    if engine not in ("claude", "openrouter", "ollama", "tesseract"):
        raise HTTPException(400, "неизвестный движок")
    run = EngineRun(document_id=doc.id, engine=f"{engine}:{model}" if model else engine)
    s.add(run)
    s.commit()
    background.add_task(_run_engine, run.id)
    return RedirectResponse(f"/documents/{doc.id}/compare", status_code=303)


@app.get("/api/engine-runs/{run_id}/status")
def api_run_status(run_id: int, s: Session = Depends(db)):
    run = s.get(EngineRun, run_id) or _404()
    return {"status": run.status, "engine": run.engine, "error": run.error}


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


def _age(birth: date | None) -> int | None:
    if not birth:
        return None
    today = date.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def chart(rows: list[ObservationRow], a: Analyte | None = None, w: int = 320, h: int = 112) -> str:
    """График динамики показателя: референсная полоса, линия, точки с цветом флага, даты по краям."""
    vals = [r.value for r in rows]
    if not vals:
        return ""
    lows = [r.ref_low for r in rows if r.ref_low is not None]
    highs = [r.ref_high for r in rows if r.ref_high is not None]
    lo = min(vals + lows[-1:])
    hi = max(vals + highs[-1:])
    pad = (hi - lo) * 0.18 or abs(hi) * 0.1 or 1
    lo, hi = lo - pad, hi + pad
    left, right, top, bottom = 8, 8, 10, 24
    plot_w, plot_h = w - left - right, h - top - bottom

    def x(i: int) -> float:
        return left + (plot_w / 2 if len(vals) == 1 else i * plot_w / (len(vals) - 1))

    def y(v: float) -> float:
        return top + (hi - v) / (hi - lo) * plot_h

    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="динамика">']
    if lows and highs:
        y1, y2 = y(highs[-1]), y(lows[-1])
        parts.append(f'<rect class="band" x="{left}" y="{y1:.1f}" width="{plot_w}" height="{max(y2 - y1, 1):.1f}" rx="2"/>')
        parts.append(f'<line class="band-edge" x1="{left}" x2="{left + plot_w}" y1="{y1:.1f}" y2="{y1:.1f}"/>')
        parts.append(f'<line class="band-edge" x1="{left}" x2="{left + plot_w}" y1="{y2:.1f}" y2="{y2:.1f}"/>')
    pts = [(x(i), y(v)) for i, v in enumerate(vals)]
    if len(pts) > 1:
        parts.append('<polyline class="line" points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + '"/>')
    for (px, py), r in zip(pts, rows, strict=True):
        parts.append(f'<circle class="pt flag-{r.flag or "none"}" cx="{px:.1f}" cy="{py:.1f}" r="4"/>')
    lx, ly = pts[-1]
    anchor = "end" if len(pts) > 1 else "middle"
    parts.append(f'<text class="val" x="{lx:.1f}" y="{max(ly - 8, 9):.1f}" text-anchor="{anchor}">'
                 f'{_fmt(vals[-1], a.code if a else None)}</text>')
    dates = [r.collected_at for r in rows]
    labels = [(pts[0][0], dates[0], "start")] if len(pts) > 1 else [(pts[0][0], dates[0], "middle")]
    if len(pts) > 1:
        labels.append((pts[-1][0], dates[-1], "end"))
    for px, d, anc in labels:
        if d:
            parts.append(f'<text class="axis" x="{px:.1f}" y="{h - 7}" text-anchor="{anc}">{d.strftime("%d.%m.%y")}</text>')
    parts.append("</svg>")
    return "".join(parts)
