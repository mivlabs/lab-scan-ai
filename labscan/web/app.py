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

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..catalog import ANALYTES, BY_CODE, UNIT_LABELS_RU, Analyte
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
async def upload(file: UploadFile = File(...), s: Session = Depends(db)):
    doc = _process(await file.read(), file.filename or "scan", s)
    return RedirectResponse(f"/documents/{doc.id}", status_code=303)


@app.post("/try/{doc_id}")
def try_sample(doc_id: str, s: Session = Depends(db)):
    """Распознать готовый синтетический бланк: демо без поиска файла на диске."""
    sample = _sample(doc_id)
    doc = _process((SAMPLES / sample["file"]).read_bytes(), sample["file"], s)
    return RedirectResponse(f"/documents/{doc.id}", status_code=303)


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
    n = doc.normalized
    need = sum(o["status"] == "review" for o in n["observations"])
    total = len(n["observations"])
    stats = {"rows": total, "need": need, "ok": total - need,
             "manual_sec": total * MANUAL_SEC_PER_ROW, "review_sec": need * REVIEW_SEC_PER_ROW,
             "manual_per_row": MANUAL_SEC_PER_ROW, "review_per_row": REVIEW_SEC_PER_ROW}
    return templates.TemplateResponse(request, "review.html", {
        "doc": doc, "n": n, "pages": range(pages), "analytes": ANALYTES, "extra": range(EXTRA_ROWS), "stats": stats})


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
