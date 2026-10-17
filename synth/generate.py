"""Генератор синтетических бланков анализов с эталонной разметкой.

Все пациенты и лаборатории вымышленные. Три вёрстки:
  citolab  — таблица с сеткой, русские названия и единицы;
  senim    — строки с точками-заполнителями, «Название (АББР) .... значение ед. (референс)»;
  labone   — англоязычный бланк с американскими единицами (g/dL, mg/dL, 10^3/uL),
             проверяет перевод единиц.
Каждый бланк сохраняется как скан (PNG), «фото с телефона» (JPG с поворотом,
размытием, шумом и неровным светом) или многостраничный PDF.

Запуск: python -m synth.generate --n 50 --out data/synth
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from labscan.catalog import BY_CODE, UNIT_LABELS_RU, Analyte

FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation")
W, H = 1240, 1754  # A4 при 150 dpi

SURNAMES = [("Ахметов", "Ахметова"), ("Иванов", "Иванова"), ("Сериков", "Серикова"), ("Ким", "Ким"),
            ("Нурланов", "Нурланова"), ("Петров", "Петрова"), ("Жумабаев", "Жумабаева"), ("Ли", "Ли")]
NAMES_M = ["Ерлан", "Сергей", "Данияр", "Алексей", "Тимур", "Асхат"]
NAMES_F = ["Айгерим", "Мария", "Динара", "Елена", "Жанна", "Алия"]
PATRON = [("Болатович", "Болатовна"), ("Петрович", "Петровна"), ("Серикович", "Сериковна"),
          ("Ахметович", "Ахметовна")]

CBC = ["HGB", "RBC", "WBC", "PLT", "HCT", "MCV", "NEUT%", "LYMPH%", "ESR"]
BIO = ["GLU", "CHOL", "HDL", "LDL", "TG", "CREA", "UREA", "ALT", "AST", "TBIL", "CRP", "HBA1C", "TSH"]

# Единицы, которыми печатает американская вёрстка
US_UNITS = {"HGB": "g/dL", "RBC": "10^6/uL", "WBC": "10^3/uL", "PLT": "10^3/uL", "GLU": "mg/dL",
            "CHOL": "mg/dL", "HDL": "mg/dL", "LDL": "mg/dL", "TG": "mg/dL", "CREA": "mg/dL", "TBIL": "mg/dL"}
US_LABELS = {"g/dL": "g/dL", "10^6/uL": "x10^6/uL", "10^3/uL": "x10^3/uL", "mg/dL": "mg/dL",
             "mmol/L": "mmol/L", "U/L": "U/L", "mg/L": "mg/L", "%": "%", "fL": "fL", "mm/h": "mm/h",
             "mIU/L": "mIU/L", "umol/L": "umol/L", "g/L": "g/L", "10^9/L": "x10^9/L", "10^12/L": "x10^12/L"}

LABS = {
    "citolab": "ЦитоЛаб — медицинская лаборатория",
    "senim": "КДЛ «Сенiм», г. Астана",
    "labone": "Lab One Diagnostics",
}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


@dataclass
class Row:
    code: str
    printed_name: str
    printed_value: str
    printed_unit: str
    printed_ref: str
    unit_key: str  # канонический ключ единицы, в которой напечатано


def _fmt(v: float, decimals: int, comma: bool) -> str:
    s = f"{v:.{decimals}f}"
    return s.replace(".", ",") if comma else s


def _sample_value(a: Analyte, rng: random.Random) -> float:
    lo, hi = a.ref
    if rng.random() < 0.25:  # отклонение от нормы
        if rng.random() < 0.5 and lo > 0:
            v = lo * rng.uniform(0.6, 0.95)
        else:
            v = hi * rng.uniform(1.05, 1.25 if a.unit == "%" else 1.8)
    else:
        v = rng.uniform(lo, hi) if hi > lo else hi * rng.random()
    return min(max(v, a.plausible[0]), a.plausible[1] * 0.9)


def make_rows(layout: str, codes: list[str], rng: random.Random) -> list[Row]:
    rows = []
    for code in codes:
        a = BY_CODE[code]
        canon = _sample_value(a, rng)
        if layout == "labone":
            unit_key = US_UNITS.get(code, a.unit)
            k = 1.0 if unit_key == a.unit else a.conversions[unit_key]
            if unit_key == "g/dL":
                dec = 1
            elif unit_key == "mg/dL":
                dec = 2 if code in ("CREA", "TBIL") else 0
            else:
                dec = a.decimals
            val = canon / k
            ref_lo, ref_hi = a.ref[0] / k, a.ref[1] / k
            name = code.replace("%", " %") if code in ("NEUT%", "LYMPH%") else code
            name = {"HBA1C": "HbA1c", "TBIL": "Bilirubin, total", "CREA": "Creatinine", "GLU": "Glucose",
                    "CHOL": "Cholesterol, total", "TG": "Triglycerides", "CRP": "CRP"}.get(code, name)
            rows.append(Row(code, name, _fmt(val, dec, False), US_LABELS[unit_key],
                            f"{_fmt(ref_lo, dec, False)} - {_fmt(ref_hi, dec, False)}", unit_key))
        else:
            comma = layout == "citolab"
            abbr = code.rstrip("%") if layout == "senim" else None
            name = a.name_ru + (f" ({abbr})" if abbr and abbr.lower() != a.name_ru.lower() else "")
            if layout == "citolab" and code in ("NEUT%", "LYMPH%"):
                name = a.name_ru + ", %"
            ref = (f"до {_fmt(a.ref[1], a.decimals, comma)}" if a.ref[0] == 0
                   else f"{_fmt(a.ref[0], a.decimals, comma)} - {_fmt(a.ref[1], a.decimals, comma)}")
            rows.append(Row(code, name, _fmt(canon, a.decimals, comma), UNIT_LABELS_RU[a.unit], ref, a.unit))
    return rows


def truth_value(row: Row) -> float:
    a = BY_CODE[row.code]
    v = float(row.printed_value.replace(",", "."))
    k = 1.0 if row.unit_key == a.unit else a.conversions[row.unit_key]
    return round(v * k, max(a.decimals, 2))


# ---------- вёрстки ----------

def _header(d: ImageDraw.ImageDraw, layout: str, meta: dict, y: int) -> int:
    big, mid = font("DejaVuSans-Bold.ttf", 34), font("DejaVuSans.ttf", 24)
    if layout == "labone":
        d.text((80, y), LABS[layout], font=big, fill=0)
        y += 60
        lines = [f"Patient: {meta['patient_lat']}", f"Date of birth: {meta['birth'].strftime('%Y-%m-%d')}",
                 f"Collected: {meta['collected'].strftime('%Y-%m-%d')}", f"Order #: {meta['order']}"]
    else:
        d.text((80, y), LABS[layout], font=big, fill=0)
        y += 60
        lines = [f"Пациент: {meta['patient']}", f"Дата рождения: {meta['birth'].strftime('%d.%m.%Y')}",
                 f"Дата взятия биоматериала: {meta['collected'].strftime('%d.%m.%Y')}", f"Заказ №: {meta['order']}"]
    for line in lines:
        d.text((80, y), line, font=mid, fill=0)
        y += 36
    return y + 20


def render_page(layout: str, meta: dict, title: str, rows: list[Row]) -> Image.Image:
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    y = _header(d, layout, meta, 70)
    tfont = font("DejaVuSans-Bold.ttf", 28)
    d.text((80, y), title, font=tfont, fill=0)
    y += 56
    if layout == "citolab":
        f, fb = font("DejaVuSans.ttf", 23), font("DejaVuSans-Bold.ttf", 23)
        cols = [80, 560, 760, 930, 1160]
        heads = ["Показатель", "Результат", "Ед. изм.", "Референс"]
        d.rectangle((cols[0], y, cols[-1], y + 44), outline=0, width=2)
        for i, h in enumerate(heads):
            d.text((cols[i] + 10, y + 9), h, font=fb, fill=0)
        y += 44
        for r in rows:
            d.rectangle((cols[0], y, cols[-1], y + 42), outline=0, width=1)
            for c in cols[1:-1]:
                d.line((c, y, c, y + 42), fill=0, width=1)
            for i, txt in enumerate([r.printed_name, r.printed_value, r.printed_unit, r.printed_ref]):
                d.text((cols[i] + 10, y + 9), txt, font=f, fill=0)
            y += 42
    elif layout == "senim":
        f = font("DejaVuSerif.ttf", 24)
        for r in rows:
            left = r.printed_name + " "
            right = f"  {r.printed_value} {r.printed_unit}   ({r.printed_ref})"
            lw = d.textlength(left, font=f)
            dots = "." * max(3, int((640 - lw) / d.textlength(".", font=f)))
            d.text((80, y), left + dots, font=f, fill=0)
            d.text((740, y), right, font=f, fill=0)
            y += 44
    else:  # labone
        f, fb = font("LiberationSans-Regular.ttf", 25), font("LiberationSans-Bold.ttf", 25)
        cols = [80, 520, 720, 900]
        for i, h in enumerate(["Test", "Result", "Units", "Reference range"]):
            d.text((cols[i], y), h, font=fb, fill=0)
        y += 40
        d.line((80, y, 1160, y), fill=0, width=2)
        y += 12
        for r in rows:
            for i, txt in enumerate([r.printed_name, r.printed_value, r.printed_unit, r.printed_ref]):
                d.text((cols[i], y), txt, font=f, fill=0)
            y += 42
    sm = font("DejaVuSans.ttf", 18)
    d.text((80, H - 90), "Синтетический бланк для демонстрации прототипа. Данные вымышлены.", font=sm, fill=90)
    return img


# ---------- «фото с телефона» ----------

def photo_effect(img: Image.Image, rng: random.Random) -> Image.Image:
    import numpy as np

    angle = rng.uniform(-4, 4)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=235)
    # лёгкая перспектива
    w, h = img.size
    dx, dy = w * 0.03, h * 0.02
    quad = (rng.uniform(0, dx), rng.uniform(0, dy), rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), h - rng.uniform(0, dy), w - rng.uniform(0, dx), rng.uniform(0, dy))
    img = img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=235)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.1)))
    arr = np.asarray(img, dtype=np.float32)
    # неровный свет: градиент от угла
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(w, h)
    arr = arr * (1 - 0.35 * dist) + rng.uniform(-10, 10)
    arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, 7, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=rng.randint(50, 70))
    return Image.open(io.BytesIO(buf.getvalue()))


# ---------- сборка ----------

def make_patient(rng: random.Random) -> dict:
    female = rng.random() < 0.5
    sur = rng.choice(SURNAMES)[1 if female else 0]
    name = rng.choice(NAMES_F if female else NAMES_M)
    pat = rng.choice(PATRON)[1 if female else 0]
    birth = date(1950, 1, 1) + timedelta(days=rng.randint(0, 365 * 55))
    return {"patient": f"{sur} {name} {pat}", "birth": birth, "sex": "F" if female else "M"}


TRANSLIT = str.maketrans({
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y",
    "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh", "Ы": "Y", "Э": "E", "Ю": "Yu", "Я": "Ya",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "ы": "y", "э": "e", "ю": "yu", "я": "ya",
    "ь": "", "ъ": "",
})


def generate(n: int, out: Path, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    patients = [make_patient(rng) for _ in range(max(3, n // 3))]  # у пациентов по несколько бланков
    for i in range(n):
        layout = ["citolab", "senim", "labone"][i % 3]
        p = rng.choice(patients)
        collected = date(2026, 1, 10) + timedelta(days=rng.randint(0, 270))
        meta = {**p, "patient_lat": p["patient"].translate(TRANSLIT), "collected": collected,
                "order": f"{rng.randint(100000, 999999)}"}
        kind = rng.choices(["scan", "photo", "pdf"], weights=[4, 4, 2])[0]
        pages = []
        truth_rows = []
        panels = [("Общий анализ крови" if layout != "labone" else "Complete Blood Count", CBC),
                  ("Биохимический анализ крови" if layout != "labone" else "Chemistry Panel", BIO)]
        chosen = panels if kind == "pdf" else [rng.choice(panels)]
        for title, pool in chosen:
            codes = sorted(rng.sample(pool, k=rng.randint(min(6, len(pool)), len(pool))), key=pool.index)
            rows = make_rows(layout, codes, rng)
            truth_rows += rows
            pages.append(render_page(layout, meta, title, rows))

        doc_id = f"{i + 1:03d}_{layout}_{kind}"
        if kind == "pdf":
            path = out / f"{doc_id}.pdf"
            pages[0].convert("RGB").save(path, "PDF", resolution=150, save_all=True,
                                         append_images=[pg.convert("RGB") for pg in pages[1:]])
        elif kind == "photo":
            path = out / f"{doc_id}.jpg"
            photo_effect(pages[0], rng).save(path, "JPEG", quality=92)
        else:
            path = out / f"{doc_id}.png"
            pages[0].save(path)

        truth = {
            "doc_id": doc_id, "file": path.name, "layout": layout, "kind": kind,
            "patient_name": meta["patient_lat"] if layout == "labone" else meta["patient"],
            "birth_date": p["birth"].isoformat(), "collected_at": collected.isoformat(),
            "results": [{"code": r.code, "value": truth_value(r), "printed": {
                "name": r.printed_name, "value": r.printed_value, "unit": r.printed_unit, "reference": r.printed_ref}}
                for r in truth_rows],
        }
        (out / f"{doc_id}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append(truth)
    (out / "index.json").write_text(json.dumps([t["doc_id"] for t in index], indent=2), encoding="utf-8")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    docs = generate(args.n, args.out, args.seed)
    print(f"Сгенерировано {len(docs)} бланков в {args.out}")


if __name__ == "__main__":
    main()
