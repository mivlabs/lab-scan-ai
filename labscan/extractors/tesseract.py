"""Базовая линия без LLM: Tesseract OCR + разбор строк регулярками.

Нужна, чтобы честно показать, что даёт LLM: на чистых сканах OCR справляется
неплохо, на фото и нестандартных единицах ломается. Работает офлайн и бесплатно.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps

from ..catalog import match_analyte
from ..schema import RawReport, RawResult
from .base import Extraction, merge_pages

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_VALUE_TOKEN = re.compile(rf"^[<>]?{_NUM}$")
_REF_TAIL = re.compile(rf"\(?\s*((?:до\s*|<\s*)?{_NUM}(?:\s*[-–—]\s*{_NUM})?)\s*\)?\s*$", re.I)
_HEADER = {
    "patient_name": re.compile(r"(?:пациент|patient)\s*[:;]?\s*(.+)", re.I),
    "birth_date": re.compile(r"(?:дата рождения|date of birth|д\.р\.)\s*[:;]?\s*(.+)", re.I),
    "collected_at": re.compile(r"(?:дата взятия[^:]*|collected|дата забора)\s*[:;]?\s*(.+)", re.I),
}
LOW_CONF = 60


def available() -> bool:
    return shutil.which("tesseract") is not None


def deskew(img: Image.Image) -> Image.Image:
    """Подбор угла поворота по максимуму «полосатости» горизонтальной проекции."""
    small = ImageOps.grayscale(img)
    small.thumbnail((800, 800))
    arr = 255 - np.asarray(small, dtype=np.float32)
    best, best_angle = -1.0, 0.0
    for angle in np.arange(-5, 5.01, 0.5):
        rot = np.asarray(Image.fromarray(arr.astype("uint8")).rotate(float(angle), fillcolor=0), dtype=np.float32)
        score = float(np.var(rot.sum(axis=1)))
        if score > best:
            best, best_angle = score, float(angle)
    if abs(best_angle) < 0.25:
        return img
    return img.rotate(best_angle, resample=Image.BICUBIC, expand=True, fillcolor="white")


def remove_rules(g: Image.Image, min_len: int = 80) -> Image.Image:
    """Стереть линии таблицы: Tesseract склеивает их с текстом и теряет строки."""
    arr = np.asarray(g, dtype=np.uint8)
    dark = arr < 128
    mask = np.zeros_like(dark)
    for axis in (1, 0):
        c = np.cumsum(dark, axis=axis, dtype=np.int32)
        c = np.concatenate([np.zeros_like(c.take([0], axis=axis)), c], axis=axis)
        n = dark.shape[axis]
        if n <= min_len:
            continue
        win = c.take(range(min_len, n + 1), axis=axis) - c.take(range(0, n - min_len + 1), axis=axis)
        full = win >= min_len  # окно целиком тёмное -> начало линии
        run = np.zeros_like(dark)
        for k in range(min_len):
            sl = [slice(None)] * 2
            sl[axis] = slice(k, k + full.shape[axis])
            run[tuple(sl)] |= full
        mask |= run
    mask = mask | np.roll(mask, 1, 0) | np.roll(mask, -1, 0) | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
    out = arr.copy()
    out[mask] = 255
    return Image.fromarray(out)


def preprocess(img: Image.Image) -> Image.Image:
    g = remove_rules(ImageOps.autocontrast(ImageOps.grayscale(deskew(img)), cutoff=1))
    if g.width < 2000:
        g = g.resize((int(g.width * 1.6), int(g.height * 1.6)), Image.LANCZOS)
    return ImageOps.autocontrast(g, cutoff=1)


def _ocr_lines(img: Image.Image) -> list[list[tuple[str, float]]]:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "rus+eng", "--psm", "6", "tsv"],
        input=buf.getvalue(), capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace")
    lines: dict[tuple[int, int, int], list[tuple[str, float]]] = {}
    for row in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if not text or text in {"|", "‘", "—"}:
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        lines.setdefault(key, []).append((text.strip("|"), float(row["conf"])))
    return [ws for _, ws in sorted(lines.items()) if ws]


def parse_line(words: list[tuple[str, float]]) -> RawResult | None:
    tokens = [w for w, _ in words if w]
    for i, tok in enumerate(tokens):
        if i == 0 or not _VALUE_TOKEN.match(tok):
            continue
        name = " ".join(tokens[:i])
        name = re.sub(r"\.{2,}.*$", "", name).strip(" .:")  # точки-заполнители
        if not re.search(r"[A-Za-zА-Яа-яЁё]{2,}", name):
            continue
        analyte, score = match_analyte(name)
        if analyte is None or score < 0.6:
            return None
        rest = " ".join(tokens[i + 1:])
        m = _REF_TAIL.search(rest)
        ref = m.group(1) if m else None
        unit = (rest[:m.start()] if m else rest).strip(" ()") or None
        conf = words[i][1]
        return RawResult(name=name, value=tok, unit=unit, reference=ref, legible=conf >= LOW_CONF)
    return None


def parse_page(lines: list[list[tuple[str, float]]]) -> RawReport:
    report = RawReport()
    for words in lines:
        text = " ".join(w for w, _ in words)
        for field, rx in _HEADER.items():
            m = rx.search(text)
            if m and getattr(report, field) is None:
                setattr(report, field, m.group(1).strip())
                break
        else:
            r = parse_line(words)
            if r:
                report.results.append(r)
    if lines:
        report.lab_name = " ".join(w for w, _ in lines[0])
    return report


class TesseractExtractor:
    name = "tesseract"

    def extract(self, pages: list[Image.Image]) -> Extraction:
        t0 = time.perf_counter()
        reports = [parse_page(_ocr_lines(preprocess(p))) for p in pages]
        return Extraction(merge_pages(reports), self.name, time.perf_counter() - t0)
