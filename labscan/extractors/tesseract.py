"""Базовая линия без LLM: Tesseract OCR + разбор строк регулярками.

Нужна, чтобы честно показать, что даёт LLM: на чистых сканах OCR справляется
неплохо, на фото и нестандартных единицах ломается. Работает офлайн и бесплатно.
Координаты строк из TSV пересчитываются обратно в исходную страницу, чтобы экран
проверки мог подсветить, откуда взято значение.
"""

from __future__ import annotations

import csv
import io
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

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

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 в пикселях
Word = tuple[str, float] | tuple[str, float, Box]


def available() -> bool:
    return shutil.which("tesseract") is not None


@dataclass(frozen=True)
class Transform:
    """Что сделал preprocess с картинкой, чтобы вернуть координаты OCR в исходную страницу."""

    angle: float  # угол deskew в градусах, как в Image.rotate (против часовой), 0 — не поворачивали
    orig: tuple[int, int]  # размер исходной страницы
    rotated: tuple[int, int]  # размер после rotate(expand=True)
    scale: float = 1.0  # во сколько раз увеличили после поворота

    def to_page(self, box: Box) -> list[float]:
        """Прямоугольник в обработанном изображении -> доли ширины и высоты исходной страницы."""
        x0, y0, x1, y1 = (v / self.scale for v in box)
        pts = [self._unrotate(x, y) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        w, h = self.orig
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [round(min(max(v, 0.0), 1.0), 4) for v in (min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h)]

    def _unrotate(self, x: float, y: float) -> tuple[float, float]:
        if not self.angle:
            return x, y
        # Image.rotate поворачивает вокруг центра и центрирует результат на расширенном холсте.
        # Обратный поворот на тот же угол возвращает точку в систему исходной страницы.
        a = math.radians(self.angle)
        cx, cy = self.rotated[0] / 2, self.rotated[1] / 2
        dx, dy = x - cx, y - cy
        ox, oy = self.orig[0] / 2, self.orig[1] / 2
        return ox + dx * math.cos(a) - dy * math.sin(a), oy + dx * math.sin(a) + dy * math.cos(a)


def deskew_angle(img: Image.Image) -> float:
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
    return 0.0 if abs(best_angle) < 0.25 else best_angle


def deskew(img: Image.Image) -> Image.Image:
    angle = deskew_angle(img)
    return img if not angle else img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")


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


def preprocess(img: Image.Image) -> tuple[Image.Image, Transform]:
    angle = deskew_angle(img)
    rot = img if not angle else img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
    g = remove_rules(ImageOps.autocontrast(ImageOps.grayscale(rot), cutoff=1))
    scale = 1.0
    if g.width < 2000:
        scale = 1.6
        g = g.resize((int(g.width * scale), int(g.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(g, cutoff=1), Transform(angle, img.size, rot.size, scale)


def parse_tsv(out: str) -> list[list[Word]]:
    """TSV Tesseract -> строки из слов (текст, уверенность, прямоугольник)."""
    lines: dict[tuple[int, int, int], list[Word]] = {}
    for row in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if not text or text in {"|", "‘", "—"}:
            continue
        key = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        left, top = float(row["left"]), float(row["top"])
        box = (left, top, left + float(row["width"]), top + float(row["height"]))
        lines.setdefault(key, []).append((text.strip("|"), float(row["conf"]), box))
    return [ws for _, ws in sorted(lines.items()) if ws]


def _ocr_lines(img: Image.Image) -> list[list[Word]]:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "rus+eng", "--psm", "6", "tsv"],
        input=buf.getvalue(), capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace")
    return parse_tsv(out)


def _union(words: list[Word]) -> Box | None:
    boxes = [w[2] for w in words if len(w) > 2]
    if not boxes:
        return None
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def parse_line(words: list[Word]) -> RawResult | None:
    tokens = [w[0] for w in words if w[0]]
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
        box = _union(words)
        return RawResult(name=name, value=tok, unit=unit, reference=ref, legible=conf >= LOW_CONF,
                         box=list(box) if box else None)
    return None


def parse_page(lines: list[list[Word]], transform: Transform | None = None) -> RawReport:
    report = RawReport()
    for words in lines:
        text = " ".join(w[0] for w in words)
        for field, rx in _HEADER.items():
            m = rx.search(text)
            if m and getattr(report, field) is None:
                setattr(report, field, m.group(1).strip())
                break
        else:
            r = parse_line(words)
            if r:
                if r.box and transform:
                    r.box = transform.to_page(tuple(r.box))
                report.results.append(r)
    if lines:
        report.lab_name = " ".join(w[0] for w in lines[0])
    return report


class TesseractExtractor:
    name = "tesseract"

    def extract(self, pages: list[Image.Image]) -> Extraction:
        t0 = time.perf_counter()
        reports = []
        for p in pages:
            g, tr = preprocess(p)
            reports.append(parse_page(_ocr_lines(g), tr))
        return Extraction(merge_pages(reports), self.name, time.perf_counter() - t0)
