"""Файл (PDF, PNG, JPG) -> список страниц-изображений."""

from __future__ import annotations

import io

from PIL import Image, ImageOps

MAX_SIDE = 2400  # больше не нужно ни OCR, ни vision-модели, а запрос дешевле


def load_pages(data: bytes, filename: str = "") -> list[Image.Image]:
    if data[:4] == b"%PDF" or filename.lower().endswith(".pdf"):
        import pymupdf

        pages = []
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return [_limit(p) for p in pages]
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")  # фото с телефона часто повёрнуто через EXIF
    return [_limit(img)]


def _limit(img: Image.Image) -> Image.Image:
    if max(img.size) > MAX_SIDE:
        img = img.copy()
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
