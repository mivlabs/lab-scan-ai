from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from ..schema import RawReport


@dataclass
class Extraction:
    report: RawReport
    engine: str
    seconds: float = 0.0
    cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)


class Extractor(Protocol):
    name: str

    def extract(self, pages: list[Image.Image]) -> Extraction: ...


def merge_pages(reports: list[RawReport]) -> RawReport:
    """Шапку берём с первой страницы, где она есть; показатели склеиваем."""
    merged = RawReport()
    for i, r in enumerate(reports):
        for f in ("lab_name", "patient_name", "birth_date", "collected_at"):
            if getattr(merged, f) is None and getattr(r, f):
                setattr(merged, f, getattr(r, f))
        for res in r.results:
            if res.page is None:
                res.page = i
        merged.results.extend(r.results)
    return merged
