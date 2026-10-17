"""Распознавание через Claude (vision + структурированный вывод по JSON-схеме)."""

from __future__ import annotations

import base64
import json
import os
import time

from PIL import Image

from ..ingest import to_png_bytes
from ..schema import RawReport
from .base import Extraction, merge_pages
from .llm_common import SYSTEM_PROMPT, USER_PROMPT, report_json_schema

DEFAULT_MODEL = "claude-opus-5-5"
# $ за 1M токенов (вход, выход) — для оценки стоимости страницы в eval
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


class ExtractionRefused(RuntimeError):
    pass


class ClaudeExtractor:
    name = "claude"

    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        self.model = model or os.getenv("LABSCAN_CLAUDE_MODEL", DEFAULT_MODEL)
        # Переписывание бланка — не задача на рассуждение, low хватает и дешевле
        self.effort = effort or os.getenv("LABSCAN_CLAUDE_EFFORT", "low")
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.schema = report_json_schema()

    def _page(self, img: Image.Image) -> tuple[RawReport, dict]:
        data = base64.standard_b64encode(to_png_bytes(img)).decode()
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": USER_PROMPT},
            ]}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}},
            # Если классификатор безопасности отклонит запрос, API сам повторит его на запасной модели
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise ExtractionRefused(getattr(response.stop_details, "explanation", None) or "модель отказалась")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("ответ обрезан по max_tokens")
        text = next(b.text for b in response.content if b.type == "text")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return RawReport.model_validate(json.loads(text)), usage

    def extract(self, pages: list[Image.Image]) -> Extraction:
        t0 = time.perf_counter()
        reports, tin, tout = [], 0, 0
        for p in pages:
            r, u = self._page(p)
            reports.append(r)
            tin += u["input_tokens"]
            tout += u["output_tokens"]
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        cost = tin / 1e6 * pin + tout / 1e6 * pout
        return Extraction(merge_pages(reports), f"{self.name}:{self.model}", time.perf_counter() - t0, cost,
                          {"input_tokens": tin, "output_tokens": tout})
