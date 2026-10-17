"""Vision-модели через OpenRouter (OpenAI-совместимый API, оплата без зарубежной карты).

Нужен OPENROUTER_API_KEY. Модель задаётся LABSCAN_OPENROUTER_MODEL; подойдёт любая
vision-модель, которая поддерживает response_format = json_schema.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request

from PIL import Image

from ..ingest import to_png_bytes
from ..schema import RawReport
from .base import Extraction, merge_pages
from .llm_common import SYSTEM_PROMPT, USER_PROMPT, report_json_schema

DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"
URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterExtractor:
    name = "openrouter"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 180):
        self.model = model or os.getenv("LABSCAN_OPENROUTER_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.schema = report_json_schema()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _page(self, img: Image.Image) -> tuple[RawReport, float]:
        data_url = "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode()
        data = self._post({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "lab_report", "strict": True, "schema": self.schema}},
            "usage": {"include": True},
        })
        if "error" in data:
            raise RuntimeError(f"OpenRouter: {data['error']}")
        content = data["choices"][0]["message"]["content"]
        return RawReport.model_validate(json.loads(content)), float(data.get("usage", {}).get("cost") or 0.0)

    def extract(self, pages: list[Image.Image]) -> Extraction:
        t0 = time.perf_counter()
        reports, cost = [], 0.0
        for p in pages:
            r, c = self._page(p)
            reports.append(r)
            cost += c
        return Extraction(merge_pages(reports), f"{self.name}:{self.model}", time.perf_counter() - t0, cost)
