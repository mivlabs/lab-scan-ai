"""Локальная vision-модель через Ollama: данные пациента не покидают сервер клиники.

Нужен запущенный Ollama (https://ollama.com) и скачанная модель, например
`ollama pull qwen2.5vl:7b`. API: POST /api/chat с полем format = JSON-схема.
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

DEFAULT_MODEL = "qwen2.5vl:7b"


class OllamaExtractor:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 300):
        self.model = model or os.getenv("LABSCAN_OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.schema = report_json_schema()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _page(self, img: Image.Image) -> RawReport:
        data = self._post({
            "model": self.model,
            "stream": False,
            "format": self.schema,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT,
                 "images": [base64.b64encode(to_png_bytes(img)).decode()]},
            ],
        })
        return RawReport.model_validate(json.loads(data["message"]["content"]))

    def extract(self, pages: list[Image.Image]) -> Extraction:
        t0 = time.perf_counter()
        reports = [self._page(p) for p in pages]
        return Extraction(merge_pages(reports), f"{self.name}:{self.model}", time.perf_counter() - t0)
