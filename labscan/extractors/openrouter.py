"""Vision-модели через OpenRouter (OpenAI-совместимый API, оплата без зарубежной карты).

Нужен OPENROUTER_API_KEY. Модель задаётся LABSCAN_OPENROUTER_MODEL; подойдёт любая
vision-модель, которая поддерживает response_format = json_schema.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image

from ..ingest import to_png_bytes
from ..schema import RawReport
from .base import Extraction, merge_pages
from .llm_common import SYSTEM_PROMPT, USER_PROMPT, report_json_schema

DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
# Переписывание бланка — не задача на рассуждение. Рассуждающие модели (Nemotron, Qwen3-thinking, Gemini)
# с полным reasoning тратят весь лимит на размышления и отдают пустой JSON; с выключенным (off) Nemotron
# помечал каждое значение как неуверенное. По умолчанию короткое рассуждение: off | low | medium | high | on
REASONING = os.getenv("LABSCAN_OPENROUTER_REASONING", "low")
_supported_params: dict[str, set[str]] = {}
RETRIES = 3
RETRY_CODES = {408, 429, 500, 502, 503, 504}
# Бланк на 20 строк — это ~1000 токенов JSON. Без явного max_tokens OpenRouter резервирует весь лимит модели
# (100k+ токенов) и отказывает при маленьком балансе: «requires more credits, or fewer max_tokens».
MAX_TOKENS = int(os.getenv("LABSCAN_OPENROUTER_MAX_TOKENS", "4000"))
MAX_TOKENS_REASONING = int(os.getenv("LABSCAN_OPENROUTER_MAX_TOKENS_REASONING", "12000"))  # размышления входят в лимит


def supported_parameters(model: str) -> set[str]:
    """Какие параметры принимает модель (по каталогу OpenRouter). Пусто, если каталог недоступен."""
    if model not in _supported_params:
        try:
            with urllib.request.urlopen(MODELS_URL, timeout=15) as resp:
                for m in json.loads(resp.read())["data"]:
                    _supported_params[m["id"]] = set(m.get("supported_parameters") or [])
        except (OSError, ValueError, KeyError):
            pass
        _supported_params.setdefault(model, set())
    return _supported_params[model]


def reasoning_param(model: str) -> dict | None:
    """Параметр reasoning только для моделей, которые его понимают: иначе require_parameters отсечёт провайдеров."""
    if REASONING == "on" or "reasoning" not in supported_parameters(model):
        return None
    return {"enabled": False} if REASONING == "off" else {"effort": REASONING}


class OpenRouterExtractor:
    name = "openrouter"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 180):
        self.model = model or os.getenv("LABSCAN_OPENROUTER_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.schema = report_json_schema()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/mivlabs/lab-scan-ai", "X-Title": "Lab Scan AI"})
        delay = 2.0
        for attempt in range(RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code in RETRY_CODES and attempt < RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < RETRIES:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"OpenRouter: {e}") from None
        raise RuntimeError("OpenRouter: исчерпаны попытки")

    def _page(self, img: Image.Image) -> tuple[RawReport, dict]:
        data_url = "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode()
        reasoning = reasoning_param(self.model)
        data = self._post({
            "model": self.model,
            "temperature": 0,
            "max_tokens": MAX_TOKENS_REASONING if reasoning else MAX_TOKENS,
            **({"reasoning": reasoning} if reasoning else {}),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "lab_report", "strict": True, "schema": self.schema}},
            # Только провайдеры, которые реально поддерживают response_format: иначе OpenRouter молча уберёт схему
            "provider": {"require_parameters": True},
            "usage": {"include": True},
        })
        if "error" in data:
            raise RuntimeError(f"OpenRouter: {data['error']}")
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError("ответ обрезан по max_tokens")
        content = choice["message"]["content"]
        usage = data.get("usage") or {}
        return RawReport.model_validate(json.loads(content)), {
            "cost": float(usage.get("cost") or 0.0),
            "input_tokens": int(usage.get("prompt_tokens") or 0), "output_tokens": int(usage.get("completion_tokens") or 0)}

    def extract(self, pages: list[Image.Image]) -> Extraction:
        t0 = time.perf_counter()
        reports, cost, tin, tout = [], 0.0, 0, 0
        for p in pages:
            r, u = self._page(p)
            reports.append(r)
            cost += u["cost"]
            tin += u["input_tokens"]
            tout += u["output_tokens"]
        return Extraction(merge_pages(reports), f"{self.name}:{self.model}", time.perf_counter() - t0, cost,
                          {"input_tokens": tin, "output_tokens": tout})
