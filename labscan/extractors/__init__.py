"""Выбор движка распознавания.

LABSCAN_ENGINE = claude | openrouter | ollama | tesseract | auto (по умолчанию).
auto: Claude, если есть ключ Anthropic; иначе OpenRouter, если есть его ключ;
иначе локальный Ollama, если запущен; иначе Tesseract.
"""

from __future__ import annotations

import os
import urllib.request


def _ollama_up() -> bool:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1):
            return True
    except OSError:
        return False


def get_extractor(name: str | None = None):
    name = (name or os.getenv("LABSCAN_ENGINE", "auto")).lower()
    if name == "auto":
        if os.getenv("ANTHROPIC_API_KEY"):
            name = "claude"
        elif os.getenv("OPENROUTER_API_KEY"):
            name = "openrouter"
        elif _ollama_up():
            name = "ollama"
        else:
            name = "tesseract"
    if name == "claude":
        from .claude import ClaudeExtractor
        return ClaudeExtractor()
    if name == "openrouter":
        from .openrouter import OpenRouterExtractor
        return OpenRouterExtractor()
    if name == "ollama":
        from .ollama import OllamaExtractor
        return OllamaExtractor()
    if name == "tesseract":
        from .tesseract import TesseractExtractor
        return TesseractExtractor()
    raise ValueError(f"неизвестный движок: {name}")
