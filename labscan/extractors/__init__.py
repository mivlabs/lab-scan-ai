"""Выбор движка распознавания.

LABSCAN_ENGINE = claude | openrouter | ollama | tesseract | auto (по умолчанию).
auto: Claude, если есть ключ Anthropic; иначе OpenRouter, если есть его ключ;
иначе локальный Ollama, если запущен; иначе Tesseract.
"""

from __future__ import annotations

import os
import urllib.request


def available_engines() -> list[dict]:
    """Какие движки можно запустить прямо сейчас: для режима сравнения на одном бланке."""
    out = []
    if os.getenv("ANTHROPIC_API_KEY"):
        out.append({"engine": "claude", "model": os.getenv("LABSCAN_CLAUDE_MODEL", "claude-opus-5-5")})
    if os.getenv("OPENROUTER_API_KEY"):
        default = os.getenv("LABSCAN_OPENROUTER_MODEL", "qwen/qwen2.5-vl-72b-instruct")
        models = os.getenv("LABSCAN_COMPARE_MODELS", f"{default},google/gemini-2.5-flash,anthropic/claude-haiku-5.5")
        out += [{"engine": "openrouter", "model": m.strip()} for m in models.split(",") if m.strip()]
    if _ollama_up():
        out.append({"engine": "ollama", "model": os.getenv("LABSCAN_OLLAMA_MODEL", "qwen2.5vl:7b")})
    from .tesseract import available

    if available():
        out.append({"engine": "tesseract", "model": None})
    return out


def _ollama_up() -> bool:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1):
            return True
    except OSError:
        return False


def get_extractor(name: str | None = None, model: str | None = None):
    """Движок по имени; model переопределяет модель для claude, openrouter и ollama (режим сравнения)."""
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
        return ClaudeExtractor(model=model)
    if name == "openrouter":
        from .openrouter import OpenRouterExtractor
        return OpenRouterExtractor(model=model)
    if name == "ollama":
        from .ollama import OllamaExtractor
        return OllamaExtractor(model=model)
    if name == "tesseract":
        from .tesseract import TesseractExtractor
        return TesseractExtractor()
    raise ValueError(f"неизвестный движок: {name}")
