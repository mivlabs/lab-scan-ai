"""Адаптеры МИС. В прототипе — запись Bundle в папку «исходящие».

Под реальную МИС (Damumed и др.) пишется класс с тем же методом send():
авторизация, поиск пациента по ИИН, маппинг кодов показателей на справочник МИС,
повторы при сбоях и идемпотентность по номеру заказа, чтобы повторная отправка
не задвоила результаты в карте.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class MISAdapter(Protocol):
    def send(self, bundle: dict, idempotency_key: str) -> str: ...


class OutboxMIS:
    """Пишет Bundle в файл. Повторная отправка с тем же ключом перезаписывает тот же файл."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def send(self, bundle: dict, idempotency_key: str) -> str:
        path = self.root / f"{idempotency_key}.json"
        path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
