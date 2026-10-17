"""Подхватить переменные из .env в текущей папке, если они ещё не заданы.

Без зависимости от python-dotenv: ключи и модель лежат в .env (см. .env.example),
docker compose читает его сам, а uvicorn и eval запускаются из папки проекта.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str = ".env") -> dict[str, str]:
    """Вернуть, что реально добавили в окружение. Уже заданные переменные не трогаем."""
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded
