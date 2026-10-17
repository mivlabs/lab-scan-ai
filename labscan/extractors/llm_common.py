"""Общее для vision-LLM: промпт и JSON-схема ответа.

Модель только переписывает бланк в структуру «как напечатано». Сопоставление
со справочником, перевод единиц и проверки делает детерминированный код
(normalize.py): так ошибки модели видны и проверяемы, а логику можно тестировать.
"""

from __future__ import annotations

import copy

from ..schema import RawReport

SYSTEM_PROMPT = """Ты переписываешь бланк лабораторного анализа в структурированный вид.

Правила:
- Переписывай ровно то, что напечатано: названия, значения, единицы и референсы как есть, без перевода единиц и без исправлений.
- Одна строка таблицы результатов — один элемент results. Строки шапки, подписи и примечания в results не включай.
- Десятичный разделитель оставляй как на бланке (запятая или точка).
- Если значение читается неуверенно (размыто, обрезано, перекрыто), всё равно перепиши лучшее прочтение и поставь legible=false.
- Если поля нет на бланке, ставь null. Ничего не придумывай.
"""

USER_PROMPT = "Перепиши этот бланк анализа по схеме."


def _inline_refs(schema: dict) -> dict:
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def _strictify(node):
    """Структурированный вывод требует: все поля required, additionalProperties=false, без default."""
    if isinstance(node, dict):
        node.pop("default", None)
        node.pop("title", None)
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
        for v in node.values():
            _strictify(v)
    elif isinstance(node, list):
        for v in node:
            _strictify(v)
    return node


def report_json_schema() -> dict:
    return _strictify(_inline_refs(RawReport.model_json_schema()))
