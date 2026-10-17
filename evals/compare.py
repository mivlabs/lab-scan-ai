"""Сводная таблица по всем прогонам eval: Tesseract против vision-моделей.

Читает evals/results/*.json и пишет evals/results/comparison.md. Ту же таблицу
вставляем в README. Сравнивать честно можно только прогоны на одном наборе бланков:
скрипт показывает число бланков в каждой колонке и предупреждает, если наборы разные.

Запуск: python -m evals.compare [--results evals/results] [--only tesseract,openrouter_...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = (
    ("Показатель распознан верно", lambda s: s["correct_auto"] + s["correct_review"]),
    ("Ошибка поймана проверкой", lambda s: s["caught"]),
    ("Тихая ошибка", lambda s: s["silent"]),
    ("Показатель пропущен", lambda s: s["missed"]),
    ("Бланк без единой пометки и полностью верный", lambda s: s["auto_and_perfect"]),
    ("Бланк без пометок, но часть строк молча пропущена", lambda s: s["auto_but_missed"]),
    ("ФИО и дата взятия верны", lambda s: s["header_ok"]),
)
SHORT = {"tesseract": "Tesseract"}


def label(res: dict) -> str:
    engine = res["engine"]
    if engine in SHORT:
        return SHORT[engine]
    model = engine.split(":", 1)[-1]
    return model.split("/", 1)[-1].replace("-instruct", "")


def load(results: Path, only: set[str] | None) -> list[dict]:
    runs = []
    for p in sorted(results.glob("*.json")):
        if only and p.stem not in only:
            continue
        res = json.loads(p.read_text(encoding="utf-8"))
        if "summary" in res and "все" in res["summary"]:
            res["_file"] = p.stem
            runs.append(res)
    return sorted(runs, key=lambda r: (r["engine"] != "tesseract", r["engine"]))  # базовая линия первой


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def table(runs: list[dict], subset: str | None = None) -> str:
    """Общая таблица или срез по типу файла (скан / фото / PDF)."""
    key = subset or "все"
    cols = [r for r in runs if key in r["summary"]]
    if not cols:
        return ""
    head = "| | " + " | ".join(label(r) for r in cols) + " |"
    lines = [head, "|---|" + "---|" * len(cols)]
    lines.append("| Бланков в прогоне | " + " | ".join(str(r["summary"][key]["docs"]) for r in cols) + " |")
    for title, fn in ROWS:
        lines.append(f"| {title} | " + " | ".join(pct(fn(r["summary"][key])) for r in cols) + " |")
    if not subset:
        lines.append("| Секунд на страницу | " + " | ".join(f"{r['seconds_per_page']:.1f}" for r in cols) + " |")
        lines.append("| Цена страницы, $ | "
                     + " | ".join(f"{r['cost_per_page']:.4f}" if r["cost_per_page"] else "0" for r in cols) + " |")
    return "\n".join(lines)


def build(runs: list[dict]) -> str:
    docs = {r["summary"]["все"]["docs"] for r in runs}
    out = ["# Сравнение движков", ""]
    if len(docs) > 1:
        out += ["> Внимание: прогоны на разном числе бланков, сравнивайте с оглядкой на строку «Бланков в прогоне». "
                "Для честного сравнения запустите все движки с одним `--limit`.", ""]
    out += [table(runs), "", "## Только фото с телефона", "", table(runs, "фото"), "",
            "## Только сканы", "", table(runs, "скан"), "", "## Только PDF", "", table(runs, "PDF"), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("evals/results"))
    ap.add_argument("--only", help="имена файлов без .json через запятую")
    args = ap.parse_args()
    runs = load(args.results, set(args.only.split(",")) if args.only else None)
    md = build(runs)
    (args.results / "comparison.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
