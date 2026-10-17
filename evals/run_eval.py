"""Оценка качества распознавания на синтетических бланках с эталоном.

Главная метрика для медицины — «тихие ошибки»: неверное значение, которое система
НЕ пометила на проверку. Ошибка, пойманная проверкой, стоит врачу 5 секунд;
тихая ошибка попадает в карту пациента.

Запуск:
  python -m evals.run_eval --engine tesseract
  python -m evals.run_eval --engine claude --limit 20      # нужен ANTHROPIC_API_KEY
  python -m evals.run_eval --engine ollama                  # нужен запущенный Ollama
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from labscan.env import load_dotenv
from labscan.extractors import get_extractor
from labscan.ingest import load_pages
from labscan.normalize import normalize_report
from labscan.schema import RawReport

CATS = ("correct_auto", "correct_review", "caught", "silent", "missed")
CAT_RU = {
    "correct_auto": "верно, без проверки",
    "correct_review": "верно, но отправлено на проверку",
    "caught": "ошибка, поймана проверкой",
    "silent": "ТИХАЯ ОШИБКА (неверно, без пометки)",
    "missed": "показатель пропущен",
}


def _same(a: float | None, b: float) -> bool:
    return a is not None and abs(a - b) <= max(0.011, abs(b) * 0.002)


def _name_key(s: str | None) -> str:
    return " ".join((s or "").lower().replace("ё", "е").split())


def score_doc(truth: dict, norm) -> dict:
    pred = {}
    for o in norm.observations:
        if o.code and o.code not in pred:
            pred[o.code] = o
    cats = Counter()
    rows = []
    for t in truth["results"]:
        o = pred.get(t["code"])
        if o is None:
            cat = "missed"
        elif _same(o.value, t["value"]):
            cat = "correct_auto" if o.status == "ok" else "correct_review"
        else:
            cat = "caught" if o.status == "review" else "silent"
        cats[cat] += 1
        if cat != "correct_auto":
            rows.append({"code": t["code"], "truth": t["value"], "got": o.value if o else None, "cat": cat,
                         "issues": o.issues if o else [], "raw": o.raw.model_dump() if o else None})
    truth_codes = {t["code"] for t in truth["results"]}
    extras = [o.code or o.raw.name for o in norm.observations if (o.code or "") not in truth_codes]
    extras_raw = [[o.code, o.raw.name, o.raw.value] for o in norm.observations if (o.code or "") not in truth_codes]
    silent_extras = [o.code for o in norm.observations if o.code and o.code not in truth_codes and o.status == "ok"]
    header_ok = (_name_key(norm.patient_name) == _name_key(truth["patient_name"])
                 and norm.collected_at == date.fromisoformat(truth["collected_at"]))
    perfect = cats["correct_auto"] + cats["correct_review"] == len(truth["results"]) and not extras and header_ok
    header = {"patient_name": norm.patient_name, "collected_at": str(norm.collected_at or ""),
              "truth_name": truth["patient_name"], "truth_collected_at": truth["collected_at"]}
    return {"cats": dict(cats), "n": len(truth["results"]), "extras": extras, "extras_raw": extras_raw,
            "silent_extras": silent_extras, "header_ok": header_ok, "header": header, "perfect": perfect,
            "auto": not norm.needs_review, "errors": rows}



def run(engine: str, data: Path, limit: int | None, only: list[str] | None = None,
        previous: dict | None = None, rescore: bool = False) -> dict:
    """previous — прошлый результат этого же движка: его удачные бланки не перепрогоняются (--retry),
    заново идут только упавшие (сеть, лимиты) и те, которых в нём не было.
    rescore — не звать модель вовсе: взять её сырой ответ из прошлого прогона и заново прогнать
    через нормализацию. Так правки normalize.py и справочника проверяются бесплатно."""
    ex = get_extractor(engine)
    ids = json.loads((data / "index.json").read_text())
    ids = [i for i in ids if i in only] if only else ids[:limit]
    done = {d["doc_id"]: d for d in (previous or {}).get("docs", []) if not d.get("failed") and "seconds" in d}
    per_doc = []
    for doc_id in ids:
        truth = json.loads((data / f"{doc_id}.json").read_text(encoding="utf-8"))
        if doc_id in done and rescore and done[doc_id].get("raw_report"):
            old = done[doc_id]
            s = score_doc(truth, normalize_report(RawReport.model_validate(old["raw_report"])))
            s.update(doc_id=doc_id, kind=truth["kind"], layout=truth["layout"], seconds=old["seconds"],
                     cost_usd=old["cost_usd"], pages=old["pages"], raw_report=old["raw_report"])
            per_doc.append(s)
            continue
        if doc_id in done and not rescore:
            per_doc.append(done[doc_id])
            continue
        raw = (data / truth["file"]).read_bytes()
        pages = load_pages(raw, truth["file"])
        t0 = time.perf_counter()
        try:
            result = ex.extract(pages)
        except Exception as e:  # noqa: BLE001 — в eval падение документа = все показатели пропущены
            print(f"{doc_id}: ошибка движка: {e}")
            per_doc.append({"doc_id": doc_id, "kind": truth["kind"], "layout": truth["layout"], "failed": str(e),
                            "cats": {"missed": len(truth["results"])}, "n": len(truth["results"]),
                            "extras": [], "silent_extras": [], "header_ok": False, "perfect": False,
                            "auto": False, "errors": []})
            continue
        s = score_doc(truth, normalize_report(result.report))
        s.update(doc_id=doc_id, kind=truth["kind"], layout=truth["layout"],
                 seconds=time.perf_counter() - t0, cost_usd=result.cost_usd, pages=len(pages),
                 raw_report=result.report.model_dump())
        per_doc.append(s)
        print(f"{doc_id}: {s['cats']}{' лишние: ' + str(s['extras']) if s['extras'] else ''}"
              + ("" if s["header_ok"] else f" шапка: {s['header']['patient_name']!r} / {s['header']['collected_at']!r}"
                 f" (эталон {s['header']['truth_name']!r} / {s['header']['truth_collected_at']})"))
    total_pages = sum(d.get("pages", 0) for d in per_doc)
    return {"engine": getattr(ex, "model", None) and f"{ex.name}:{ex.model}" or ex.name, "docs": per_doc,
            "seconds_per_page": sum(d.get("seconds", 0.0) for d in per_doc) / max(total_pages, 1),
            "cost_per_page": sum(d.get("cost_usd", 0.0) for d in per_doc) / max(total_pages, 1),
            "pages": total_pages, "failed": [d["doc_id"] for d in per_doc if d.get("failed")]}


def summarize(res: dict) -> str:
    docs = res["docs"]

    def block(subset: list[dict]) -> dict:
        c = Counter()
        for d in subset:
            c.update(d["cats"])
        n = sum(d["n"] for d in subset) or 1
        return {"n_fields": n, **{k: c[k] / n for k in CATS},
                "silent_extras": sum(len(d["silent_extras"]) for d in subset),
                "perfect": sum(d["perfect"] for d in subset) / max(len(subset), 1),
                "auto": sum(d["auto"] for d in subset) / max(len(subset), 1),
                "auto_and_perfect": sum(d["auto"] and d["perfect"] for d in subset) / max(len(subset), 1),
                "auto_but_missed": sum(d["auto"] and d["cats"].get("missed", 0) > 0 for d in subset) / max(len(subset), 1),
                "header_ok": sum(d["header_ok"] for d in subset) / max(len(subset), 1), "docs": len(subset)}

    groups = {"все": docs}
    by = defaultdict(list)
    for d in docs:
        by[{"scan": "скан", "photo": "фото", "pdf": "PDF"}[d["kind"]]].append(d)
        by[d["layout"]].append(d)
    groups.update(dict(sorted(by.items())))
    stats = {k: block(v) for k, v in groups.items()}
    res["summary"] = stats

    def pct(x):
        return f"{x * 100:.1f}%"

    a = stats["все"]
    failed = res.get("failed") or []
    lines = [
        f"# Eval: {res['engine']}", "",
        f"Бланков: {a['docs']}, страниц: {res['pages']}, показателей в эталоне: {a['n_fields']}. "
        f"Время на страницу: {res['seconds_per_page']:.1f} с"
        + (f", стоимость на страницу: ${res['cost_per_page']:.4f}" if res["cost_per_page"] else ""), "",
        *([f"Не распознаны из-за ошибки движка (все показатели посчитаны пропущенными): {', '.join(failed)}. "
           "Повторить только их: `--retry`.", ""] if failed else []),
        "## Показатели", "", "| | доля |", "|---|---|",
        *[f"| {CAT_RU[k]} | {pct(a[k])} |" for k in CATS],
        f"| лишние строки без пометки (шт.) | {a['silent_extras']} |", "",
        "## Бланки", "", "| | доля |", "|---|---|",
        f"| распознан полностью верно | {pct(a['perfect'])} |",
        f"| можно записать без проверки (нет ни одной пометки) | {pct(a['auto'])} |",
        f"| без проверки и полностью верно | {pct(a['auto_and_perfect'])} |",
        f"| без пометок, но часть строк молча пропущена | {pct(a['auto_but_missed'])} |",
        f"| ФИО и дата взятия верны | {pct(a['header_ok'])} |", "",
        "## По типу и вёрстке", "",
        "| группа | бланков | верно всего | тихие ошибки | пропущено | бланков без проверки |", "|---|---|---|---|---|---|",
    ]
    for k, s in stats.items():
        if k == "все":
            continue
        lines.append(f"| {k} | {s['docs']} | {pct(s['correct_auto'] + s['correct_review'])} | {pct(s['silent'])} | "
                     f"{pct(s['missed'])} | {pct(s['auto'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--data", type=Path, default=Path("data/synth"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="id бланков через запятую, например 002_senim_scan,007_citolab_scan")
    ap.add_argument("--retry", action="store_true",
                    help="взять прошлый результат этого движка из --out и перепрогнать только упавшие бланки")
    ap.add_argument("--rescore", action="store_true",
                    help="не звать модель: пересчитать прошлый прогон из её сырых ответов новой нормализацией")
    ap.add_argument("--out", type=Path, default=Path("evals/results"))
    args = ap.parse_args()
    load_dotenv()
    ex = get_extractor(args.engine)
    tag = (getattr(ex, "model", None) and f"{ex.name}:{ex.model}" or ex.name).replace(":", "_").replace("/", "_")
    previous = None
    if (args.retry or args.rescore) and (args.out / f"{tag}.json").exists():
        previous = json.loads((args.out / f"{tag}.json").read_text(encoding="utf-8"))
        ok = [d for d in previous["docs"] if not d.get("failed")]
        print(f"--retry: беру {len(ok)} удачных бланков из прошлого прогона, остальные заново")
    res = run(args.engine, args.data, args.limit, args.only.split(",") if args.only else None, previous, args.rescore)
    md = summarize(res)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}.md").write_text(md, encoding="utf-8")
    (args.out / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
