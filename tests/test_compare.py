from evals.compare import build, label, table


def _run(engine, docs, correct=0.9, missed=0.05, cost=0.0):
    s = {"docs": docs, "correct_auto": correct, "correct_review": 0.0, "caught": 0.03, "silent": 0.02, "missed": missed,
         "auto_and_perfect": 0.5, "auto_but_missed": 0.1, "header_ok": 0.9}
    return {"engine": engine, "seconds_per_page": 3.0, "cost_per_page": cost,
            "summary": {"все": s, "фото": s, "скан": s, "PDF": s}}


def test_labels_and_table():
    runs = [_run("tesseract", 50, correct=0.6), _run("openrouter:qwen/qwen2.5-vl-72b-instruct", 20, cost=0.003)]
    assert [label(r) for r in runs] == ["Tesseract", "qwen2.5-vl-72b"]
    md = table(runs)
    assert "| | Tesseract | qwen2.5-vl-72b |" in md
    assert "| Бланков в прогоне | 50 | 20 |" in md
    assert "| Показатель распознан верно | 60.0% | 90.0% |" in md
    assert "| Цена страницы, $ | 0 | 0.0030 |" in md
    full = build(runs)
    assert "Внимание" in full and "## Только фото с телефона" in full
    assert "Внимание" not in build([_run("tesseract", 20), _run("openrouter:x/y", 20)])


def test_limit_recomputes_summary_on_common_subset(tmp_path):
    import json

    from evals.compare import load
    from evals.run_eval import summarize

    data = tmp_path / "synth"
    data.mkdir()
    (data / "index.json").write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    base = {"extras": [], "silent_extras": [], "header_ok": True, "perfect": True, "auto": True, "errors": [],
            "kind": "scan", "layout": "citolab"}
    docs = [{"doc_id": "a", "n": 10, "cats": {"correct_auto": 10}, "seconds": 2.0, "cost_usd": 0.01, "pages": 1, **base},
            {"doc_id": "b", "n": 10, "cats": {"correct_auto": 10}, "seconds": 2.0, "cost_usd": 0.01, "pages": 1, **base},
            {"doc_id": "c", "n": 10, "cats": {"missed": 10}, "seconds": 6.0, "cost_usd": 0.04, "pages": 1,
             **{**base, "perfect": False}}]
    res = {"engine": "tesseract", "docs": docs, "seconds_per_page": 3.3, "cost_per_page": 0.02, "pages": 3}
    summarize(res)
    results = tmp_path / "results"
    results.mkdir()
    (results / "tesseract.json").write_text(json.dumps(res), encoding="utf-8")
    full = load(results, None)[0]
    assert abs(full["summary"]["все"]["missed"] - 1 / 3) < 1e-9
    cut = load(results, None, limit=2, data=data)[0]
    assert cut["summary"]["все"]["docs"] == 2 and cut["summary"]["все"]["missed"] == 0
    assert cut["seconds_per_page"] == 2.0 and cut["cost_per_page"] == 0.01
