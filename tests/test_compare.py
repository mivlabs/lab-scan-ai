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
