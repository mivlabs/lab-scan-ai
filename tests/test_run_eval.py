import json
from pathlib import Path

from evals import run_eval
from labscan.extractors.base import Extraction
from labscan.schema import RawReport, RawResult
from synth.generate import generate


class FlakyTruth:
    """Отдаёт эталон бланка; на указанных бланках падает, как при обрыве сети."""

    name = "flaky"

    def __init__(self, data: Path, fail: set[str]):
        self.data, self.fail, self.calls, self.current = data, fail, [], ""

    def extract(self, pages):
        doc = self.current
        self.calls.append(doc)
        if doc in self.fail:
            raise RuntimeError("getaddrinfo failed")
        t = json.loads((self.data / f"{doc}.json").read_text(encoding="utf-8"))
        return Extraction(RawReport(patient_name=t["patient_name"], collected_at=t["collected_at"], results=[
            RawResult(name=r["printed"]["name"], value=r["printed"]["value"], unit=r["printed"]["unit"],
                      reference=r["printed"]["reference"]) for r in t["results"]]), "flaky", 0.5, 0.001)


def test_retry_reruns_only_failed_docs(tmp_path, monkeypatch):
    docs = generate(4, tmp_path, seed=3)
    ids = [d["doc_id"] for d in docs]
    ex = FlakyTruth(tmp_path, fail={ids[1], ids[2]})
    monkeypatch.setattr(run_eval, "get_extractor", lambda engine: ex)
    real_load = run_eval.load_pages

    def load(raw, filename):  # движок не знает, какой бланк ему дали: подсказываем через load_pages
        ex.current = Path(filename).stem
        return real_load(raw, filename)

    monkeypatch.setattr(run_eval, "load_pages", load)
    first = run_eval.run("flaky", tmp_path, None)
    assert first["failed"] == [ids[1], ids[2]] and first["pages"] >= 2
    assert all("seconds" in d and "cost_usd" in d for d in first["docs"] if not d.get("failed"))
    assert "Не распознаны из-за ошибки движка" in run_eval.summarize(first)

    ex.fail, ex.calls = set(), []
    second = run_eval.run("flaky", tmp_path, None, previous=first)
    assert ex.calls == [ids[1], ids[2]]  # удачные бланки не перепрогонялись
    assert second["failed"] == [] and [d["doc_id"] for d in second["docs"]] == ids
    assert all(d["perfect"] for d in second["docs"])
    assert second["pages"] == sum(d["pages"] for d in second["docs"])
    assert "Не распознаны" not in run_eval.summarize(second)


def test_rescore_uses_stored_raw_report_without_calling_engine(tmp_path, monkeypatch):
    docs = generate(2, tmp_path, seed=5)
    ids = [d["doc_id"] for d in docs]
    ex = FlakyTruth(tmp_path, fail=set())
    monkeypatch.setattr(run_eval, "get_extractor", lambda engine: ex)
    real_load = run_eval.load_pages

    def load(raw, filename):
        ex.current = Path(filename).stem
        return real_load(raw, filename)

    monkeypatch.setattr(run_eval, "load_pages", load)
    first = run_eval.run("flaky", tmp_path, None)
    assert all(d["raw_report"]["results"] for d in first["docs"])
    ex.calls = []
    again = run_eval.run("flaky", tmp_path, None, previous=first, rescore=True)
    assert ex.calls == [] and [d["doc_id"] for d in again["docs"]] == ids and all(d["perfect"] for d in again["docs"])
    assert again["pages"] == first["pages"] and again["cost_per_page"] == first["cost_per_page"]
