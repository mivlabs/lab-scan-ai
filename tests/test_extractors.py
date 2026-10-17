import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from labscan.extractors.claude import ClaudeExtractor, ExtractionRefused
from labscan.extractors.llm_common import report_json_schema
from labscan.extractors.tesseract import parse_line
from labscan.ingest import load_pages
from labscan.normalize import normalize_report
from synth.generate import generate


def test_schema_is_strict():
    s = report_json_schema()
    assert "$defs" not in json.dumps(s)
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    item = s["properties"]["results"]["items"]
    assert item["additionalProperties"] is False and "legible" in item["required"]


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            stop_reason=self.stop_reason, stop_details=SimpleNamespace(explanation="нет"),
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(input_tokens=1500, output_tokens=400))


def _client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


PAYLOAD = {"lab_name": "ЦитоЛаб", "patient_name": "Иванова Мария", "birth_date": "01.02.1990",
           "collected_at": "10.10.2026", "results": [
               {"name": "Гемоглобин", "value": "131", "unit": "г/л", "reference": "120 - 160", "legible": True},
               {"name": "Лейкоциты", "value": "6,2", "unit": "×10⁹/л", "reference": "4,0 - 9,0", "legible": True}]}


def test_claude_extractor_request_and_parse():
    from PIL import Image

    msgs = FakeMessages(PAYLOAD)
    ex = ClaudeExtractor(model="claude-opus-5-5", client=_client(msgs))
    res = ex.extract([Image.new("RGB", (200, 100), "white"), Image.new("RGB", (200, 100), "white")])
    assert len(msgs.calls) == 2  # по запросу на страницу
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["messages"][0]["content"][0]["type"] == "image"
    assert len(res.report.results) == 4  # две страницы склеены
    assert res.cost_usd == pytest.approx(2 * (1500 * 4 + 400 * 20) / 1e6)
    n = normalize_report(res.report)
    assert [o.code for o in n.observations][:2] == ["HGB", "WBC"]


def test_claude_refusal_raises():
    from PIL import Image

    ex = ClaudeExtractor(client=_client(FakeMessages(PAYLOAD, stop_reason="refusal")))
    with pytest.raises(ExtractionRefused):
        ex.extract([Image.new("RGB", (10, 10))])


def test_parse_line_with_dot_leaders():
    words = [(w, 95.0) for w in "Гемоглобин (HGB) .......... 135 г/л (120 - 160)".split()]
    r = parse_line(words)
    assert r.name == "Гемоглобин (HGB)" and r.value == "135" and r.unit == "г/л" and r.reference == "120 - 160"


def test_parse_line_skips_header():
    assert parse_line([("Заказ", 90.0), ("№:", 90.0), ("677814", 90.0)]) is None


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract не установлен")
def test_tesseract_on_clean_scan(tmp_path: Path):
    from labscan.extractors.tesseract import TesseractExtractor

    docs = generate(3, tmp_path, seed=1)
    scan = next(d for d in docs if d["kind"] == "scan")
    res = TesseractExtractor().extract(load_pages((tmp_path / scan["file"]).read_bytes(), scan["file"]))
    n = normalize_report(res.report)
    got = {o.code: o.value for o in n.observations}
    truth = {r["code"]: r["value"] for r in scan["results"]}
    correct = sum(1 for c, v in truth.items() if got.get(c) == v)
    assert correct / len(truth) >= 0.6


def test_openrouter_request_and_parse(monkeypatch):
    from PIL import Image

    from labscan.extractors.openrouter import OpenRouterExtractor

    sent = []
    ex = OpenRouterExtractor(model="qwen/qwen2.5-vl-72b-instruct", api_key="k")
    monkeypatch.setattr(ex, "_post", lambda payload: sent.append(payload) or {
        "choices": [{"message": {"content": json.dumps(PAYLOAD)}, "finish_reason": "stop"}],
        "usage": {"cost": 0.0012, "prompt_tokens": 1800, "completion_tokens": 300}})
    res = ex.extract([Image.new("RGB", (50, 50), "white")])
    body = sent[0]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}  # без этого OpenRouter может молча убрать схему
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert res.cost_usd == pytest.approx(0.0012) and len(res.report.results) == 2
    assert res.meta == {"input_tokens": 1800, "output_tokens": 300}


def test_openrouter_http_error_is_readable(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from PIL import Image

    from labscan.extractors import openrouter

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 402, "Payment Required", {}, io.BytesIO(b'{"error":"no credits"}'))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(openrouter, "RETRIES", 0)
    with pytest.raises(RuntimeError, match="HTTP 402.*no credits"):
        openrouter.OpenRouterExtractor(api_key="k").extract([Image.new("RGB", (10, 10), "white")])
