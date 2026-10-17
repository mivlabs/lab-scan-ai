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
    assert body["max_tokens"] == 4000  # иначе OpenRouter резервирует весь лимит модели и отказывает при малом балансе
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


def test_transform_maps_ocr_box_back_to_page():
    """Точка на странице -> поворот как в preprocess -> её координаты в OCR -> обратно в доли страницы."""
    import numpy as np
    from PIL import Image, ImageDraw

    from labscan.extractors.tesseract import Transform

    w, h, px, py = 600, 800, 450, 200
    for angle in (3.0, -4.5, 0.0):
        img = Image.new("L", (w, h), 255)
        ImageDraw.Draw(img).rectangle((px - 3, py - 3, px + 3, py + 3), fill=0)
        rot = img if not angle else img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
        ys, xs = np.nonzero(np.asarray(rot) < 128)
        scale = 1.6
        tr = Transform(angle, img.size, rot.size, scale)
        fx, fy = xs.mean() * scale, ys.mean() * scale
        x0, y0, x1, y1 = tr.to_page((fx - 5, fy - 5, fx + 5, fy + 5))
        assert abs((x0 + x1) / 2 * w - px) < 1.5 and abs((y0 + y1) / 2 * h - py) < 1.5
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def test_parse_tsv_gives_line_boxes_in_page_fractions():
    from labscan.extractors.tesseract import Transform, parse_page, parse_tsv

    head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [(1, "Гемоглобин", 100, 500, 200, 30, 96), (1, "135", 420, 500, 50, 30, 92), (1, "г/л", 500, 502, 40, 28, 90),
            (1, "(120", 600, 500, 60, 30, 88), (1, "-", 670, 500, 10, 30, 80), (1, "160)", 690, 500, 60, 30, 91),
            (2, "Заказ", 100, 600, 80, 30, 95), (2, "№:", 190, 600, 30, 30, 95), (2, "677814", 230, 600, 90, 30, 95)]
    tsv = "\n".join([head] + [f"5\t1\t1\t1\t{ln}\t{i}\t{left}\t{t}\t{w}\t{h}\t{c}\t{txt}"
                              for i, (ln, txt, left, t, w, h, c) in enumerate(rows)])
    lines = parse_tsv(tsv)
    assert [len(ws) for ws in lines] == [6, 3] and lines[0][0][2] == (100.0, 500.0, 300.0, 530.0)
    report = parse_page(lines, Transform(0.0, (1000, 2000), (1000, 2000), 1.6))
    assert len(report.results) == 1
    r = report.results[0]
    assert r.name == "Гемоглобин" and r.value == "135" and r.unit == "г/л" and r.reference == "120 - 160"
    # объединённый прямоугольник строки 100..750 x 500..530 в OCR-пикселях, делённый на 1.6 и на размер страницы
    expected = [100 / 1.6 / 1000, 500 / 1.6 / 2000, 750 / 1.6 / 1000, 530 / 1.6 / 2000]
    assert r.box == [round(v, 4) for v in expected]


def test_schema_box_only_on_request(monkeypatch):
    monkeypatch.delenv("LABSCAN_LLM_BOX", raising=False)
    default = report_json_schema()["properties"]["results"]["items"]
    assert "box" not in default["properties"] and "page" not in default["properties"]
    with_box = report_json_schema(with_box=True)["properties"]["results"]["items"]
    assert "box" in with_box["properties"] and "box" in with_box["required"] and "page" not in with_box["properties"]
    monkeypatch.setenv("LABSCAN_LLM_BOX", "1")
    assert "box" in report_json_schema()["properties"]["results"]["items"]["properties"]


def test_merge_pages_numbers_pages():
    from labscan.extractors.base import merge_pages
    from labscan.schema import RawReport, RawResult

    merged = merge_pages([RawReport(results=[RawResult(name="a", value="1")]),
                          RawReport(results=[RawResult(name="b", value="2"), RawResult(name="c", value="3", page=7)])])
    assert [r.page for r in merged.results] == [0, 1, 7]
