import os

from labscan.env import load_dotenv


def test_dotenv_fills_only_missing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# комментарий\nOPENROUTER_API_KEY=sk-test\nLABSCAN_ENGINE=\n  QUOTED = 'x y' \nBROKEN LINE\n",
                   encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("QUOTED", "already")
    loaded = load_dotenv(env)
    assert loaded == {"OPENROUTER_API_KEY": "sk-test"}
    assert os.environ["OPENROUTER_API_KEY"] == "sk-test" and os.environ["QUOTED"] == "already"
    assert "LABSCAN_ENGINE" not in loaded  # пустое значение не задаём
    assert load_dotenv(tmp_path / "нет") == {}
