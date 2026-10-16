from pathlib import Path

import pytest

from codecast.config import VoiceConfig
from codecast.errors import ConfigError
from codecast.voice import engines
from codecast.voice.engines import SayEngine, select_engine, wav_duration
from codecast.voice.fetch_voice import fetch_voice, voice_urls

from tests.fakes import SilenceEngine


def test_wav_duration(tmp_path):
    p = tmp_path / "a.wav"
    SilenceEngine(0.5).synthesize("two words", p)
    assert wav_duration(p) == pytest.approx(1.0)


def test_say_passes_text_via_file(monkeypatch, tmp_path):
    """Review Focus 3: text never becomes an argv flag."""
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["text"] = Path(argv[argv.index("-f") + 1]).read_text()
        Path(argv[argv.index("-o") + 1]).write_bytes(b"")
    monkeypatch.setattr(engines.subprocess, "run", fake_run)
    monkeypatch.setattr(engines, "convert_to_wav48k", lambda src, dst: None)
    SayEngine("Samantha").synthesize('-v Evil "quoted" <tag> & 🎬', tmp_path / "o.wav")
    # `<`/`>` are spoken as words (see SayEngine); everything else reaches say verbatim.
    assert seen["text"] == '-v Evil "quoted"  less than tag greater than  & 🎬'
    assert "-v Evil" not in seen["argv"] and "<" not in seen["text"]
    assert all("quoted" not in a for a in seen["argv"])


def cfg(tmp_path):
    return VoiceConfig(voices_dir=str(tmp_path / "voices"))


def test_select_prefers_piper_when_model_present(monkeypatch, tmp_path):
    c = cfg(tmp_path)
    monkeypatch.setattr(engines, "_piper_installed", lambda: True)
    engines.piper_model_path(c).parent.mkdir(parents=True)
    engines.piper_model_path(c).write_bytes(b"x")
    monkeypatch.setattr(engines, "PiperEngine", lambda path: type("P", (), {"name": "piper"})())
    assert select_engine(c, allow_network=False).name == "piper"


def test_select_falls_back_to_say(monkeypatch, tmp_path):
    monkeypatch.setattr(engines, "_piper_installed", lambda: False)
    monkeypatch.setattr(engines, "_say_available", lambda voice: True)
    assert select_engine(cfg(tmp_path), allow_network=False).name == "say"


def test_edge_only_with_network_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(engines, "_piper_installed", lambda: False)
    monkeypatch.setattr(engines, "_say_available", lambda voice: False)
    monkeypatch.setattr(engines, "_edge_installed", lambda: True)
    with pytest.raises(ConfigError, match="fetch-voice"):
        select_engine(cfg(tmp_path), allow_network=False)
    assert select_engine(cfg(tmp_path), allow_network=True).name == "edge-tts"


def test_engine_status(monkeypatch, tmp_path):
    monkeypatch.setattr(engines, "_piper_installed", lambda: True)
    monkeypatch.setattr(engines, "_say_available", lambda voice: True)
    monkeypatch.setattr(engines, "_edge_installed", lambda: False)
    st = engines.engine_status(cfg(tmp_path), allow_network=False)
    assert st["piper"].startswith("voice missing") and st["say"] == "ok" and st["edge-tts"] != "ok"


def test_voice_urls():
    urls = voice_urls("en_US-lessac-medium")
    assert urls[0].endswith("/en/en_US/lessac/medium/en_US-lessac-medium.onnx")
    assert urls[1].endswith(".onnx.json")


def test_fetch_voice_without_piper_is_a_notice(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(engines, "_piper_installed", lambda: False)
    assert fetch_voice(cfg(tmp_path)) is None
    assert "say" in capsys.readouterr().out


def test_fetch_voice_downloads_atomically(monkeypatch, tmp_path):
    monkeypatch.setattr(engines, "_piper_installed", lambda: True)

    class Resp:
        def __init__(self, url): self.url = url
        def read(self, n=-1):
            data, self.url = (self.url.encode(), "") if self.url else (b"", "")
            return data
        def __enter__(self): return self
        def __exit__(self, *a): return False
    path = fetch_voice(cfg(tmp_path), opener=lambda url, timeout=60: Resp(url))
    assert path.name == "en_US-lessac-medium.onnx" and path.read_bytes().endswith(b".onnx")
    assert not list(path.parent.glob("*.part"))
