from codecast import preflight
from codecast.config import load_config


def patch_all_ok(monkeypatch):
    monkeypatch.setattr(preflight, "ffmpeg_problems", lambda: [])
    monkeypatch.setattr(preflight, "_chromium_problem", lambda: None)
    monkeypatch.setattr(preflight, "engine_status", lambda cfg, allow_network: {"piper": "x", "say": "ok", "edge-tts": "x"})


def test_missing_key_is_warning_in_doctor(monkeypatch):
    patch_all_ok(monkeypatch)
    checks = preflight.run_checks(load_config(), require_key=False)
    key = [c for c in checks if c.name == "llm api key"][0]
    assert not key.ok and not key.fatal
    assert preflight.fatal_problems(load_config(), require_key=False, allow_network_tts=False) == []


def test_missing_key_is_fatal_for_run(monkeypatch):
    patch_all_ok(monkeypatch)
    problems = preflight.fatal_problems(load_config(), require_key=True, allow_network_tts=False)
    assert any("ANTHROPIC_API_KEY" in p for p in problems)


def test_ffmpeg_and_tts_failures_are_fatal_with_hints(monkeypatch):
    monkeypatch.setattr(preflight, "ffmpeg_problems", lambda: ["ffmpeg not found (brew install ffmpeg)"])
    monkeypatch.setattr(preflight, "_chromium_problem", lambda: "Chromium missing")
    monkeypatch.setattr(preflight, "engine_status", lambda cfg, allow_network: {"piper": "no", "say": "no", "edge-tts": "no"})
    checks = {c.name: c for c in preflight.run_checks(load_config(), require_key=False)}
    assert checks["ffmpeg"].fatal and "brew install ffmpeg" in checks["ffmpeg"].fix
    assert checks["chromium"].fatal and "playwright install chromium" in checks["chromium"].fix
    assert checks["tts"].fatal and "fetch-voice" in checks["tts"].fix
