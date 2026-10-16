"""Preflight checks (spec §9.3). Shared by `codecast doctor` and the start of `codecast run`."""

from __future__ import annotations

import os
from dataclasses import dataclass

from codecast.compose.ffmpeg import ffmpeg_problems
from codecast.config import Config
from codecast.voice.engines import engine_status


@dataclass
class Check:
    name: str
    ok: bool
    message: str
    fix: str | None
    fatal: bool


def _chromium_problem() -> str | None:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return None
    except Exception as e:  # noqa: BLE001
        return f"Playwright Chromium cannot start: {str(e).splitlines()[0]}"


def run_checks(cfg: Config, *, require_key: bool, allow_network_tts: bool = False) -> list[Check]:
    checks = []
    ff = ffmpeg_problems()
    checks.append(Check("ffmpeg", not ff, "; ".join(ff) or "ffmpeg >= 7 with overlay, loudnorm, -/filter_complex",
                        "brew install ffmpeg" if ff else None, True))
    chrome = _chromium_problem()
    checks.append(Check("chromium", chrome is None, chrome or "headless Chromium starts",
                        ".venv/bin/python -m playwright install chromium" if chrome else None, True))
    st = engine_status(cfg.voice, allow_network=allow_network_tts)
    chosen = next((name for name in ("piper", "say", "edge-tts") if st[name] == "ok"), None)
    checks.append(Check("tts", chosen is not None, f"engine: {chosen}" if chosen else f"no engine: {st}",
                        None if chosen else "codecast fetch-voice  (or run on macOS with `say`)", True))
    has_key = bool(os.environ.get(cfg.llm.api_key_env))
    checks.append(Check("llm api key", has_key,
                        f"{cfg.llm.api_key_env} is set" if has_key else f"{cfg.llm.api_key_env} is not set",
                        None if has_key else f"export {cfg.llm.api_key_env}=...", require_key))
    return checks


def fatal_problems(cfg: Config, *, require_key: bool, allow_network_tts: bool) -> list[str]:
    return [f"{c.name}: {c.message} (fix: {c.fix})" for c in run_checks(cfg, require_key=require_key,
                                                                       allow_network_tts=allow_network_tts)
            if c.fatal and not c.ok]
