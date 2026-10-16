"""Download the piper voice once into ~/.cache/codecast/voices (spec §8)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from codecast.config import VoiceConfig
from codecast.voice import engines

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


def voice_urls(name: str) -> list[str]:
    lang_region, speaker, quality = name.split("-", 2)
    lang = lang_region.split("_")[0]
    stem = f"{BASE}/{lang}/{lang_region}/{speaker}/{quality}/{name}"
    return [f"{stem}.onnx", f"{stem}.onnx.json"]


def _download(url: str, dest: Path, opener) -> None:
    part = dest.with_name(dest.name + ".part")
    with opener(url, timeout=60) as resp, open(part, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    part.replace(dest)


def fetch_voice(cfg: VoiceConfig, *, opener=urllib.request.urlopen) -> Path | None:
    if not engines._piper_installed():
        print("piper-tts is not installed; CodeCast will use macOS `say` for narration.")
        return None
    dest_dir = cfg.voices_path
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        for url in voice_urls(cfg.piper_voice):
            dest = dest_dir / url.rsplit("/", 1)[-1]
            if not (dest.is_file() and dest.stat().st_size > 0):
                _download(url, dest, opener)
    except OSError as e:
        print(f"voice download failed ({e}); CodeCast will fall back to macOS `say`.")
        return None
    model = engines.piper_model_path(cfg)
    print(f"piper voice ready: {model}")
    return model
