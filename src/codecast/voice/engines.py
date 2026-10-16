"""Local-first TTS engines (spec §8): piper → say → edge-tts (network, opt-in). One engine per video."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Protocol

from codecast.config import VoiceConfig
from codecast.errors import CodecastError, ConfigError


class TTSEngine(Protocol):
    name: str

    def synthesize(self, text: str, out_wav: Path) -> None: ...


def convert_to_wav48k(src: Path, dst: Path) -> None:
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "48000",
                        "-c:a", "pcm_s16le", str(dst)], check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise CodecastError(f"audio conversion failed: {e.stderr.decode(errors='replace')[-300:]}", stage="voice") from e


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


class SayEngine:
    name = "say"

    def __init__(self, voice: str = "Samantha"):
        self.voice = voice

    def synthesize(self, text: str, out_wav: Path) -> None:
        # macOS `say` treats `<word>` as embedded markup and silently drops the rest of the text,
        # so angle brackets are spoken as words before the text is handed over.
        spoken = text.replace("<", " less than ").replace(">", " greater than ")
        with tempfile.TemporaryDirectory() as td:
            txt, aiff = Path(td) / "in.txt", Path(td) / "out.aiff"
            txt.write_text(spoken, encoding="utf-8")
            subprocess.run(["say", "-v", self.voice, "-o", str(aiff), "-f", str(txt)],
                           check=True, capture_output=True, timeout=120)
            convert_to_wav48k(aiff, out_wav)


class PiperEngine:
    name = "piper"

    def __init__(self, model_path: Path):
        from piper import PiperVoice
        self._voice = PiperVoice.load(str(model_path))

    def synthesize(self, text: str, out_wav: Path) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.wav"
            with wave.open(str(raw), "wb") as wf:
                synth = getattr(self._voice, "synthesize_wav", None) or self._voice.synthesize
                synth(text, wf)
            convert_to_wav48k(raw, out_wav)


class EdgeEngine:
    name = "edge-tts"

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice

    def synthesize(self, text: str, out_wav: Path) -> None:
        with tempfile.TemporaryDirectory() as td:
            txt, mp3 = Path(td) / "in.txt", Path(td) / "out.mp3"
            txt.write_text(text, encoding="utf-8")
            subprocess.run([sys.executable, "-m", "edge_tts", "--voice", self.voice, "--file", str(txt),
                            "--write-media", str(mp3)], check=True, capture_output=True, timeout=120)
            convert_to_wav48k(mp3, out_wav)


def _piper_installed() -> bool:
    return importlib.util.find_spec("piper") is not None


def _edge_installed() -> bool:
    return importlib.util.find_spec("edge_tts") is not None


def _say_available(voice: str) -> bool:
    if shutil.which("say") is None:
        return False
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=30).stdout
    return any(line.split()[0] == voice for line in out.splitlines() if line.strip())


def piper_model_path(cfg: VoiceConfig) -> Path:
    return cfg.voices_path / f"{cfg.piper_voice}.onnx"


def engine_status(cfg: VoiceConfig, *, allow_network: bool) -> dict[str, str]:
    status = {}
    if not _piper_installed():
        status["piper"] = "piper-tts not installed"
    elif not piper_model_path(cfg).is_file():
        status["piper"] = "voice missing; run `codecast fetch-voice`"
    else:
        status["piper"] = "ok"
    status["say"] = "ok" if _say_available(cfg.say_voice) else f"macOS say or voice {cfg.say_voice!r} unavailable"
    if not _edge_installed():
        status["edge-tts"] = "edge-tts not installed"
    else:
        status["edge-tts"] = "ok" if allow_network else "disabled (needs --allow-network-tts)"
    return status


def select_engine(cfg: VoiceConfig, *, allow_network: bool) -> TTSEngine:
    st = engine_status(cfg, allow_network=allow_network)
    if st["piper"] == "ok":
        return PiperEngine(piper_model_path(cfg))
    if st["say"] == "ok":
        return SayEngine(cfg.say_voice)
    if st["edge-tts"] == "ok":
        return EdgeEngine(cfg.edge_voice)
    raise ConfigError("no TTS engine available: run `codecast fetch-voice` (piper), use macOS `say`, "
                      f"or pass --allow-network-tts with edge-tts installed. Status: {st}")
