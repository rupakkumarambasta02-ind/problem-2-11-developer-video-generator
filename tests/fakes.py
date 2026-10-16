"""Shared offline fakes."""

import wave
from pathlib import Path


class SilenceEngine:
    name = "silence"

    def __init__(self, seconds_per_word: float = 0.4):
        self.seconds_per_word = seconds_per_word
        self.calls: list[str] = []

    def synthesize(self, text: str, out_wav: Path) -> None:
        self.calls.append(text)
        frames = int(48_000 * max(1, len(text.split())) * self.seconds_per_word)
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48_000)
            w.writeframes(b"\x00\x00" * frames)
