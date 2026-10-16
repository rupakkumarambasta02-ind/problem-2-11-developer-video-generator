"""Measured-audio timeline, caption cues and SRT (spec §8)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from pydantic import BaseModel

from codecast.errors import CodecastError
from codecast.grounding.validator import MAX_WORDS, MIN_WORDS
from codecast.models import DURATION_RANGE_S, Storyboard, scene_id
from codecast.voice.engines import TTSEngine, wav_duration

FPS = 30
MAX_CUE_CHARS = 84


class SentenceAudio(BaseModel):
    text: str
    wav: str
    start: float
    duration: float


class Cue(BaseModel):
    text: str
    start: float
    end: float


class SceneTiming(BaseModel):
    scene_id: str
    start: float
    duration: float
    sentences: list[SentenceAudio]
    cues: list[Cue]


class Timeline(BaseModel):
    scenes: list[SceneTiming]
    total: float

    def per_scene_seconds(self) -> list[float]:
        return [s.duration for s in self.scenes]

    @property
    def narration_seconds(self) -> float:
        return sum(a.duration for s in self.scenes for a in s.sentences)


def split_cues(text: str, start: float, duration: float, max_chars: int = MAX_CUE_CHARS) -> list[Cue]:
    if len(text) <= max_chars:
        return [Cue(text=text, start=start, end=start + duration)]
    chunks, cur = [], ""
    for word in text.split():
        candidate = f"{cur} {word}".strip()
        if cur and len(candidate) > max_chars:
            chunks.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    total_chars = sum(len(c) for c in chunks)
    cues, t = [], start
    for i, c in enumerate(chunks):
        end = start + duration if i == len(chunks) - 1 else t + duration * len(c) / total_chars
        cues.append(Cue(text=c, start=round(t, 6), end=round(end, 6)))
        t = end
    return cues


def build_timeline(scenes: list[tuple[str, list[tuple[str, Path, float]]]], *, gap_s: float, tail_s: float) -> Timeline:
    out, t = [], 0.0
    for sid, sentences in scenes:
        start, cur = t, t
        audio, cues = [], []
        for k, (text, wav, dur) in enumerate(sentences):
            if k:
                cur += gap_s
            audio.append(SentenceAudio(text=text, wav=str(wav), start=round(cur, 6), duration=dur))
            cues += split_cues(text, round(cur, 6), dur)
            cur += dur
        raw = cur - start + tail_s
        duration = math.ceil(round(raw * FPS, 6)) / FPS
        out.append(SceneTiming(scene_id=sid, start=round(start, 6), duration=duration, sentences=audio, cues=cues))
        t = start + duration
    return Timeline(scenes=out, total=round(t, 6))


def srt_timestamp(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(timeline: Timeline, path: Path) -> None:
    blocks, n = [], 0
    for scene in timeline.scenes:
        for cue in scene.cues:
            n += 1
            blocks.append(f"{n}\n{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}\n{cue.text}\n")
    Path(path).write_text("\n".join(blocks), encoding="utf-8")


def synthesize_storyboard(sb: Storyboard, engine: TTSEngine, cache_dir: Path, *, gap_s: float, tail_s: float) -> Timeline:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scenes = []
    for i, sc in enumerate(sb.scenes):
        items = []
        for sent in sc.narration:
            key = hashlib.sha256(f"{engine.name}\x00{sent.text}".encode()).hexdigest()[:20]
            wav = cache_dir / f"{key}.wav"
            if not wav.is_file():
                tmp = cache_dir / f"{key}.tmp.wav"
                try:
                    engine.synthesize(sent.text, tmp)
                except Exception as e:  # noqa: BLE001 - any engine failure is a voice-stage failure
                    raise CodecastError(f"TTS ({engine.name}) failed for {sent.text[:60]!r}: {e}", stage="voice") from e
                tmp.replace(wav)
            items.append((sent.text, wav, wav_duration(wav)))
        scenes.append((scene_id(i), items))
    return build_timeline(scenes, gap_s=gap_s, tail_s=tail_s)


def duration_feedback(timeline: Timeline, sb: Storyboard, *, target: tuple[float, float] = DURATION_RANGE_S) -> str:
    words = sum(len(s.text.split()) for sc in sb.scenes for s in sc.narration)
    wps = words / timeline.narration_seconds if timeline.narration_seconds else 2.5
    lo, hi = target
    goal = (lo + hi) / 2
    naive = words + round((goal - timeline.total) * wps)
    target_words = min(max(naive, MIN_WORDS), MAX_WORDS)
    delta_words = target_words - words
    verb = "shorten" if timeline.total > goal else "lengthen"
    slow = ("" if verb == "lengthen" else
            " Numbers, identifiers, snake_case names and long lists are spoken slowly (each takes about as long "
            "as 3 plain words), so drop or simplify those first; the per-scene seconds show where the time goes."
            + (f" You cannot go below {MIN_WORDS} words, so you must cut slow content, not just words."
               if naive < MIN_WORDS else ""))
    return json.dumps({
        "per_scene_seconds": [round(s, 2) for s in timeline.per_scene_seconds()],
        "total": round(timeline.total, 6),
        "target": [lo, hi],
        "spoken_words": words,
        "target_words": target_words,
        "instruction": (f"The video is {timeline.total:.1f}s; it must be {lo:.0f}-{hi:.0f}s. "
                        f"{verb.capitalize()} the narration by about {abs(delta_words)} words, from {words} to "
                        f"about {target_words} words, and resubmit the full storyboard with "
                        f"submit_storyboard; unchanged narration will fail again. Keep every rule; to {verb}, "
                        f"edit or drop sentences, do not invent facts.{slow}"),
    })
