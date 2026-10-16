"""ffmpeg execution (argv lists only, never a shell) and the ffprobe post-check (spec §9.2)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from codecast.compose.segments import build_audio, concat_list, ken_burns_frames, segment_command
from codecast.errors import CodecastError, RenderError
from codecast.models import DURATION_RANGE_S, SceneFrames
from codecast.voice.timeline import Timeline

MAX_AV_DRIFT = 0.1


def _run_streaming(argv: list[str], chunks: Iterable[bytes], timeout: float) -> tuple[int, bytes, bool]:
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    err_parts: list[bytes] = []
    fired = threading.Event()

    def drain() -> None:
        err_parts.append(proc.stderr.read())

    def on_timeout() -> None:
        fired.set()
        proc.kill()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timer = threading.Timer(timeout, on_timeout)
    timer.start()
    try:
        try:
            for chunk in chunks:
                proc.stdin.write(chunk)
        except BrokenPipeError:
            pass
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        proc.wait()
        reader.join(5)
        return proc.returncode, b"".join(err_parts), fired.is_set()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        timer.cancel()
        for pipe in (proc.stdin, proc.stderr):
            try:
                pipe.close()
            except Exception:
                pass


def run_ffmpeg(argv: list[str], *, stdin_chunks: Iterable[bytes] | None = None, timeout: float = 600) -> None:
    try:
        if stdin_chunks is None:
            res = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
            code, err = res.returncode, res.stderr
        else:
            code, err, timed_out = _run_streaming(argv, stdin_chunks, timeout)
            if timed_out:
                raise RenderError(f"{argv[0]} timed out after {timeout:.0f}s", stage="compose")
    except FileNotFoundError as e:
        raise RenderError(f"{argv[0]} not found; install it with `brew install ffmpeg`", stage="compose") from e
    except subprocess.TimeoutExpired as e:
        raise RenderError(f"{argv[0]} timed out after {timeout:.0f}s", stage="compose") from e
    except CodecastError:
        raise
    except Exception as e:
        raise RenderError(f"{argv[0]} input stream failed: {e}", stage="compose") from e
    if code != 0:
        raise RenderError(f"{argv[0]} failed ({code}): {err.decode(errors='replace')[-800:]}", stage="compose")


def probe(path: Path) -> dict:
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration,size:stream=codec_type,codec_name,duration,width,height,pix_fmt,"
                          "r_frame_rate,sample_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RenderError(f"ffprobe failed: {res.stderr[-400:]}", stage="compose")
    return json.loads(res.stdout)


def media_durations(info: dict) -> dict:
    out = {"format": float(info["format"]["duration"]), "video": None, "audio": None}
    for s in info.get("streams", []):
        if s.get("codec_type") in ("video", "audio") and s.get("duration") is not None:
            out[s["codec_type"]] = float(s["duration"])
    return out


def post_check(mp4: Path, *, lo: float = DURATION_RANGE_S[0], hi: float = DURATION_RANGE_S[1]) -> dict:
    info = probe(mp4)
    d = media_durations(info)
    if d["video"] is None or d["audio"] is None:
        raise RenderError("output is missing a video or audio stream", stage="compose")
    if not lo <= d["format"] <= hi:
        raise RenderError(f"duration {d['format']:.2f}s outside [{lo}, {hi}]", stage="compose")
    if abs(d["video"] - d["audio"]) >= MAX_AV_DRIFT:
        raise RenderError(f"audio/video drift {abs(d['video'] - d['audio']):.3f}s >= {MAX_AV_DRIFT}s", stage="compose")
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {"duration_s": d["format"], "video_s": d["video"], "audio_s": d["audio"],
            "size_bytes": int(info["format"].get("size") or Path(mp4).stat().st_size),
            "width": video.get("width"), "height": video.get("height")}


def ffmpeg_problems() -> list[str]:
    problems = []
    try:
        ver = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=30).stdout
    except FileNotFoundError:
        return ["ffmpeg not found (brew install ffmpeg)"]
    m = re.search(r"ffmpeg version n?(\d+)\.", ver)
    if not m or int(m.group(1)) < 7:
        problems.append(f"ffmpeg >= 7 required (found: {ver.splitlines()[0] if ver else 'unknown'})")
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30).stdout
    for name in ("overlay", "loudnorm", "scale", "format"):
        if not re.search(rf"^\s*\S+\s+{name}\s", filters, re.MULTILINE):
            problems.append(f"ffmpeg filter '{name}' missing")
    try:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "f.txt"
            f.write_text("[0:v]format=yuv420p[vout]")
            run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                        "-/filter_complex", str(f), "-map", "[vout]", "-f", "null", "-"], timeout=60)
    except RenderError as e:
        problems.append(f"ffmpeg does not accept -/filter_complex <file>: {e.message[-200:]}")
    if shutil.which("ffprobe") is None:
        problems.append("ffprobe not found (brew install ffmpeg)")
    return problems


def _segment(sf: SceneFrames, workdir: Path) -> Path:
    out = workdir / f"{sf.scene_id}.mp4"
    filt = workdir / f"{sf.scene_id}.filter.txt"
    argv, ftext, extra = segment_command(sf, out, filt, workdir)
    filt.write_text(ftext, encoding="utf-8")
    for path, text in extra.items():
        path.write_text(text, encoding="utf-8")
    stdin = ken_burns_frames(Path(sf.frames[0]), sf) if sf.motion == "ken_burns" else None
    run_ffmpeg(argv, stdin_chunks=stdin)
    return out


def compose_video(frames: list[SceneFrames], timeline: Timeline, out_mp4: Path, workdir: Path, *,
                  lo: float = DURATION_RANGE_S[0], hi: float = DURATION_RANGE_S[1], workers: int = 4) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        segments = list(pool.map(lambda sf: _segment(sf, workdir), frames))
    lst = workdir / "segments.txt"
    lst.write_text(concat_list(segments), encoding="utf-8")
    video = workdir / "video.mp4"
    run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)])
    raw = workdir / "narration_raw.wav"
    build_audio(timeline, raw)
    audio = workdir / "narration.m4a"
    run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar", "48000", "-c:a", "aac", "-b:a", "160k", str(audio)])
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
                "-c", "copy", "-movflags", "+faststart", str(out_mp4)])
    return post_check(out_mp4, lo=lo, hi=hi)
