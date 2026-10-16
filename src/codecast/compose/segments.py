"""Pure builders for compose (spec §9.2): filter scripts, ffmpeg argv, concat lists, Ken Burns, audio."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterator

from PIL import Image

from codecast.errors import RenderError
from codecast.models import Box, CaptionImage, SceneFrames
from codecast.voice.timeline import Timeline

FPS, W, H, CAPTION_Y = 30, 1920, 1080, 920
SAMPLE_RATE = 48_000
BASE_FILTER = f"scale={W}:{H}:flags=lanczos,setsar=1,fps={FPS},format=rgba"


def n_frames(duration: float) -> int:
    return int(round(duration * FPS))


def _q(path: str) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _entry(path) -> str:
    """A concat-list entry. ffmpeg resolves relative entries against the list file's folder, so make them absolute."""
    return _q(str(Path(path).absolute()))


def build_filter(base: str, captions: list[CaptionImage], first_caption_input: int = 1) -> str:
    lines = [f"[0:v]{base}[v0]"]
    for k, cap in enumerate(captions):
        lines.append(f"[v{k}][{first_caption_input + k}:v]overlay=0:{CAPTION_Y}:"
                     f"enable='between(t,{cap.start:.3f},{cap.end:.3f})'[v{k + 1}]")
    lines.append(f"[v{len(captions)}]format=yuv420p[vout]")
    return ";\n".join(lines)


def sequence_list(frames: list[str], duration: float) -> str:
    each = duration / len(frames)
    out = "".join(f"file {_entry(f)}\nduration {each:.6f}\n" for f in frames)
    return out + f"file {_entry(frames[-1])}\n"


def concat_list(paths: list[Path]) -> str:
    return "".join(f"file {_entry(p)}\n" for p in paths)


def segment_command(sf: SceneFrames, out_mp4: Path, filter_file: Path, workdir: Path) -> tuple[list[str], str, dict[Path, str]]:
    dur = f"{sf.duration:.3f}"
    argv = ["ffmpeg", "-y", "-v", "error"]
    extra: dict[Path, str] = {}
    if sf.motion == "static":
        argv += ["-loop", "1", "-framerate", str(FPS), "-t", dur, "-i", sf.frames[0]]
    elif sf.motion == "ken_burns":
        argv += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-"]
    elif sf.motion == "sequence":
        lst = Path(workdir) / f"{sf.scene_id}_frames.txt"
        extra[lst] = sequence_list(sf.frames, sf.duration)
        argv += ["-f", "concat", "-safe", "0", "-i", str(lst)]
    else:
        raise RenderError(f"unknown motion {sf.motion!r}", stage="compose")
    for cap in sf.captions:
        argv += ["-loop", "1", "-framerate", str(FPS), "-t", dur, "-i", cap.path]
    ftext = build_filter(BASE_FILTER, sf.captions)
    argv += ["-/filter_complex", str(filter_file), "-map", "[vout]", "-t", dur, "-r", str(FPS),
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(out_mp4)]
    return argv, ftext, extra


def fit_16_9(x: float, y: float, w: float, h: float, bound_w: float, bound_h: float) -> tuple[float, float, float, float]:
    target = 16 / 9
    cx, cy = x + w / 2, y + h / 2
    if w / h < target:
        w = h * target
    else:
        h = w / target
    if w > bound_w:
        w, h = bound_w, bound_w / target
    if h > bound_h:
        h, w = bound_h, bound_h * target
    x = min(max(cx - w / 2, 0.0), bound_w - w)
    y = min(max(cy - h / 2, 0.0), bound_h - h)
    return (x, y, w, h)


def frame_rect(rect, box: Box, scale: float) -> tuple[float, float, float, float]:
    rx, ry, rw, rh = rect
    x = (box.x + rx * box.w) * scale
    y = (box.y + ry * box.h) * scale
    return fit_16_9(x, y, rw * box.w * scale, rh * box.h * scale, W * scale, H * scale)


def ken_burns_boxes(from_box, to_box, count: int) -> list[tuple]:
    if count <= 1:
        return [tuple(from_box)]
    out = []
    for i in range(count):
        p = i / (count - 1)
        out.append(tuple(a + (b - a) * p for a, b in zip(from_box, to_box)))
    return out


def ken_burns_frames(png: Path, sf: SceneFrames) -> Iterator[bytes]:
    if sf.image_box is None or sf.ken_burns is None:
        raise RenderError(f"{sf.scene_id}: ken_burns needs image_box and motion", stage="compose")
    with Image.open(png) as im:
        img = im.convert("RGB")
    scale = img.width / W
    start = frame_rect(sf.ken_burns.from_rect, sf.image_box, scale)
    end = frame_rect(sf.ken_burns.to_rect, sf.image_box, scale)
    for x, y, w, h in ken_burns_boxes(start, end, n_frames(sf.duration)):
        yield img.resize((W, H), Image.Resampling.LANCZOS, box=(x, y, x + w, y + h)).tobytes()


def build_audio(timeline: Timeline, out_wav: Path) -> None:
    total = round(timeline.total * SAMPLE_RATE)
    buf = bytearray(total * 2)
    for scene in timeline.scenes:
        for s in scene.sentences:
            with wave.open(s.wav) as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                    raise RenderError(f"{s.wav} is not 48 kHz mono 16-bit", stage="compose")
                data = w.readframes(w.getnframes())
            off = round(s.start * SAMPLE_RATE) * 2
            data = data[: max(0, len(buf) - off)]
            buf[off: off + len(data)] = data
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(buf))
