from pathlib import Path

import pytest
from PIL import Image

from codecast.compose.ffmpeg import compose_video, probe
from codecast.models import Box, CaptionImage, KenBurnsMotion, SceneFrames
from codecast.voice.timeline import build_timeline

from tests.fakes import SilenceEngine

pytestmark = pytest.mark.integration


def png(path: Path, size, colour):
    Image.new("RGB", size, colour).save(path)
    return str(path)


def cap(path: Path):
    im = Image.new("RGBA", (1920, 160), (0, 0, 0, 0))
    for x in range(600, 1320):
        for y in range(40, 120):
            im.putpixel((x, y), (255, 255, 255, 255))
    im.save(path)
    return str(path)


def test_three_scene_compose_passes_ffprobe(tmp_path):
    eng = SilenceEngine(0.5)
    wavs = []
    for i in range(3):
        w = tmp_path / f"a{i}.wav"
        eng.synthesize("a b c", w)          # 1.5 s each
        wavs.append(w)
    tl = build_timeline([(f"scene0{i + 1}", [("a b c", wavs[i], 1.5)]) for i in range(3)], gap_s=0.15, tail_s=0.5)
    frames = [
        SceneFrames(scene_id="scene01", frames=[png(tmp_path / "s1.png", (1920, 1080), (13, 17, 23))],
                    motion="static", duration=tl.scenes[0].duration,
                    captions=[CaptionImage(path=cap(tmp_path / "c1.png"), start=0.0, end=1.5)]),
        SceneFrames(scene_id="scene02", frames=[png(tmp_path / "s2.png", (3840, 2160), (40, 80, 120))],
                    motion="ken_burns", duration=tl.scenes[1].duration, image_box=Box(x=80, y=56, w=1760, h=844),
                    ken_burns=KenBurnsMotion(from_rect=(0, 0, 1, 1), to_rect=(0.25, 0.25, 0.5, 0.5))),
        SceneFrames(scene_id="scene03", frames=[png(tmp_path / "t1.png", (1920, 1080), (0, 0, 0)),
                                                png(tmp_path / "t2.png", (1920, 1080), (90, 90, 90))],
                    motion="sequence", duration=tl.scenes[2].duration),
    ]
    out = tmp_path / "explainer.mp4"
    result = compose_video(frames, tl, out, tmp_path / "work", lo=tl.total - 0.1, hi=tl.total + 0.1)
    assert abs(result["video_s"] - result["audio_s"]) < 0.1
    assert result["duration_s"] == pytest.approx(tl.total, abs=0.1)
    v = [s for s in probe(out)["streams"] if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"], v["pix_fmt"], v["r_frame_rate"]) == (1920, 1080, "yuv420p", "30/1")
    a = [s for s in probe(out)["streams"] if s["codec_type"] == "audio"][0]
    assert a["codec_name"] == "aac" and a["sample_rate"] == "48000"
