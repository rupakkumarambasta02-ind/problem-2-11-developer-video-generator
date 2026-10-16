import wave
from pathlib import Path

import pytest
from PIL import Image

from codecast.compose.segments import (
    build_audio, build_filter, concat_list, fit_16_9, frame_rect, ken_burns_boxes, ken_burns_frames,
    n_frames, segment_command, sequence_list,
)
from codecast.models import Box, CaptionImage, KenBurnsMotion, SceneFrames
from codecast.voice.timeline import build_timeline

from tests.fakes import SilenceEngine


def caps():
    return [CaptionImage(path="/w/c0.png", start=0.0, end=1.5), CaptionImage(path="/w/c1.png", start=1.65, end=3.0)]


def test_build_filter_chains_caption_overlays():
    f = build_filter("scale=1920:1080,setsar=1,fps=30,format=rgba", caps())
    assert f == (
        "[0:v]scale=1920:1080,setsar=1,fps=30,format=rgba[v0];\n"
        "[v0][1:v]overlay=0:920:enable='between(t,0.000,1.500)'[v1];\n"
        "[v1][2:v]overlay=0:920:enable='between(t,1.650,3.000)'[v2];\n"
        "[v2]format=yuv420p[vout]")


def test_static_segment_command(tmp_path):
    sf = SceneFrames(scene_id="scene01", frames=["/w/s1.png"], motion="static", duration=3.5, captions=caps())
    argv, ftext, extra = segment_command(sf, tmp_path / "s1.mp4", tmp_path / "s1.txt", tmp_path)
    assert argv[0] == "ffmpeg" and "-/filter_complex" in argv
    assert argv[argv.index("-/filter_complex") + 1] == str(tmp_path / "s1.txt")
    assert argv[argv.index("-i") + 1] == "/w/s1.png" and argv.count("-loop") == 3
    assert argv[argv.index("-t") + 1] == "3.500"
    for flag, val in (("-c:v", "libx264"), ("-crf", "20"), ("-pix_fmt", "yuv420p"), ("-r", "30")):
        assert argv[argv.index(flag) + 1] == val
    assert extra == {} and "[vout]" in ftext


def test_ken_burns_segment_reads_stdin(tmp_path):
    sf = SceneFrames(scene_id="scene02", frames=["/w/s2.png"], motion="ken_burns", duration=2.0,
                     image_box=Box(x=80, y=56, w=1760, h=844),
                     ken_burns=KenBurnsMotion(from_rect=(0, 0, 1, 1), to_rect=(0.25, 0.25, 0.5, 0.5)))
    argv, _, _ = segment_command(sf, tmp_path / "o.mp4", tmp_path / "f.txt", tmp_path)
    i = argv.index("-f")
    assert argv[i:i + 8] == ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-framerate", "30"]
    assert argv[argv.index("-i") + 1] == "-"


def test_sequence_segment_writes_concat_list(tmp_path):
    sf = SceneFrames(scene_id="scene03", frames=["/w/a.png", "/w/b'c.png"], motion="sequence", duration=3.0)
    argv, _, extra = segment_command(sf, tmp_path / "o.mp4", tmp_path / "f.txt", tmp_path)
    lst = tmp_path / "scene03_frames.txt"
    assert argv[argv.index("-i") + 1] == str(lst)
    assert extra[lst] == sequence_list(sf.frames, 3.0)
    assert sequence_list(sf.frames, 3.0) == (
        "file '/w/a.png'\nduration 1.500000\nfile '/w/b'\\''c.png'\nduration 1.500000\nfile '/w/b'\\''c.png'\n")


def test_concat_list():
    assert concat_list([Path("/w/a.mp4"), Path("/w/b.mp4")]) == "file '/w/a.mp4'\nfile '/w/b.mp4'\n"


def test_lists_make_relative_paths_absolute():
    # ffmpeg's concat demuxer resolves relative entries against the list file's folder, not the cwd.
    cwd = Path.cwd()
    assert concat_list([Path("ws/a.mp4")]) == f"file '{cwd / 'ws/a.mp4'}'\n"
    assert sequence_list(["ws/a.png"], 2.0) == (
        f"file '{cwd / 'ws/a.png'}'\nduration 2.000000\nfile '{cwd / 'ws/a.png'}'\n")


@pytest.mark.parametrize("rect,expected", [
    ((0, 0, 1920, 1080), (0, 0, 1920, 1080)),
    ((900, 500, 160, 160), (837.7777, 500, 284.4444, 160)),      # widen to 16:9 around centre
    ((1800, 1000, 400, 100), (1520, 855, 400, 225)),              # clamp inside frame
])
def test_fit_16_9(rect, expected):
    assert fit_16_9(*rect, 1920, 1080) == pytest.approx(expected, abs=1e-3)


def test_frame_rect_maps_image_relative_to_2x_pixels():
    box = Box(x=80, y=56, w=1760, h=844)
    x, y, w, h = frame_rect((0, 0, 1, 1), box, 2.0)
    assert w / h == pytest.approx(16 / 9) and 0 <= x and x + w <= 3840 + 1e-6 and y + h <= 2160 + 1e-6


def test_ken_burns_boxes_interpolate():
    boxes = ken_burns_boxes((0, 0, 100, 50), (50, 25, 50, 25), 3)
    assert boxes == [(0, 0, 100, 50), (25, 12.5, 75, 37.5), (50, 25, 50, 25)]
    assert ken_burns_boxes((0, 0, 1, 1), (2, 2, 1, 1), 1) == [(0, 0, 1, 1)]


def test_ken_burns_frames_size_and_count(tmp_path):
    png = tmp_path / "k.png"
    Image.new("RGB", (3840, 2160), (10, 20, 30)).save(png)
    sf = SceneFrames(scene_id="s", frames=[str(png)], motion="ken_burns", duration=0.2,
                     image_box=Box(x=80, y=56, w=1760, h=844),
                     ken_burns=KenBurnsMotion(from_rect=(0, 0, 1, 1), to_rect=(0.2, 0.2, 0.5, 0.5)))
    frames = list(ken_burns_frames(png, sf))
    assert len(frames) == n_frames(0.2) == 6
    assert all(len(f) == 1920 * 1080 * 3 for f in frames)


def test_build_audio_places_sentences(tmp_path):
    eng = SilenceEngine(0.5)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    eng.synthesize("x", a)
    eng.synthesize("y", b)
    tl = build_timeline([("scene01", [("x", a, 0.5), ("y", b, 0.5)])], gap_s=0.15, tail_s=0.5)
    out = tmp_path / "narration.wav"
    build_audio(tl, out)
    with wave.open(str(out)) as w:
        assert w.getframerate() == 48_000 and w.getnchannels() == 1
        assert w.getnframes() == round(tl.total * 48_000)
