import pytest

from codecast.compose import ffmpeg as ff
from codecast.errors import RenderError


def info(fmt, v, a):
    return {"format": {"duration": str(fmt), "size": "1000"},
            "streams": [{"codec_type": "video", "duration": str(v), "width": 1920, "height": 1080},
                        {"codec_type": "audio", "duration": str(a)}]}


def test_media_durations():
    assert ff.media_durations(info(55.0, 55.0, 55.02)) == {"format": 55.0, "video": 55.0, "audio": 55.02}


@pytest.mark.parametrize("fmt,v,a,ok", [
    (55.0, 55.0, 55.02, True),
    (49.9, 49.9, 49.9, False),
    (60.2, 60.2, 60.2, True),
    (80.2, 80.2, 80.2, False),
    (55.0, 55.0, 55.2, False),
])
def test_post_check(monkeypatch, tmp_path, fmt, v, a, ok):
    mp4 = tmp_path / "x.mp4"
    mp4.write_bytes(b"x")
    monkeypatch.setattr(ff, "probe", lambda p: info(fmt, v, a))
    if ok:
        assert ff.post_check(mp4)["duration_s"] == fmt
    else:
        with pytest.raises(RenderError) as ei:
            ff.post_check(mp4)
        assert ei.value.exit_code == 6


def test_run_ffmpeg_failure_carries_stderr(tmp_path):
    with pytest.raises(RenderError, match="No such file"):
        ff.run_ffmpeg(["ffmpeg", "-v", "error", "-i", str(tmp_path / "missing.png"), str(tmp_path / "o.mp4")])


def _gone(pid):
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_generator_error_wrapped_and_child_reaped(monkeypatch):
    import subprocess
    import sys
    procs = []
    real = subprocess.Popen

    def spy(*a, **k):
        p = real(*a, **k)
        procs.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", spy)

    def gen():
        yield b"x"
        raise ValueError("bad image")

    with pytest.raises(RenderError, match="bad image") as ei:
        ff.run_ffmpeg([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin_chunks=gen())
    assert ei.value.exit_code == 6
    assert procs[0].poll() is not None and _gone(procs[0].pid)


def test_large_stderr_no_deadlock():
    import sys
    code = "import sys; sys.stderr.write('x'*300000); sys.stdin.buffer.read(); sys.exit(3)"
    chunks = (b"y" * 65536 for _ in range(64))
    with pytest.raises(RenderError, match="failed \\(3\\)"):
        ff.run_ffmpeg([sys.executable, "-c", code], stdin_chunks=chunks, timeout=30)


def test_timeout_kills_child(monkeypatch):
    import subprocess
    import sys
    procs = []
    real = subprocess.Popen

    def spy(*a, **k):
        p = real(*a, **k)
        procs.append(p)
        return p

    monkeypatch.setattr(subprocess, "Popen", spy)
    with pytest.raises(RenderError, match="timed out"):
        ff.run_ffmpeg([sys.executable, "-c", "import time; time.sleep(30)"], stdin_chunks=iter([b"x"]), timeout=1)
    assert procs[0].poll() is not None
