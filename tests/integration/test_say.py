import shutil
import wave

import pytest

from codecast.voice.engines import SayEngine, wav_duration

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS say not available")
def test_say_produces_48k_mono_wav(tmp_path):
    out = tmp_path / "s.wav"
    SayEngine("Samantha").synthesize('- Hello <world> & "friends" 🎬, version 0.735.', out)
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (48_000, 1, 2)
    assert wav_duration(out) > 1.0
