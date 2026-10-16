from pathlib import Path

import pytest
from PIL import Image

from codecast.errors import RenderError
from codecast.models import Storyboard, Submission
from codecast.render.renderer import render_all
from codecast.voice.timeline import synthesize_storyboard

from tests.fakes import SilenceEngine
from tests.unit.storyboards import valid_submission

pytestmark = pytest.mark.integration


def build(raw):
    sub = Submission.model_validate(raw)
    return Storyboard(repo_url="https://github.com/o/tiny", commit_sha="c", title=sub.title,
                      sources={s.id: s for s in sub.sources}, scenes=sub.scenes)


def test_image_scene_reports_box_and_draws_highlight(renderer, tiny_repo, tmp_path):
    rs = renderer.render_scene(build(valid_submission()), 4, Path(tiny_repo.root), tmp_path)
    box = rs.image_box
    # 400x300 image contained in 1760x844 at (80,56): height-bound → 1125.33 x 844
    assert box.h == pytest.approx(844, abs=1) and box.w == pytest.approx(1125.3, abs=1)
    with Image.open(rs.frames[0]) as im:
        x = int(box.x + 0.1 * box.w) + 2
        y = int(box.y + 0.25 * box.h)
        assert im.convert("RGB").getpixel((x, y)) == (242, 204, 96)   # highlight border colour


def test_ken_burns_renders_at_2x(renderer, tiny_repo, tmp_path):
    raw = valid_submission()
    raw["scenes"][4]["motion"] = {"kind": "ken_burns", "from_rect": [0, 0, 1, 1], "to_rect": [0.2, 0.2, 0.5, 0.5]}
    rs = renderer.render_scene(build(raw), 4, Path(tiny_repo.root), tmp_path)
    with Image.open(rs.frames[0]) as im:
        assert im.size == (3840, 2160)


def test_metrics_table_and_diagram(renderer, tiny_repo, tmp_path):
    raw = valid_submission()
    raw["scenes"][4] = {"type": "diagram", "mermaid": "flowchart LR\n  A[Clip] --> B[Score] --> C[Label]",
                        "narration": [{"text": "Flow.", "source_ids": ["readme"]}]}
    sb = build(raw)
    for i in (3, 4):
        rs = renderer.render_scene(sb, i, Path(tiny_repo.root), tmp_path)
        with Image.open(rs.frames[0]) as im:
            assert im.size == (1920, 1080)
    assert "0.735" in (tmp_path / "scene04.html").read_text()


def test_terminal_sequence_frames(renderer, tiny_repo, tmp_path):
    raw = valid_submission()
    raw["sources"] += [
        {"kind": "text", "id": "c1", "file": "README.md", "start_line": 3, "end_line": 3,
         "text": "Run `python app.py --top 3` to predict."},
        {"kind": "text", "id": "c2", "file": "README.md", "start_line": 1, "end_line": 1, "text": "# Tiny"},
        {"kind": "text", "id": "log", "file": "logs/run.log", "start_line": 1, "end_line": 1, "text": "accuracy 0.91"},
    ]
    raw["scenes"][4] = {"type": "terminal", "command_span_ids": ["c1", "c2"], "output_span_id": "log",
                        "narration": [{"text": "Run it.", "source_ids": ["c1"]}]}
    rs = renderer.render_scene(build(raw), 4, Path(tiny_repo.root), tmp_path)
    assert len(rs.frames) == 2
    assert "accuracy 0.91" in (tmp_path / "scene05_f02.html").read_text()
    assert "accuracy 0.91" not in (tmp_path / "scene05_f01.html").read_text()


def test_caption_png_is_transparent_band(renderer, tmp_path):
    out = tmp_path / "cap.png"
    renderer.render_caption('He said "<hi>" & left 🎬', out)
    with Image.open(out) as im:
        assert im.size == (1920, 160) and im.mode == "RGBA"
        assert im.getpixel((5, 5))[3] == 0              # transparent corner
        assert im.getpixel((960, 80))[3] > 0             # text pill in the middle


def test_mermaid_check(renderer):
    assert renderer.mermaid_check("flowchart LR\n  A --> B") is None
    assert renderer.mermaid_check("flowchart LR\n  A --> -->") is not None


def test_render_all_builds_scene_frames(renderer, tiny_repo, tmp_path):
    sb = build(valid_submission())
    tl = synthesize_storyboard(sb, SilenceEngine(0.4), tmp_path / "audio", gap_s=0.15, tail_s=0.5)
    frames = render_all(sb, tl, Path(tiny_repo.root), tmp_path / "frames", renderer)
    assert [f.scene_id for f in frames] == [s.scene_id for s in tl.scenes]
    assert all(f.motion == "static" for f in frames)
    first = frames[0]
    assert first.duration == tl.scenes[0].duration
    assert first.captions[0].start == 0.0
    assert all(c.end <= first.duration for c in first.captions)
    assert frames[4].image_box is not None


def _code_scene(tiny_repo, n_lines, width):
    lines = [f"value_{i:02d} = compute_something_long(argument_one, argument_two, " .ljust(width - 3, "x") + f"{i:02d}"
             for i in range(1, n_lines + 1)]
    (Path(tiny_repo.root) / "src" / "long.py").write_text("\n".join(lines) + "\n")
    raw = valid_submission()
    raw["sources"].append({"kind": "text", "id": "long", "file": "src/long.py", "start_line": 1,
                           "end_line": n_lines, "text": "\n".join(lines)})
    raw["scenes"][2] = {"type": "code", "span_id": "long", "highlight_lines": [n_lines],
                        "narration": [{"text": "Long listing.", "source_ids": ["long"]}]}
    return build(raw)


def test_code_scene_of_18_wide_lines_fits_frame(renderer, tiny_repo, tmp_path):
    sb = _code_scene(tiny_repo, 18, 90)
    rs = renderer.render_scene(sb, 2, Path(tiny_repo.root), tmp_path)
    html = (tmp_path / "scene03.html").read_text()
    assert "18" in html
    # the rendered page reports no overflow: its code block ends inside the 920px stage
    info = renderer._shoot(html, tmp_path / "probe.png", repo_root=Path(tiny_repo.root))
    assert info["code_bottom"] <= 920
    assert 18 <= info["code_font_px"] <= 29
    assert len(rs.frames) == 1


def test_code_scene_that_cannot_fit_raises(renderer, tiny_repo, tmp_path):
    sb = _code_scene(tiny_repo, 18, 400)
    with pytest.raises(RenderError, match="minimum font size"):
        renderer.render_scene(sb, 2, Path(tiny_repo.root), tmp_path)


def test_code_scene_at_the_validator_width_limit_renders(renderer, tiny_repo, tmp_path):
    from codecast.grounding.validator import MAX_CODE_COLS
    sb = _code_scene(tiny_repo, 18, MAX_CODE_COLS)
    assert len(renderer.render_scene(sb, 2, Path(tiny_repo.root), tmp_path).frames) == 1
