from pathlib import Path

import pytest
from PIL import Image

from codecast.models import Storyboard, Submission

from tests.unit.storyboards import valid_submission

pytestmark = pytest.mark.integration


def sb_for(repo):
    sub = Submission.model_validate(valid_submission())
    return Storyboard(repo_url=repo.repo_url, commit_sha="c", title=sub.title,
                      sources={s.id: s for s in sub.sources}, scenes=sub.scenes)


@pytest.mark.parametrize("index", [0, 1, 2, 5])   # title, bullets, code, outro
def test_basic_scenes_render_full_hd(renderer, tiny_repo, tmp_path, index):
    sb = sb_for(tiny_repo)
    rs = renderer.render_scene(sb, index, Path(tiny_repo.root), tmp_path)
    assert len(rs.frames) == 1
    with Image.open(rs.frames[0]) as im:
        assert im.size == (1920, 1080)
        band = im.crop((0, 920, 1920, 1080)).convert("RGB")
        assert band.getextrema() == ((13, 13), (17, 17), (23, 23))   # caption band left empty (bg #0d1117)


def test_text_is_escaped_in_rendered_page(renderer, tiny_repo, tmp_path):
    raw = valid_submission()
    raw["scenes"][1]["heading"] = '<img src=x onerror="document.body.dataset.error=1"> & co'
    sub = Submission.model_validate(raw)
    sb = Storyboard(repo_url="u", commit_sha="c", title="t", sources={s.id: s for s in sub.sources}, scenes=sub.scenes)
    renderer.render_scene(sb, 1, Path(tiny_repo.root), tmp_path)    # would raise RenderError if the onerror ran
    assert "&lt;img" in (tmp_path / "scene02.html").read_text()
