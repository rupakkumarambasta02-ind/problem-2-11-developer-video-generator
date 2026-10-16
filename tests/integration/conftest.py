import pytest

from codecast.render.renderer import Renderer


@pytest.fixture(scope="session")
def renderer():
    with Renderer() as r:
        yield r
