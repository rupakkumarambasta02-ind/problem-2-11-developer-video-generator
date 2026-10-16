"""Playwright + Jinja2 renderer (spec §9.1). Pages are served from a fake origin; the network is never used."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename
from pygments.util import ClassNotFound

from codecast.errors import RenderError
from codecast.grounding.spans import code_display_text, format_cell, load_data_decimal, resolve_pointer
from codecast.ingest.sandbox import Sandbox, SandboxError
from codecast.models import (
    Box, BulletsScene, CellRef, CodeScene, DiagramScene, ImageScene, MetricsTableScene, OutroScene,
    Storyboard, TerminalScene, TextSpan, TitleScene, scene_id,
)
from codecast.models import CaptionImage, SceneFrames
from codecast.voice.timeline import Timeline

PKG = Path(str(resources.files("codecast.render")))
TEMPLATES = PKG / "templates"
STATIC = PKG / "static"
ORIGIN = "http://codecast.local"
STATIC_URL = f"{ORIGIN}/static"
WIDTH, HEIGHT, CAPTION_HEIGHT = 1920, 1080, 160

_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))


def highlight_code(span: TextSpan, highlight_lines: list[int]) -> tuple[str, str]:
    try:
        lexer = get_lexer_for_filename(span.file, stripnl=False)
    except ClassNotFound:
        lexer = TextLexer(stripnl=False)
    fmt = HtmlFormatter(linenos="table", linenostart=span.start_line, cssclass="hl", style="github-dark",
                        hl_lines=[h - span.start_line + 1 for h in highlight_lines])
    return highlight(code_display_text(span.text), lexer, fmt), fmt.get_style_defs(".hl")


def _cell(cell, sb: Storyboard, repo_root: Path) -> dict:
    if isinstance(cell, str):
        return {"text": cell, "num": False}
    span = sb.sources[cell.span_id]
    value = resolve_pointer(load_data_decimal(repo_root / span.file), span.pointer)
    text = format_cell(value, cell.decimals)
    if text is None:
        raise RenderError(f"table cell {cell.span_id} does not resolve to a number")
    return {"text": text, "num": True}


def _command_text(text: str) -> str:
    """Command lines to type on screen. When the span holds a ``` fence (e.g. README prose around a code
    block), only the fenced lines are commands; the surrounding prose is not shown in the terminal."""
    raw = text.splitlines()
    if any(ln.strip().startswith("```") for ln in raw):
        fenced, inside = [], False
        for ln in raw:
            if ln.strip().startswith("```"):
                inside = not inside
            elif inside:
                fenced.append(ln)
        raw = fenced
    lines = [ln.strip() for ln in raw if ln.strip()]
    return "\n".join(ln[2:] if ln.startswith("$ ") else ln for ln in lines)


def scene_context(sb: Storyboard, index: int, repo_root: Path) -> tuple[str, dict]:
    sc = sb.scenes[index]
    base = {"static": STATIC_URL, "kicker": sb.title, "scene_number": index + 1, "scene_count": len(sb.scenes)}
    if isinstance(sc, TitleScene):
        return "title.html", {**base, "heading": sc.heading, "subheading": sc.subheading, "repo_url": sb.repo_url}
    if isinstance(sc, BulletsScene):
        return "bullets.html", {**base, "heading": sc.heading, "items": sc.items}
    if isinstance(sc, CodeScene):
        span = sb.sources[sc.span_id]
        code_html, code_css = highlight_code(span, sc.highlight_lines)
        return "code.html", {**base, "file": span.file, "start": span.start_line, "end": span.end_line,
                             "code_html": code_html, "code_css": code_css}
    if isinstance(sc, ImageScene):
        span = sb.sources[sc.image_id]
        return "image.html", {**base, "src": f"{ORIGIN}/repo/{quote(span.file)}", "file": span.file,
                              "highlights": [list(r) for r in sc.highlights]}
    if isinstance(sc, MetricsTableScene):
        rows = [[_cell(c, sb, repo_root) for c in row] for row in sc.rows]
        num_cols = [any(i < len(r) and r[i]["num"] for r in rows) for i in range(len(sc.columns))]
        return "metrics_table.html", {**base, "columns": sc.columns, "rows": rows, "num_cols": num_cols}
    if isinstance(sc, DiagramScene):
        return "diagram.html", {**base, "mermaid": sc.mermaid}
    if isinstance(sc, TerminalScene):
        commands = [_command_text(sb.sources[s].text) for s in sc.command_span_ids]
        output = sb.sources[sc.output_span_id].text if sc.output_span_id else None
        return "terminal.html", {**base, "commands": commands, "output": output}
    if isinstance(sc, OutroScene):
        return "outro.html", {**base, "repo_url": sc.repo_url, "cta": sc.cta, "title": sb.title}
    raise RenderError(f"unknown scene type {sc.type!r}")


class RenderedScene(BaseModel):
    frames: list[str]
    image_box: Box | None = None


def _route_handler(pages: dict[str, str], repo_root: Path | None):
    sandbox = Sandbox(repo_root) if repo_root else None
    static_root = STATIC.resolve()

    def handle(route):
        url = route.request.url
        if not url.startswith(ORIGIN + "/"):
            return route.abort()
        path = unquote(urlparse(url).path)
        if path in pages:
            return route.fulfill(status=200, body=pages[path], content_type="text/html; charset=utf-8")
        target = None
        if path.startswith("/static/"):
            cand = (static_root / path[len("/static/"):]).resolve()
            if static_root in cand.parents:
                target = cand
        elif path.startswith("/repo/") and sandbox is not None:
            try:
                target = sandbox.resolve(path[len("/repo/"):])
            except SandboxError:
                target = None
        if target is not None and target.is_file():
            return route.fulfill(path=str(target))
        return route.fulfill(status=404, body="not found")
    return handle


class Renderer:
    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self._check_ctx = None
        self._check_page = None

    def __enter__(self) -> "Renderer":
        from playwright.sync_api import sync_playwright
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch()
        except Exception as e:  # noqa: BLE001
            self.close()
            raise RenderError(f"could not start Chromium ({e}); run: python -m playwright install chromium") from e
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for obj, method in ((self._check_ctx, "close"), (self._browser, "close"), (self._pw, "stop")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:  # noqa: BLE001
                    pass
        self._pw = self._browser = self._check_ctx = self._check_page = None

    def _shoot(self, html: str, out_png: Path, *, repo_root: Path | None = None, width: int = WIDTH,
               height: int = HEIGHT, scale: int = 1, transparent: bool = False) -> dict:
        from playwright.sync_api import Error as PlaywrightError
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_png.with_suffix(".html").write_text(html, encoding="utf-8")   # kept for debugging
        ctx = self._browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale)
        try:
            ctx.route("**/*", _route_handler({"/page.html": html}, repo_root))
            page = ctx.new_page()
            page.goto(f"{ORIGIN}/page.html")
            page.wait_for_selector("body[data-ready='1']", state="attached", timeout=20_000)
            err = page.evaluate("() => document.body.dataset.error || null")
            if err:
                raise RenderError(f"{out_png.name}: {err}")
            info = page.evaluate("() => window.__codecast || {}")
            page.screenshot(path=str(out_png), omit_background=transparent)
            return info
        except PlaywrightError as e:
            raise RenderError(f"{out_png.name}: {e}") from e
        finally:
            ctx.close()

    def render_scene(self, sb: Storyboard, index: int, repo_root: Path, out_dir: Path) -> RenderedScene:
        name, ctx = scene_context(sb, index, Path(repo_root))
        tpl = _env.get_template(name)
        sid = scene_id(index)
        sc = sb.scenes[index]
        out_dir = Path(out_dir)
        if isinstance(sc, TerminalScene):
            frames = []
            n = len(ctx["commands"])
            for k in range(1, n + 1):
                png = out_dir / f"{sid}_f{k:02d}.png"
                self._shoot(tpl.render(**ctx, visible=k, show_output=(k == n)), png, repo_root=repo_root)
                frames.append(str(png))
            return RenderedScene(frames=frames)
        scale = 2 if isinstance(sc, ImageScene) and sc.motion.kind == "ken_burns" else 1
        png = out_dir / f"{sid}.png"
        info = self._shoot(tpl.render(**ctx), png, repo_root=repo_root, scale=scale)
        box = Box(**info["image_box"]) if info.get("image_box") else None
        return RenderedScene(frames=[str(png)], image_box=box)

    def render_caption(self, text: str, out_png: Path) -> None:
        html = _env.get_template("caption.html").render(static=STATIC_URL, text=text)
        self._shoot(html, out_png, height=CAPTION_HEIGHT, transparent=True)

    def mermaid_check(self, code: str) -> str | None:
        if self._check_page is None:
            self._check_ctx = self._browser.new_context()
            page = '<!doctype html><html><body><script src="/static/mermaid.min.js"></script></body></html>'
            self._check_ctx.route("**/*", _route_handler({"/check.html": page}, None))
            self._check_page = self._check_ctx.new_page()
            self._check_page.goto(f"{ORIGIN}/check.html")
            self._check_page.wait_for_function("() => window.mermaid !== undefined", timeout=20_000)
            self._check_page.evaluate("() => mermaid.initialize({ startOnLoad: false })")
        return self._check_page.evaluate(
            "async (code) => { try { await mermaid.parse(code); return null; }"
            " catch (e) { return String((e && e.message) || e); } }", code)


def render_all(sb: Storyboard, timeline: Timeline, repo_root: Path, out_dir: Path, renderer: Renderer) -> list[SceneFrames]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for i, (sc, timing) in enumerate(zip(sb.scenes, timeline.scenes)):
        rendered = renderer.render_scene(sb, i, Path(repo_root), out_dir)
        captions = []
        for k, cue in enumerate(timing.cues):
            png = out_dir / f"{timing.scene_id}_cap{k:02d}.png"
            renderer.render_caption(cue.text, png)
            captions.append(CaptionImage(path=str(png), start=round(cue.start - timing.start, 3),
                                         end=round(cue.end - timing.start, 3)))
        if isinstance(sc, ImageScene) and sc.motion.kind == "ken_burns":
            motion, kb = "ken_burns", sc.motion
        elif isinstance(sc, TerminalScene):
            motion, kb = "sequence", None
        else:
            motion, kb = "static", None
        result.append(SceneFrames(scene_id=timing.scene_id, frames=rendered.frames, motion=motion,
                                  duration=timing.duration, image_box=rendered.image_box, ken_burns=kb,
                                  captions=captions))
    return result
