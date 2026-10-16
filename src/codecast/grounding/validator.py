"""Storyboard validator (spec §7). Pure: no I/O beyond reading cited repo files."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from codecast.grounding.numbers import (
    decimals as literal_decimals, extract_claims, identifier_grounded, number_grounded,
    source_number_literals,
)
from codecast.grounding.spans import SourceIndex, check_sources, code_display_text, format_cell
from codecast.models import (
    BulletsScene, CellRef, CodeScene, DiagramScene, ImageScene, Issue, JsonSpan, MetricsTableScene,
    OutroScene, RepoMap, Storyboard, Submission, TerminalScene, TitleScene,
)


@dataclass
class ValidationContext:
    repomap: RepoMap
    llm: dict[str, str]
    output_allowlist: list[str]
    mermaid_check: Callable[[str], str | None] | None = None


@dataclass
class ValidationResult:
    storyboard: Storyboard | None
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return self.storyboard is not None and not self.issues


def _loc(loc: tuple) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "storyboard"


def scene_source_ids(scene) -> set[str]:
    ids = {sid for s in scene.narration for sid in s.source_ids}
    if isinstance(scene, CodeScene):
        ids.add(scene.span_id)
    elif isinstance(scene, ImageScene):
        ids.add(scene.image_id)
    elif isinstance(scene, MetricsTableScene):
        ids |= {c.span_id for row in scene.rows for c in row if isinstance(c, CellRef)}
    elif isinstance(scene, TerminalScene):
        ids |= set(scene.command_span_ids)
        if scene.output_span_id:
            ids.add(scene.output_span_id)
    return ids


def on_screen_texts(scene) -> list[tuple[str, str]]:
    if isinstance(scene, TitleScene):
        out = [("heading", scene.heading)]
        if scene.subheading:
            out.append(("subheading", scene.subheading))
        return out
    if isinstance(scene, BulletsScene):
        return [("heading", scene.heading)] + [(f"items[{k}]", t) for k, t in enumerate(scene.items)]
    if isinstance(scene, MetricsTableScene):
        out = [(f"columns[{k}]", t) for k, t in enumerate(scene.columns)]
        out += [(f"rows[{r}][{c}]", cell) for r, row in enumerate(scene.rows)
                for c, cell in enumerate(row) if isinstance(cell, str)]
        return out
    if isinstance(scene, DiagramScene):
        return [("mermaid", scene.mermaid)]
    if isinstance(scene, OutroScene):
        return [("cta", scene.cta)]
    return []


def _reference_issues(sb: Storyboard, index: SourceIndex) -> list[Issue]:
    issues: list[Issue] = []

    def need(sid: str, path: str, kind: str | None, hint: str = "") -> None:
        span = sb.sources.get(sid)
        if span is None:
            issues.append(Issue(path=path, code="UNKNOWN_SOURCE_ID", message=f"source {sid!r} is not declared"))
        elif kind and span.kind != kind:
            issues.append(Issue(path=path, code="WRONG_SOURCE_KIND",
                                message=f"source {sid!r} is {span.kind}, expected {kind}{hint}"))

    for i, sc in enumerate(sb.scenes):
        for j, sent in enumerate(sc.narration):
            for sid in sent.source_ids:
                need(sid, f"scenes[{i}].narration[{j}]", None)
        if isinstance(sc, CodeScene):
            need(sc.span_id, f"scenes[{i}].span_id", "text")
        elif isinstance(sc, ImageScene):
            need(sc.image_id, f"scenes[{i}].image_id", "image")
        elif isinstance(sc, MetricsTableScene):
            for r, row in enumerate(sc.rows):
                for c, cell in enumerate(row):
                    if isinstance(cell, CellRef):
                        need(cell.span_id, f"scenes[{i}].rows[{r}][{c}]", "json", _CELL_KIND_HINT)
        elif isinstance(sc, TerminalScene):
            for k, sid in enumerate(sc.command_span_ids):
                need(sid, f"scenes[{i}].command_span_ids[{k}]", "text")
            if sc.output_span_id:
                need(sc.output_span_id, f"scenes[{i}].output_span_id", "text")
    return issues


_CELL_KIND_HINT = ("; a {\"span_id\": ...} cell needs a json source (a .json/.yml/.yaml file + pointer). If the number "
                   "only appears in a text span, write it as a plain string cell (e.g. \"57.1\", copied exactly) "
                   "and cite that text span in this scene's narration")


def _ground(text: str, ids: list[str], index: SourceIndex, path: str) -> list[Issue]:
    texts = [index.texts[s] for s in ids if s in index.texts]
    literals = [lit for t in texts for lit in source_number_literals(t)]
    issues, seen = [], set()
    for claim in extract_claims(text):
        if claim.token in seen:
            continue
        seen.add(claim.token)
        if claim.kind == "number" and not number_grounded(claim.token, literals):
            issues.append(Issue(path=path, code="UNGROUNDED_NUMBER",
                                message=f"{claim.token} not found in cited sources {ids}; "
                                        f"{_holders(claim, ids, index, path)}"))
        elif claim.kind == "identifier" and not identifier_grounded(claim.token, texts):
            issues.append(Issue(path=path, code="UNGROUNDED_IDENTIFIER",
                                message=f"{claim.token} not found (case-sensitive, whole token) "
                                        f"in cited sources {ids}; {_holders(claim, ids, index, path)}"))
    return issues


def _holders(claim, ids: list[str], index: SourceIndex, path: str) -> str:
    """Tell the agent how to fix the claim: which declared sources would ground it, and whether it only
    appears in a cited json source's pointer (a pointer is not citable text)."""
    if claim.kind == "number":
        found = [s for s, t in index.texts.items() if number_grounded(claim.token, source_number_literals(t))]
    else:
        found = [s for s, t in index.texts.items() if identifier_grounded(claim.token, [t])]
    in_pointer = [s for s in ids if isinstance(index.spans.get(s), JsonSpan) and claim.token in index.spans[s].pointer]
    note = (f"{claim.token} appears only in the pointer of {', '.join(repr(s) for s in in_pointer)}, and a "
            f"pointer is not citable text; " if in_pointer else "")
    if not found:
        return f"{note}no declared source contains it: find a span that literally does, or remove it"
    where = "this sentence's source_ids" if ".narration[" in path else "a narration sentence of this scene"
    return (f"{note}declared sources that contain it: {sorted(found)}; add one of {sorted(found)} to {where}, "
            f"or remove {claim.token}")


def _grounding_issues(sb: Storyboard, index: SourceIndex) -> list[Issue]:
    issues: list[Issue] = []
    for i, sc in enumerate(sb.scenes):
        for j, sent in enumerate(sc.narration):
            issues += _ground(sent.text, sent.source_ids, index, f"scenes[{i}].narration[{j}]")
        scene_ids = sorted(scene_source_ids(sc))
        for suffix, text in on_screen_texts(sc):
            issues += _ground(text, scene_ids, index, f"scenes[{i}].{suffix}")
    return issues


MIN_SCENES, MAX_SCENES = 4, 10
MIN_WORDS, MAX_WORDS = 110, 200  # ~45-80 s of piper speech plus scene pauses
TARGET_WORDS = 155
MAX_SENTENCE_WORDS = 30
# Widest code line the code template fits at its 18px minimum font (measured: 144 with 4-digit line numbers).
MAX_CODE_COLS = 140
MAX_BULLETS_TOTAL = 12
_MULTI_SENTENCE = re.compile(r"[.!?]\s+\S")


def output_allowed(path: str, patterns: list[str]) -> bool:
    for p in patterns:
        if fnmatch.fnmatch(path, p):
            return True
        if p.startswith("**/") and fnmatch.fnmatch(path, p[3:]):
            return True
    return False


def structure_issues(sb: Storyboard) -> list[Issue]:
    issues: list[Issue] = []
    n = len(sb.scenes)
    if not MIN_SCENES <= n <= MAX_SCENES:
        issues.append(Issue(path="scenes", code="SCENE_COUNT", message=f"{n} scenes; need {MIN_SCENES}-{MAX_SCENES}"))
    if n and sb.scenes[0].type != "title":
        issues.append(Issue(path="scenes[0]", code="FIRST_SCENE", message="first scene must be 'title'"))
    if n and sb.scenes[-1].type != "outro":
        issues.append(Issue(path=f"scenes[{n - 1}]", code="LAST_SCENE", message="last scene must be 'outro'"))
    words = 0
    for i, sc in enumerate(sb.scenes):
        if not sc.narration:
            issues.append(Issue(path=f"scenes[{i}].narration", code="NARRATION_EMPTY",
                                message="every scene needs at least one narration sentence"))
        for j, sent in enumerate(sc.narration):
            path = f"scenes[{i}].narration[{j}]"
            count = len(sent.text.split())
            words += count
            if count == 0:
                issues.append(Issue(path=path, code="EMPTY_SENTENCE", message="sentence text is empty"))
            if count > MAX_SENTENCE_WORDS:
                issues.append(Issue(path=path, code="SENTENCE_TOO_LONG",
                                    message=f"{count} words; max {MAX_SENTENCE_WORDS}"))
            if not sent.source_ids:
                issues.append(Issue(path=path, code="SENTENCE_UNCITED", message="cite at least one source"))
    if not MIN_WORDS <= words <= MAX_WORDS:
        issues.append(Issue(path="scenes", code="WORD_COUNT",
                            message=f"{words} spoken words; need {MIN_WORDS}-{MAX_WORDS}; aim for ~{TARGET_WORDS} words"))
    return issues


def _rect_ok(r) -> bool:
    x, y, w, h = r
    return 0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0 and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9


def _too_long(path: str, text: str, limit: int) -> list[Issue]:
    if len(text) > limit:
        return [Issue(path=path, code="TEXT_TOO_LONG", message=f"{len(text)} chars; max {limit}")]
    return []


def scene_rule_issues(sb: Storyboard, index: SourceIndex, ctx: ValidationContext) -> list[Issue]:
    issues: list[Issue] = []
    bullets_total = 0
    for i, sc in enumerate(sb.scenes):
        p = f"scenes[{i}]"
        if isinstance(sc, TitleScene):
            issues += _too_long(f"{p}.heading", sc.heading, 60)
            texts = [("heading", sc.heading)]
            if sc.subheading is not None:
                issues += _too_long(f"{p}.subheading", sc.subheading, 90)
                texts.append(("subheading", sc.subheading))
            for suffix, text in texts:
                if _MULTI_SENTENCE.search(text):
                    issues.append(Issue(path=f"{p}.{suffix}", code="NOT_ONE_SENTENCE", message="use one sentence"))
        elif isinstance(sc, BulletsScene):
            bullets_total += len(sc.items)
            if len(sc.items) > 5:
                issues.append(Issue(path=f"{p}.items", code="TOO_MANY_ITEMS", message=f"{len(sc.items)} items; max 5"))
            for k, item in enumerate(sc.items):
                issues += _too_long(f"{p}.items[{k}]", item, 60)
        elif isinstance(sc, CodeScene):
            span = sb.sources.get(sc.span_id)
            if span is not None and span.kind == "text":
                n_lines = span.end_line - span.start_line + 1
                if n_lines > 18:
                    issues.append(Issue(path=f"{p}.span_id", code="CODE_TOO_LONG", message=f"{n_lines} lines; max 18"))
                width = max((len(ln) for ln in code_display_text(span.text or "").splitlines()), default=0)
                if width > MAX_CODE_COLS:
                    issues.append(Issue(path=f"{p}.span_id", code="CODE_TOO_WIDE",
                                        message=f"widest line is {width} characters after removing shared "
                                                f"indentation; max {MAX_CODE_COLS}. Pick a range without the "
                                                f"long lines"))
                bad = [h for h in sc.highlight_lines if not span.start_line <= h <= span.end_line]
                if bad:
                    issues.append(Issue(path=f"{p}.highlight_lines", code="HIGHLIGHT_OUT_OF_RANGE",
                                        message=f"{bad} outside {span.start_line}-{span.end_line}"))
        elif isinstance(sc, ImageScene):
            rects = [(f"{p}.highlights[{k}]", r) for k, r in enumerate(sc.highlights)]
            if sc.motion.kind == "ken_burns":
                rects += [(f"{p}.motion.from_rect", sc.motion.from_rect), (f"{p}.motion.to_rect", sc.motion.to_rect)]
            for path, r in rects:
                if not _rect_ok(r):
                    issues.append(Issue(path=path, code="RECT_OUT_OF_BOUNDS",
                                        message="rect must be [x,y,w,h] in 0-1 with x+w<=1, y+h<=1, w,h>0"))
        elif isinstance(sc, MetricsTableScene):
            for r, row in enumerate(sc.rows):
                if len(row) != len(sc.columns):
                    issues.append(Issue(path=f"{p}.rows[{r}]", code="TABLE_SHAPE",
                                        message=f"{len(row)} cells; table has {len(sc.columns)} columns"))
                for c, cell in enumerate(row):
                    if not isinstance(cell, CellRef) or cell.span_id not in index.values:
                        continue
                    value = index.values[cell.span_id]
                    rendered = format_cell(value, cell.decimals)
                    cp = f"{p}.rows[{r}][{c}]"
                    if rendered is None:
                        issues.append(Issue(path=cp, code="CELL_NOT_NUMERIC", message="cell span must resolve to a number"))
                        continue
                    literal = str(value)
                    if cell.decimals is not None and (cell.decimals < 0 or cell.decimals >= literal_decimals(literal)
                                                      or not number_grounded(rendered, [literal])):
                        issues.append(Issue(path=cp, code="CELL_UNGROUNDED",
                                            message=f"{literal} rounded to {cell.decimals} decimals is not allowed"))
        elif isinstance(sc, DiagramScene):
            if ctx.mermaid_check is not None:
                err = ctx.mermaid_check(sc.mermaid)
                if err:
                    issues.append(Issue(path=f"{p}.mermaid", code="MERMAID_SYNTAX", message=err))
        elif isinstance(sc, TerminalScene):
            if not sc.command_span_ids:
                issues.append(Issue(path=f"{p}.command_span_ids", code="TERMINAL_NO_COMMANDS",
                                    message="cite at least one command span"))
            out = sb.sources.get(sc.output_span_id) if sc.output_span_id else None
            if out is not None and not output_allowed(out.file, ctx.output_allowlist):
                issues.append(Issue(path=f"{p}.output_span_id", code="OUTPUT_NOT_ALLOWED",
                                    message=f"{out.file} is not an allowed output file {ctx.output_allowlist}"))
        elif isinstance(sc, OutroScene):
            if sc.repo_url != ctx.repomap.repo_url:
                issues.append(Issue(path=f"{p}.repo_url", code="REPO_URL_MISMATCH",
                                    message=f"must be exactly {ctx.repomap.repo_url}"))
    if bullets_total > MAX_BULLETS_TOTAL:
        issues.append(Issue(path="scenes", code="TOO_MANY_BULLETS_TOTAL",
                            message=f"{bullets_total} bullet items; max {MAX_BULLETS_TOTAL} per video"))
    return issues


def validate_submission(raw: Any, ctx: ValidationContext) -> ValidationResult:
    if not isinstance(raw, dict):
        return ValidationResult(None, [Issue(path="storyboard", code="SCHEMA",
                                             message="storyboard must be a JSON object")])
    try:
        sub = Submission.model_validate(raw)
    except ValidationError as e:
        return ValidationResult(None, [Issue(path=_loc(err["loc"]), code="SCHEMA", message=err["msg"])
                                       for err in e.errors()[:20]])
    issues, index = check_sources(sub.sources, ctx.repomap)
    sb = Storyboard(repo_url=ctx.repomap.repo_url, commit_sha=ctx.repomap.commit_sha, title=sub.title,
                    sources={s.id: index.spans.get(s.id, s) for s in sub.sources}, scenes=sub.scenes, llm=ctx.llm)
    issues += _reference_issues(sb, index)
    issues += _grounding_issues(sb, index)
    issues += structure_issues(sb)
    issues += scene_rule_issues(sb, index, ctx)
    return ValidationResult(None if issues else sb, issues)
