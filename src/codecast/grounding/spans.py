"""Source integrity (spec §7 step 2): JSON Pointer, exact span text, decimal-safe JSON."""

from __future__ import annotations

import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

import yaml

from codecast.models import ImageRef, Issue, JsonSpan, RepoMap, Source, TextSpan


class PointerError(Exception):
    pass


def resolve_pointer(doc: Any, pointer: str) -> Any:
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise PointerError("pointer must be empty or start with '/'")
    cur = doc
    toks = [raw.replace("~1", "/").replace("~0", "~") for raw in pointer[1:].split("/")]
    for i, tok in enumerate(toks):
        if isinstance(cur, dict):
            if tok not in cur:
                raise PointerError(f"key {tok!r} not found{_slash_key_hint(cur, toks, i)}")
            cur = cur[tok]
        elif isinstance(cur, list):
            # Bounded digits keep int() far below the int-string conversion limit.
            if not re.fullmatch(r"0|[1-9]\d{0,17}", tok) or int(tok) >= len(cur):
                raise PointerError(f"invalid array index {tok!r}")
            cur = cur[int(tok)]
        else:
            raise PointerError(f"cannot descend into {type(cur).__name__} at {tok!r}")
    return cur


def _slash_key_hint(obj: dict, toks: list[str], i: int) -> str:
    """Point out a key containing '/' that the pointer split into several tokens."""
    for j in range(len(toks), i + 1, -1):
        key = "/".join(toks[i:j])
        if key in obj:
            escaped = "/" + "/".join(t.replace("~", "~0").replace("/", "~1") for t in [key, *toks[j:]])
            return f"; key {key!r} exists: escape '/' inside keys as '~1', e.g. {escaped!r}"
    return ""


def read_text_normalized(path: Path) -> str:
    text = Path(path).read_bytes().decode("utf-8", errors="replace")
    return text[1:] if text.startswith("﻿") else text


def read_lines(path: Path) -> list[str]:
    return read_text_normalized(path).splitlines()


def code_display_text(text: str) -> str:
    """Code as a code scene shows it: tabs expanded to 8 columns, indentation shared by all lines removed."""
    return textwrap.dedent(text.expandtabs(8))


def load_json_decimal(path: Path) -> Any:
    text = read_text_normalized(path)
    try:
        return json.loads(text, parse_float=Decimal)
    except json.JSONDecodeError as strict_error:
        relaxed = _strip_jsonc(text)
        if relaxed == text:
            raise
        try:  # JSON with comments / trailing commas, as found in many repos' config and data files
            return json.loads(relaxed, parse_float=Decimal)
        except json.JSONDecodeError:
            raise strict_error from None


def _strip_jsonc(text: str) -> str:
    """Drop // and /* */ comments and trailing commas outside strings; newlines are kept."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        else:
            out.append(c)
            i += 1
    stripped = "".join(out)
    return re.sub(r',(\s*[}\]])|("(?:[^"\\]|\\.)*")', lambda m: m.group(2) or m.group(1), stripped)


class _DecimalYamlLoader(yaml.SafeLoader):
    """SafeLoader that keeps floats exact (Decimal from the literal) and timestamps as their literal text."""


def _yaml_float(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Any:
    try:
        return Decimal(loader.construct_scalar(node).replace("_", ""))
    except InvalidOperation:  # .inf / .nan spellings
        return loader.construct_yaml_float(node)


_DecimalYamlLoader.add_constructor("tag:yaml.org,2002:float", _yaml_float)
_DecimalYamlLoader.add_constructor("tag:yaml.org,2002:timestamp", yaml.SafeLoader.construct_scalar)

YAML_SUFFIXES = (".yml", ".yaml")


def load_data_decimal(path: Path) -> Any:
    """Load a JSON or YAML data file (chosen by suffix) with exact Decimal floats."""
    if Path(path).suffix.lower() not in YAML_SUFFIXES:
        return load_json_decimal(path)
    try:
        return yaml.load(read_text_normalized(path), Loader=_DecimalYamlLoader)
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML: {e}") from None


def dumps_decimal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, Decimal)):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(dumps_decimal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(str(k), ensure_ascii=False)}: {dumps_decimal(v)}"
                               for k, v in value.items()) + "}"
    return json.dumps(str(value))


def _json_safe(value: Any) -> Any:
    """Resolved (Decimal-bearing) JSON as plain Python values that round-trip through values_equal.

    Raises ValueError for a number that cannot be stored exactly as an int or float.
    """
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite number")
        if value == value.to_integral_value() and "." not in str(value) and "E" not in str(value).upper():
            return int(value)
        f = float(value)
        if math.isfinite(f) and Decimal(repr(f)) == value:
            return f
        raise ValueError(f"{value} is not exactly representable as a float")
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def _num_equal(a: Any, b: Any) -> bool:
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except (ValueError, InvalidOperation):
        return False


def _show(value: Any) -> str:
    try:
        return dumps_decimal(value)
    except (RecursionError, ValueError):
        return "<value too large to show>"


def values_equal(agent_value: Any, resolved: Any) -> bool:
    if _is_number(resolved):
        return _is_number(agent_value) and _num_equal(agent_value, resolved)
    if isinstance(resolved, list):
        return (isinstance(agent_value, list) and len(agent_value) == len(resolved)
                and all(values_equal(a, r) for a, r in zip(agent_value, resolved)))
    if isinstance(resolved, dict):
        return (isinstance(agent_value, dict) and agent_value.keys() == resolved.keys()
                and all(values_equal(agent_value[k], resolved[k]) for k in resolved))
    return type(agent_value) is type(resolved) and agent_value == resolved


@dataclass
class SourceIndex:
    spans: dict[str, Source] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)


def _issue(i: int, code: str, message: str) -> Issue:
    return Issue(path=f"sources[{i}]", code=code, message=message)


def _exact_range(got: list[str], lines: list[str], start_line: int) -> tuple[int, int] | None:
    """The 1-based line range whose lines equal `got` exactly, nearest to start_line; None if none."""
    n = len(got)
    hits = [s for s in range(len(lines) - n + 1) if lines[s:s + n] == got]
    if not hits:
        return None
    s = min(hits, key=lambda h: abs(h + 1 - start_line))
    return s + 1, s + n


def _mismatch_hint(claimed: str, actual: str, start_line: int, lines: list[str]) -> str:
    """Say where claimed span text first departs from the file, so the agent can fix it in one round."""
    got, want = claimed.split("\n"), actual.split("\n")
    exact = _exact_range(got, lines, start_line)
    if exact is not None:
        return (f"your text is exactly lines {exact[0]}-{exact[1]}: set start_line {exact[0]} and "
                f"end_line {exact[1]}, keeping the text")
    if len(got) != len(want):
        return (f"your text has {len(got)} line(s); lines {start_line}-{start_line + len(want) - 1} "
                f"are {len(want)} lines")
    k = next(k for k, (a, b) in enumerate(zip(got, want)) if a != b)
    shown = want[k] if len(want[k]) <= 160 else want[k][:160] + "..."
    return f"first difference at line {start_line + k}: the file has {shown!r}, keep its leading whitespace"


def check_sources(sources: list[Source], repomap: RepoMap) -> tuple[list[Issue], SourceIndex]:
    root = Path(repomap.root)
    issues: list[Issue] = []
    index = SourceIndex()
    for i, src in enumerate(sources):
        if src.id in index.spans:
            issues.append(_issue(i, "DUPLICATE_SOURCE_ID", f"source id {src.id!r} declared twice"))
            continue
        index.spans[src.id] = src
        entry = repomap.file(src.file)
        if isinstance(src, ImageRef):
            if entry is None or entry.kind != "image":
                issues.append(_issue(i, "IMAGE_NOT_FOUND", f"{src.file} is not an image in the repo"))
            index.texts[src.id] = ""
            continue
        if entry is None:
            issues.append(_issue(i, "SPAN_FILE_MISSING", f"{src.file} is not a readable file in the repo"))
            continue
        if entry.kind != "text":
            issues.append(_issue(i, "SPAN_NOT_TEXT", f"{src.file} is {entry.kind}, not text"))
            continue
        path = root / src.file
        if isinstance(src, TextSpan):
            try:
                lines = read_lines(path)
            except OSError as e:
                issues.append(_issue(i, "SPAN_FILE_MISSING", f"{src.file} could not be read: {e}"))
                continue
            if not (1 <= src.start_line <= src.end_line <= len(lines)):
                issues.append(_issue(i, "SPAN_RANGE_INVALID",
                                     f"lines {src.start_line}-{src.end_line} outside 1-{len(lines)}"))
                continue
            actual = "\n".join(lines[src.start_line - 1:src.end_line])
            if src.text is None:
                index.spans[src.id] = src.model_copy(update={"text": actual})
                index.texts[src.id] = actual
                continue
            claimed = src.text.replace("\r\n", "\n")
            # One terminal newline is tolerated, but only if dropping it is what makes the text match:
            # a span ending on a blank line is legitimately "...\n" (lines joined with "\n").
            if claimed != actual and claimed.endswith("\n"):
                claimed = claimed[:-1]
            if claimed != actual:
                issues.append(_issue(i, "SPAN_TEXT_MISMATCH",
                                     f"text differs from {src.file}:{src.start_line}-{src.end_line} "
                                     f"({_mismatch_hint(claimed, actual, src.start_line, lines)}); copy the lines "
                                     f"exactly as read_file shows them (without line numbers)"))
                continue
            index.spans[src.id] = src.model_copy(update={"text": actual})  # canonical file text
            index.texts[src.id] = actual
        elif isinstance(src, JsonSpan):
            try:
                doc = load_data_decimal(path)
            except (ValueError, UnicodeDecodeError, RecursionError, OSError) as e:
                suffix = path.suffix.lower()
                if suffix not in (".json", *YAML_SUFFIXES):
                    issues.append(_issue(i, "NOT_A_DATA_FILE",
                                         f"{src.file} is not a JSON or YAML data file; json sources need .json, .yml "
                                         f"or .yaml. Cite {src.file} as kind text (file + start_line + end_line) and, "
                                         f"for a table, write the number as a plain string cell"))
                    continue
                fmt = "YAML" if suffix in YAML_SUFFIXES else "JSON"
                issues.append(_issue(i, "JSON_PARSE_ERROR", f"{src.file} could not be parsed as {fmt}: {e}"))
                continue
            try:
                resolved = resolve_pointer(doc, src.pointer)
            except PointerError as e:
                issues.append(_issue(i, "JSON_POINTER_UNRESOLVED", f"{src.pointer}: {e}"))
                continue
            provided = "value" in src.model_fields_set
            try:
                equal = values_equal(src.value, resolved) if provided else True
            except RecursionError:
                equal = False
            if not equal:
                issues.append(_issue(i, "JSON_VALUE_MISMATCH",
                                     f"value at {src.pointer} is {_show(resolved)}"))
                continue
            if not provided:
                try:
                    safe = _json_safe(resolved)
                except ValueError as e:
                    issues.append(_issue(i, "JSON_VALUE_UNREPRESENTABLE",
                                         f"value at {src.pointer} cannot be stored exactly ({e}); cite something else"))
                    continue
                index.spans[src.id] = src.model_copy(update={"value": safe}, deep=True)
            index.values[src.id] = resolved
            index.texts[src.id] = resolved if isinstance(resolved, str) else _show(resolved)
    return issues, index


def format_cell(value: Any, decimals: int | None) -> str | None:
    if not (isinstance(value, (int, Decimal)) and not isinstance(value, bool)):
        return None
    if decimals is not None and decimals < 0:
        return None
    d = Decimal(value)  # Decimal(int) has no digit limit, unlike str(int)
    if not d.is_finite():
        return None
    if decimals is None:
        try:
            return str(value)
        except ValueError:  # int beyond the int-string conversion limit
            return None
    with localcontext() as ctx:
        # Size precision to the operand so wide values cannot raise InvalidOperation.
        ctx.prec = max(28, d.adjusted() + 1 + decimals + 10)
        return f"{d.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP):f}"
