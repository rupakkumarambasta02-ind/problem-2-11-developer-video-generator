"""codecast eval (spec §10.4): targets table + independent grounding re-check against a fresh clone."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from codecast.compose.ffmpeg import media_durations, probe
from codecast.config import load_config
from codecast.errors import IngestError
from codecast.grounding.validator import ValidationContext, validate_submission
from codecast.ingest.repomap import build_repomap
from codecast.models import DURATION_RANGE_S, RepoMap, Storyboard
from codecast.report import RunReport

_TS = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)")


@dataclass
class EvalRow:
    metric: str
    target: str
    value: str
    ok: bool | None


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def srt_covered_seconds(path: Path) -> float:
    spans = sorted((_secs(*m.groups()[:4]), _secs(*m.groups()[4:])) for m in _TS.finditer(Path(path).read_text()))
    total, cur_s, cur_e = 0.0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _git(args: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=180,
                         env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"})
    if res.returncode != 0:
        raise IngestError(f"git {args[0]} failed: {res.stderr.strip()[-300:]}")
    return res.stdout.strip()


def clone_at(source: str, commit_sha: str, dest: Path, repo_url: str) -> RepoMap:
    src = f"file://{Path(source).resolve()}" if Path(source).exists() else source
    _git(["clone", "--depth", "1", "--no-recurse-submodules", "--quiet", src, str(dest)])
    if _git(["rev-parse", "HEAD"], cwd=dest) != commit_sha:
        _git(["fetch", "--depth", "1", "--quiet", "origin", commit_sha], cwd=dest)
        _git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=dest)
    if _git(["rev-parse", "HEAD"], cwd=dest) != commit_sha:
        raise IngestError(f"could not check out {commit_sha}")
    return build_repomap(dest, repo_url, commit_sha)


def _mark(ok: bool | None) -> str:
    return "manual" if ok is None else ("✓" if ok else "✗")


def evaluate(out_dir: Path, *, reclone: bool = True, source: str | None = None,
             eval_log: Path = Path("docs/eval.md")) -> int:
    out_dir = Path(out_dir)
    report = RunReport.model_validate_json((out_dir / "run_report.json").read_text())
    sb = Storyboard.model_validate_json((out_dir / "storyboard.json").read_text())
    dur = media_durations(probe(out_dir / "explainer.mp4"))
    rows: list[EvalRow] = []

    ungrounded, other = None, None
    if reclone:
        with tempfile.TemporaryDirectory() as td:
            repomap = clone_at(source or sb.repo_url, sb.commit_sha, Path(td) / "repo", sb.repo_url)
            cfg = load_config()
            res = validate_submission(sb.to_submission(), ValidationContext(
                repomap=repomap, llm=sb.llm, output_allowlist=cfg.render.terminal_output_allowlist))
            ungrounded = sum(i.code.startswith("UNGROUNDED") for i in res.issues)
            other = len(res.issues) - ungrounded
            for i in res.issues[:20]:
                print(f"  re-check: {i.path} {i.code}: {i.message}")
    rows.append(EvalRow("Ungrounded numbers / identifiers", "0", "skipped" if ungrounded is None else str(ungrounded),
                        None if ungrounded is None else ungrounded == 0))
    rows.append(EvalRow("Other re-check issues", "0", "skipped" if other is None else str(other),
                        None if other is None else other == 0))
    lo, hi = DURATION_RANGE_S
    rows.append(EvalRow("Duration (ffprobe)", f"{lo:.0f}–{hi:.0f} s", f"{dur['format']:.1f} s", lo <= dur["format"] <= hi))
    narr = report.video.narration_s if report.video else 0.0
    coverage = srt_covered_seconds(out_dir / "explainer.srt") / narr if narr else 0.0
    rows.append(EvalRow("Caption coverage of narration", "≥ 95%", f"{coverage:.0%}", coverage >= 0.95))
    cost = report.cost_usd
    rows.append(EvalRow("Cost per video", "< $0.10",
                        f"${cost:.4f}" if cost is not None else f"unknown ({report.cost_unknown_reason})",
                        cost < 0.10 if cost is not None else False))
    total_s = sum(report.stage_seconds.values())
    rows.append(EvalRow("End-to-end time", "< 5 min", f"{total_s / 60:.1f} min", total_s < 300))
    rows.append(EvalRow("Human clarity rating (1–5)", "> 4", "record below", None))

    table = "| Metric | Target | Value | OK |\n|---|---|---|---|\n" + "\n".join(
        f"| {r.metric} | {r.target} | {r.value} | {_mark(r.ok)} |" for r in rows)
    print(table)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    eval_log = Path(eval_log)
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    prior = eval_log.read_text() if eval_log.exists() else ""
    entry = (f"## {stamp} — {sb.repo_url} @ {sb.commit_sha[:7]}\n\n"
             f"LLM: {report.llm.provider} / {report.llm.model} · TTS: {report.tts_engine} · run `{report.run_id}`\n\n"
             f"{table}\n\nClarity rating: _fill in_ (rater, score, notes)\n\n")
    eval_log.write_text(entry + prior)
    hard = [rows[0], rows[1], rows[2], rows[3]]
    return 0 if all(r.ok for r in hard if r.ok is not None) else 1
