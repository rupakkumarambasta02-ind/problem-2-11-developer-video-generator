"""codecast run | doctor | fetch-voice | eval (spec §10.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codecast.config import load_config
from codecast.errors import CodecastError
from codecast.pipeline import RunOptions
from codecast.pipeline import run as pipeline_run
from codecast.preflight import run_checks


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codecast", description="GitHub repo → grounded explainer video")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate a video")
    r.add_argument("--repo", required=True, help="public https GitHub URL or local git path")
    r.add_argument("--out", default="output", type=Path)
    r.add_argument("--config", type=Path)
    r.add_argument("--max-steps", type=int)
    r.add_argument("--allow-network-tts", action="store_true")
    r.add_argument("--from-stage", choices=["voice", "render", "compose"])
    r.add_argument("--run", dest="run_id")
    r.add_argument("--workspace", type=Path, default=Path("workspace"))
    d = sub.add_parser("doctor", help="check the local setup")
    d.add_argument("--config", type=Path)
    f = sub.add_parser("fetch-voice", help="download the piper voice")
    f.add_argument("--config", type=Path)
    e = sub.add_parser("eval", help="evaluate a generated output directory")
    e.add_argument("out_dir", type=Path)
    e.add_argument("--no-reclone", action="store_true", help="skip the independent re-check against a fresh clone")
    e.add_argument("--source", help="clone the re-check from this path or URL instead of repo_url")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(getattr(args, "config", None))
        if args.cmd == "doctor":
            checks = run_checks(cfg, require_key=False)
            for c in checks:
                tag = "OK  " if c.ok else ("FAIL" if c.fatal else "WARN")
                print(f"[{tag}] {c.name}: {c.message}" + (f"\n       fix: {c.fix}" if c.fix and not c.ok else ""))
            return 7 if any(c.fatal and not c.ok for c in checks) else 0
        if args.cmd == "fetch-voice":
            from codecast.voice.fetch_voice import fetch_voice
            fetch_voice(cfg.voice)
            return 0
        if args.cmd == "eval":
            from codecast.eval import evaluate
            return evaluate(args.out_dir, reclone=not args.no_reclone, source=args.source)
        opts = RunOptions(repo=args.repo, out_dir=args.out, workspace=args.workspace, max_steps=args.max_steps,
                          allow_network_tts=args.allow_network_tts, from_stage=args.from_stage, run_id=args.run_id)
        report = pipeline_run(opts, cfg)
        status = "OK" if report.exit_code == 0 else f"FAILED (exit {report.exit_code})"
        detail = (f"{report.video.duration_s:.1f}s → {report.video.path}" if report.video
                  else f"{report.failure.stage}: {report.failure.message}")
        print(f"codecast {status}: {detail}\nrun report: {args.workspace / report.run_id / 'run_report.json'}")
        return report.exit_code
    except CodecastError as e:
        print(f"codecast: {e.message}", file=sys.stderr)
        return e.exit_code


def main_entry() -> None:
    sys.exit(main())
