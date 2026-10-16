"""Stage orchestration, resume and atomic outputs (spec §2, §10.1)."""

from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from codecast.agent.loop import DirectorAgent, default_max_steps, initial_messages, load_director_prompt
from codecast.agent.submit import Submitter
from codecast.agent.tools import ToolBox
from codecast.agent.transcript import Transcript
from codecast.compose.ffmpeg import compose_video
from codecast.config import Config
from codecast.errors import CodecastError, ConfigError, DurationError
from codecast.grounding.validator import ValidationContext
from codecast.ingest.clone import ingest, repo_name
from codecast.llm.pricing import estimate_cost
from codecast.llm.registry import create_provider
from codecast.models import DURATION_RANGE_S, RepoMap, SceneFrames, Storyboard
from codecast.render.renderer import Renderer, render_all
from codecast.report import ReportBuilder, RunReport, VideoInfo
from codecast.voice.engines import select_engine
from codecast.voice.timeline import Timeline, duration_feedback, synthesize_storyboard, write_srt

STAGES = ("ingest", "agent", "voice", "render", "compose")
MAX_DURATION_ROUNDS = 2


@dataclass
class RunOptions:
    repo: str
    out_dir: Path
    workspace: Path = Path("workspace")
    max_steps: int | None = None
    allow_network_tts: bool = False
    from_stage: str | None = None
    run_id: str | None = None
    duration_range: tuple[float, float] = DURATION_RANGE_S


def safe_mermaid_check(check):
    """Wrap Renderer.mermaid_check so a browser hiccup skips the check instead of crashing the agent."""
    def wrapped(code: str) -> str | None:
        try:
            return check(code)
        except Exception:  # noqa: BLE001 - invalid diagrams still fail at render (exit 6)
            return None
    return wrapped


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_or_fail(path: Path, what: str, stage: str) -> str:
    if not path.is_file():
        raise ConfigError(f"cannot resume: {what} not found at {path}", stage=stage)
    return path.read_text(encoding="utf-8")


def run(opts: RunOptions, cfg: Config, *, provider=None, engine=None, renderer_factory=Renderer,
        preflight: bool = True) -> RunReport:
    run_id = opts.run_id or _new_run_id()
    ws = Path(opts.workspace) / run_id
    rb = ReportBuilder(run_id, opts.repo, cfg)
    rep = rb.report
    out_root = Path(opts.out_dir)
    try:
        name = repo_name(opts.repo)
    except Exception:  # noqa: BLE001 - reported properly at the ingest stage
        name = "unknown"
    current = "preflight"
    lo, hi = opts.duration_range
    try:
        if opts.from_stage and opts.from_stage not in STAGES[2:]:
            raise ConfigError(f"--from-stage must be one of {STAGES[2:]}")
        if opts.from_stage and not opts.run_id:
            raise ConfigError("--from-stage requires --run <run_id>")
        resume_at = STAGES.index(opts.from_stage) if opts.from_stage else 0
        ws.mkdir(parents=True, exist_ok=True)
        if resume_at > 0:
            _carry_forward(rep, ws / "run_report.json", STAGES[:resume_at])  # read before anything overwrites it
        if preflight:
            from codecast.preflight import fatal_problems
            problems = fatal_problems(cfg, require_key=resume_at <= 1, allow_network_tts=opts.allow_network_tts)
            if problems:
                raise ConfigError("preflight failed: " + "; ".join(problems))
        if engine is None:
            engine = select_engine(cfg.voice, allow_network=opts.allow_network_tts)
        rep.tts_engine = engine.name

        # [1] ingest
        current = "ingest"
        if resume_at <= 0:
            with rb.stage("ingest"):
                repomap = ingest(opts.repo, ws / "repo")
                _write_json(ws / "repomap.json", repomap.model_dump_json(indent=1))
        else:
            repomap = RepoMap.model_validate_json(_load_or_fail(ws / "repomap.json", "repomap.json", "ingest"))
        rep.repo_url, rep.commit_sha, rep.ingest.n_files = repomap.repo_url, repomap.commit_sha, repomap.n_files
        rep.ingest.skipped_by_reason = dict(Counter(s.reason for s in repomap.skipped))
        name = repo_name(repomap.repo_url)

        with renderer_factory() as renderer:
            # [2] agent
            current = "agent"
            agent = submitter = None
            if resume_at <= 1:
                with rb.stage("agent"):
                    if provider is None:
                        provider = create_provider(cfg.llm)
                    rep.llm.model = provider.model
                    submitter = Submitter(ValidationContext(
                        repomap=repomap, llm={"provider": provider.name, "model": provider.model},
                        output_allowlist=cfg.render.terminal_output_allowlist, mermaid_check=safe_mermaid_check(renderer.mermaid_check)))
                    max_steps = opts.max_steps or cfg.agent.max_steps or default_max_steps(repomap.n_files)
                    agent = DirectorAgent(provider, ToolBox(repomap, submitter), max_steps=max_steps,
                                          max_total_tokens=cfg.agent.max_total_tokens,
                                          wall_clock_s=cfg.agent.wall_clock_s, revision_steps=cfg.agent.revision_steps,
                                          temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens,
                                          timeout_s=cfg.llm.timeout_s, transcript=Transcript(ws / "transcript.jsonl"))
                    try:
                        sb = agent.run(initial_messages(repomap, load_director_prompt()))
                    finally:
                        _agent_stats(rep, agent, submitter, cfg)
                    _write_json(ws / "storyboard.json", sb.model_dump_json(indent=1))
            else:
                sb = Storyboard.model_validate_json(_load_or_fail(ws / "storyboard.json", "storyboard.json", "agent"))

            # [3] voice (+ duration loop)
            current = "voice"
            if resume_at <= 2:
                with rb.stage("voice"):
                    round_no = 0
                    timeline = synthesize_storyboard(sb, engine, ws / "audio" / f"round{round_no}",
                                                     gap_s=cfg.voice.gap_s, tail_s=cfg.voice.tail_s)
                    while not lo <= timeline.total <= hi:
                        if agent is None or round_no >= MAX_DURATION_ROUNDS:
                            raise DurationError(f"duration {timeline.total:.1f}s outside [{lo:.0f}, {hi:.0f}] "
                                                f"after {round_no} revision rounds")
                        round_no += 1
                        rep.validation.duration_rounds = round_no
                        feedback = duration_feedback(timeline, sb, target=(lo, hi))
                        agent.transcript.log_event(agent.stats.steps_used, "duration_feedback", json.loads(feedback))
                        submitter.begin_revision(sb, target_words=json.loads(feedback).get("target_words"))
                        current = "agent"
                        with rb.stage("agent"):
                            try:
                                sb = agent.revise(feedback)
                            finally:
                                _agent_stats(rep, agent, submitter, cfg)
                        current = "voice"
                        _write_json(ws / "storyboard.json", sb.model_dump_json(indent=1))
                        timeline = synthesize_storyboard(sb, engine, ws / "audio" / f"round{round_no}",
                                                         gap_s=cfg.voice.gap_s, tail_s=cfg.voice.tail_s)
                    _write_json(ws / "timeline.json", timeline.model_dump_json(indent=1))
            else:
                timeline = Timeline.model_validate_json(_load_or_fail(ws / "timeline.json", "timeline.json", "voice"))

            # [4] render
            current = "render"
            if resume_at <= 3:
                with rb.stage("render"):
                    frames = render_all(sb, timeline, Path(repomap.root), ws / "frames", renderer)
                    _write_json(ws / "frames.json", json.dumps([f.model_dump() for f in frames], indent=1))
            else:
                frames = [SceneFrames.model_validate(f) for f in json.loads(_load_or_fail(ws / "frames.json", "frames.json", "render"))]

        # [5] compose
        current = "compose"
        with rb.stage("compose"):
            staging = out_root / name / f".staging-{run_id}"
            staging.mkdir(parents=True, exist_ok=True)
            result = compose_video(frames, timeline, staging / "explainer.mp4", ws / "compose", lo=lo, hi=hi)
            write_srt(timeline, staging / "explainer.srt")
            _write_json(staging / "storyboard.json", sb.model_dump_json(indent=1))
        rep.video = VideoInfo(path=str(out_root / name / "explainer.mp4"), duration_s=result["duration_s"],
                              size_bytes=result["size_bytes"], narration_s=round(timeline.narration_seconds, 3))
        final = rb.finish()
        _write_json(staging / "run_report.json", final.model_dump_json(indent=1))
        for f in ("explainer.mp4", "explainer.srt", "storyboard.json", "run_report.json"):
            (staging / f).replace(out_root / name / f)
        shutil.rmtree(staging, ignore_errors=True)
        _write_json(ws / "run_report.json", final.model_dump_json(indent=1))
        return final
    except BaseException as e:  # noqa: BLE001 - every failure becomes a typed report
        if isinstance(e, KeyboardInterrupt):
            rb.fail(CodecastError("interrupted by user", stage=current), stage=current)
        else:
            rb.fail(e, stage=current)
        final = rb.finish()
        text = final.model_dump_json(indent=1)
        for target in (ws / "run_report.json", out_root / name / "last_failed_run_report.json"):
            try:
                _write_json(target, text)
            except OSError:
                pass
        shutil.rmtree(out_root / name / f".staging-{run_id}", ignore_errors=True)
        return final


def _carry_forward(rep: RunReport, path: Path, kept_stages: tuple[str, ...]) -> None:
    """Resume: keep the earlier run's agent accounting and timings for the stages that are not re-run."""
    try:
        prior = RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    rep.llm, rep.budget, rep.usage = prior.llm, prior.budget, prior.usage
    rep.cost_usd, rep.cost_unknown_reason, rep.context = prior.cost_usd, prior.cost_unknown_reason, prior.context
    rep.validation, rep.ingest = prior.validation, prior.ingest
    rep.repo_url, rep.commit_sha = prior.repo_url, prior.commit_sha
    rep.stage_seconds.update({k: v for k, v in prior.stage_seconds.items() if k in kept_stages})


def _agent_stats(rep, agent: DirectorAgent, submitter: Submitter, cfg: Config) -> None:
    st = agent.stats
    rep.budget.max_steps, rep.budget.steps_used = st.max_steps, st.steps_used
    rep.budget.tool_calls = dict(st.tool_calls)
    rep.usage.input, rep.usage.output = st.usage.input_tokens, st.usage.output_tokens
    rep.usage.cache_read_input, rep.usage.cache_write_input = st.usage.cache_read_input_tokens, st.usage.cache_write_input_tokens
    rep.context = st.context
    rep.validation.storyboard_rounds = submitter.rounds_total
    rep.validation.failures_by_code = dict(submitter.failures_by_code)
    rep.cost_usd, rep.cost_unknown_reason = estimate_cost(agent.provider.model, st.usage, cfg.pricing)
