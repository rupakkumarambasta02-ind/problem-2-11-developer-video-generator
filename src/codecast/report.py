"""run_report.json (spec §6.4)."""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from codecast.config import Config
from codecast.errors import CodecastError


class LLMInfo(BaseModel):
    provider: str
    model: str | None
    temperature: float


class Failure(BaseModel):
    stage: str
    code: str
    message: str
    traceback: str | None = None


class UsageInfo(BaseModel):
    input: int = 0
    output: int = 0
    cache_read_input: int = 0
    cache_write_input: int = 0


class BudgetInfo(BaseModel):
    max_steps: int = 0
    steps_used: int = 0
    tool_calls: dict[str, int] = Field(default_factory=dict)


class ValidationInfo(BaseModel):
    storyboard_rounds: int = 0
    duration_rounds: int = 0
    failures_by_code: dict[str, int] = Field(default_factory=dict)


class IngestInfo(BaseModel):
    n_files: int = 0
    skipped_by_reason: dict[str, int] = Field(default_factory=dict)


class VideoInfo(BaseModel):
    path: str
    duration_s: float
    size_bytes: int
    narration_s: float


class RunReport(BaseModel):
    run_id: str
    started_at: str
    ended_at: str | None = None
    repo_url: str
    commit_sha: str | None = None
    llm: LLMInfo
    status: str = "ok"
    exit_code: int = 0
    failure: Failure | None = None
    ingest: IngestInfo = Field(default_factory=IngestInfo)
    budget: BudgetInfo = Field(default_factory=BudgetInfo)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    cost_usd: float | None = None
    cost_unknown_reason: str | None = None
    context: dict[str, Any] | None = None
    validation: ValidationInfo = Field(default_factory=ValidationInfo)
    tts_engine: str | None = None
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    video: VideoInfo | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReportBuilder:
    def __init__(self, run_id: str, repo: str, cfg: Config):
        self.report = RunReport(run_id=run_id, started_at=_now(), repo_url=repo,
                                llm=LLMInfo(provider=cfg.llm.provider, model=cfg.llm.model,
                                            temperature=cfg.llm.temperature))

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.report.stage_seconds[name] = round(self.report.stage_seconds.get(name, 0) + time.monotonic() - t0, 3)

    def fail(self, exc: BaseException, stage: str) -> None:
        if isinstance(exc, CodecastError):
            code, message, st, tb = exc.exit_code, exc.message, exc.stage or stage, None
            if exc.details.get("issues"):
                message += f" (first issues: {exc.details['issues'][:5]})"
        else:
            code, message, st = 1, f"{type(exc).__name__}: {exc}", stage
            tb = "".join(traceback.format_exception(exc))
        self.report.status = "failed"
        self.report.exit_code = code
        self.report.failure = Failure(stage=st, code=type(exc).__name__, message=message, traceback=tb)

    def finish(self) -> RunReport:
        self.report.ended_at = _now()
        return self.report
