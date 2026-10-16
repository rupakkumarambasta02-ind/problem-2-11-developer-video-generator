"""Typed failures. Every CodecastError maps to one process exit code (spec §6.5)."""

from __future__ import annotations

from typing import Any


class CodecastError(Exception):
    exit_code: int = 1
    default_stage: str = "unknown"

    def __init__(self, message: str, *, stage: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.stage = stage or self.default_stage
        self.details = details or {}


class IngestError(CodecastError):
    exit_code = 2
    default_stage = "ingest"


class DurationError(CodecastError):
    exit_code = 3
    default_stage = "voice"


class BudgetExceeded(CodecastError):
    exit_code = 4
    default_stage = "agent"


class ValidationFailed(CodecastError):
    exit_code = 5
    default_stage = "agent"


class RenderError(CodecastError):
    exit_code = 6
    default_stage = "render"


class ConfigError(CodecastError):
    exit_code = 7
    default_stage = "preflight"
