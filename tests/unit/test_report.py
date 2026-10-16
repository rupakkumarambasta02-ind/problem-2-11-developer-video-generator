from pathlib import Path

from codecast.config import load_config
from codecast.errors import DurationError
from codecast.report import ReportBuilder


def test_report_records_failure_and_stages():
    rb = ReportBuilder("r1", "https://github.com/o/r", load_config())
    with rb.stage("ingest"):
        pass
    rb.fail(DurationError("63.2s after 2 rounds"), stage="voice")
    rep = rb.finish()
    assert rep.status == "failed" and rep.exit_code == 3
    assert rep.failure.stage == "voice" and rep.failure.code == "DurationError"
    assert "ingest" in rep.stage_seconds
    assert rep.llm.provider == "anthropic" and rep.llm.temperature == 0.3


def test_unhandled_exception_is_exit_1():
    rb = ReportBuilder("r1", "x", load_config())
    rb.fail(ZeroDivisionError("boom"), stage="render")
    rep = rb.finish()
    assert rep.exit_code == 1 and rep.failure.traceback


def test_ok_report_shape():
    rb = ReportBuilder("r1", "x", load_config())
    rep = rb.finish()
    d = rep.model_dump()
    for key in ("run_id", "started_at", "ended_at", "repo_url", "commit_sha", "llm", "status", "exit_code",
                "ingest", "budget", "usage", "cost_usd", "validation", "tts_engine", "stage_seconds", "video"):
        assert key in d
    assert rep.status == "ok" and rep.exit_code == 0


def test_safe_mermaid_check_swallows_browser_errors():
    from codecast.pipeline import safe_mermaid_check

    def boom(code):
        raise RuntimeError("browser died")

    assert safe_mermaid_check(boom)("graph TD; A-->B") is None
    assert safe_mermaid_check(lambda c: "bad syntax")("x") == "bad syntax"
