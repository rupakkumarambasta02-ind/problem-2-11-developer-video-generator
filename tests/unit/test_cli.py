from codecast import cli


def test_doctor_exit_codes(monkeypatch, capsys):
    from codecast.preflight import Check
    monkeypatch.setattr(cli, "run_checks", lambda cfg, require_key, allow_network_tts=False:
                        [Check("ffmpeg", True, "ok", None, True), Check("llm api key", False, "unset", "export it", False)])
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "WARN" in out and "llm api key" in out
    monkeypatch.setattr(cli, "run_checks", lambda cfg, require_key, allow_network_tts=False:
                        [Check("ffmpeg", False, "missing", "brew install ffmpeg", True)])
    assert cli.main(["doctor"]) == 7


def test_run_returns_report_exit_code(monkeypatch, tmp_path):
    from codecast.report import ReportBuilder
    from codecast.config import load_config
    from codecast.errors import IngestError

    def fake_run(opts, cfg, **kw):
        rb = ReportBuilder("r", opts.repo, cfg)
        rb.fail(IngestError("nope"), stage="ingest")
        return rb.finish()
    monkeypatch.setattr(cli, "pipeline_run", fake_run)
    assert cli.main(["run", "--repo", "x", "--out", str(tmp_path)]) == 2


def test_bad_config_is_exit_7(tmp_path):
    assert cli.main(["run", "--repo", "x", "--out", str(tmp_path), "--config", str(tmp_path / "nope.yaml")]) == 7


def test_from_stage_requires_run_id(tmp_path):
    assert cli.main(["run", "--repo", "x", "--out", str(tmp_path), "--from-stage", "render",
                     "--workspace", str(tmp_path / "ws")]) == 7
