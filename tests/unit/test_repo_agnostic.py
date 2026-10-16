from pathlib import Path

FORBIDDEN = ("video-classifier", "video_classifier", "eval_results", "production.json", "pareto")


def test_no_demo_specific_strings_in_package():
    root = Path(__file__).resolve().parents[2] / "src" / "codecast"
    offenders = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".html", ".yaml", ".css"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            offenders += [f"{p.relative_to(root)}: {w}" for w in FORBIDDEN if w in text]
    assert offenders == []
