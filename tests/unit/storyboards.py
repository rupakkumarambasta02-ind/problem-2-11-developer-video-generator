"""A grounded 6-scene submission against the tiny_repo fixture (130 spoken words)."""

import copy


def valid_submission() -> dict:
    return copy.deepcopy({
        "title": "Tiny in a minute",
        "sources": [
            {"kind": "text", "id": "readme", "file": "README.md", "start_line": 1, "end_line": 3,
             "text": "# Tiny\nTiny classifies clips with ViT-B-32 at 0.7349 F1.\n"
                     "Run `python app.py --top 3` to predict."},
            {"kind": "text", "id": "code", "file": "src/app.py", "start_line": 1, "end_line": 2,
             "text": 'def main():\n    print("hello")'},
            {"kind": "json", "id": "models", "file": "metrics.json", "pointer": "/models",
             "value": {"small": {"f1": 0.677, "latency_ms": 420},
                       "large@8": {"f1": 0.7349, "latency_ms": 891.2}}},
            {"kind": "json", "id": "f1", "file": "metrics.json", "pointer": "/models/large@8/f1",
             "value": 0.7349},
            {"kind": "image", "id": "plot", "file": "plot.png"},
        ],
        "scenes": [
            {"type": "title", "heading": "Tiny", "subheading": "Sorting clips, simply",
             "narration": [
                 {"text": "Tiny is a small project that sorts short video clips into a handful of "
                          "everyday categories for you.", "source_ids": ["readme"]},
                 {"text": "It keeps the whole pipeline simple, readable and easy to run on an "
                          "ordinary laptop.", "source_ids": ["readme"]}]},
            {"type": "bullets", "heading": "How it works",
             "items": ["Reads a clip", "Scores each frame", "Picks the best label"],
             "narration": [
                 {"text": "First it reads a clip, then it scores every frame, and finally it picks "
                          "the most likely label.", "source_ids": ["readme"]},
                 {"text": "Each step lives in its own small function, so the flow is easy to follow.",
                  "source_ids": ["readme"]}]},
            {"type": "code", "span_id": "code", "highlight_lines": [2],
             "narration": [{"text": "The entry point is a tiny main function that prints a friendly "
                                    "greeting.", "source_ids": ["code"]}]},
            {"type": "metrics_table", "columns": ["Model", "F1"],
             "rows": [["large@8", {"span_id": "f1", "decimals": 3}]],
             "narration": [{"text": "The large@8 model reaches an F1 score of 0.735 on the held out set.",
                            "source_ids": ["models", "readme"]}]},
            {"type": "image", "image_id": "plot", "highlights": [[0.1, 0.1, 0.3, 0.3]],
             "motion": {"kind": "static"},
             "narration": [{"text": "This plot compares the models side by side, so you can see the "
                                    "trade off at a glance.", "source_ids": ["plot"]}]},
            {"type": "outro", "repo_url": "https://github.com/o/tiny", "cta": "Star the repo and try it",
             "narration": [{"text": "That is the whole tour, so clone the repository and give it a try "
                                    "on your own clips today.", "source_ids": ["readme"]}]},
        ],
    })
