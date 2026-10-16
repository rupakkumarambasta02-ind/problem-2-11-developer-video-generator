# CodeCast: GitHub repo → grounded 60-second explainer video

**Result:** `codecast run --repo <url>` turns a repository into a 1920×1080 explainer (MP4 + SRT + storyboard + run report) whose every number and identifier is checked against the repo's own files. The committed demo, [`output/video-classifier/explainer.mp4`](output/video-classifier/explainer.mp4) for [rkumar-bengaluru/video-classifier](https://github.com/rkumar-bengaluru/video-classifier), is **57.5 s, 0 ungrounded claims, 100% caption coverage, $0.051** of LLM spend (gpt-5.4-mini, 8 agent steps, 3 validation rounds, 1 duration revision). Numbers come from `output/video-classifier/run_report.json` and [`docs/eval.md`](docs/eval.md).

## 1. Problem and approach

Docs are walls of text; a new engineer needs half an hour to absorb a README, a `run.yaml` and a results JSON. Hand-made screen recordings go stale and are not tied to the code. CodeCast generates the explainer automatically, and **refuses to say anything the repo does not literally contain.**

| Stage | What happens | Key choice |
|---|---|---|
| 1 Ingest | `git clone --depth 1` at HEAD → sandboxed file map (skip rules, 1 MB / 1 GB caps) | citations pin the commit SHA |
| 2 Director | LLM agent with 5 read-only tools (`list_dir`, `read_file`, `grep`, `read_json`, `view_image`) explores the repo and calls `submit_storyboard` | agent decides, **code checks** |
| 3 Voice | local TTS per sentence (piper, fallback macOS `say`) → measured, frame-aligned timeline | measured audio drives timing |
| 4 Render | Playwright renders 8 scene templates + caption PNGs (fonts and mermaid.js vendored, no network) | captions as PNG overlays (Homebrew ffmpeg has no libass) |
| 5 Compose | ffmpeg segments + overlays + `loudnorm` audio → ffprobe post-check (50–80 s, A/V drift < 0.1 s) | fail loudly with typed exit codes 0–7 |

The LLM sits behind a provider-neutral adapter (Anthropic, OpenAI and OpenAI-compatible endpoints, plus a Fake provider that replays transcripts for offline tests). The agent never writes HTML or ffmpeg commands; it picks scenes from a fixed catalog: title, bullets, code, image (with highlights / Ken Burns), metrics table, mermaid diagram, terminal, outro.

## 2. Grounding: the core idea

Every sentence and every on-screen label must cite sources: file + line range, JSON Pointer, or image. The validator (pure code, never the LLM) extracts each **number** (`-?\d+(,\d{3})*(\.\d+)?%?`, number words included: "six" → 6) and each **identifier** (any token with a digit: `large@8`, `ViT-B-32`) and accepts it only if it appears in a cited source.

- Numbers are compared as `Decimal` from the source's literal text. Rounding is allowed (0.7349 → 0.735 / 0.73 / 73.5%) only if ≥ 2 significant figures survive. **Derived numbers (sums, counts, differences) are rejected**, so the agent must find a span that literally contains them.
- Identifiers match case-sensitively as whole tokens. Spans are cited by reference; the tool fills in the exact file text, so a citation cannot drift from the file.
- Errors go back to the agent as tool results (`UNGROUNDED_NUMBER: 0.74 not found in cited sources [s3]`); it gets 4 rounds, then exit 5. `codecast eval` re-runs the same rules against a **fresh clone** at the recorded SHA.

## 3. Results

| Metric (demo) | Target | Value |
|---|---|---|
| Ungrounded numbers / identifiers (independent re-check) | 0 | **0** |
| Duration (ffprobe) | 50–80 s | **57.5 s** |
| Caption coverage of narration | ≥ 95% | **100%** |
| Cost per video | < $0.10 | **$0.051** (32,946 input + 112,128 cached + 3,960 output tokens) |
| End-to-end time | < 5 min | not measured for this run (see §5) |
| Human clarity (1–5) | > 4 | not yet rated |

The 7-scene, 118-word storyboard cites 17 sources: README lines, `artifacts/eval_results.json` pointers, the Pareto plot, `src/predict.py` and the `run.yaml` commands. **Reliability across 10 live runs** (gpt-5.4-mini, $1.02 total):

| Runs | Outcome | What changed next |
|---|---|---|
| 1–6 | 0 videos: 5 × exit 5 (validation), 1 × exit 3 (duration) | the model copied cited lines inexactly and overshot the word budget → **spans by reference**, validated worked example, word target |
| 7–10 | 1 video, 2 × exit 3, 1 × exit 5 | the video run hit a real compose bug (relative concat paths), fixed and finished by `--from-stage` resume |

The strict validator worked as intended: no run produced a video with an invented number. The cost is that a small model needs several rounds, and about 1 run in 4 succeeds; the main remaining failure is duration (piper speaks number-heavy sentences at about 1.1 words/s versus 3 for prose, and the model shortens too little).

## 4. Engineering notes

- **Safety:** tools are read-only and confined by `realpath` to the clone; repo text is treated as untrusted data; ffmpeg runs from argv lists with filter graphs in files (`-/filter_complex`), never a shell.
- **Reproducibility:** every stage persists to `workspace/<run_id>/` (repo map, `transcript.jsonl` with image sidecars, storyboard, timeline, frames); `--from-stage voice|render|compose --run <id>` resumes and carries the agent's accounting forward. Outputs are staged and swapped in only after the post-check, so a failed run never replaces the committed demo.
- **Tests:** 364 offline tests (`pytest -q`, no API key): table-driven grounding rules, sandbox escapes, adapters via mocked HTTP, the agent loop via the Fake provider, real Chromium renders, real ffmpeg compose, and offline end-to-end runs of the whole pipeline. `pytest -m live` adds real-LLM runs (demo repo plus a smoke repo without metrics).
- **Cost control:** the step budget is `30 + min(30, n_files // 50)`, prompt caching (75–90% of input tokens were cache hits), and pricing comes from the provider's official page; an unknown model reports `cost_usd: null` with a reason rather than a guess.

## 5. Limitations and how to run

**Limitations.** One demo repo and 10 live runs is a small sample; about 1 in 4 runs succeeds with gpt-5.4-mini, mostly failing the 50–60 s window it had then (now widened to 50–80 s). The demo was completed by an offline resume after the compose bug fix, so its ingest, agent and voice timings were not recorded and end-to-end time is unmeasured; its agent cost was reconstructed from the transcript. On screen the metrics table shows per-run JSON values (0.741 / 0.842) while the narration quotes the README averages (0.735 / 0.843): both are grounded, but they differ. The plot highlight sits over an empty area. Terminal scenes show real commands from cited lines, never executed output. Output is 16:9 only; GitHub Action, private repos, 9:16 and multi-language narration are Phase B. Design spec: [`docs/superpowers/specs/2026-10-15-codecast-design.md`](docs/superpowers/specs/2026-10-15-codecast-design.md).

**Run** (from `run.yaml`; macOS, Python 3.12+, `brew install ffmpeg`, `OPENAI_API_KEY`; or omit `--config` to use the default Anthropic model with `ANTHROPIC_API_KEY`):

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m playwright install chromium
.venv/bin/codecast fetch-voice && .venv/bin/codecast doctor
.venv/bin/codecast run --repo https://github.com/rkumar-bengaluru/video-classifier --out output/ --config config/openai.yaml
.venv/bin/codecast eval output/video-classifier/
.venv/bin/python -m pytest -q
```
