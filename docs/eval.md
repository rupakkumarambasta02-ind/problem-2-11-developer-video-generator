# CodeCast eval log

Newest entries first. Each entry is written by `codecast eval output/<repo>/`; fill in the clarity rating by hand after watching the video.

## 2026-10-15 16:49 UTC — https://github.com/rkumar-bengaluru/video-classifier @ 12863e2

LLM: openai / gpt-5.4-mini · TTS: piper · run `20261015T163800Z-988caa`

| Metric | Target | Value | OK |
|---|---|---|---|
| Ungrounded numbers / identifiers | 0 | 0 | ✓ |
| Other re-check issues | 0 | 0 | ✓ |
| Duration (ffprobe) | 50–80 s | 57.5 s | ✓ |
| Caption coverage of narration | ≥ 95% | 100% | ✓ |
| Cost per video | < $0.10 | $0.0509 | ✓ |
| End-to-end time | < 5 min | not measured (see note) | — |
| Human clarity rating (1–5) | > 4 | not yet rated | manual |

Notes:
- This run failed at compose (ffmpeg concat lists used relative paths; fixed in commit 3eb0461) and was completed with `codecast run --from-stage render --run 20261015T163800Z-988caa`. The resume recorded only render (1.5 s) and compose (7.5 s); the ingest, agent and voice timings were not recorded, so the eval tool's computed "0.1 min" is not an end-to-end time and is shown as not measured.
- Agent usage and cost were reconstructed from `workspace/20261015T163800Z-988caa/transcript.jsonl` (8 steps; 16 read_file, 1 view_image, 3 submit_storyboard calls) and priced at the official gpt-5.4-mini rates in `src/codecast/defaults.yaml`.
- Live-run history: 10 runs with gpt-5.4-mini, $1.02 total; this is the only run that produced a video. See README §3.

Clarity rating: not yet rated.
