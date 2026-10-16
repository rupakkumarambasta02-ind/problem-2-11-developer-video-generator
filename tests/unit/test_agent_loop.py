import pytest

from codecast.agent.loop import DirectorAgent, default_max_steps, initial_messages, load_director_prompt
from codecast.agent.submit import Submitter
from codecast.agent.tools import ToolBox
from codecast.agent.transcript import Transcript, load_responses
from codecast.errors import BudgetExceeded, ConfigError, ValidationFailed
from codecast.grounding.validator import ValidationContext
from codecast.llm.errors import AuthError, ContextLengthError
from codecast.llm.fake import FAKE_CAPS, FakeProvider, say, tool_call
from codecast.llm.types import ProviderCaps, ToolResultPart, Usage

from tests.unit.storyboards import valid_submission


def bad():
    raw = valid_submission()
    raw["scenes"][3]["narration"][0]["text"] = "The large@8 model reaches an F1 score of 0.74 overall."
    return raw


def make(repo, script, caps=FAKE_CAPS, **kw):
    provider = FakeProvider(script, caps=caps)
    submitter = Submitter(ValidationContext(repomap=repo, llm={"provider": "fake", "model": "m"}, output_allowlist=["*.log"]))
    agent = DirectorAgent(provider, ToolBox(repo, submitter), max_steps=kw.pop("max_steps", 10),
                          max_total_tokens=kw.pop("max_total_tokens", 10_000_000), wall_clock_s=kw.pop("wall_clock_s", 600),
                          sleep=lambda s: None, **kw)
    return agent, provider, submitter


def msgs(repo):
    return initial_messages(repo, "SYSTEM")


def test_default_max_steps():
    assert default_max_steps(10) == 30
    assert default_max_steps(500) == 40
    assert default_max_steps(10_000) == 60


def test_prompt_mentions_catalog_and_rules():
    p = load_director_prompt()
    for word in ("submit_storyboard", "metrics_table", "ken_burns", "110", "200", "50–80", "letters only"):
        assert word in p


def test_prompt_cites_by_reference_and_forbids_spelling():
    p = load_director_prompt()
    assert "omit `text`" in p and "omit `value`" in p and "spell" in p


def test_happy_path(tiny_repo):
    agent, provider, _ = make(tiny_repo, [
        tool_call("list_dir", {"path": "."}, "a"),
        tool_call("read_file", {"path": "README.md"}, "b"),
        tool_call("submit_storyboard", {"storyboard": valid_submission()}, "c"),
    ])
    sb = agent.run(msgs(tiny_repo))
    assert sb.title == "Tiny in a minute"
    assert agent.stats.steps_used == 3
    assert agent.stats.tool_calls == {"list_dir": 1, "read_file": 1, "submit_storyboard": 1}
    assert provider.requests[0].messages[1].text().startswith("Repository: https://github.com/o/tiny")


def test_self_correction(tiny_repo):
    agent, provider, sub = make(tiny_repo, [
        tool_call("submit_storyboard", {"storyboard": bad()}, "a"),
        tool_call("submit_storyboard", {"storyboard": valid_submission()}, "b"),
    ])
    agent.run(msgs(tiny_repo))
    last = provider.requests[1].messages[-1]
    assert last.role == "tool"
    res = last.parts[0]
    assert isinstance(res, ToolResultPart) and res.is_error and "UNGROUNDED_NUMBER" in res.parts[0].text
    assert sub.rounds_total == 2


def test_nudge_forces_submit(tiny_repo):
    agent, provider, _ = make(tiny_repo, [say("Let me think."),
                                          tool_call("submit_storyboard", {"storyboard": valid_submission()})])
    agent.run(msgs(tiny_repo))
    assert provider.requests[0].tool_choice == "auto"
    assert provider.requests[1].tool_choice == {"name": "submit_storyboard"}
    assert agent.stats.nudges == 1


def test_three_text_turns_exhaust_nudges(tiny_repo):
    agent, _, _ = make(tiny_repo, [say("a"), say("b"), say("c")])
    with pytest.raises(BudgetExceeded, match="nudge"):
        agent.run(msgs(tiny_repo))


def test_step_budget(tiny_repo):
    agent, _, _ = make(tiny_repo, [tool_call("list_dir", {"path": "."})] * 5, max_steps=2)
    with pytest.raises(BudgetExceeded, match="step budget"):
        agent.run(msgs(tiny_repo))


def test_token_budget(tiny_repo):
    big = Usage(9_000, 1_000, 0, 0)
    agent, _, _ = make(tiny_repo, [tool_call("list_dir", {"path": "."}, usage=big)] * 3, max_total_tokens=15_000)
    with pytest.raises(BudgetExceeded, match="token"):
        agent.run(msgs(tiny_repo))


def test_wall_clock(tiny_repo):
    ticks = iter([0, 0, 1000, 1000])
    agent, _, _ = make(tiny_repo, [tool_call("list_dir", {"path": "."})] * 3, wall_clock_s=10,
                       clock=lambda: next(ticks))
    with pytest.raises(BudgetExceeded, match="wall"):
        agent.run(msgs(tiny_repo))


def test_validation_exhaustion(tiny_repo):
    agent, _, _ = make(tiny_repo, [tool_call("submit_storyboard", {"storyboard": bad()})] * 4)
    with pytest.raises(ValidationFailed):
        agent.run(msgs(tiny_repo))


def test_vision_filtering(tiny_repo):
    no_vision = ProviderCaps(False, True, False, True, 100_000)
    agent, provider, _ = make(tiny_repo, [tool_call("submit_storyboard", {"storyboard": valid_submission()})], caps=no_vision)
    agent.run(msgs(tiny_repo))
    assert "view_image" not in [t.name for t in provider.requests[0].tools]


def test_no_tool_use_is_config_error(tiny_repo):
    with pytest.raises(ConfigError):
        make(tiny_repo, [], caps=ProviderCaps(True, False, False, True, 100_000))


def test_context_overflow_records_context(tiny_repo):
    agent, _, _ = make(tiny_repo, [tool_call("read_file", {"path": "README.md"}, "r1", usage=Usage(150_000, 10, 0, 0)),
                                   ContextLengthError("too long")])
    with pytest.raises(BudgetExceeded, match="context"):
        agent.run(msgs(tiny_repo))
    ctx = agent.stats.context
    assert ctx["window"] == 200_000 and ctx["input_tokens_at_failure"] == 150_000
    assert ctx["last_turn_tool_outputs"][0]["tool"] == "read_file" and ctx["last_turn_tool_outputs"][0]["bytes"] > 0


def test_auth_error_is_config_error(tiny_repo):
    agent, _, _ = make(tiny_repo, [AuthError("bad key")])
    with pytest.raises(ConfigError):
        agent.run(msgs(tiny_repo))


def test_revise_continues_conversation(tiny_repo, tmp_path):
    t = Transcript(tmp_path / "transcript.jsonl")
    agent, provider, sub = make(tiny_repo, [
        tool_call("submit_storyboard", {"storyboard": valid_submission()}, "a"),
        tool_call("submit_storyboard", {"storyboard": valid_submission()}, "b"),
    ], transcript=t)
    agent.run(msgs(tiny_repo))
    sub.begin_revision()
    sb = agent.revise('{"total": 63.2, "target": [50, 60]}')
    assert sb is not None
    assert provider.requests[1].messages[-1].text().startswith('{"total": 63.2')
    assert len(load_responses(tmp_path / "transcript.jsonl")) == 2


def test_worked_example_is_valid_and_grounded():
    import json
    import re
    from codecast.grounding.numbers import extract_claims, identifier_grounded, number_grounded, source_number_literals
    prompt = load_director_prompt()
    block = re.search(r"# Worked example.*?```json\n(.*?)\n```", prompt, re.S).group(1)
    sb = json.loads(block)
    texts = {"readme": "tinyq is a tiny job queue.\nRun it with `tinyq serve --port 8080`.\n",
             "core": "def run(job):\n    return handler(job)", "p50": "12.5"}
    everything = list(texts.values())
    words = 0
    for sc in sb["scenes"]:
        for sent in sc["narration"]:
            words += len(sent["text"].split())
            cited = [texts[i] for i in sent["source_ids"]]
            lits = [x for t in cited for x in source_number_literals(t)]
            for c in extract_claims(sent["text"]):
                ok = number_grounded(c.token, lits) if c.kind == "number" else identifier_grounded(c.token, cited)
                assert ok, (c, sent["text"])
        shown = [sc[k] for k in ("heading", "subheading", "cta") if k in sc] + sc.get("items", [])
        for text in shown:
            for c in extract_claims(text):
                assert identifier_grounded(c.token, everything), (c, text)
    assert 120 <= words <= 140, words
