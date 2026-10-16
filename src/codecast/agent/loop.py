"""Director agent loop (spec §5.1)."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from importlib import resources
from typing import Callable

from codecast.agent.tools import ToolBox
from codecast.agent.transcript import Transcript
from codecast.errors import BudgetExceeded, CodecastError, ConfigError
from codecast.llm.base import LLMProvider, complete_with_retries
from codecast.llm.errors import AuthError, ContextLengthError, ProviderError, RetryableError
from codecast.llm.types import ChatRequest, ChatResponse, Message, TextPart, ToolResultPart, Usage
from codecast.models import RepoMap, Storyboard

NUDGE = ("You did not call a tool. Finish now by calling submit_storyboard with your best fully grounded "
         "storyboard.")
MAX_NUDGES = 2


def default_max_steps(n_files: int) -> int:
    return 30 + min(30, n_files // 50)


def load_director_prompt() -> str:
    return resources.files("codecast.agent").joinpath("prompts/director.md").read_text(encoding="utf-8")


def initial_messages(repomap: RepoMap, system_prompt: str) -> list[Message]:
    user = (f"Repository: {repomap.repo_url}\nCommit: {repomap.commit_sha}\n\n"
            f"File tree (depth-limited, with sizes):\n{repomap.tree_text}\n\n"
            f"Explore with the tools, then call submit_storyboard. "
            f"The outro repo_url must be exactly {repomap.repo_url}.")
    return [Message("system", [TextPart(system_prompt)]), Message("user", [TextPart(user)])]


@dataclass
class AgentStats:
    max_steps: int
    steps_used: int = 0
    nudges: int = 0
    tool_calls: Counter = field(default_factory=Counter)
    usage: Usage = field(default_factory=Usage)
    context: dict | None = None


class DirectorAgent:
    def __init__(self, provider: LLMProvider, toolbox: ToolBox, *, max_steps: int, max_total_tokens: int,
                 wall_clock_s: float, revision_steps: int = 15, temperature: float = 0.3,
                 max_tokens: int = 4096, timeout_s: float = 120, transcript: Transcript | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        if not provider.caps.tool_use:
            raise ConfigError(f"{provider.name} model {provider.model} does not support tool use")
        self.provider = provider
        self.toolbox = toolbox
        self.tools = toolbox.specs(vision=provider.caps.vision)
        self.max_total_tokens = max_total_tokens
        self.wall_clock_s = wall_clock_s
        self.revision_steps = revision_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.transcript = transcript
        self.clock = clock
        self.sleep = sleep
        self.stats = AgentStats(max_steps=max_steps)
        self.messages: list[Message] = []
        self._logged = 0
        self._t0 = 0.0
        self._last_usage = Usage()

    def run(self, messages: list[Message]) -> Storyboard:
        self.messages = list(messages)
        self._logged = 0
        self._t0 = self.clock()
        return self._loop(self.stats.max_steps)

    def revise(self, feedback: str) -> Storyboard:
        self.messages.append(Message("user", [TextPart(feedback)]))
        self._t0 = self.clock()
        return self._loop(self.revision_steps)

    def _check_budget(self) -> None:
        used = self.stats.usage.total_input + self.stats.usage.output_tokens
        if used > self.max_total_tokens:
            raise BudgetExceeded(f"token budget exhausted ({used} > {self.max_total_tokens})")
        if self.clock() - self._t0 > self.wall_clock_s:
            raise BudgetExceeded(f"wall clock limit of {self.wall_clock_s:.0f}s exhausted")

    def _last_turn_outputs(self) -> list[dict]:
        names: dict[str, str] = {}
        for m in reversed(self.messages):
            if m.role == "assistant":
                names = {c.id: c.name for c in m.tool_calls()}
                break
        for m in reversed(self.messages):
            if m.role == "tool":
                out = []
                for r in m.parts:
                    if isinstance(r, ToolResultPart):
                        size = sum(len(q.text.encode()) if isinstance(q, TextPart) else len(q.data) for q in r.parts)
                        out.append({"tool": names.get(r.call_id, "?"), "call_id": r.call_id, "bytes": size})
                return out
        return []

    def _complete(self, req: ChatRequest) -> ChatResponse:
        try:
            return complete_with_retries(self.provider, req, sleep=self.sleep)
        except ContextLengthError as e:
            self.stats.context = {"window": self.provider.caps.context_window,
                                  "input_tokens_at_failure": self._last_usage.total_input,
                                  "last_turn_tool_outputs": self._last_turn_outputs()}
            raise BudgetExceeded(f"context window exhausted: {e}", details=self.stats.context) from e
        except AuthError as e:
            raise ConfigError(f"LLM authentication failed: {e}") from e
        except RetryableError as e:
            raise CodecastError(f"LLM provider unavailable after retries: {e}", stage="agent") from e
        except ProviderError as e:
            raise CodecastError(f"LLM provider error: {e}", stage="agent") from e

    def _loop(self, steps: int) -> Storyboard:
        nudges = 0
        choice: str | dict = "auto"
        for _ in range(steps):
            self._check_budget()
            self.stats.steps_used += 1
            step = self.stats.steps_used
            req = ChatRequest(messages=list(self.messages), tools=self.tools, tool_choice=choice,
                              max_tokens=self.max_tokens, temperature=self.temperature, timeout_s=self.timeout_s)
            if self.transcript:
                self.transcript.log_request(step, req, self.messages[self._logged:])
                self._logged = len(self.messages)
            resp = self._complete(req)
            self._last_usage = resp.usage
            self.stats.usage = self.stats.usage + resp.usage
            if self.transcript:
                self.transcript.log_response(step, resp)
            self.messages.append(resp.message)
            self._logged = len(self.messages)
            calls = resp.message.tool_calls()
            if not calls:
                nudges += 1
                self.stats.nudges += 1
                if nudges > MAX_NUDGES:
                    raise BudgetExceeded(f"model stopped without calling submit_storyboard after {MAX_NUDGES} nudges")
                self.messages.append(Message("user", [TextPart(NUDGE)]))
                choice = {"name": "submit_storyboard"}
                continue
            choice = "auto"
            results, accepted = [], None
            for call in calls:
                self.stats.tool_calls[call.name] += 1
                ex = self.toolbox.execute(call)
                results.append(ex.result)
                if ex.storyboard is not None:
                    accepted = ex.storyboard
            self.messages.append(Message("tool", results))
            if accepted is not None:
                if self.transcript:
                    self.transcript.log_messages(step, self.messages[self._logged:])
                    self._logged = len(self.messages)
                return accepted
        raise BudgetExceeded(f"step budget of {steps} exhausted without an accepted storyboard")
