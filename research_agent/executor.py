"""Executor: runs ONE plan step as a bounded think → act → observe loop."""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Optional

from . import prompts
from .config import Settings
from .llm import LLM, LLMOutputError
from .models import AgentAction, PlanStep, StepResult, ToolCallRecord
from .tools import ToolRegistry, ToolResult
from .trace import Tracer


_URL = re.compile(r"https?://[^\s<>\"'`\]]+")


def urls_in(text: str) -> list[str]:
    """URLs written inline in a step's output, minus trailing punctuation and an unbalanced ')'."""
    urls = []
    for url in _URL.findall(text):
        url = url.rstrip(".,;:!?*")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1].rstrip(".,;:!?*")
        urls.append(url)
    return urls


def format_prior(results: list[StepResult], per_step_chars: int = 700) -> str:
    if not results:
        return "(none yet)"
    lines = []
    for r in results:
        lines.append(f"- Step {r.step_id} [{r.status}] {r.title}: {r.output[:per_step_chars]}")
        if r.sources:
            lines.append(f"  sources: {', '.join(r.sources[:6])}")
    return "\n".join(lines)


class Executor:
    def __init__(self, llm: LLM, registry: ToolRegistry, settings: Settings, tracer: Tracer):
        self.llm, self.registry, self.settings, self.tracer = llm, registry, settings, tracer
        self.seen_sources: set[str] = set()  # every URL any tool returned during the run
        self.current: Optional[StepResult] = None  # the attempt in progress, so an interrupted step keeps its calls

    def run_step(self, goal: str, step: PlanStep, total: int, prior: list[StepResult],
                 feedback: Optional[str] = None) -> StepResult:
        max_actions = self.settings.max_actions_per_step
        system = prompts.EXECUTOR_SYSTEM.format(today=dt.date.today().isoformat(),
                                                tools=self.registry.describe(), max_actions=max_actions)
        user = prompts.EXECUTOR_STEP.format(
            goal=goal, step_id=step.id, total=total, title=step.title, instruction=step.instruction,
            criteria=step.success_criteria, tool_hint=step.tool_hint,
            feedback=f"\nREVIEWER FEEDBACK ON YOUR PREVIOUS ATTEMPT (address it!): {feedback}\n" if feedback else "",
            prior=format_prior(prior))
        if feedback and step.tool_hint != "none":
            # on a retry the model tends to "remember" data it no longer has and finish without tools
            user += ("\nThe tool results from your previous attempt are NOT available now: call tools again "
                     "to gather the data. Finishing without a tool call will be rejected.")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        result = self.current = StepResult(step_id=step.id, title=step.title)
        called: set[str] = set()
        actions_used = 0

        while True:
            out_of_actions = actions_used >= max_actions
            if out_of_actions:
                self.tracer.recovery("forced_finish", limit=max_actions)
                messages.append({"role": "user", "content": "You have no tool calls left. Reply now with "
                                 "action 'finish', summarising what you found and what is missing."})
            try:
                action = self.llm.complete_json(messages, AgentAction, max_tokens=1600)
            except LLMOutputError as e:
                result.output = f"Step aborted: the model kept returning invalid actions ({e})."
                return result
            messages.append({"role": "assistant", "content": action.model_dump_json(exclude_none=True)})
            self.tracer.thought(action.thought)

            if action.action == "finish":
                result.output = action.output.strip()
                # URLs written next to each item count as sources, so the model needn't repeat them
                cited = [s for s in action.sources if isinstance(s, str)] + urls_in(result.output)
                result.sources = list(dict.fromkeys(cited))
                self.tracer.step_output(result.output, result.sources)
                return result

            if out_of_actions:  # model ignored the instruction: salvage what the tools returned
                result.output = self._salvage(result)
                return result

            actions_used += 1
            observation = self._act(action, called, result)
            messages.append({"role": "user", "content": f"OBSERVATION (tool call {actions_used}/{max_actions}):\n"
                                                        f"{observation}"})

    def _act(self, action: AgentAction, called: set[str], result: StepResult) -> str:
        key = f"{action.tool}:{json.dumps(action.args, sort_keys=True, default=str)}"
        if key in called:
            self.tracer.recovery("duplicate_call", tool=action.tool)
            return ("You already made exactly this call in this step; its result is above. "
                    "Use a different query/tool, or finish.")
        called.add(key)

        if self.registry.get(action.tool) is None:
            self.tracer.recovery("invalid_action", error=f"unknown tool {action.tool!r}")
        self.tracer.tool_call(action.tool, action.args)
        res: ToolResult = self.registry.execute(action.tool, action.args)
        self.tracer.tool_result(res)
        self.seen_sources.update(res.sources)
        result.tool_calls.append(ToolCallRecord(tool=res.tool, args=res.args, ok=res.ok, error=res.error,
                                                attempts=res.attempts, latency_ms=res.latency_ms,
                                                preview=res.text[:200]))
        if res.ok:
            limit = self.settings.max_observation_chars
            text = res.text if len(res.text) <= limit else res.text[:limit] + "\n...(truncated)"
            return f"{res.tool} OK:\n{text}"
        return (f"{res.tool} FAILED after {res.attempts} attempt(s): {res.error}\n"
                f"Change your approach: try another tool, a different query, or another source.")

    @staticmethod
    def _salvage(result: StepResult) -> str:
        ok = [c for c in result.tool_calls if c.ok]
        if not ok:
            return "Step incomplete: no tool call succeeded."
        return "Step incomplete (no summary produced). Raw tool results:\n" + "\n".join(c.preview for c in ok)
