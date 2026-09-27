"""Planner: turns a goal into an ordered, checkable plan, and re-plans after failures."""
from __future__ import annotations

import datetime as dt

from . import prompts
from .config import Settings
from .llm import LLM, LLMOutputError
from .models import Plan, PlanStep, Replan, StepResult
from .tools import ToolRegistry
from .trace import Tracer

MIN_STEPS = 2
MAX_REPLAN_STEPS = 3


class Planner:
    def __init__(self, llm: LLM, registry: ToolRegistry, settings: Settings, tracer: Tracer):
        self.llm, self.registry, self.settings, self.tracer = llm, registry, settings, tracer

    def _fmt(self, template: str, **extra) -> str:
        return template.format(today=dt.date.today().isoformat(), tools=self.registry.describe(),
                               tool_names=", ".join(self.registry.names()),
                               min_steps=MIN_STEPS, max_steps=self.settings.max_plan_steps, **extra)

    def _check_steps(self, steps: list[PlanStep], min_steps: int = 1) -> None:
        valid = set(self.registry.names()) | {"none"}
        bad = [s.tool_hint for s in steps if s.tool_hint not in valid]
        if bad:
            raise ValueError(f"unknown tool_hint {bad}; use one of {sorted(valid)}")
        if not min_steps <= len(steps) <= self.settings.max_plan_steps:
            raise ValueError(f"plan must have {min_steps}-{self.settings.max_plan_steps} steps, got {len(steps)}")

    def create_plan(self, goal: str) -> Plan:
        messages = [
            {"role": "system", "content": self._fmt(prompts.PLANNER_SYSTEM)},
            {"role": "user", "content": f"GOAL: {goal}"},
        ]
        try:
            plan = self.llm.complete_json(messages, Plan, check=lambda p: self._check_steps(p.steps, MIN_STEPS),
                                          max_tokens=1500)
        except LLMOutputError as e:
            self.tracer.recovery("fallback_plan", reason=str(e)[:200])
            plan = fallback_plan(goal)
        plan.steps = self._trim(plan.steps, analysis_steps=1)
        for i, step in enumerate(plan.steps, 1):
            step.id = i
        return plan

    def replan(self, goal: str, results: list[StepResult], failed: StepResult,
               remaining: list[PlanStep], next_id: int) -> tuple[str, list[PlanStep]]:
        """Return (reasoning, new steps). Returns ([], ...) if re-planning itself fails."""
        done = "\n".join(f"- Step {r.step_id} [{r.status}] {r.title}: {r.output[:400]}" for r in results
                         if r.step_id != failed.step_id) or "(none)"
        calls = "\n".join(f"  - {c.tool}({c.args}) -> {'ok' if c.ok else 'FAILED: ' + (c.error or '')}"
                          for c in failed.tool_calls[-6:]) or "  (no tool calls)"
        rest = "\n".join(f"- {s.title} ({s.tool_hint})" for s in remaining) or "(none)"
        user = (f"GOAL: {goal}\n\nCOMPLETED STEPS:\n{done}\n\n"
                f"FAILED STEP {failed.step_id}: {failed.title}\n"
                f"Why it failed: {failed.verdict.reason if failed.verdict else 'unknown'}\n"
                f"Its tool calls:\n{calls}\nIts partial output: {failed.output[:400]}\n\n"
                f"STEPS THAT WERE STILL PLANNED:\n{rest}")
        messages = [
            {"role": "system", "content": self._fmt(prompts.REPLANNER_SYSTEM, next_id=next_id)},
            {"role": "user", "content": user},
        ]
        try:
            rp = self.llm.complete_json(messages, Replan, check=lambda r: self._check_steps(r.steps), max_tokens=1200)
        except LLMOutputError as e:
            return f"re-planning failed: {e}", []
        steps = self._trim(rp.steps, analysis_steps=0, max_steps=MAX_REPLAN_STEPS)
        for i, step in enumerate(steps):
            step.id = next_id + i
        return rp.reasoning, steps

    def _trim(self, steps: list[PlanStep], analysis_steps: int, max_steps: int = 0) -> list[PlanStep]:
        """Drop steps that only summarise or rank (keeping the last ``analysis_steps`` of them): the
        report writer does that anyway, and on a rate-limited LLM every step costs real time."""
        analysis = [s for s in steps if s.tool_hint == "none"]
        if len(analysis) < len(steps) and len(analysis) > analysis_steps:
            drop = {id(s) for s in analysis[:len(analysis) - analysis_steps]}
            self.tracer.info("skipping analysis-only steps (the final report summarises and ranks anyway): "
                             + "; ".join(s.title for s in steps if id(s) in drop))
            steps = [s for s in steps if id(s) not in drop]
        if max_steps and len(steps) > max_steps:
            self.tracer.info(f"re-plan capped at {max_steps} steps; dropped: "
                             + "; ".join(s.title for s in steps[max_steps:]))
            steps = steps[:max_steps]
        return steps


def fallback_plan(goal: str) -> Plan:
    """Deterministic plan used when the LLM can't produce a valid one."""
    return Plan(
        understanding=f"Research the goal using a generic search-read-analyse plan: {goal}",
        assumptions=["The planner LLM output was unusable, so a generic research plan is used."],
        steps=[
            PlanStep(id=1, title="Search recent news", tool_hint="news_search",
                     instruction=f"Search recent news relevant to: {goal}",
                     success_criteria="At least 3 relevant, dated results with URLs"),
            PlanStep(id=2, title="Background web search", tool_hint="web_search",
                     instruction=f"Search the web for background and primary sources about: {goal}",
                     success_criteria="At least 2 relevant sources with URLs"),
            PlanStep(id=3, title="Read key sources", tool_hint="fetch_page",
                     instruction="Fetch the 1-2 most relevant publisher URLs found so far and extract key facts",
                     success_criteria="Concrete facts (names, dates, numbers) from at least one page"),
            PlanStep(id=4, title="Analyse and rank findings", tool_hint="none",
                     instruction="Combine earlier results and pick the most important findings for the goal",
                     success_criteria="A ranked list of findings, each tied to a source"),
        ],
    )
