"""Orchestrator: plan → (execute → verify → retry/replan)* → report."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from .config import Settings
from .executor import Executor
from .failures import FailureInjector
from .llm import LLM, BudgetExceeded, LLMError
from .models import Plan, PlanStep, StepResult
from .planner import Planner
from .reporter import Reporter
from .tools import ToolRegistry, build_default_registry
from .trace import Tracer
from .verifier import Verifier

ClarifyFn = Callable[[list[str]], Optional[str]]


@dataclass
class RunResult:
    goal: str
    plan: Plan
    results: list[StepResult]
    report: dict


class ResearchAgent:
    def __init__(self, settings: Settings, llm: LLM, registry: ToolRegistry, tracer: Tracer):
        self.settings, self.llm, self.registry, self.tracer = settings, llm, registry, tracer
        self.planner = Planner(llm, registry, settings, tracer)
        self.executor = Executor(llm, registry, settings, tracer)
        self.verifier = Verifier(llm, tracer, settings.review_model, returned=self.executor.seen_sources)
        self.reporter = Reporter(llm, tracer)
        self._interrupted_calls: list = []

    @classmethod
    def build(cls, settings: Settings, injector: Optional[FailureInjector] = None,
              tracer: Optional[Tracer] = None, llm_client=None) -> "ResearchAgent":
        tracer = tracer or Tracer()
        injector = injector or FailureInjector()
        llm = LLM(settings, injector, emit=tracer.emit, client=llm_client)
        registry = build_default_registry(settings, injector, emit=tracer.emit)
        return cls(settings, llm, registry, tracer)

    def run(self, goal: str, clarify: Optional[ClarifyFn] = None) -> RunResult:
        t = self.tracer
        model = self.settings.model
        if self.llm.lanes:
            model += f" (on {' + '.join([self.llm.primary] + [lane.name for lane in self.llm.lanes])})"
        t.goal(goal, model, self.registry.injector.describe())

        plan = self.planner.create_plan(goal)
        if plan.ambiguities and clarify:
            answer = clarify(plan.ambiguities)
            if answer:
                goal = f"{goal}\n(User clarification: {answer})"
                t.info("re-planning with the user's clarification")
                plan = self.planner.create_plan(goal)
        t.plan(plan)

        queue: deque[PlanStep] = deque(plan.steps)
        results: list[StepResult] = []
        replans: list[dict] = []
        try:
            while queue:
                step = queue.popleft()
                total = queue[-1].id if queue else step.id  # ids keep increasing across re-plans
                result = self._run_step(goal, step, total, results)
                results.append(result)
                t.step_done(result)
                if result.status == "failed" and len(replans) < self.settings.max_replans:
                    next_id = max([s.id for s in queue] + [step.id]) + 1
                    t.recovery("replan", failed_step=step.id, reason=(result.verdict.reason if result.verdict else "")[:120])
                    reasoning, new_steps = self.planner.replan(goal, results, result, list(queue), next_id)
                    if new_steps:
                        replans.append({"after_step": step.id, "reasoning": reasoning,
                                        "new_steps": [s.title for s in new_steps]})
                        queue = deque(new_steps)
                        t.replan(reasoning, new_steps)
                    else:
                        t.info(f"re-planning produced nothing ({reasoning}); continuing with original plan")
        except LLMError as e:
            # Budget spent, or the LLM keeps failing (quota, outage): stop executing, but still report.
            budget = isinstance(e, BudgetExceeded)
            t.recovery("budget" if budget else "llm_unavailable", detail=str(e)[:200])
            interrupted = not results or results[-1].step_id != step.id
            if interrupted:
                queue.appendleft(step)
            why = "token budget exhausted" if budget else "LLM unavailable"
            for i, skipped in enumerate(queue):
                # the interrupted step keeps the tool calls it made, so the run's stats count them
                calls = self._interrupted_calls if interrupted and i == 0 else []
                results.append(StepResult(step_id=skipped.id, title=skipped.title, status="skipped",
                                          output=f"Skipped: {why}.", tool_calls=calls))

        t.section("Writing final report…")
        report = self.reporter.build(goal=goal, plan=plan, results=results,
                                     seen_sources=self.executor.seen_sources, model=self.settings.model,
                                     usage=self.llm.usage, injected=self.registry.injector.describe(),
                                     replans=replans, duration_s=t.elapsed)
        t.event("report", report=report)
        return RunResult(goal, plan, results, report)

    def _run_step(self, goal: str, step: PlanStep, total: int, prior: list[StepResult]) -> StepResult:
        feedback: Optional[str] = None
        all_calls = []
        result = StepResult(step_id=step.id, title=step.title)
        for attempt in range(1, self.settings.max_step_retries + 2):
            self.tracer.step_start(step, attempt, total)
            try:
                result = self.executor.run_step(goal, step, total, prior, feedback)
            except LLMError:
                self._interrupted_calls = all_calls + self.executor.current.tool_calls
                raise
            all_calls += result.tool_calls
            verdict = self.verifier.review(step, result)
            result.verdict, result.attempts = verdict, attempt
            if verdict.passed:
                result.status = "passed"
                break
            if attempt <= self.settings.max_step_retries:
                feedback = f"{verdict.reason} Suggestion: {verdict.suggestion}".strip()
                self.tracer.recovery("step_retry", reason=verdict.reason[:120])
        result.tool_calls = all_calls
        return result
