"""Verifier: decides whether a step result meets its success criteria.

Cheap deterministic checks run first; an LLM reviewer judges the rest. If the
reviewer itself fails, we fall back to the deterministic verdict rather than
blocking the run.
"""
from __future__ import annotations

import datetime as dt
from typing import Collection, Optional
from urllib.parse import unquote, urlsplit

from . import prompts
from .llm import LLM, LLMError
from .models import PlanStep, StepResult, ToolCallRecord, Verdict
from .trace import Tracer

MIN_OUTPUT_CHARS = 40


def _url_key(url: str) -> str:
    """Host and path: a URL copied without its scheme, ``www.`` or query string still matches."""
    parts = urlsplit(url.strip())
    return parts.netloc.lower().removeprefix("www.") + unquote(parts.path).rstrip("/")


def ungrounded(cited: list[str], returned: Collection[str]) -> list[str]:
    """Cited URLs that no tool returned. A bare site counts if a tool returned a page on it."""
    keys = {_url_key(u) for u in returned}
    hosts = {k.split("/", 1)[0] for k in keys}
    return [u for u in cited if _url_key(u) not in keys | hosts]


def deterministic_check(step: PlanStep, result: StepResult,
                        returned: Optional[Collection[str]] = None) -> Optional[Verdict]:
    """Return a failing verdict for obviously bad results, else None.

    ``returned`` is every URL a tool returned during the run. When it is given, citing anything else
    fails the step: asked for "at least 5 items" when a search found 1, models make up the other 4.
    """
    if len(result.output.strip()) < MIN_OUTPUT_CHARS:
        return Verdict(passed=False, reason="the step produced no meaningful output",
                       suggestion="call a tool and report concrete findings")
    if step.tool_hint != "none" and result.tool_calls and not any(c.ok for c in result.tool_calls):
        errors = "; ".join(sorted({(c.error or "")[:80] for c in result.tool_calls}))
        return Verdict(passed=False, reason=f"every tool call failed ({errors}), so the result is not grounded",
                       suggestion="use a different tool or query")
    if step.tool_hint != "none" and not result.tool_calls:
        return Verdict(passed=False, reason="no tool was called, so the result is not grounded in data",
                       suggestion=f"call {step.tool_hint} before finishing")
    invented = ungrounded(result.sources, returned) if returned is not None else []
    if invented:
        return Verdict(passed=False,
                       reason=f"{len(invented)} of {len(result.sources)} cited URLs were not returned by any tool "
                              f"in this run, so those items may be invented: {', '.join(u[:90] for u in invented[:3])}",
                       suggestion="report only items a tool returned; if that is too few, search again with "
                                  "other words or a wider time window")
    return None


def _call_line(c: ToolCallRecord) -> str:
    if not c.ok:
        return f"- {c.tool}({c.args}) -> FAILED: {c.error or ''}"
    headline = c.preview.strip().splitlines()[0][:160] if c.preview.strip() else ""
    return f"- {c.tool}({c.args}) -> ok" + (f": {headline}" if headline else "")


class Verifier:
    def __init__(self, llm: LLM, tracer: Tracer, model: str = "", returned: Optional[Collection[str]] = None):
        self.llm, self.tracer = llm, tracer
        self.model = model or None  # a second model has its own rate limit, so reviews don't queue behind steps
        self.returned = returned  # the executor's live set of URLs tools returned

    def review(self, step: PlanStep, result: StepResult) -> Verdict:
        verdict = deterministic_check(step, result, self.returned)
        if verdict is None:
            verdict = self._llm_review(step, result)
        self.tracer.verdict(step.id, verdict)
        return verdict

    def _llm_review(self, step: PlanStep, result: StepResult) -> Verdict:
        calls = "\n".join(_call_line(c) for c in result.tool_calls) or "- (none)"
        messages = [
            {"role": "system", "content": prompts.VERIFIER_SYSTEM.format(today=dt.date.today().isoformat())},
            {"role": "user", "content": prompts.VERIFIER_USER.format(
                title=step.title, instruction=step.instruction, criteria=step.success_criteria,
                calls=calls, output=result.output[:2500],
                sources="\n".join(f"- {u}" for u in result.sources[:10]) or "- (none)")},
        ]
        try:
            return self.llm.complete_json(messages, Verdict, max_tokens=400, temperature=0.0, model=self.model)
        except LLMError as e:
            self.tracer.info(f"reviewer unavailable ({str(e)[:100]}); accepting on deterministic checks")
            return Verdict(passed=True, reason="LLM reviewer unavailable; deterministic checks passed")
