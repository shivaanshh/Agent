"""Typed data passed between agent components.

LLM outputs are parsed into these pydantic models, so a malformed response is
caught at the boundary (and repaired) instead of breaking code further down.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ── Planning ──────────────────────────────────────────────────────────────────
class PlanStep(BaseModel):
    id: int
    title: str = Field(min_length=3)
    instruction: str = Field(min_length=10)
    tool_hint: str = "none"
    success_criteria: str = Field(min_length=5)


class Plan(BaseModel):
    understanding: str = Field(min_length=5)
    ambiguities: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(min_length=1)


class Replan(BaseModel):
    reasoning: str
    steps: list[PlanStep] = Field(min_length=1)


# ── Execution ─────────────────────────────────────────────────────────────────
class AgentAction(BaseModel):
    """One decision of the executor: call a tool, or finish the step."""

    thought: str = ""
    action: Literal["call_tool", "finish"]
    tool: Optional[str] = None
    args: dict[str, Any] = Field(default_factory=dict)
    output: Optional[str] = None
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _infer_action(cls, data: Any) -> Any:
        # Smaller models often drop "action" or put the tool name in it; the intent is still clear,
        # and inferring it here saves an LLM repair round-trip.
        if isinstance(data, dict) and data.get("action") not in ("call_tool", "finish"):
            data, act = dict(data), data.get("action")
            if data.get("tool"):
                data["action"] = "call_tool"
            elif data.get("output"):
                data["action"] = "finish"
            elif isinstance(act, str) and act.strip():
                data["tool"], data["action"] = act.strip(), "call_tool"  # {"action": "web_search", "args": {...}}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> "AgentAction":
        if self.action == "call_tool" and not self.tool:
            raise ValueError("'tool' is required when action is 'call_tool'")
        if self.action == "finish" and not (self.output or "").strip():
            raise ValueError("'output' is required when action is 'finish'")
        return self


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    error: Optional[str] = None
    attempts: int = 1
    latency_ms: int = 0
    preview: str = ""


class Verdict(BaseModel):
    passed: bool
    reason: str = ""
    suggestion: str = ""


class StepResult(BaseModel):
    step_id: int
    title: str
    status: Literal["passed", "failed", "skipped"] = "failed"
    output: str = ""
    sources: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    attempts: int = 0
    verdict: Optional[Verdict] = None


# ── Reporting ─────────────────────────────────────────────────────────────────
class Finding(BaseModel):
    title: str
    summary: str
    date: Optional[str] = None
    why_it_matters: Optional[str] = None
    sources: list[str] = Field(default_factory=list)


class ReportDraft(BaseModel):
    title: str
    executive_summary: str
    findings: list[Finding] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    confidence_reason: str = ""
