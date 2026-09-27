"""Shared fixtures: a scripted fake LLM client and fake tools, so tests run offline."""
from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace
from typing import Any, Callable, Union

import pytest

from research_agent.config import Settings
from research_agent.failures import FailureInjector
from research_agent.tools import Param, Tool, ToolOutput, ToolRegistry
from research_agent.tools.calculator import calculator

Reply = Union[str, dict, Exception, Callable[[list[dict]], Any]]


class Cut(str):
    """A scripted reply that stopped at max_tokens (finish_reason "length")."""


def role_of(messages: list[dict]) -> str:
    system = messages[0]["content"]
    if "Part of the current plan failed" in system:
        return "replan"
    for marker, role in (("PLANNER", "plan"), ("EXECUTOR", "exec"), ("REVIEWER", "review"), ("FINAL REPORT", "report")):
        if marker in system:
            return role
    raise AssertionError(f"unknown prompt: {system[:80]}")


class ScriptedClient:
    """Mimics ``openai.OpenAI().chat.completions.create`` with per-role reply queues."""

    def __init__(self, script: dict[str, list[Reply]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, list[dict]]] = []
        self.response_formats: list[bool] = []
        self.models: list[str] = []
        self.max_tokens: list[int] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        messages = kwargs["messages"]
        self.response_formats.append("response_format" in kwargs)
        self.models.append(kwargs.get("model"))
        self.max_tokens.append(kwargs.get("max_tokens"))
        role = role_of(messages)
        self.calls.append((role, messages))
        queue = self.script.get(role)
        if not queue:
            raise AssertionError(f"no scripted reply left for role '{role}'")
        reply = queue.pop(0)
        if callable(reply) and not isinstance(reply, dict):
            reply = reply(messages)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        finish = "length" if isinstance(reply, Cut) else "stop"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=str(content)),
                                                        finish_reason=finish)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))

    def count(self, role: str) -> int:
        return sum(1 for r, _ in self.calls if r == role)


NEWS_URL = "https://example.com/news/agents-launch"
WEB_URL = "https://example.org/background"


def _fake_news(args, ctx):
    return ToolOutput(text=f"1 news result for '{args['query']}':\n[1] Agents launch (2026-09-20)\n    url: {NEWS_URL}",
                      data=[{"url": NEWS_URL}], sources=[NEWS_URL])


def _fake_web(args, ctx):
    return ToolOutput(text=f"1 web result:\n[1] Background\n    url: {WEB_URL}", data=[{"url": WEB_URL}],
                      sources=[WEB_URL])


def fake_tools() -> list[Tool]:
    return [
        Tool("news_search", "fake news", {"query": Param("string", "q", required=True)}, _fake_news),
        Tool("web_search", "fake web", {"query": Param("string", "q", required=True)}, _fake_web),
        calculator,
    ]


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="test", backoff_base=0.0, tool_retries=2, llm_retries=2,
                    max_actions_per_step=3, max_step_retries=1, max_replans=1, token_budget=100_000,
                    tpm_limit=0)


@pytest.fixture
def make_agent(settings):
    """Build a ResearchAgent wired to a scripted LLM and fake tools."""
    from research_agent.agent import ResearchAgent
    from research_agent.llm import LLM
    from research_agent.trace import Tracer

    def _make(script: dict, faults: list[str] = (), **overrides):
        s = Settings(**{**settings.__dict__, **overrides})
        tracer = Tracer(quiet=True)
        injector = FailureInjector.from_strings(list(faults))
        client = ScriptedClient(script)
        llm = LLM(s, injector, emit=tracer.emit, client=client, sleep=lambda _: None)
        registry = ToolRegistry(injector, timeout=5, retries=s.tool_retries, backoff_base=0,
                                emit=tracer.emit, sleep=lambda _: None)
        for t in fake_tools():
            registry.register(t)
        return ResearchAgent(s, llm, registry, tracer), client

    return _make


def plan_json(*steps: tuple[str, str], ambiguities=(), assumptions=()) -> dict:
    return {
        "understanding": "Find recent developments in AI agents",
        "ambiguities": list(ambiguities),
        "assumptions": list(assumptions),
        "steps": [{"id": i, "title": title, "instruction": f"Do: {title} carefully",
                   "tool_hint": tool, "success_criteria": "concrete results with URLs"}
                  for i, (title, tool) in enumerate(steps, 1)],
    }


def call(tool: str, **args) -> dict:
    return {"thought": f"use {tool}", "action": "call_tool", "tool": tool, "args": args}


def finish(output: str, sources=()) -> dict:
    return {"thought": "done", "action": "finish", "output": output, "sources": list(sources)}


PASS = {"passed": True, "reason": "meets criteria"}
FAIL = {"passed": False, "reason": "too vague", "suggestion": "search again"}


def report_json(sources: list[str]) -> dict:
    return {
        "title": "AI agents this week",
        "executive_summary": "Agents launched.",
        "findings": [{"title": "Agents launch", "summary": "Something launched.",
                      "date": (dt.date.today() - dt.timedelta(days=2)).isoformat(),
                      "why_it_matters": "It matters.", "sources": sources}],
        "limitations": [],
        "confidence": "medium",
        "confidence_reason": "one source",
    }
