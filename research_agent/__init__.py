"""Autonomous research agent: plan → act (tools) → verify → recover → report."""
from .agent import ResearchAgent, RunResult
from .config import Settings

__all__ = ["ResearchAgent", "RunResult", "Settings"]
