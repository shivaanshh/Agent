"""Run tracing: pretty console output + machine-readable event log.

Every meaningful thing the agent does becomes an event (goal, plan, thought,
tool call/result, verdict, recovery, replan, report). Events are printed with
rich as they happen, and saved as ``trace.jsonl`` plus a plain-text
``transcript.txt`` for each run.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .models import Plan, PlanStep, StepResult, Verdict

RECOVERY_LABELS = {
    "tool_retry": "retrying tool",
    "provider_fallback": "switching search provider",
    "llm_retry": "retrying LLM call",
    "llm_rate_limit": "rate limited, backing off",
    "llm_throttle": "pacing under tokens-per-minute limit",
    "reasoning_salvage": "recovered JSON from model reasoning",
    "json_mode_off": "provider JSON mode misfiring, switching to plain JSON",
    "llm_unavailable": "LLM unavailable, finishing with what we have",
    "json_repair": "repairing invalid LLM JSON",
    "llm_truncated": "reply cut off at the token limit, asking again with more room",
    "model_fallback": "secondary model unavailable, using the main model",
    "lane_fallback": "second provider failed this call, sending it to the primary",
    "lane_disabled": "second provider unusable, continuing on the primary only",
    "invalid_action": "correcting invalid action",
    "duplicate_call": "blocked repeated tool call",
    "forced_finish": "action limit reached, forcing summary",
    "step_retry": "retrying step with reviewer feedback",
    "replan": "re-planning remaining work",
    "fallback_plan": "using fallback plan",
    "fallback_report": "using fallback report",
    "budget": "token budget exhausted",
}


def _short(value: Any, n: int = 90) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _ensure_utf8_stdout() -> None:
    # Windows consoles default to cp1252, which can't encode the trace's symbols.
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if enc.replace("-", "") != "utf8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Tracer:
    def __init__(self, console: Optional[Console] = None, quiet: bool = False):
        if console is None and not quiet:
            _ensure_utf8_stdout()
        # quiet: record the transcript without printing it
        self.console = console or Console(record=True, width=110, soft_wrap=False,
                                          file=io.StringIO() if quiet else None)
        self.quiet = quiet and console is not None
        self.listeners: list[Callable[[dict], None]] = []  # e.g. the web UI's live stream
        self.events: list[dict] = []
        self.recoveries: list[dict] = []
        self.current_step: Optional[int] = None
        self._t0 = time.monotonic()

    # core
    def event(self, event_kind: str, **data) -> None:
        e = {"t": round(time.monotonic() - self._t0, 2), "event": event_kind, **data}
        self.events.append(e)
        for listener in self.listeners:
            listener(e)

    def _print(self, *args, **kwargs) -> None:
        if not self.quiet:
            self.console.print(*args, **kwargs)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    # callback used by tools / LLM for recovery events
    def emit(self, kind: str, **data) -> None:
        self.recovery(kind, **data)

    # typed events
    def goal(self, goal: str, model: str, faults: list[str]) -> None:
        self.event("goal", goal=goal, model=model, injected_faults=faults)
        body = f"[bold]{escape(goal)}[/bold]\n[dim]model: {model}"
        if faults:
            body += f"  |  injected faults: {', '.join(faults)}"
        self._print(Panel(body + "[/dim]", title="GOAL", border_style="cyan"))

    def plan(self, plan: Plan, title: str = "PLAN") -> None:
        self.event("plan", plan=plan.model_dump())
        self._print(f"\n[bold cyan]{title}[/bold cyan]  [dim]{escape(plan.understanding)}[/dim]")
        for label, items in (("ambiguities", plan.ambiguities), ("assumptions", plan.assumptions)):
            for a in items:
                self._print(f"  [yellow]{label[:-1]}:[/yellow] {escape(a)}")
        self.steps_table(plan.steps)

    def steps_table(self, steps: list[PlanStep]) -> None:
        table = Table(show_header=True, header_style="bold", expand=False, pad_edge=False)
        table.add_column("#", justify="right")
        table.add_column("step")
        table.add_column("tool")
        table.add_column("done when")
        for s in steps:
            table.add_row(str(s.id), escape(s.title), s.tool_hint, escape(_short(s.success_criteria, 60)))
        self._print(table)

    def step_start(self, step: PlanStep, attempt: int, total: int) -> None:
        self.current_step = step.id
        self.event("step_start", step_id=step.id, title=step.title, attempt=attempt, total=total)
        suffix = f" (attempt {attempt})" if attempt > 1 else ""
        self._print(f"\n[bold]── Step {step.id}/{total}: {escape(step.title)}{suffix}[/bold]")

    def thought(self, text: str) -> None:
        if text:
            self.event("thought", step_id=self.current_step, text=text)
            self._print(f"  [magenta]think:[/magenta] {escape(_short(text, 200))}")

    def tool_call(self, tool: str, args: dict) -> None:
        self.event("tool_call", step_id=self.current_step, tool=tool, args=args)
        arg_s = ", ".join(f"{k}={_short(v, 60)!s}" for k, v in args.items())
        self._print(f"  [blue]→ {tool}[/blue]({escape(arg_s)})")

    def tool_result(self, result) -> None:
        self.event("tool_result", step_id=self.current_step, tool=result.tool, ok=result.ok,
                   error=result.error, attempts=result.attempts, latency_ms=result.latency_ms,
                   preview=_short(result.text, 300) if result.ok else "")
        meta = f"[dim]({result.latency_ms} ms{', ' + str(result.attempts) + ' attempts' if result.attempts > 1 else ''})[/dim]"
        if result.ok:
            first = result.text.splitlines()[0] if result.text else ""
            self._print(f"    [green]✓[/green] {escape(_short(first, 90))} {meta}")
        else:
            self._print(f"    [red]✗ {escape(_short(result.error or 'failed', 140))}[/red] {meta}")

    def step_output(self, output: str, sources: Optional[list[str]] = None) -> None:
        self.event("step_output", step_id=self.current_step, output=output, sources=sources or [])
        self._print(f"  [cyan]result:[/cyan] {escape(_short(output, 260))}")

    def verdict(self, step_id: int, verdict: Verdict) -> None:
        self.event("verdict", step_id=step_id, **verdict.model_dump())
        if verdict.passed:
            self._print(f"  [green]✔ review passed[/green] [dim]{escape(_short(verdict.reason, 120))}[/dim]")
        else:
            self._print(f"  [red]✘ review failed:[/red] {escape(_short(verdict.reason, 160))}")

    def step_done(self, result: StepResult) -> None:
        self.event("step_done", step_id=result.step_id, status=result.status, attempts=result.attempts)

    def recovery(self, kind: str, **data) -> None:
        entry = {"kind": kind, "step_id": self.current_step, **data}
        self.recoveries.append(entry)
        self.event("recovery", **entry)
        label = RECOVERY_LABELS.get(kind, kind)
        detail = ", ".join(f"{k}={_short(v, 70)}" for k, v in data.items())
        self._print(f"  [yellow]↻ RECOVERY[/yellow] [bold yellow]{label}[/bold yellow] [dim]{escape(detail)}[/dim]")

    def replan(self, reasoning: str, steps: list[PlanStep]) -> None:
        self.event("replan", reasoning=reasoning, steps=[s.model_dump() for s in steps])
        self._print(f"  [yellow]new plan:[/yellow] {escape(reasoning)}")
        self.steps_table(steps)

    def section(self, title: str) -> None:
        self.event("section", title=title)
        self._print(f"\n[bold cyan]{escape(title)}[/bold cyan]")

    def info(self, msg: str) -> None:
        self.event("info", message=msg)
        self._print(f"[dim]{escape(msg)}[/dim]")

    def error(self, msg: str) -> None:
        self.event("error", message=msg)
        self._print(f"[bold red]ERROR:[/bold red] {escape(msg)}")

    # persistence
    def export(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "trace.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        (run_dir / "transcript.txt").write_text(self.console.export_text(clear=False), encoding="utf-8")
