"""Command-line entry point: ``python -m research_agent "your goal"``."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.markdown import Markdown

from .agent import ResearchAgent
from .config import Settings
from .demo import DemoClient, demo_settings
from .failures import MODES, TARGETS, FailureInjector
from .llm import LLMError
from .reporter import render_markdown
from .trace import Tracer

EXAMPLES = """examples:
  python -m research_agent "Research and summarize the top 3 developments in AI agents from the last week"
  python -m research_agent "Compare the latest quarterly revenue of Nvidia and AMD" --inject ddgs:error:*
  python -m research_agent "Tell me the latest about Python" --interactive
  python -m research_agent "Top 3 AI agent developments this week" --demo --inject news_search:error:*
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "run"


def _ask_user(ambiguities: list[str]) -> Optional[str]:
    print("\nThe goal is ambiguous:")
    for a in ambiguities:
        print(f"  - {a}")
    try:
        answer = input("Clarify (or press Enter to let the agent assume): ").strip()
    except EOFError:
        return None
    return answer or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research_agent", description="Autonomous research agent (plan → act → verify → report).",
                                epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("goal", nargs="?", help="natural-language goal (prompted for if omitted)")
    p.add_argument("--inject", action="append", metavar="TARGET:MODE[:COUNT]",
                   help="deliberately inject a failure (repeatable). See --list-faults")
    p.add_argument("--list-faults", action="store_true", help="list failure-injection modes and exit")
    p.add_argument("--interactive", action="store_true", help="ask the user when the goal is ambiguous")
    p.add_argument("--model", help="LLM model id (default from GROQ_MODEL or settings)")
    p.add_argument("--demo", action="store_true",
                   help="no API key: a rule-based stand-in makes the LLM's decisions; tools and checks stay real")
    p.add_argument("--budget", type=int, help="soft token budget for the run")
    p.add_argument("--out", default="runs", help="directory for run artefacts (default: runs/)")
    p.add_argument("--quiet", action="store_true", help="don't stream the trace; print only the report path")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.list_faults:
        print("Fault spec: TARGET:MODE[:COUNT]   (COUNT defaults to 1, '*' = every call)")
        print(f"Targets: {', '.join(TARGETS)}\n")
        for mode, desc in MODES.items():
            print(f"  {mode:<13} {desc}")
        return 0

    load_dotenv()
    settings = Settings.from_env(model=args.model, token_budget=args.budget)
    if args.demo:
        settings = demo_settings(settings)
    elif not settings.api_key:
        print("GROQ_API_KEY is not set. Put it in .env (see .env.example) or export it, or try --demo.",
              file=sys.stderr)
        return 2
    try:
        injector = FailureInjector.from_strings(args.inject, targets=TARGETS)
    except ValueError as e:
        print(f"--inject: {e}", file=sys.stderr)
        return 2

    try:
        goal = (args.goal if args.goal is not None else input("Goal: ")).strip()
    except (EOFError, KeyboardInterrupt):
        goal = ""
    if not goal:
        print("No goal given.", file=sys.stderr)
        return 2

    tracer = Tracer(quiet=args.quiet)
    agent = ResearchAgent.build(settings, injector, tracer, llm_client=DemoClient() if args.demo else None)
    run_dir = Path(args.out) / f"{dt.datetime.now():%Y%m%d-%H%M%S}_{'demo_' if args.demo else ''}{_slug(goal)}"

    code = 0
    try:
        result = agent.run(goal, clarify=_ask_user if args.interactive else None)
        md = render_markdown(result.report)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "report.md").write_text(md, encoding="utf-8")
        if not args.quiet:
            tracer.console.rule("FINAL REPORT")
            tracer.console.print(Markdown(md))
    except LLMError as e:
        tracer.error(f"run aborted: {e}")
        code = 1
    except KeyboardInterrupt:
        tracer.error("interrupted by user")
        code = 130
    finally:
        tracer.export(run_dir)
    print(f"\nArtefacts saved to {run_dir}/ (report.md, report.json, transcript.txt, trace.jsonl)")
    return code
