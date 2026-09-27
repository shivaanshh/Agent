"""Run the agent on a small goal set and tabulate behaviour metrics.

Usage:  python -m evals.run_evals [--only id1,id2] [--budget 40000]

Each goal is a real run (live tools + Groq), so it spends tokens and time: on
the free tier (8K tokens/minute) each goal takes several minutes because the
agent paces itself under the limit. Use --only to run a subset.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from dotenv import load_dotenv

from research_agent.agent import ResearchAgent
from research_agent.config import Settings
from research_agent.failures import TARGETS, FailureInjector
from research_agent.llm import LLMError
from research_agent.trace import Tracer

HERE = Path(__file__).parent


def evaluate(case: dict, settings: Settings, out_dir: Path) -> dict:
    tracer = Tracer(quiet=True)
    agent = ResearchAgent.build(settings, FailureInjector.from_strings(case["inject"], targets=TARGETS), tracer)
    row = {"id": case["id"], "inject": ",".join(case["inject"]) or "-"}
    try:
        run = agent.run(case["goal"])
    except LLMError as e:
        row.update(status="aborted", error=str(e)[:120])
        return row
    finally:
        tracer.export(out_dir / case["id"])
    r = run.report
    (out_dir / case["id"] / "report.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    s, g = r["stats"], r["grounding"]
    row.update(
        status="ok",
        plan_steps=len(run.plan.steps),
        steps_passed=f"{s['steps_passed']}/{s['steps_total']}",
        findings=len(r["findings"]),
        recoveries=s["recoveries"],
        recovery_kinds=",".join(sorted({x["kind"] for x in r["execution"]["recoveries"]})) or "-",
        grounding=g["rate"],
        confidence=r["confidence"],
        tokens=s["tokens"],
        seconds=s["duration_s"],
    )
    return row


def to_markdown(rows: list[dict]) -> str:
    cols = ["id", "status", "inject", "plan_steps", "steps_passed", "findings", "recoveries",
            "recovery_kinds", "grounding", "confidence", "tokens", "seconds"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated goal ids")
    p.add_argument("--budget", type=int, default=40_000, help="token budget per goal")
    args = p.parse_args()

    load_dotenv()
    settings = Settings.from_env(token_budget=args.budget)
    cases = json.loads((HERE / "goals.json").read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    out_dir = HERE / "results" / f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    rows = []
    for case in cases:
        print(f"running {case['id']} …", flush=True)
        rows.append(evaluate(case, settings, out_dir))
        print(f"  -> {rows[-1]}")
    table = to_markdown(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
