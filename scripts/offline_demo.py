"""Offline demo: live tools + injected failures, with the rule-based demo stand-in for the LLM (no API key).

The same as running
    python -m research_agent "<goal>" --demo --inject ddgs:error:* --inject web_search:timeout:1
It shows the control loop and the recovery machinery against live search providers, not LLM
quality. For real runs, drop --demo and set GROQ_API_KEY.

    python scripts/offline_demo.py [out_dir]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_agent.cli import main  # noqa: E402

GOAL = "Research and summarize the top 3 developments in AI agents from the last week"
FAULTS = ["ddgs:error:*", "web_search:timeout:1"]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "runs" / "offline_demo")
    raise SystemExit(main([GOAL, "--demo", "--out", out, *(a for f in FAULTS for a in ("--inject", f))]))
