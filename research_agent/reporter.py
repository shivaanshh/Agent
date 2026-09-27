"""Reporter: turns step results into the final structured report (JSON + Markdown).

The narrative comes from the LLM; everything factual about the run itself
(plan status, recoveries, stats, source grounding) is filled in by code so it
can't be hallucinated.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from . import prompts
from .llm import LLM, LLMError
from .models import Finding, Plan, ReportDraft, StepResult
from .trace import Tracer


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = "&".join(q for q in parts.query.split("&") if q and not q.lower().startswith(("utm_", "oc=")))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower().removeprefix("www."),
                       parts.path.rstrip("/"), query, ""))


def requested_window_days(goal: str) -> Optional[int]:
    """How far back the goal asks to look ("last week" -> 7), or None if it doesn't say."""
    g = goal.lower()
    m = re.search(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", g)
    if m:
        return int(m.group(1))
    for pattern, days in ((r"\b(?:last|past)\s+24\s+hours\b|\btoday\b", 1),
                          (r"\b(?:this|last|past)\s+week\b", 7),
                          (r"\b(?:this|last|past)\s+month\b", 31)):
        if re.search(pattern, g):
            return days
    return None


def outside_window(date: Optional[str], window_days: Optional[int], today: dt.date) -> bool:
    """True only for a parseable ISO date older than the requested window (one day of slack for time zones)."""
    if not date or not window_days:
        return False
    try:
        d = dt.date.fromisoformat(date.strip()[:10])
    except ValueError:
        return False
    return d < today - dt.timedelta(days=window_days + 1)


_LEVELS = ["low", "medium", "high"]


def _cap(confidence: str, reason: str, ceiling: str, why: str) -> tuple[str, str]:
    """Lower ``confidence`` to ``ceiling`` if it is above it, saying why."""
    if _LEVELS.index(confidence) <= _LEVELS.index(ceiling):
        return confidence, reason
    return ceiling, f"{reason} (Capped at {ceiling}: {why}.)".strip()


def fallback_draft(goal: str, results: list[StepResult]) -> ReportDraft:
    """Report built without the LLM, straight from step outputs."""
    passed = [r for r in results if r.status == "passed"] or results
    findings = [Finding(title=r.title, summary=r.output[:800] or "(no output)", sources=r.sources[:5])
                for r in passed] or [Finding(title="No results", summary="No step produced output.")]
    return ReportDraft(
        title=f"Research notes: {goal[:80]}",
        executive_summary="The report writer was unavailable, so this report lists the raw step results.",
        findings=findings,
        limitations=["Final synthesis by the LLM failed; findings are unedited step outputs."],
        confidence="low",
        confidence_reason="No synthesis or cross-checking was possible.",
    )


class Reporter:
    def __init__(self, llm: LLM, tracer: Tracer):
        self.llm, self.tracer = llm, tracer

    def draft(self, goal: str, plan: Plan, results: list[StepResult], seen: set[str]) -> ReportDraft:
        # Only offer URLs a tool really returned: given nothing citable, models invent plausible-looking links.
        def citable(r: StepResult) -> str:
            return ", ".join([u for u in r.sources if normalize_url(u) in seen][:8]) or "none (do not invent any)"
        steps = "\n\n".join(
            f"### Step {r.step_id} [{r.status}] {r.title}\n{r.output[:1500]}\nSources: {citable(r)}"
            for r in results)
        messages = [
            {"role": "system", "content": prompts.REPORTER_SYSTEM.format(today=dt.date.today().isoformat())},
            {"role": "user", "content": prompts.REPORTER_USER.format(
                goal=goal, assumptions="; ".join(plan.assumptions) or "none", steps=steps)},
        ]
        try:
            return self.llm.complete_json(messages, ReportDraft, max_tokens=2000, critical=True)
        except LLMError as e:
            self.tracer.recovery("fallback_report", reason=str(e)[:200])
            return fallback_draft(goal, results)

    def build(self, *, goal: str, plan: Plan, results: list[StepResult], seen_sources: set[str],
              model: str, usage: dict, injected: list[str], replans: list[dict], duration_s: float) -> dict[str, Any]:
        seen = {normalize_url(u) for u in seen_sources}
        draft = self.draft(goal, plan, results, seen)
        window, today = requested_window_days(goal), dt.date.today()
        cited = verified = 0
        findings = []
        for f in draft.findings:
            sources = []
            for url in dict.fromkeys(f.sources):
                ok = normalize_url(url) in seen
                cited += 1
                verified += ok
                sources.append({"url": url, "verified": ok})
            item = {**f.model_dump(exclude={"sources"}), "sources": sources}
            if outside_window(f.date, window, today):
                item["outside_requested_window"] = True
            findings.append(item)

        # The LLM's confidence is capped by what code can check: steps passed, recency and source grounding.
        limitations, confidence = list(draft.limitations), draft.confidence
        confidence_reason = draft.confidence_reason.strip() or "The model gave no reason."
        passed = sum(r.status == "passed" for r in results)
        if results and passed < len(results) / 2:  # the draft reads just as well when most of the work failed
            limitations.insert(0, f"Only {passed} of {len(results)} steps passed review, so the findings rest on "
                                  f"partial evidence.")
            confidence, confidence_reason = _cap(confidence, confidence_reason, "low",
                                                 "fewer than half of the steps passed review")
        stale = [f["title"] for f in findings if f.get("outside_requested_window")]
        if stale:  # the LLM happily calls a 2-week-old story "this week"
            limitations.insert(0, f"{len(stale)} finding(s) dated before the requested {window}-day window: "
                                  + "; ".join(stale))
            confidence, confidence_reason = _cap(confidence, confidence_reason, "medium",
                                                 "not every finding is inside the requested time window")
        if cited and verified / cited < 0.5:
            unverified = [s["url"] for f in findings for s in f["sources"] if not s["verified"]]
            limitations.insert(0, f"{cited - verified} of {cited} cited URLs were not returned by any tool in this "
                                  f"run and may be invented: " + ", ".join(unverified[:5]))
            confidence, confidence_reason = _cap(confidence, confidence_reason, "low",
                                                 "most cited URLs could not be verified")

        tool_calls = [c for r in results for c in r.tool_calls]
        return {
            "goal": goal,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "title": draft.title,
            "executive_summary": draft.executive_summary,
            "findings": findings,
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "understanding": plan.understanding,
            "ambiguities": plan.ambiguities,
            "assumptions": plan.assumptions,
            "limitations": limitations,
            "execution": {
                "steps": [{"id": r.step_id, "title": r.title, "status": r.status, "attempts": r.attempts,
                           "tools": [f"{c.tool}{'' if c.ok else ' (failed)'}" for c in r.tool_calls],
                           "review": r.verdict.reason if r.verdict else None} for r in results],
                "replans": replans,
                "recoveries": self.tracer.recoveries,
                "injected_faults": injected,
            },
            "grounding": {"cited_sources": cited, "verified_sources": verified,
                          "rate": round(verified / cited, 2) if cited else None,
                          "note": "verified = the URL was actually returned by a tool during this run",
                          "window_days": window, "findings_outside_window": len(stale)},
            "stats": {
                "duration_s": round(duration_s, 1),
                "steps_passed": sum(r.status == "passed" for r in results),
                "steps_total": len(results),
                "tool_calls": len(tool_calls),
                "tool_failures": sum(not c.ok for c in tool_calls),
                "recoveries": len(self.tracer.recoveries),
                "llm_calls": usage.get("calls", 0),
                "llm_calls_by_provider": usage.get("by_provider", {}),
                "tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            },
        }


ROUTINE_RECOVERIES = {"llm_rate_limit", "llm_throttle", "lane_fallback"}


def render_markdown(report: dict[str, Any]) -> str:
    s = report["stats"]
    g = report["grounding"]
    lines = [
        f"# {report['title']}",
        "",
        f"> **Goal:** {report['goal']}  ",
        f"> Generated {report['generated_at']} · model `{report['model']}` · confidence **{report['confidence']}**",
        "",
        "## Summary",
        report["executive_summary"],
        "",
        "## Findings",
    ]
    for i, f in enumerate(report["findings"], 1):
        date = f" ({f['date']})" if f.get("date") else ""
        if f.get("outside_requested_window"):
            date += " ⚠️ outside requested window"
        lines += [f"### {i}. {f['title']}{date}", f["summary"]]
        if f.get("why_it_matters"):
            lines.append(f"\n*Why it matters:* {f['why_it_matters']}")
        if f["sources"]:
            lines.append("\nSources: " + " · ".join(
                f"[{urlsplit(src['url']).netloc or src['url']}]({src['url']})" + ("" if src["verified"] else " ⚠️ unverified")
                for src in f["sources"]))
        lines.append("")

    lines += ["## Confidence", f"**{report['confidence']}**: {report['confidence_reason']}", ""]
    if report["assumptions"] or report["ambiguities"]:
        lines.append("## Assumptions & ambiguities")
        lines += [f"- Ambiguity: {a}" for a in report["ambiguities"]]
        lines += [f"- Assumed: {a}" for a in report["assumptions"]]
        lines.append("")
    if report["limitations"]:
        lines += ["## Limitations"] + [f"- {x}" for x in report["limitations"]] + [""]

    ex = report["execution"]
    lines += ["## How the agent got here", "", "| # | Step | Status | Attempts | Tools |", "|---|---|---|---|---|"]
    icon = {"passed": "✅ passed", "failed": "❌ failed", "skipped": "⏭️ skipped"}
    for st in ex["steps"]:
        lines.append(f"| {st['id']} | {st['title']} | {icon.get(st['status'], st['status'])} | {st['attempts']} | "
                     f"{', '.join(st['tools']) or '-'} |")
    lines.append("")
    if ex["recoveries"]:
        counts = Counter(r["kind"] for r in ex["recoveries"])
        lines.append("**Recoveries:** " + ", ".join(f"`{k}` ×{n}" for k, n in counts.most_common()))
        lines.append("")
        # waiting on provider rate limits is routine; list the substantive recoveries individually
        for r in (r for r in ex["recoveries"] if r["kind"] not in ROUTINE_RECOVERIES):
            detail = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("kind", "step_id"))
            where = f"step {r['step_id']}" if r.get("step_id") else "run"
            lines.append(f"- `{r['kind']}` ({where}): {detail[:200]}")
        lines.append("")
    for rp in ex["replans"]:
        lines.append(f"**Re-plan after step {rp['after_step']}:** {rp['reasoning']}")
        lines.append("")
    if ex["injected_faults"]:
        lines.append(f"**Deliberately injected faults:** {', '.join(f'`{x}`' for x in ex['injected_faults'])}\n")

    rate = f"{g['verified_sources']}/{g['cited_sources']}" if g["cited_sources"] else "n/a"
    providers = s.get("llm_calls_by_provider") or {}
    split = f" ({', '.join(f'{p} {n}' for p, n in providers.items())})" if len(providers) > 1 else ""
    lines += [
        "## Run stats",
        f"- Steps passed: {s['steps_passed']}/{s['steps_total']} · tool calls: {s['tool_calls']} "
        f"({s['tool_failures']} failed) · recoveries: {s['recoveries']}",
        f"- LLM calls: {s['llm_calls']}{split} · tokens: {s['tokens']:,} · duration: {s['duration_s']}s",
        f"- Source grounding: {rate} cited URLs were returned by tools during the run",
    ]
    if g.get("window_days"):
        n = g.get("findings_outside_window", 0)
        lines.append(f"- Recency: {'all dated findings are' if not n else f'{n} finding(s) are not'} "
                     f"inside the requested {g['window_days']}-day window")
    return "\n".join(lines) + "\n"
