"""A rule-based stand-in for the LLM, so the agent can be demonstrated without an API key.

It speaks the OpenAI chat-completions interface and reads the same prompts the real model
gets, then answers with simple rules: search with the step's suggested tool, switch tools
when one fails, cite only URLs a tool returned, and rank what earlier steps found by date.
The control loop, tools, checks, retries, re-plans and grounding are the agent's real code;
only the model's decisions are canned, so the prose is plain and the ranking naive.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlsplit

from .config import Settings

MODEL_NAME = "demo (rule-based stand-in, no LLM)"

ALTERNATIVE = {"news_search": "web_search", "web_search": "news_search", "fetch_page": "web_search"}
STOP = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "from", "about", "me", "tell", "what", "whats",
        "is", "are", "was", "were", "latest", "last", "this", "week", "month", "year", "past", "recent", "top",
        "research", "summarize", "summarise", "find", "developments", "news", "with", "their", "compare",
        "give", "list", "show", "happened", "please", "most", "important", "key", "new", "by", "vs"}
# a search result as the tools print it:  [1] Title (Source | 2026-09-22)\n    url: https://...
RESULT = re.compile(r"^\[\d+\] (.+?)(?: \(([^()]*)\))?\n\s+url: (\S+)", re.M)
# a result as this stand-in lists it in a step's output:  - Title | Source, 2026-09-22 | https://...
LISTED = re.compile(r"^- (.+) \| ([^|]*) \| (https?://\S+)$", re.M)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def demo_settings(settings: Settings) -> Settings:
    """Settings for a demo run: no key, and no tokens-per-minute pacing (nothing is sent anywhere)."""
    return replace(settings, api_key="demo", cerebras_api_key="", model=MODEL_NAME, review_model="", tpm_limit=0)


def _role(system: str) -> str:
    if "Part of the current plan failed" in system:
        return "replan"
    for marker, role in (("PLANNER", "plan"), ("EXECUTOR", "exec"), ("REVIEWER", "review"),
                         ("FINAL REPORT", "report")):
        if marker in system:
            return role
    raise ValueError("demo client: unrecognised prompt")


def _line(text: str, label: str) -> str:
    m = re.search(rf"^{re.escape(label)}:? (.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def _date(meta: str) -> str:
    m = DATE.search(meta)
    return m.group(0) if m else ""


def _source(meta: str, url: str) -> str:
    """'Reported by Reuters on 2026-09-22.' from a listed item's meta, or the site's name when there is none."""
    name = meta.split(",")[0].strip()
    name = "" if name == "undated" or DATE.fullmatch(name) else name
    site = re.sub(r"^www\.", "", urlsplit(url).netloc)
    when = f" on {_date(meta)}" if _date(meta) else ""
    return f"Reported by {name}{when}." if name else f"Found on {site}{when}." if site else "Found by a web search."


def topic(goal: str, words: int = 5) -> str:
    """A short keyword query from the goal (the search tools work best with 2-6 words)."""
    goal = goal.split("\n(User clarification")[0]
    kept = [w for w in re.findall(r"[A-Za-z0-9][\w.+#-]*", goal) if w.lower() not in STOP and not w.isdigit()]
    return " ".join(kept[:words]) or goal[:60]


def _step(i: int, title: str, tool: str, instruction: str, criteria: str) -> dict:
    return {"id": i, "title": title, "tool_hint": tool, "instruction": instruction, "success_criteria": criteria}


class DemoClient:
    """Drop-in for ``openai.OpenAI()`` with rule-based replies; ``pause`` slows it down enough to watch."""

    def __init__(self, pause: float = 0.6):
        self.pause = pause
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        messages = kwargs["messages"]
        if self.pause:
            time.sleep(self.pause)
        # a JSON-repair turn: answer the original question again (the rules are deterministic)
        while len(messages) > 2 and messages[-1]["content"].startswith("Your reply was invalid"):
            messages = messages[:-2]
        reply = getattr(self, f"_{_role(messages[0]['content'])}")(messages)
        content = json.dumps(reply)
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=prompt_chars // 4,
                                                     completion_tokens=len(content) // 4))

    # ── planner ──────────────────────────────────────────────────────────────
    def _plan(self, messages: list[dict]) -> dict:
        goal, q = _line(messages[1]["content"], "GOAL"), topic(_line(messages[1]["content"], "GOAL"))
        tools = set(re.findall(r"\w+", _line(messages[0]["content"], "- tool_hint = the main tool for the "
                                                                     "step, one of")))
        steps = [
            _step(0, "Search recent news", "news_search", f"Search news from the last 7 days for '{q}'.",
                  "At least 3 dated results with URLs"),
            _step(0, "Read the top story", "fetch_page", "Fetch the most recent article found in step 1 and "
                  "pull out its key facts.", "Key facts from the article, with its URL"),
            _step(0, "Background web search", "web_search", f"Search the web for background on '{q}'.",
                  "At least 2 relevant results with URLs"),
            _step(0, "Rank findings", "none", "Rank the items found so far by recency and source coverage.",
                  "A ranked list that uses only items found in earlier steps"),
        ]
        steps = [s for s in steps if s["tool_hint"] == "none" or not tools or s["tool_hint"] in tools]
        for i, s in enumerate(steps, 1):
            s["id"] = i
        return {"understanding": f"Find and rank the most recent, well-sourced items about: {q}.",
                "ambiguities": [],
                "assumptions": ["Demo mode: a rule-based stand-in makes the model's decisions; the tools, "
                                "checks, retries and report pipeline are real.", "'Recent' means the last 7 days."],
                "steps": steps}

    def _replan(self, messages: list[dict]) -> dict:
        next_id = int(re.search(r"ids starting at (\d+)", messages[0]["content"]).group(1))
        user = messages[1]["content"]
        q = topic(_line(user, "GOAL"), words=3)
        failed = _line(user, "Why it failed") or "it produced nothing usable"
        return {"reasoning": f"The step failed because {failed}. A broader web search usually returns "
                             f"readable pages, so the remaining work switches to web_search.",
                "steps": [_step(next_id, "Broader web search", "web_search",
                                f"Search the web for '{q}' and collect the most relevant pages.",
                                "At least 2 relevant results with URLs"),
                          _step(next_id + 1, "Rank findings", "none", "Rank everything found so far by recency "
                                "and source coverage.", "A ranked list that uses only items found earlier")]}

    # ── executor ─────────────────────────────────────────────────────────────
    def _exec(self, messages: list[dict]) -> dict:
        task = messages[1]["content"]
        goal, hint = _line(task, "GOAL"), _line(task, "Suggested tool")
        retry = "REVIEWER FEEDBACK" in task
        calls = [a for a in (json.loads(m["content"]) for m in messages[2:] if m["role"] == "assistant")
                 if a.get("action") == "call_tool"]
        tried = [c["tool"] for c in calls]
        last = messages[-1]["content"] if len(messages) > 2 else ""

        if "no tool calls left" in last or "already made exactly this call" in last:
            return self._give_up(messages, "I ran out of tool calls for this step.")
        if hint == "none":
            return self._rank(task)
        if not calls:
            tool = ALTERNATIVE.get(hint, hint) if retry and hint != "fetch_page" else hint
            thought = (f"The reviewer rejected my last attempt, so I will try {tool} instead." if tool != hint
                       else "I will read the first article link an earlier step found." if tool == "fetch_page"
                       else f"The plan suggests {tool} for this step.")
            return self._call(tool, task, goal, calls, thought)

        tool = tried[-1]
        if f"{tool} FAILED" in last:
            error = last.split(": ", 1)[-1].split("\n")[0][:120]
            if tool == "fetch_page" and self._next_url(task, calls):
                return self._call("fetch_page", task, goal, calls,
                                  f"fetch_page failed ({error}); I will read the next article instead.")
            alt = ALTERNATIVE.get(tool)
            if alt and alt not in tried:
                return self._call(alt, task, goal, calls, f"{tool} failed ({error}); switching to {alt}.")
            return self._give_up(messages, f"Every tool I tried failed; the last error was: {error}")

        if tool == "fetch_page":
            url = calls[-1]["args"]["url"]
            body = " ".join(last.split("OK:\n", 1)[-1].split())[:600]
            return {"thought": "The page loaded, so I will pull out its opening facts and cite it.",
                    "action": "finish", "output": f"Read {url}. Key facts: {body}", "sources": [url]}
        results = RESULT.findall(last)
        if not results:
            alt = ALTERNATIVE.get(tool)
            if alt and alt not in tried:
                return self._call(alt, task, goal, calls, f"{tool} returned nothing usable; trying {alt}.")
            return self._give_up(messages, f"{tool} returned no results.")
        top = results[:5]
        listed = "\n".join(f"- {t} | {(meta or 'undated').replace(' | ', ', ')} | {u}" for t, meta, u in top)
        return {"thought": f"{tool} returned {len(results)} results, which meets the step's criteria; "
                           f"I will report the top {len(top)} with their URLs.",
                "action": "finish", "output": f"Found {len(results)} results:\n{listed}",
                "sources": [u for _, _, u in top]}

    def _call(self, tool: str, task: str, goal: str, calls: list[dict], thought: str) -> dict:
        if tool == "fetch_page":
            url = self._next_url(task, calls)
            if url is None:
                why = ("the only article links so far are Google News redirects, which don't open the article"
                       if "news.google.com" in task else "no earlier step gave me an article URL to read")
                tool, thought = "web_search", f"I can't read a page because {why}, so I will search the web instead."
            else:
                return {"thought": thought, "action": "call_tool", "tool": tool, "args": {"url": url}}
        broad = tool in [c["tool"] for c in calls] or "REVIEWER FEEDBACK" in task
        q = topic(goal, words=3 if broad else 5)
        args = {"query": q, "days": 7, "max_results": 6} if tool == "news_search" else {"query": q, "max_results": 5}
        return {"thought": thought, "action": "call_tool", "tool": tool, "args": args}

    @staticmethod
    def _next_url(task: str, calls: list[dict]) -> Optional[str]:
        fetched = {c.get("args", {}).get("url") for c in calls}
        for line in re.findall(r"^\s+sources: (.+)$", task, re.M):
            for url in (u.strip() for u in line.split(",")):
                if url not in fetched and "news.google.com" not in url:  # Google News links are redirects
                    return url
        return None

    @staticmethod
    def _give_up(messages: list[dict], reason: str) -> dict:
        seen = [u for m in messages if m["content"].startswith("OBSERVATION")
                for u in re.findall(r"url: (\S+)", m["content"])]
        return {"thought": "I cannot meet the criteria with the tools that work, so I will say what is missing.",
                "action": "finish", "output": f"Step incomplete. {reason}", "sources": seen[:3]}

    @staticmethod
    def _rank(task: str) -> dict:
        unique: dict[str, tuple] = {}
        for t, meta, u in LISTED.findall(task.split("Results of earlier steps:", 1)[-1]):
            unique.setdefault(u, (t, meta, u))  # one entry per URL, in the order found
        items = list(unique.values())
        if not items:
            return {"thought": "Earlier steps found nothing I can rank.", "action": "finish",
                    "output": "Nothing to rank: no earlier step produced results with URLs.", "sources": []}
        items.sort(key=lambda it: _date(it[1]), reverse=True)  # stable: undated items keep the order found
        ranked = "\n".join(f"{i}. {t} ({meta})" for i, (t, meta, _) in enumerate(items[:5], 1))
        if any(_date(meta) for _, meta, _ in items):
            thought, heading = (f"Ranking the {len(items)} distinct items from earlier steps, newest first.",
                                "Ranked by recency, newest first:")
        else:
            thought, heading = (f"None of the {len(items)} distinct items has a date, so I keep the order the "
                                f"searches returned them in.", "In the order the searches returned them (undated):")
        return {"thought": thought, "action": "finish", "output": f"{heading}\n{ranked}",
                "sources": [u for _, _, u in items[:5]]}

    # ── reviewer ─────────────────────────────────────────────────────────────
    @staticmethod
    def _review(messages: list[dict]) -> dict:
        user = messages[1]["content"]
        analysis = "Tool calls in this attempt:\n- (none)" in user
        if "Step incomplete" in user:
            return {"passed": False, "reason": "the result says the step is incomplete",
                    "suggestion": "use a different tool or a broader query"}
        if user.rstrip().endswith("- (none)") and not analysis:
            return {"passed": False, "reason": "the result cites no URLs from the tool results",
                    "suggestion": "use web_search with a broader query and cite the URLs it returns"}
        return {"passed": True, "reason": "a ranking of earlier results" if analysis
                else "concrete results with URLs returned by the tools"}

    # ── report ───────────────────────────────────────────────────────────────
    @staticmethod
    def _report(messages: list[dict]) -> dict:
        user = messages[1]["content"]
        goal = _line(user, "GOAL")
        m = re.search(r"\btop (\d+)", goal, re.I)
        n = int(m.group(1)) if m else 3
        found, seen = [], set()
        for block in user.split("### Step ")[1:]:
            citable = set(re.findall(r"https?://[^\s,]+", block.rsplit("Sources:", 1)[-1]))
            for title, meta, url in LISTED.findall(block):
                if url in citable and url not in seen:  # only URLs the agent says a tool returned
                    seen.add(url)
                    found.append((title, meta, url))
        found.sort(key=lambda it: _date(it[1]), reverse=True)
        dated = any(_date(meta) for _, meta, _ in found)
        findings = [{"title": t, "date": _date(meta) or None, "sources": [u],
                     "summary": f"{_source(meta, u)} The demo stand-in lists headlines; it does not summarise "
                                f"the article text.",
                     "why_it_matters": "One of the most recent items the tools returned for this goal." if dated
                                       else "One of the first results the searches returned for this goal."}
                    for t, meta, u in found[:n]]
        if not findings:
            findings = [{"title": "No usable results", "summary": "No step returned results that could be cited.",
                         "sources": []}]
        picked = (f"the {len(findings)} newest are listed below. They were chosen by date alone" if dated
                  else f"none had a date, so the first {len(findings)} found are listed below. They were chosen "
                       f"by search order alone")
        return {"title": f"Demo report: {topic(goal)}",
                "executive_summary": f"The agent searched live sources and kept {len(found)} cited items; "
                                     f"{picked}, because demo mode has no language model to judge importance.",
                "findings": findings,
                "limitations": ["Demo mode: headlines were picked by date or search order, not read and judged "
                                "by a model."],
                "confidence": "low",
                "confidence_reason": "A rule-based stand-in chose the findings; nothing was checked for significance."}
