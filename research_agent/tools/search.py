"""Live search tools with provider fallback.

news_search: DuckDuckGo News -> Bing News RSS -> Google News RSS
web_search:  DuckDuckGo Web  -> Wikipedia search API

Every provider is free and needs no API key. If the primary provider fails
(rate limit, timeout, outage), the tool automatically tries the next one and
emits a ``provider_fallback`` event so the recovery is visible in the trace.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Callable, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from .base import Param, Tool, ToolContext, ToolOutput, TransientToolError

# Wikimedia rejects user agents without a contact URL/e-mail (policy: meta.wikimedia.org/wiki/User-Agent_policy).
USER_AGENT = f"ResearchAgent/1.0 ({os.environ.get('AGENT_CONTACT', 'https://github.com/research-agent')}) httpx"
HTTP_TIMEOUT = 12.0

Provider = Callable[[dict], list[dict]]


# ── helpers ───────────────────────────────────────────────────────────────────
def _parse_date(raw: Optional[str]) -> Optional[dt.date]:
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(raw).date()
    except (TypeError, ValueError):
        return None


def _timelimit(days: int) -> str:
    return "d" if days <= 1 else "w" if days <= 7 else "m" if days <= 31 else "y"


# DuckDuckGo mixes sponsored results into web results; their URLs are ad-click redirects, not pages.
_AD_URL = re.compile(r"^https?://(?:[\w-]+\.)*(?:bing\.com/aclick|duckduckgo\.com/y\.js)", re.I)


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = (it.get("title") or "").strip().lower()[:80]
        if key and key not in seen and it.get("url"):
            seen.add(key)
            out.append(it)
    return out


def _count(items: list, noun: str) -> str:
    return f"{len(items)} {noun}{'' if len(items) == 1 else 's'}"


def _format(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        meta = " | ".join(x for x in (it.get("source"), it.get("date")) if x)
        lines.append(f"[{i}] {it['title']}" + (f" ({meta})" if meta else ""))
        lines.append(f"    url: {it['url']}")
        if it.get("snippet"):
            lines.append(f"    {it['snippet'][:240]}")
    return "\n".join(lines)


def _run_providers(tool: str, providers: list[tuple[str, Provider]], args: dict,
                   ctx: ToolContext) -> tuple[int, list[dict]]:
    """Try providers in order; return the index of the one that answered and its items.

    Every provider coming back empty is an answer (no items); raise only if *every* provider errored.
    """
    errors = []
    for i, (name, provider) in enumerate(providers):
        try:
            ctx.injector.before_call(name)
            items = [it for it in provider(args) or [] if not _AD_URL.match(it.get("url") or "")]
            if items:
                return i, items
            errors.append(f"{name}: 0 results")
        except Exception as e:  # noqa: BLE001 - any provider failure triggers fallback
            # DuckDuckGo raises "No results found." for an empty answer; that is not an outage worth retrying
            errors.append(f"{name}: 0 results" if "no results" in str(e).lower()
                          else f"{name}: {type(e).__name__}: {str(e)[:120]}")
        if i + 1 < len(providers):
            ctx.emit("provider_fallback", tool=tool, failed=name, next=providers[i + 1][0], error=errors[-1])
    if all("0 results" in e for e in errors):
        return len(providers) - 1, []
    raise TransientToolError(f"all providers failed -> {'; '.join(errors)}")


# ── news_search ───────────────────────────────────────────────────────────────
def _ddgs_news(args: dict) -> list[dict]:
    from ddgs import DDGS

    raw = DDGS(timeout=HTTP_TIMEOUT).news(args["query"], timelimit=_timelimit(args["days"]),
                                          max_results=args["max_results"] * 2)
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "source": r.get("source", ""),
             "date": r.get("date", ""), "snippet": r.get("body", "")} for r in raw or []]


def _rss(content: bytes, provider: str) -> ET.Element:
    try:
        return ET.fromstring(content)
    except ET.ParseError as e:
        raise TransientToolError(f"{provider} returned invalid XML: {e}")


def _bing_news(args: dict) -> list[dict]:
    params = {"q": args["query"], "format": "rss"}
    days = args["days"]
    if days <= 31:  # Bing's freshness filter: 7 = past day, 8 = past week, 9 = past month
        params["qft"] = f'interval="{7 if days <= 1 else 8 if days <= 7 else 9}"'
    r = httpx.get("https://www.bing.com/news/search", params=params, timeout=HTTP_TIMEOUT,
                  headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    r.raise_for_status()
    items = []
    for node in _rss(r.content, "Bing News").iter("item"):
        link = (node.findtext("link") or "").strip()
        # links are click-tracking redirects that carry the publisher's URL in ?url=
        url = parse_qs(urlparse(link).query).get("url", [link])[0]
        # <News:Source>, in a namespace whose URI is the query URL itself
        source = next((c.text or "" for c in node if c.tag.endswith("}Source")), "")
        items.append({
            "title": (node.findtext("title") or "").strip(),
            "url": url,
            "source": source.strip(),
            "date": node.findtext("pubDate") or "",
            "snippet": (node.findtext("description") or "").strip(),
        })
    return items


def _google_news(args: dict) -> list[dict]:
    q = quote_plus(f"{args['query']} when:{args['days']}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    r = httpx.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    r.raise_for_status()
    root = _rss(r.content, "Google News")
    items = []
    for node in root.iter("item"):
        src = node.find("source")
        items.append({
            "title": (node.findtext("title") or "").strip(),
            "url": (node.findtext("link") or "").strip(),
            "source": src.text if src is not None else "",
            "date": node.findtext("pubDate") or "",
            "snippet": "",
        })
    return items


def _fresh(items: list[dict], cutoff: dt.date) -> list[dict]:
    """Items dated on or after ``cutoff`` (undated ones are kept), with ISO dates."""
    out = []
    for it in items:
        d = _parse_date(it.get("date"))
        if d is not None and d < cutoff:
            continue  # provider returned something outside the requested window
        out.append({**it, "date": d.isoformat() if d else ""})
    return out


def _news(args: dict, ctx: ToolContext) -> ToolOutput:
    providers = [("ddgs", _ddgs_news), ("bing_news", _bing_news), ("google_news", _google_news)]
    cutoff = dt.date.today() - dt.timedelta(days=args["days"] + 1)  # +1 day for timezone slack
    want = min(3, args["max_results"])
    fresh: list[dict] = []
    while providers:
        try:
            used, items = _run_providers("news_search", providers, args, ctx)
        except TransientToolError:
            if fresh:
                break  # the top-up failed; what we have is still an answer
            raise
        fresh = _dedupe(fresh + _fresh(items, cutoff))
        name, providers = providers[used][0], providers[used + 1:]
        if len(fresh) >= want or not providers:
            break
        # DuckDuckGo sometimes answers a "past week" query with mostly older stories; a model handed one
        # item for a "top 5" pads the list from memory, so top up from the next provider instead
        ctx.emit("provider_fallback", tool="news_search", failed=name, next=providers[0][0],
                 error=f"{name}: {'only ' + _count(fresh, 'result') if fresh else 'no results'} "
                       f"from the last {args['days']} days")
    fresh = sorted(fresh, key=lambda x: x["date"], reverse=True)[: args["max_results"]]
    if not fresh:
        return ToolOutput(text=f"No news found for '{args['query']}' in the last {args['days']} days. "
                               f"Try a broader or differently worded query.", data=[])
    header = f"{_count(fresh, 'news result')} for '{args['query']}' (last {args['days']} days, newest first):\n"
    if any(it["url"].startswith("https://news.google.com/") for it in fresh):
        header += ("(news.google.com links redirect to the publisher: cite them as they are. fetch_page "
                   "can't open them; to read one, web_search its headline.)\n")
    if any(urlparse(it["url"]).netloc.endswith("msn.com") for it in fresh):
        header += ("(msn.com pages are drawn by JavaScript, so fetch_page reads nothing there: cite them as they "
                   "are, and web_search the headline to read the publisher's own copy.)\n")
    return ToolOutput(text=header + _format(fresh), data=fresh, sources=[it["url"] for it in fresh])


news_search = Tool(
    name="news_search",
    description="Search recent NEWS articles (dated). Best for 'latest', 'this week', 'recent developments'.",
    params={
        "query": Param("string", "short keyword query, 2-6 words works best", required=True),
        "days": Param("integer", "only articles from the last N days", default=7, minimum=1, maximum=365),
        "max_results": Param("integer", "number of results", default=8, minimum=1, maximum=15),
    },
    func=_news,
)


# ── web_search ────────────────────────────────────────────────────────────────
def _ddgs_text(args: dict) -> list[dict]:
    from ddgs import DDGS

    raw = DDGS(timeout=HTTP_TIMEOUT).text(args["query"], max_results=args["max_results"])
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in raw or []]


def _wikipedia(args: dict) -> list[dict]:
    r = httpx.get("https://en.wikipedia.org/w/api.php", timeout=HTTP_TIMEOUT,
                  headers={"User-Agent": USER_AGENT},
                  params={"action": "query", "list": "search", "srsearch": args["query"],
                          "srlimit": args["max_results"], "format": "json"})
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    return [{"title": h["title"],
             "url": "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
             "source": "Wikipedia",
             "snippet": re.sub(r"<[^>]+>", "", h.get("snippet", ""))} for h in hits]


def _web(args: dict, ctx: ToolContext) -> ToolOutput:
    _, items = _run_providers("web_search", [("ddgs", _ddgs_text), ("wikipedia", _wikipedia)], args, ctx)
    items = _dedupe(items)[: args["max_results"]]
    if not items:
        return ToolOutput(text=f"No web results for '{args['query']}'. Try different keywords.", data=[])
    return ToolOutput(text=f"{_count(items, 'web result')} for '{args['query']}':\n" + _format(items),
                      data=items, sources=[it["url"] for it in items])


web_search = Tool(
    name="web_search",
    description="General web search (not date-filtered). Good for background, definitions, official pages.",
    params={
        "query": Param("string", "search query", required=True),
        "max_results": Param("integer", "number of results", default=6, minimum=1, maximum=10),
    },
    func=_web,
)
