"""Tool registry, calculator, fault injection and parsing tests (no network)."""
import socket
import time

import httpx
import pytest

from research_agent.failures import TARGETS, FailureInjector, FaultSpec
from research_agent.tools import Param, Tool, ToolOutput, ToolRegistry, TransientToolError
from research_agent.tools import search as search_mod
from research_agent.tools.base import ToolContext, ToolError, validate_args
from research_agent.tools.calculator import evaluate
from research_agent.tools import fetch as fetch_mod
from research_agent.tools.fetch import _check_url, clean_url, extract_text, split_urls


def registry(*tools, faults=(), **kw):
    reg = ToolRegistry(FailureInjector.from_strings(list(faults)), backoff_base=0, sleep=lambda _: None, **kw)
    for t in tools:
        reg.register(t)
    return reg


def echo_tool(func=None, **params):
    params = params or {"q": Param("string", "q", required=True)}
    return Tool("echo", "echo", params, func or (lambda a, c: ToolOutput(text=f"echo {a}")))


# ── calculator ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", 14), ("(1250 - 980) / 980 * 100", pytest.approx(27.551, rel=1e-3)),
    ("1,250,000 * 2", 2_500_000), ("sqrt(16) + max(1, 5)", 9), ("2 ^ 10", 1024),
    ("15% * 200", 30), ("10 % 3", 1), ("-5 + pi", pytest.approx(-1.8584, rel=1e-3)),
])
def test_calculator_valid(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "1/0", "2 ** 100000", "a + 1", "(1,2)", "1 +",
                                  "((2 ** 1000) ** 1000) ** 1000", "0.1 ** -1000", "exp(1000)"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


# ── argument validation ───────────────────────────────────────────────────────
def test_validate_args_coerces_clamps_and_defaults():
    tool = echo_tool(q=Param("string", "q", required=True),
                     n=Param("integer", "n", default=5, minimum=1, maximum=10))
    assert validate_args(tool, {"q": "hi", "n": "50", "junk": 1}) == {"q": "hi", "n": 10}
    assert validate_args(tool, {"q": "hi"}) == {"q": "hi", "n": 5}
    with pytest.raises(ToolError, match="missing required"):
        validate_args(tool, {"n": 3})
    with pytest.raises(ToolError, match="should be integer"):
        validate_args(tool, {"q": "x", "n": "lots"})


# ── registry: errors, retries, timeouts ───────────────────────────────────────
def test_unknown_tool_and_bad_args_return_errors_not_exceptions():
    reg = registry(echo_tool())
    assert not reg.execute("nope", {}).ok
    res = reg.execute("echo", {})
    assert not res.ok and "missing required" in res.error


def test_transient_error_is_retried_then_succeeds():
    calls = []

    def flaky(a, c):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectTimeout("slow")
        return ToolOutput(text="ok")

    events = []
    reg = registry(echo_tool(flaky), emit=lambda k, **d: events.append(k))
    res = reg.execute("echo", {"q": "x"})
    assert res.ok and res.attempts == 2 and events == ["tool_retry"]


def test_permanent_error_is_not_retried():
    def broken(a, c):
        raise ToolError("404 not found")

    res = registry(echo_tool(broken)).execute("echo", {"q": "x"})
    assert not res.ok and res.attempts == 1


def test_http_status_classification():
    req = httpx.Request("GET", "https://x.test/")
    for code, transient in ((404, False), (503, True), (429, True)):
        def f(a, c, code=code):
            raise httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, request=req))
        res = registry(echo_tool(f)).execute("echo", {"q": "x"})
        assert (res.attempts == 3) is transient, code


def test_unexpected_exception_in_tool_is_contained():
    def bug(a, c):
        raise KeyError("oops")

    res = registry(echo_tool(bug)).execute("echo", {"q": "x"})
    assert not res.ok and "KeyError" in res.error


def test_timeout_is_enforced():
    def slow(a, c):
        time.sleep(1.0)
        return ToolOutput(text="late")

    reg = registry(echo_tool(slow), timeout=0.1, retries=0)
    t0 = time.monotonic()
    res = reg.execute("echo", {"q": "x"})
    assert not res.ok and "timed out" in res.error
    assert time.monotonic() - t0 < 0.8


# ── fault injection ───────────────────────────────────────────────────────────
def test_fault_spec_parsing():
    s = FaultSpec.parse("news_search:timeout:2")
    assert (s.target, s.mode, s.remaining) == ("news_search", "timeout", 2)
    assert FaultSpec.parse("ddgs:error:*").remaining is None
    for bad in ("nocolon", "x:explode", "fetch_page:bad_json", "ddgs:error:0", "ddgs:error:-1", "ddgs:error:two"):
        with pytest.raises(ValueError):
            FaultSpec.parse(bad)


def test_a_misspelt_fault_target_is_refused_when_the_targets_are_known():
    with pytest.raises(ValueError, match="unknown fault target 'news_serch'"):
        FailureInjector.from_strings(["news_serch:error"], targets=TARGETS)
    assert FailureInjector.from_strings(["ddgs:error:*"], targets=TARGETS).describe() == ["ddgs:error:*"]


@pytest.mark.parametrize("mode", ["timeout", "error", "bad_response"])
def test_injected_fault_once_is_recovered_by_retry(mode):
    res = registry(echo_tool(), faults=[f"echo:{mode}:1"]).execute("echo", {"q": "x"})
    assert res.ok and res.attempts == 2


def test_injected_fault_always_exhausts_retries():
    res = registry(echo_tool(), faults=["echo:timeout:*"]).execute("echo", {"q": "x"})
    assert not res.ok and res.attempts == 3 and "[injected]" in res.error


def test_injected_empty_result():
    res = registry(echo_tool(), faults=["echo:empty"]).execute("echo", {"q": "x"})
    assert res.ok and res.text == "No results found." and res.sources == []


# ── search tool: provider fallback + date filtering ──────────────────────────
def test_news_search_falls_back_and_filters_old_items(monkeypatch):
    import datetime as dt

    today = dt.date.today()

    def ddgs_down(args):
        raise RuntimeError("ratelimited")

    def gnews(args):
        return [
            {"title": "Fresh story", "url": "https://a.test/1", "source": "A", "date": today.isoformat(), "snippet": ""},
            {"title": "Fresh story", "url": "https://a.test/dup", "source": "A", "date": today.isoformat(), "snippet": ""},
            {"title": "Old story", "url": "https://a.test/2", "source": "B",
             "date": (today - dt.timedelta(days=30)).isoformat(), "snippet": ""},
        ]

    monkeypatch.setattr(search_mod, "_ddgs_news", ddgs_down)
    monkeypatch.setattr(search_mod, "_bing_news", lambda args: [])
    monkeypatch.setattr(search_mod, "_google_news", gnews)
    events = []
    ctx = ToolContext(FailureInjector(), emit=lambda k, **d: events.append((k, d)))
    out = search_mod._news({"query": "x", "days": 7, "max_results": 5}, ctx)
    assert out.sources == ["https://a.test/1"]  # duplicate and stale items dropped
    assert [(k, d["failed"], d["next"]) for k, d in events] == [
        ("provider_fallback", "ddgs", "bing_news"), ("provider_fallback", "bing_news", "google_news")]


BING_RSS = b"""<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"
xmlns:News="https://www.bing.com/news/search?q=AI+agents&amp;format=rss"><channel><title>AI agents - BingNews</title>
<item><title>Rogue AI Agents Need Guardrails</title>
<link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;aid=&amp;url=https%3a%2f%2fa.test%2fnews%2f42%3fx%3d1&amp;c=86</link>
<description>A short summary of the story.</description><pubDate>Wed, 23 Sep 2026 07:15:00 GMT</pubDate>
<News:Source>Example Times</News:Source></item></channel></rss>"""


def test_bing_news_unwraps_click_redirects_to_publisher_urls(monkeypatch):
    seen = {}

    def fake_get(url, params=None, **kw):
        seen.update(params)
        return httpx.Response(200, content=BING_RSS, request=httpx.Request("GET", url))

    monkeypatch.setattr(search_mod.httpx, "get", fake_get)
    [item] = search_mod._bing_news({"query": "AI agents", "days": 7, "max_results": 5})
    assert item == {"title": "Rogue AI Agents Need Guardrails", "url": "https://a.test/news/42?x=1",
                    "source": "Example Times", "date": "Wed, 23 Sep 2026 07:15:00 GMT",
                    "snippet": "A short summary of the story."}
    assert seen == {"q": "AI agents", "format": "rss", "qft": 'interval="8"'}  # 8 = past week


def test_web_search_drops_ad_redirects(monkeypatch):
    ads = [{"title": "Sponsored", "url": "https://www.bing.com/aclick?ld=e8abc&u=aHR0cHM6Ly9", "snippet": ""},
           {"title": "Also an ad", "url": "https://duckduckgo.com/y.js?ad_domain=x.test", "snippet": ""}]
    monkeypatch.setattr(search_mod, "_ddgs_text", lambda args: ads + [{"title": "Real", "url": "https://a.test/p"}])
    out = search_mod._web({"query": "x", "max_results": 5}, ToolContext(FailureInjector()))
    assert out.sources == ["https://a.test/p"]

    # a page of nothing but ads counts as no results, so the next provider is tried
    monkeypatch.setattr(search_mod, "_ddgs_text", lambda args: ads)
    monkeypatch.setattr(search_mod, "_wikipedia", lambda args: [{"title": "W", "url": "https://en.wikipedia.org/wiki/W"}])
    out = search_mod._web({"query": "x", "max_results": 5}, ToolContext(FailureInjector()))
    assert out.sources == ["https://en.wikipedia.org/wiki/W"]


def test_all_providers_failing_is_transient(monkeypatch):
    def down(args):
        raise RuntimeError("down")

    monkeypatch.setattr(search_mod, "_ddgs_text", down)
    monkeypatch.setattr(search_mod, "_wikipedia", down)
    with pytest.raises(TransientToolError, match="all providers failed"):
        search_mod._web({"query": "x", "max_results": 3}, ToolContext(FailureInjector()))


def test_a_thin_news_answer_is_topped_up_from_the_next_provider(monkeypatch):
    import datetime as dt

    today, old = dt.date.today(), dt.date.today() - dt.timedelta(days=40)

    def story(n, day):
        return {"title": f"Story {n}", "url": f"https://a.test/{n}", "source": "A", "date": day.isoformat()}

    monkeypatch.setattr(search_mod, "_ddgs_news", lambda args: [story(1, today), story(2, old), story(3, old)])
    monkeypatch.setattr(search_mod, "_bing_news", lambda args: [story(1, today), story(4, today), story(5, today)])
    monkeypatch.setattr(search_mod, "_google_news", lambda args: pytest.fail("three fresh results are enough"))
    events = []
    ctx = ToolContext(FailureInjector(), emit=lambda k, **d: events.append(d))
    out = search_mod._news({"query": "x", "days": 7, "max_results": 8}, ctx)
    assert sorted(out.sources) == ["https://a.test/1", "https://a.test/4", "https://a.test/5"]
    assert [(d["failed"], d["next"], d["error"]) for d in events] == [
        ("ddgs", "bing_news", "ddgs: only 1 result from the last 7 days")]

    def down(args):
        raise RuntimeError("down")

    monkeypatch.setattr(search_mod, "_bing_news", down)
    monkeypatch.setattr(search_mod, "_google_news", down)
    out = search_mod._news({"query": "x", "days": 7, "max_results": 8}, ctx)
    assert out.sources == ["https://a.test/1"]  # a failed top-up keeps what the first provider found

    events.clear()
    monkeypatch.setattr(search_mod, "_ddgs_news", lambda args: [story(2, old)])
    monkeypatch.setattr(search_mod, "_bing_news", lambda args: [story(4, today), story(5, today), story(6, today)])
    search_mod._news({"query": "x", "days": 7, "max_results": 8}, ctx)
    assert events[0]["error"] == "ddgs: no results from the last 7 days"


def test_an_empty_answer_is_not_an_outage(monkeypatch):
    def nothing(args):
        raise RuntimeError("No results found.")  # how DuckDuckGo reports an empty answer

    monkeypatch.setattr(search_mod, "_ddgs_news", nothing)
    monkeypatch.setattr(search_mod, "_bing_news", lambda args: [])
    monkeypatch.setattr(search_mod, "_google_news", lambda args: [])
    out = search_mod._news({"query": "x", "days": 7, "max_results": 5}, ToolContext(FailureInjector()))
    assert out.text.startswith("No news found for 'x'") and out.sources == []  # an answer, so no retries


def test_news_results_warn_about_links_fetch_page_cannot_read(monkeypatch):
    import datetime as dt

    story = {"title": "Story", "url": "https://www.msn.com/en-us/news/ar-AA1", "source": "Reuters",
             "date": dt.date.today().isoformat(), "snippet": ""}
    monkeypatch.setattr(search_mod, "_ddgs_news", lambda args: [story])
    monkeypatch.setattr(search_mod, "_bing_news", lambda args: [])
    monkeypatch.setattr(search_mod, "_google_news", lambda args: [])
    out = search_mod._news({"query": "x", "days": 7, "max_results": 5}, ToolContext(FailureInjector()))
    assert out.text.startswith("1 news result for 'x'") and "msn.com pages are drawn by JavaScript" in out.text


# ── fetch helpers ─────────────────────────────────────────────────────────────
def test_extract_text_prefers_article_and_strips_chrome():
    html = ("<html><head><title>T</title><script>evil()</script></head><body><nav>menu menu menu menu menu menu "
            "menu menu</nav><article><p>" + "Real paragraph content here. " * 5 + "</p></article></body></html>")
    title, text, day = extract_text(html)
    assert title == "T" and "Real paragraph" in text and "menu" not in text and "evil" not in text and day == ""


@pytest.mark.parametrize("head,body,day", [
    ('<meta property="article:published_time" content="2026-09-25T08:00:00Z">', "", "2026-09-25"),
    ('<meta name="parsely-pub-date" content="2026-09-24">', "", "2026-09-24"),
    ('<script type="application/ld+json">{"@type": "NewsArticle", "datePublished": "2026-09-23T10:00"}</script>',
     "", "2026-09-23"),
    ("", '<article><header><time datetime="2026-09-22T09:00">Sep 22</time></header></article>', "2026-09-22"),
    ('<meta name="date" content="September 21">', "", ""),  # not ISO: better no date than a wrong one
])
def test_extract_text_finds_the_publication_day(head, body, day):
    html = f"<html><head><title>T</title>{head}</head><body>{body}<p>{'Some article text. ' * 5}</p></body></html>"
    assert extract_text(html)[2] == day


@pytest.mark.parametrize("url", [
    "ftp://x.test/a", "http://localhost:5000", "http://127.0.0.1/", "http://10.0.0.5/x", "http://printer.local/",
    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://0.0.0.0/", "http://169.254.169.254/latest/meta-data/",
    "http://224.0.0.1/",
])
def test_fetch_rejects_unsafe_urls(url):
    with pytest.raises(ToolError):
        _check_url(url)


def test_a_name_that_resolves_inside_the_network_is_refused(monkeypatch):
    monkeypatch.setattr(fetch_mod, "_resolve", lambda host: ["93.184.216.34", "192.168.1.10"])
    with pytest.raises(ToolError, match="private"):
        _check_url("https://looks-public.example/")
    monkeypatch.setattr(fetch_mod, "_resolve", lambda host: ["93.184.216.34", "2606:4700::1111", "::ffff:8.8.8.8"])
    _check_url("https://example.com/")  # public addresses pass


@pytest.mark.parametrize("raw,clean", [
    ("httpshttps://a.test/x", "https://a.test/x"), ("https://https://a.test/x", "https://a.test/x"),
    ("<https://a.test/x>.", "https://a.test/x"),
    ("https://en.wikipedia.org/wiki/Python_(language)", "https://en.wikipedia.org/wiki/Python_(language)"),
    ("https://r.jina.ai/http://a.test/x", "http://a.test/x"), ("https://r.jina.ai/a.test/x", "https://a.test/x"),
])
def test_clean_url_fixes_llm_copy_mistakes(raw, clean):
    assert clean_url(raw) == clean


@pytest.mark.parametrize("raw,urls", [
    ("https://a.test/x", ["https://a.test/x"]),
    ("https://a.test/x, https://b.test/y https://a.test/x", ["https://a.test/x", "https://b.test/y"]),
    ("['https://a.test/x', 'https://b.test/y']", ["https://a.test/x", "https://b.test/y"]),  # a list, stringified
    ("https://en.wikipedia.org/wiki/Python_(language)", ["https://en.wikipedia.org/wiki/Python_(language)"]),
    ("not a url", ["not a url"]),  # left for _check_url to reject
])
def test_split_urls(raw, urls):
    assert split_urls(raw) == urls


ARTICLE = ("<html><head><title>Page {n}</title></head><body><article><p>"
           + "Plenty of readable article text in this paragraph. " * 10 + "</p></article></body></html>")


@pytest.fixture
def web(monkeypatch):
    """Serve fetch_page from ``handler(request)`` instead of the network; *.test names get a public address."""
    real_resolve = fetch_mod._resolve
    monkeypatch.setattr(fetch_mod, "_resolve",
                        lambda host: ["93.184.216.34"] if host.endswith(".test") else real_resolve(host))
    return lambda handler: monkeypatch.setattr(fetch_mod, "_TRANSPORT", httpx.MockTransport(handler))


def fake_pages(request):
    url = str(request.url)
    if "blocked" in url:
        return httpx.Response(403)
    return httpx.Response(200, text=ARTICLE.format(n=url[-1]), headers={"content-type": "text/html"})


def test_fetch_page_reads_up_to_three_pages_in_one_call(web):
    web(fake_pages)
    out = fetch_mod._fetch({"url": "https://a.test/1 https://a.test/blocked https://a.test/2 https://a.test/3",
                            "max_chars": 500}, ToolContext(FailureInjector()))
    assert out.text.startswith("Read 2 of 3 pages (1 failed, listed last)")
    assert "[page 1/2] TITLE: Page 1" in out.text and "[page 2/2] TITLE: Page 2" in out.text
    assert out.text.endswith("FAILED https://a.test/blocked: HTTP 403")  # one bad page doesn't sink the call
    assert out.sources == ["https://a.test/1", "https://a.test/2"] and "Page 3" not in out.text


def test_a_page_that_stalls_is_not_retried(web):
    calls = []

    def stall(request):
        calls.append(request.url)
        raise httpx.ReadTimeout("The read operation timed out", request=request)

    web(stall)
    res = registry(fetch_mod.fetch_page).execute("fetch_page", {"url": "https://slow.test/a"})
    assert not res.ok and res.attempts == 1 and len(calls) == 1  # a retry would only wait another 10 s
    assert res.error.startswith("no answer from slow.test within 10 s") and "search snippet" in res.error


def test_fetch_page_fails_only_when_every_page_fails(web):
    web(fake_pages)
    with pytest.raises(httpx.HTTPStatusError):  # a real error, so the registry classifies and retries it
        fetch_mod._fetch({"url": "https://a.test/blocked https://b.test/blocked", "max_chars": 500},
                         ToolContext(FailureInjector()))


def test_a_redirect_into_the_local_network_is_never_followed(web):
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1:8765/api/config"})

    web(handler)
    with pytest.raises(ToolError, match="refusing"):
        fetch_mod._fetch_one("https://a.test/start", 500)
    assert requested == ["https://a.test/start"]


def test_a_huge_page_is_cut_off_while_downloading(web, monkeypatch):
    sent = []

    def endless():
        while True:
            sent.append(1)
            yield b"<p>" + b"x" * 1000 + b"</p>"

    web(lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=endless()))
    monkeypatch.setattr(fetch_mod, "MAX_DOWNLOAD_BYTES", 5_000)
    with pytest.raises(ToolError, match="too large"):
        fetch_mod._fetch_one("https://big.test/a", 500)
    assert len(sent) < 10  # stopped at the cap instead of reading everything


def test_an_unknown_host_fails_once_without_retries(monkeypatch):
    def no_such_host(host):
        raise socket.gaierror(11001, "getaddrinfo failed")

    monkeypatch.setattr(fetch_mod, "_resolve", no_such_host)
    res = registry(fetch_mod.fetch_page).execute("fetch_page", {"url": "https://made-up-news.example/story"})
    assert not res.ok and res.attempts == 1 and "could not resolve made-up-news.example" in res.error
