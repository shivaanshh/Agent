"""fetch_page: download a web page and extract its readable text."""
from __future__ import annotations

import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .base import Param, Tool, ToolContext, ToolError, ToolOutput
from .search import USER_AGENT

MIN_TEXT_CHARS = 300
MAX_DOWNLOAD_BYTES = 3_000_000
MAX_PAGES = 3  # pages per call: reading several in one call saves an LLM turn per page
MAX_REDIRECTS = 5
FETCH_TIMEOUT = 10.0
_TRANSPORT: Optional[httpx.BaseTransport] = None  # tests swap in an httpx.MockTransport
_URL = re.compile(r"https?://[^\s,'\"<>\[\]]+", re.I)
_READER_PROXY = re.compile(r"^https?://r\.jina\.ai/+", re.I)
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_KEYS = {"article:published_time", "og:published_time", "datepublished", "date", "pubdate", "publishdate",
              "parsely-pub-date", "sailthru.date", "dc.date", "dcterms.date"}
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 " + USER_AGENT)


def clean_url(url: str) -> str:
    """Undo common LLM copy mistakes: wrapping quotes/brackets, doubled schemes (``httpshttps://``) and
    reader proxies (``r.jina.ai/<url>``), which refuse requests without an API key: the page is read directly."""
    url = url.strip().rstrip(".,;").strip("<>'\"").rstrip(".,;")
    if _READER_PROXY.match(url):
        url = _READER_PROXY.sub("", url)
        url = url if re.match(r"https?:", url, re.I) else "https://" + url
    return re.sub(r"^(?:https?:?/*)+(?=https?://)", "", url, flags=re.I)


def _resolve(host: str) -> list[str]:
    """Every address ``host`` resolves to; an IP literal is its own answer."""
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        return [info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]


def _is_public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 is loopback too
    return ip.is_global and not ip.is_multicast


def _check_url(url: str) -> None:
    """Refuse URLs that reach this machine or its network (SSRF): URLs come from search results and the model.

    Resolved addresses are checked, not just the name, and the check runs again on every redirect. httpx
    resolves once more when it connects, so a DNS server that changes its answer in between (rebinding)
    is not covered.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError(f"not a valid http(s) URL: {url!r}")
    host = parsed.hostname
    if host == "localhost" or host.endswith((".local", ".localhost")):
        raise ToolError("refusing to fetch local addresses")
    try:
        addresses = _resolve(host)
    except (OSError, UnicodeError):
        raise ToolError(f"could not resolve {host}: check the URL") from None
    if not addresses or not all(_is_public(a) for a in addresses):
        raise ToolError("refusing to fetch local or private network addresses")


def published(soup: BeautifulSoup, html: str) -> str:
    """The page's publication day (YYYY-MM-DD) from its metadata, or "" if it doesn't say."""
    for meta in soup.find_all("meta"):
        key = (meta.get("property") or meta.get("name") or meta.get("itemprop") or "").lower()
        m = _DAY.match(meta.get("content") or "") if key in _DATE_KEYS else None
        if m:
            return m.group(0)
    m = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', html)  # JSON-LD
    if m:
        return m.group(1)
    t = (soup.find("article") or soup).find("time", attrs={"datetime": True})
    m = _DAY.match(t["datetime"]) if t else None
    return m.group(0) if m else ""


def extract_text(html: str) -> tuple[str, str, str]:
    """Title, main readable text and publication day ("" if unknown) of an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    day = published(soup, html)  # before the page chrome goes: dates often sit in <header> or <script>
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in root.find_all(["p", "h1", "h2", "h3", "li"])]
    text = "\n".join(p for p in paragraphs if len(p) > 40) or root.get_text(" ", strip=True)
    return title, re.sub(r"[ \t]+", " ", text).strip(), day


def split_urls(raw: str) -> list[str]:
    """The URLs in a ``url`` argument: one, several separated by spaces or commas, or a stringified list."""
    urls = [clean_url(u) for u in _URL.findall(raw)] or [clean_url(raw)]
    return list(dict.fromkeys(urls))


def _fetch(args: dict, ctx: ToolContext) -> ToolOutput:
    urls = split_urls(args["url"])
    if len(urls) == 1:
        return _fetch_one(urls[0], args["max_chars"])
    urls = urls[:MAX_PAGES]
    with ThreadPoolExecutor(len(urls)) as pool:
        futures = [pool.submit(_fetch_one, url, args["max_chars"]) for url in urls]
    parts, failed, pages, sources, errors = [], [], [], [], []
    for url, future in zip(urls, futures):
        try:
            out = future.result()
        except Exception as e:  # noqa: BLE001 - one bad page shouldn't lose the others
            errors.append(e)
            failed.append(f"FAILED {url}: {_describe(e)}")
            continue
        parts.append(out.text)
        pages.append(out.data)
        sources += out.sources
    if len(errors) == len(urls):
        raise errors[0]  # nothing readable: surface a real error so the registry can retry or report it
    header = f"Read {len(pages)} of {len(urls)} pages" + (f" ({len(errors)} failed, listed last)" if errors else "")
    body = [f"[page {i}/{len(pages)}] {text}" for i, text in enumerate(parts, 1)] + failed
    return ToolOutput(text="\n\n".join([header] + body), data=pages, sources=list(dict.fromkeys(sources)))


def _describe(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    return str(e) if isinstance(e, ToolError) else f"{type(e).__name__}: {str(e)[:120]}"


def _download(url: str) -> tuple[str, str]:
    """The page's HTML and final URL. Every redirect hop passes _check_url, and reading stops at the size cap."""
    hooks = {"request": [lambda request: _check_url(str(request.url))]}
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True, max_redirects=MAX_REDIRECTS,
                      headers={"User-Agent": _BROWSER_UA}, transport=_TRANSPORT, event_hooks=hooks) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                raise ToolError(f"unsupported content type '{ctype}' (only HTML/text pages can be read)")
            declared = r.headers.get("content-length", "")
            if declared.isdecimal() and int(declared) > MAX_DOWNLOAD_BYTES:
                raise ToolError("page too large")
            chunks, size = [], 0
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ToolError("page too large")
                chunks.append(chunk)
            return b"".join(chunks).decode(r.encoding or "utf-8", errors="replace"), str(r.url)


def _fetch_one(url: str, limit: int) -> ToolOutput:
    _check_url(url)
    if "news.google.com/rss/articles" in url:
        raise ToolError("Google News redirect links can't be fetched directly; search for the headline "
                        "with web_search to find the publisher's URL")
    try:
        html, final_url = _download(url)
    except httpx.TimeoutException:
        # a news site that stalls this long is usually holding off bots, so a retry just waits again
        raise ToolError(f"no answer from {urlparse(url).hostname} within {FETCH_TIMEOUT:.0f} s (probably "
                        f"blocking automated readers): use the search snippet or another source") from None
    title, text, day = extract_text(html)
    if len(text) < MIN_TEXT_CHARS:
        raise ToolError(f"page has almost no readable text ({len(text)} chars); it is probably "
                        f"JavaScript-rendered or paywalled. Try another source")
    body = text[:limit] + ("\n...(truncated)" if len(text) > limit else "")
    head = f"TITLE: {title}\nURL: {final_url}\n" + (f"PUBLISHED: {day}\n" if day else "")
    return ToolOutput(text=f"{head}\n{body}", data={"title": title, "chars": len(text), "published": day},
                      sources=list(dict.fromkeys([url, final_url])))


fetch_page = Tool(
    name="fetch_page",
    description="Download web pages and return their main readable text. Use to verify details from "
                "specific articles found via search. Read up to 3 pages in ONE call (URLs separated by "
                "spaces): much faster than one call per page.",
    params={
        "url": Param("string", "full http(s) URL taken from a search result, or up to 3 separated by spaces",
                     required=True),
        "max_chars": Param("integer", "max characters of text per page", default=2500, minimum=500, maximum=6000),
    },
    func=_fetch,
)
