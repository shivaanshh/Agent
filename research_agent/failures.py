"""Deliberate failure injection, used to demonstrate and test recovery paths.

A fault spec looks like ``target:mode[:count]``:

    news_search:timeout        first news_search call times out
    ddgs:error:*               every DuckDuckGo request fails (forces provider fallback)
    fetch_page:bad_response:1  first fetch_page returns a malformed payload
    web_search:empty:2         first two web_search calls return no results
    llm:bad_json:1             first LLM reply is truncated, invalid JSON

``target`` is a tool name, a search provider (``ddgs``, ``bing_news``,
``google_news``, ``wikipedia``) or ``llm``. ``count`` defaults to 1; ``*`` means every call.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterable, Optional

MODES = {
    "timeout": "raise a timeout before the call runs",
    "error": "raise an HTTP 503-style error before the call runs",
    "bad_response": "replace the tool's result with a malformed payload",
    "empty": "replace the tool's result with an empty result set",
    "bad_json": "truncate the LLM's reply so it is invalid JSON (target must be 'llm')",
}
TARGETS = ("news_search", "web_search", "fetch_page", "calculator", "ddgs", "bing_news", "google_news", "wikipedia",
           "llm")


class InjectedFault(Exception):
    """Raised by the injector; the caller maps it to its own transient error type."""

    def __init__(self, target: str, mode: str):
        self.target, self.mode = target, mode
        what = "timed out" if mode == "timeout" else "returned HTTP 503 Service Unavailable"
        super().__init__(f"[injected] {target} {what}")


@dataclass
class FaultSpec:
    target: str
    mode: str
    remaining: Optional[int]  # None = unlimited
    fired: int = 0

    @classmethod
    def parse(cls, spec: str) -> "FaultSpec":
        parts = spec.strip().split(":")
        if len(parts) not in (2, 3) or not all(parts):
            raise ValueError(f"bad fault spec {spec!r}; expected target:mode[:count]")
        target, mode = parts[0].lower(), parts[1].lower()
        if mode not in MODES:
            raise ValueError(f"unknown fault mode {mode!r}; choose from {sorted(MODES)}")
        if mode == "bad_json" and target != "llm":
            raise ValueError("bad_json only applies to target 'llm'")
        count_raw = parts[2] if len(parts) == 3 else "1"
        if count_raw != "*" and not (count_raw.isdecimal() and int(count_raw) > 0):
            raise ValueError(f"bad count {count_raw!r} in {spec!r}; use a positive number or '*'")
        remaining = None if count_raw == "*" else int(count_raw)
        return cls(target, mode, remaining)

    def __str__(self) -> str:
        return f"{self.target}:{self.mode}:{'*' if self.remaining is None else self.remaining + self.fired}"


@dataclass
class FailureInjector:
    specs: list[FaultSpec] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self._lock = threading.Lock()

    @classmethod
    def from_strings(cls, specs: Optional[list[str]], targets: Optional[Iterable[str]] = None) -> "FailureInjector":
        """Parse fault specs. Given ``targets``, a spec for any other target is an error instead of a fault
        that silently never fires (``news_serch:error``)."""
        parsed = [FaultSpec.parse(s) for s in (specs or [])]
        known = list(targets) if targets is not None else None
        for spec in parsed:
            if known is not None and spec.target not in known:
                raise ValueError(f"unknown fault target {spec.target!r}; choose from {', '.join(known)}")
        return cls(parsed)

    def _take(self, target: str, modes: set[str]) -> Optional[FaultSpec]:
        with self._lock:
            for spec in self.specs:
                if spec.target == target and spec.mode in modes and spec.remaining != 0:
                    if spec.remaining is not None:
                        spec.remaining -= 1
                    spec.fired += 1
                    self.log.append({"target": target, "mode": spec.mode})
                    return spec
        return None

    def before_call(self, target: str) -> None:
        spec = self._take(target, {"timeout", "error"})
        if spec:
            raise InjectedFault(target, spec.mode)

    def after_call(self, target: str, output):
        """Possibly replace a tool's output. Returns the (maybe replaced) output."""
        spec = self._take(target, {"bad_response", "empty"})
        if spec is None:
            return output
        if spec.mode == "bad_response":
            return {"unexpected": "payload", "status": "???"}  # not a ToolOutput: fails validation
        from .tools.base import ToolOutput

        return ToolOutput(text="No results found.", data=[], sources=[])

    def corrupt_llm(self, text: str) -> str:
        if self._take("llm", {"bad_json"}):
            return text[: max(1, len(text) // 2)]
        return text

    @property
    def active(self) -> bool:
        return bool(self.specs)

    def describe(self) -> list[str]:
        return [str(s) for s in self.specs]
