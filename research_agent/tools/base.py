"""Tool abstraction and the registry that runs tools safely.

Every tool call goes through ``ToolRegistry.execute``, which:
  1. rejects unknown tools and validates/coerces arguments,
  2. applies deliberate fault injection (if configured),
  3. enforces a hard timeout,
  4. retries *transient* failures with exponential backoff,
  5. validates the output shape,
  6. never raises: failures come back as ``ToolResult(ok=False, error=...)``
     so the agent can reason about them instead of crashing.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

from ..failures import FailureInjector, InjectedFault


class ToolError(Exception):
    """Permanent failure (bad input, 404, unusable page). Retrying won't help."""


class TransientToolError(ToolError):
    """Timeouts, 5xx, rate limits, malformed responses: worth retrying."""


@dataclass
class Param:
    type: str                      # "string" | "integer" | "number" | "boolean"
    description: str
    required: bool = False
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class ToolOutput:
    text: str                                   # compact text shown to the LLM
    data: Any = None                            # structured payload (for tests / reports)
    sources: list[str] = field(default_factory=list)  # URLs surfaced by this call


@dataclass
class ToolContext:
    injector: FailureInjector
    emit: Callable[..., None] = lambda kind, **data: None  # recovery/event hook


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, Param]
    func: Callable[[dict, ToolContext], ToolOutput]
    timeout: Optional[float] = None

    def signature(self) -> str:
        parts = []
        for name, p in self.params.items():
            opt = "" if p.required else f"={p.default!r}"
            parts.append(f"{name}: {p.type}{opt}")
        return f"{self.name}({', '.join(parts)})"

    def describe(self) -> str:
        lines = [f"- {self.signature()}: {self.description}"]
        for name, p in self.params.items():
            lines.append(f"    {name}: {p.description}")
        return "\n".join(lines)


@dataclass
class ToolResult:
    tool: str
    args: dict
    ok: bool
    text: str = ""
    sources: list[str] = field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 0
    latency_ms: int = 0


_COERCE = {"string": str, "integer": int, "number": float, "boolean": None}


def validate_args(tool: Tool, args: Any) -> dict:
    """Check required params, drop unknown ones, coerce types and clamp ranges."""
    if not isinstance(args, dict):
        raise ToolError(f"arguments must be an object, got {type(args).__name__}")
    clean: dict[str, Any] = {}
    for name, p in tool.params.items():
        if name not in args or args[name] in (None, ""):
            if p.required:
                raise ToolError(f"missing required argument '{name}'")
            if p.default is not None:
                clean[name] = p.default
            continue
        value = args[name]
        try:
            if p.type == "boolean":
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
            elif p.type == "integer":
                value = int(float(value))
            else:
                value = _COERCE[p.type](value)
        except (TypeError, ValueError):
            raise ToolError(f"argument '{name}' should be {p.type}, got {value!r}")
        if p.minimum is not None and value < p.minimum:
            value = type(value)(p.minimum)
        if p.maximum is not None and value > p.maximum:
            value = type(value)(p.maximum)
        if p.type == "string" and not value.strip():
            raise ToolError(f"argument '{name}' must not be empty")
        clean[name] = value
    return clean


def classify_exception(exc: BaseException) -> ToolError:
    """Map arbitrary exceptions from tool code onto transient vs permanent."""
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, InjectedFault):
        return TransientToolError(str(exc))
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return TransientToolError(f"network timeout: {exc}")
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        msg = f"HTTP {code} from {exc.request.url.host}"
        return TransientToolError(msg) if code == 429 or code >= 500 else ToolError(msg)
    if isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError, ConnectionError)):
        return TransientToolError(f"network error: {exc}")
    return ToolError(f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    def __init__(
        self,
        injector: Optional[FailureInjector] = None,
        *,
        timeout: float = 20.0,
        retries: int = 2,
        backoff_base: float = 1.0,
        emit: Callable[..., None] = lambda kind, **data: None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._tools: dict[str, Tool] = {}
        self.injector = injector or FailureInjector()
        self.timeout, self.retries, self.backoff_base = timeout, retries, backoff_base
        self.emit, self.sleep = emit, sleep
        self._pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    # registration / introspection
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def describe(self) -> str:
        return "\n".join(t.describe() for t in self._tools.values())

    # execution
    def execute(self, name: str, args: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, args if isinstance(args, dict) else {}, False,
                              error=f"unknown tool '{name}'. Available tools: {', '.join(self._tools)}")
        try:
            clean = validate_args(tool, args)
        except ToolError as e:
            return ToolResult(name, args if isinstance(args, dict) else {}, False, error=f"invalid arguments: {e}")

        start = time.monotonic()
        last_err: ToolError = ToolError("not run")
        attempt = 0
        for attempt in range(1, self.retries + 2):
            try:
                out = self._run_once(tool, clean)
                return ToolResult(name, clean, True, text=out.text, sources=list(out.sources),
                                  attempts=attempt, latency_ms=int((time.monotonic() - start) * 1000))
            except TransientToolError as e:
                last_err = e
                if attempt <= self.retries:
                    delay = self.backoff_base * (2 ** (attempt - 1))
                    self.emit("tool_retry", tool=name, attempt=attempt, error=str(e), wait_s=delay)
                    self.sleep(delay)
            except ToolError as e:
                last_err = e
                break
        return ToolResult(name, clean, False, error=str(last_err), attempts=attempt,
                          latency_ms=int((time.monotonic() - start) * 1000))

    def _run_once(self, tool: Tool, args: dict) -> ToolOutput:
        ctx = ToolContext(self.injector, self.emit)
        timeout = tool.timeout or self.timeout
        try:
            self.injector.before_call(tool.name)
            future = self._pool.submit(tool.func, args, ctx)
            out = future.result(timeout=timeout)
        except cf.TimeoutError:
            # The worker thread can't be killed, but the agent stops waiting for it.
            raise TransientToolError(f"{tool.name} timed out after {timeout:.0f}s")
        except BaseException as e:  # noqa: BLE001 - tools must never crash the agent
            raise classify_exception(e)

        out = self.injector.after_call(tool.name, out)
        if not isinstance(out, ToolOutput) or not isinstance(out.text, str) or not out.text.strip():
            raise TransientToolError(f"{tool.name} returned a malformed/empty response: {str(out)[:80]}")
        return out
