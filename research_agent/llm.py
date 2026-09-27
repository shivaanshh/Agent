"""LLM client wrapper (Groq via its OpenAI-compatible API).

Adds what a raw SDK call lacks:
  * client-side pacing under the provider's per-model tokens-per-minute limit
  * an optional second provider for the main model (``Lane``), used whenever it can take a call sooner
  * retries with backoff for rate limits (honours Retry-After), timeouts, 5xx
  * a retry with more room when a reply is cut off at max_tokens
  * a soft token budget per run
  * ``complete_json``: parse + pydantic-validate the reply, and on failure send
    the validation error back to the model to repair it (bounded attempts)
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar
from urllib.parse import urlsplit

import openai
from pydantic import BaseModel, ValidationError

from .config import Settings
from .failures import FailureInjector, InjectedFault

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM could not be reached or keeps failing."""


class LLMOutputError(LLMError):
    """The LLM replied, but the reply stayed invalid after repair attempts."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class BudgetExceeded(LLMError):
    """The run hit its token budget."""


MAX_REPLY_TOKENS = 4096  # ceiling when a cut-off reply is retried with a larger max_tokens
CUT_OFF = "reply was cut off before the JSON object closed"


def parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and text.count("{") > text.count("}"):
            raise ValueError(CUT_OFF)
        if start == -1 or end <= start:
            raise ValueError("reply contains no JSON object")
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON ({e.msg} at char {e.pos})")
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
                         for err in e.errors()[:5])
    return str(e)[:300]


class TokenBucket:
    """Client-side mirror of the provider's tokens-per-minute limit for one model.

    Groq's limit is a bucket of ``limit`` tokens that refills continuously (limit/60 per
    second). A request is admitted if its prompt + max_tokens fit in what is left; once it
    completes, only the tokens actually used stay charged.
    """

    MARGIN = 0.95  # aim just under the limit: our prompt-token estimate is approximate

    def __init__(self, limit: int):
        self.limit = limit
        self._used, self._at = 0.0, time.monotonic()

    def used(self, now: float) -> float:
        return max(0.0, self._used - max(0.0, now - self._at) * self.limit / 60)

    def charge(self, tokens: float, now: float) -> None:
        self._used, self._at = self.used(now) + tokens, now

    def wait_for(self, request: int, now: float) -> float:
        """Seconds until a request of this size fits (0 if it fits now)."""
        room = max(0.0, self.limit * self.MARGIN - request)
        return max(0.0, self.used(now) - room) * 60 / self.limit

    def sync(self, headers, reserved: int, completion: int, now: float) -> None:
        """Correct the mirror from the provider's x-ratelimit-* response headers."""
        # Groq names them ...-tokens, Cerebras ...-tokens-minute
        limit = _int_header(headers, "x-ratelimit-limit-tokens", "x-ratelimit-limit-tokens-minute")
        if limit:
            self.limit = limit  # e.g. a paid tier: pacing follows the real limit
        remaining = _int_header(headers, "x-ratelimit-remaining-tokens", "x-ratelimit-remaining-tokens-minute")
        if remaining is None:
            return
        # `remaining` can still include part of this request's max_tokens reservation, so trust it
        # only when it shows more usage than we know of (e.g. another run on the same API key).
        used = self.limit - remaining - max(0, reserved - completion)
        if used > self.used(now):
            self._used, self._at = float(used), now


RPM_WINDOW = 61.0   # a minute, plus a second of slack for clock differences
LANE_STRIKES = 2    # unexplained failures in a row before a lane is switched off for the run
LANE_FIELDS = (("response_format", r"response_format|json"), ("extra_body", r"reasoning"))


@dataclass
class Lane:
    """A second provider serving the main model, with rate limits of its own.

    Free tiers cap tokens per minute per provider, so a sequential agent spends most of a run
    waiting on one bucket. Each main-model call goes to whichever lane can take it sooner. A lane
    that misbehaves only ever costs the primary provider one call: it is rested or switched off.
    """
    name: str
    client: Any
    model: str                                   # this provider's id for the main model
    bucket: TokenBucket
    rpm: int = 0                                 # requests per minute, 0 = no limit
    sent: deque = field(default_factory=deque)   # when recent requests went out (monotonic)
    rest_until: float = 0.0                      # set by a 429
    too_big: float = float("inf")                # requests this size (prompt + max_tokens) were rejected as too long
    drop: set = field(default_factory=set)       # request fields it rejected, e.g. response_format
    strikes: int = 0
    disabled: bool = False

    def wait_for(self, request: int, now: float) -> Optional[float]:
        """Seconds until this lane can take the request, or None if it can't take it at all."""
        if self.disabled or request >= self.too_big:
            return None
        while self.sent and now - self.sent[0] >= RPM_WINDOW:
            self.sent.popleft()
        wait = max(self.bucket.wait_for(request, now), self.rest_until - now)
        if self.rpm and len(self.sent) >= self.rpm:
            wait = max(wait, self.sent[-self.rpm] + RPM_WINDOW - now)
        return max(0.0, wait)


class LLM:
    def __init__(self, settings: Settings, injector: Optional[FailureInjector] = None,
                 emit: Callable[..., None] = lambda kind, **data: None, client=None,
                 sleep: Callable[[float], None] = time.sleep, lanes: Optional[list[Lane]] = None):
        self.settings = settings
        self.injector = injector or FailureInjector()
        self.emit, self.sleep = emit, sleep
        self.client = client or openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                                              timeout=settings.llm_timeout, max_retries=0)
        # a scripted or demo client means nothing may reach a real provider, so no lanes by default
        self.lanes: list[Lane] = list(lanes) if lanes is not None else [] if client else _lanes_from(settings)
        self.primary = _provider_name(settings.base_url)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_provider": {}}
        self.buckets: dict[str, TokenBucket] = {}  # per model: each model has its own TPM limit
        self._chars_per_token = 3.0  # conservative until the first response tells us the real ratio
        self.json_mode_ok = True  # switched off for the run if the provider's JSON mode keeps misfiring
        self._unavailable: set[str] = set()  # secondary models the provider doesn't serve

    @property
    def total_tokens(self) -> int:
        return self.usage["prompt_tokens"] + self.usage["completion_tokens"]

    def chat(self, messages: list[dict], *, json_mode: bool = True, temperature: float = 0.2,
             max_tokens: int = 1200, critical: bool = False, model: Optional[str] = None) -> str:
        """One completion. ``critical`` calls (the final report) may exceed the budget.

        ``model`` routes the call to another model than the main one (it has its own rate limit).
        """
        if not critical and self.total_tokens >= self.settings.token_budget:
            raise BudgetExceeded(f"token budget of {self.settings.token_budget:,} exhausted")
        if not model or model in self._unavailable:
            model = self.settings.model
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        json_mode = json_mode and self.json_mode_ok
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.reasoning_effort and _is_reasoning_model(model):
            kwargs["extra_body"] = {"reasoning_effort": self.settings.reasoning_effort}

        last: Exception = LLMError("no attempt made")
        attempt, rate_waits, grown = 1, 0, False
        while attempt <= self.settings.llm_retries + 1:
            try:
                self.injector.before_call("llm")
                resp = self._via_lane(model, kwargs, messages, max_tokens)
                if resp is None:  # the primary provider takes it
                    self._pace(model, messages, max_tokens)
                    resp, headers = self._create(kwargs)
                    self._record_usage(resp, self.bucket(model), headers, self.primary,
                                       _prompt_chars(messages), reserved=max_tokens)
                if _finish_reason(resp) == "length" and not grown and max_tokens < MAX_REPLY_TOKENS:
                    # Cut off mid-reply: a "repair" would only resend the fragment. Ask again with room.
                    grown, new = True, min(MAX_REPLY_TOKENS, max_tokens * 2)
                    self.emit("llm_truncated", max_tokens=max_tokens, retry_with=new)
                    max_tokens = kwargs["max_tokens"] = new
                    continue
                return self.injector.corrupt_llm(self._reply_text(resp.choices[0].message))
            except openai.AuthenticationError as e:
                raise LLMError("the LLM provider rejected the API key (check GROQ_API_KEY)") from e
            except openai.NotFoundError as e:
                if model != self.settings.model:
                    # a secondary model (e.g. the reviewer's) isn't offered here: use the main one
                    self._unavailable.add(model)
                    self.emit("model_fallback", model=model, using=self.settings.model)
                    model = kwargs["model"] = self.settings.model
                    if not _is_reasoning_model(model):
                        kwargs.pop("extra_body", None)
                    last = e
                    continue
                raise LLMError(f"model '{model}' not found; set GROQ_MODEL to a model "
                               f"listed at https://console.groq.com/docs/models") from e
            except openai.BadRequestError as e:
                # Groq rejects replies that fail its own JSON check; hand the raw text to our repair loop.
                failed = _failed_generation(e)
                if failed is not None:
                    return failed
                if "extra_body" in kwargs and "reasoning" in str(e).lower():
                    kwargs.pop("extra_body")
                    last = e
                    continue
                if json_mode and "json" in str(e).lower():
                    # Typically a reasoning model (gpt-oss) that wrote its answer into the reasoning
                    # channel, leaving `content` empty. Plain mode + salvage handles that, so stop
                    # using the provider's JSON mode for the rest of the run.
                    kwargs.pop("response_format", None)
                    json_mode = self.json_mode_ok = False
                    last = e
                    self.emit("json_mode_off", error=str(e)[:120])
                    continue
                raise LLMError(f"LLM request rejected: {e}") from e
            except openai.RateLimitError as e:
                # 429s are expected on free tiers: they get their own allowance, not the error retries
                last, rate_waits = e, rate_waits + 1
                wait = _retry_after(e) or self.settings.backoff_base * 2 ** min(rate_waits, 5)
                if wait > self.settings.llm_max_wait:
                    raise LLMError(f"rate limited for {wait:.0f}s (daily quota?): {e}") from e
                if rate_waits > self.settings.llm_rate_limit_waits:
                    raise LLMError(f"still rate limited after {rate_waits - 1} waits: {e}") from e
                self.emit("llm_rate_limit", wait_s=round(wait, 1), error=_short_rate_error(e))
                self.sleep(wait)
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError,
                    InjectedFault) as e:
                last = e
                self._backoff("llm_retry", attempt, self.settings.backoff_base * 2 ** (attempt - 1), e)
                attempt += 1
        raise LLMError(f"LLM failed after {self.settings.llm_retries + 1} attempts: {last}")

    def complete_json(self, messages: list[dict], schema: Type[T], *, check: Optional[Callable[[T], None]] = None,
                      max_repairs: int = 2, **chat_kwargs) -> T:
        """Get a reply that parses into ``schema`` (and passes ``check``), repairing if needed."""
        msgs = list(messages)
        for attempt in range(max_repairs + 1):
            raw = self.chat(msgs, **chat_kwargs)
            try:
                obj = schema.model_validate(parse_json_object(raw))
                if check:
                    check(obj)
                return obj
            except (ValueError, ValidationError) as e:
                err = _short_error(e)
                if attempt == max_repairs:
                    raise LLMOutputError(f"{schema.__name__} still invalid after {max_repairs} repairs: {err}", raw)
                self.emit("json_repair", target=schema.__name__, error=err, raw=raw[:300])
                if err == CUT_OFF:  # resending the fragment would only make the next reply longer
                    msgs = msgs + [{"role": "user", "content": "Your reply was too long and got cut off. Reply "
                                                               "again with a complete but much SHORTER JSON object."}]
                    continue
                msgs = msgs + [
                    {"role": "assistant", "content": raw[:3000]},
                    {"role": "user", "content": f"Your reply was invalid: {err}\n"
                                                f"Reply again with ONLY the corrected JSON object."},
                ]
        raise AssertionError("unreachable")

    # helpers
    def _backoff(self, kind: str, attempt: int, wait: float, err: Exception) -> None:
        if attempt > self.settings.llm_retries:
            return
        self.emit(kind, attempt=attempt, wait_s=round(wait, 1), error=str(err)[:160])
        self.sleep(wait)

    def _reply_text(self, message) -> str:
        text = message.content or ""
        if not text.strip():
            # Some reasoning models (e.g. gpt-oss) occasionally leave `content` empty and put the
            # answer at the end of their reasoning. Salvage the JSON object from there.
            salvaged = _last_json_object(getattr(message, "reasoning", None) or "")
            if salvaged is not None:
                self.emit("reasoning_salvage", chars=len(salvaged))
                return salvaged
        return text

    def bucket(self, model: Optional[str] = None) -> TokenBucket:
        model = model or self.settings.model
        if model not in self.buckets:
            self.buckets[model] = TokenBucket(self.settings.tpm_limit)
        return self.buckets[model]

    def _pace(self, model: str, messages: list[dict], max_tokens: int) -> None:
        """Wait until this request fits under the model's tokens-per-minute limit."""
        if self.settings.tpm_limit <= 0:
            return
        bucket, now = self.bucket(model), time.monotonic()
        estimate = self.estimate_tokens(messages, max_tokens)
        wait = min(bucket.wait_for(estimate, now), 60.5)
        if wait <= 0:
            return
        if wait >= 1:
            self.emit("llm_throttle", wait_s=round(wait, 1), model=model, provider=self.primary,
                      used=round(bucket.used(now)), limit=bucket.limit, request_estimate=estimate)
        self.sleep(wait)

    def _via_lane(self, model: str, kwargs: dict, messages: list[dict], max_tokens: int):
        """Send a main-model call to a lane if one can take it no later than the primary provider.

        Returns None when the primary provider should take the call, including right after the
        lane failed it: the lane is rested or switched off, and the call moves on without it.
        """
        if not self.lanes or model != self.settings.model:
            return None
        now, request = time.monotonic(), self.estimate_tokens(messages, max_tokens)
        waits = [(w, lane) for lane in self.lanes if (w := lane.wait_for(request, now)) is not None]
        if not waits:
            return None
        wait, lane = min(waits, key=lambda x: x[0])
        primary_wait = self.bucket(model).wait_for(request, now) if self.settings.tpm_limit > 0 else 0.0
        if wait > primary_wait:
            return None
        wait = min(wait, 60.5)
        if wait >= 1:
            self.emit("llm_throttle", wait_s=round(wait, 1), model=lane.model, provider=lane.name,
                      used=round(lane.bucket.used(now)), limit=lane.bucket.limit, request_estimate=request)
        if wait > 0:
            self.sleep(wait)
        lane.sent.append(time.monotonic())
        try:
            resp, headers = self._create({**{k: v for k, v in kwargs.items() if k not in lane.drop},
                                          "model": lane.model}, lane.client)
        except openai.APIError as e:
            self._lane_failed(lane, e, request)
            return None
        lane.strikes = 0
        self._record_usage(resp, lane.bucket, headers, lane.name, _prompt_chars(messages), reserved=max_tokens)
        return resp

    def _lane_failed(self, lane: Lane, e: openai.APIError, request: int) -> None:
        """Adapt the lane to what went wrong. The call itself goes to the primary provider."""
        msg = str(e)
        unsupported = next((f for f, pattern in LANE_FIELDS
                            if f not in lane.drop and re.search(pattern, msg, re.I)), None)
        if isinstance(e, openai.RateLimitError):
            lane.rest_until = time.monotonic() + (_retry_after(e) or 60.0)
        elif isinstance(e, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
            lane.disabled = True
        elif isinstance(e, openai.BadRequestError) and unsupported:
            lane.drop.add(unsupported)  # e.g. no JSON mode there: later calls go without it
        elif isinstance(e, openai.BadRequestError) and re.search(r"context|too long|too large|tokens", msg, re.I):
            lane.too_big = min(lane.too_big, request)  # e.g. a smaller context window on a free tier
        else:
            lane.strikes += 1
            lane.disabled = lane.strikes >= LANE_STRIKES
        self.emit("lane_disabled" if lane.disabled else "lane_fallback", provider=lane.name,
                  using=self.primary, error=msg[:160])

    def estimate_tokens(self, messages: list[dict], max_tokens: int = 0) -> int:
        """What the provider checks against the limit before admitting a request: prompt + max_tokens."""
        return int(_prompt_chars(messages) / self._chars_per_token) + max_tokens

    def _create(self, kwargs: dict, client=None):
        """The completion plus the response headers (which carry the provider's rate-limit state)."""
        client = client or self.client
        raw_api = getattr(client.chat.completions, "with_raw_response", None)
        if raw_api is None:  # scripted/demo clients
            return client.chat.completions.create(**kwargs), {}
        raw = raw_api.create(**kwargs)
        return raw.parse(), raw.headers

    def _record_usage(self, resp, bucket: TokenBucket, headers, provider: str, prompt_chars: int = 0,
                      reserved: int = 0) -> None:
        self.usage["calls"] += 1
        by_provider = self.usage["by_provider"]
        by_provider[provider] = by_provider.get(provider, 0) + 1
        u = getattr(resp, "usage", None)
        prompt = completion = 0
        if u:
            prompt = getattr(u, "prompt_tokens", 0) or 0
            completion = getattr(u, "completion_tokens", 0) or 0
            self.usage["prompt_tokens"] += prompt
            self.usage["completion_tokens"] += completion
        if prompt and prompt_chars:
            self._chars_per_token = max(2.0, prompt_chars / prompt)  # calibrate the pacing estimate
        now = time.monotonic()
        bucket.charge(prompt + completion, now)  # the unused part of max_tokens is released
        bucket.sync(headers, reserved, completion, now)


def _prompt_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def _finish_reason(resp) -> Optional[str]:
    try:
        return getattr(resp.choices[0], "finish_reason", None)
    except (AttributeError, IndexError):
        return None


def _int_header(headers, *names: str) -> Optional[int]:
    """The first of these headers that is present, as an int."""
    for name in names:
        try:
            raw = headers.get(name)
            if raw not in (None, ""):
                return int(float(raw))
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _lanes_from(settings: Settings) -> list[Lane]:
    """The optional Cerebras lane, if a key is set and it serves the same model as the primary."""
    same_model = settings.model.split("/")[-1] == settings.cerebras_model.split("/")[-1]
    if not settings.cerebras_api_key or not same_model:
        return []
    client = openai.OpenAI(api_key=settings.cerebras_api_key, base_url=settings.cerebras_base_url,
                           timeout=settings.llm_timeout, max_retries=0)
    return [Lane("cerebras", client, settings.cerebras_model, TokenBucket(settings.cerebras_tpm_limit),
                 rpm=settings.cerebras_rpm_limit)]


def _provider_name(base_url: str) -> str:
    """'groq' for https://api.groq.com/openai/v1, and so on."""
    labels = (urlsplit(base_url).hostname or "llm").split(".")
    return labels[-2] if len(labels) > 1 else labels[0]


def _is_reasoning_model(model: str) -> bool:
    return any(tag in model.lower() for tag in ("gpt-oss", "qwen3", "deepseek-r1"))


def _last_json_object(text: str) -> Optional[str]:
    """The last top-level JSON object embedded in free text (e.g. a model's reasoning), if any."""
    decoder, found, i = json.JSONDecoder(), None, text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found = text[i:end]
        i = text.find("{", end)
    return found


def _short_rate_error(e: Exception) -> str:
    m = re.search(r"((?:output |input )?tokens per \w+|requests per \w+)[^:]*: Limit (\d+), Used (\d+), Requested (\d+)", str(e))
    return f"{m.group(1)}: limit {m.group(2)}, used {m.group(3)}, requested {m.group(4)}" if m else str(e)[:120]


def _retry_after(e: openai.APIStatusError) -> Optional[float]:
    try:
        raw = e.response.headers.get("retry-after")
        return float(raw) + 0.5 if raw else None
    except (AttributeError, ValueError):
        return None


def _failed_generation(e: openai.BadRequestError) -> Optional[str]:
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body) if isinstance(body, dict) else {}
    if isinstance(err, dict) and err.get("failed_generation"):
        return str(err["failed_generation"])
    return None
