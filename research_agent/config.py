"""Runtime settings, read from environment variables (and .env via python-dotenv)."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # LLM (Groq exposes an OpenAI-compatible endpoint, so any compatible provider works)
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    # The step reviewer ("" = the main model). On Groq every model has its own tokens-per-minute
    # limit, so reviewing on a second model spreads the load.
    review_model: str = "openai/gpt-oss-20b"
    llm_timeout: float = 60.0
    llm_retries: int = 4
    reasoning_effort: str = "low"       # for reasoning models (gpt-oss, qwen3); "" to omit
    llm_max_wait: float = 65.0          # longest rate-limit wait we accept before giving up
    llm_rate_limit_waits: int = 10      # 429 waits allowed per call (separate from error retries)
    tpm_limit: int = 8000               # provider tokens-per-minute cap to pace against; 0 = off
    token_budget: int = 60_000          # soft cap on total tokens per run

    # Optional second provider for the main model, with rate limits of its own. Cerebras' free tier
    # serves gpt-oss-120b too; each main-model call goes to whichever provider can take it sooner.
    cerebras_api_key: str = ""          # empty = Groq only
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model: str = "gpt-oss-120b"  # must be the same model as `model`, or the lane stays off
    cerebras_tpm_limit: int = 30_000
    cerebras_rpm_limit: int = 5         # free tier: 5 requests a minute

    # Tools
    tool_timeout: float = 20.0
    tool_retries: int = 2               # extra attempts for transient failures
    backoff_base: float = 1.0           # seconds; doubles each retry
    max_observation_chars: int = 2500   # tool output passed back to the LLM

    # Control loop
    max_plan_steps: int = 6
    max_actions_per_step: int = 4
    max_step_retries: int = 1           # re-run a step once with reviewer feedback
    max_replans: int = 1                # re-plan remaining work once after a hard failure

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        s = cls(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            model=os.environ.get("GROQ_MODEL", cls.model),
            review_model=os.environ.get("GROQ_REVIEW_MODEL", cls.review_model),
            llm_timeout=_float("LLM_TIMEOUT", cls.llm_timeout),
            reasoning_effort=os.environ.get("REASONING_EFFORT", cls.reasoning_effort),
            token_budget=_int("TOKEN_BUDGET", cls.token_budget),
            tpm_limit=_int("LLM_TPM_LIMIT", cls.tpm_limit),
            cerebras_api_key=os.environ.get("CEREBRAS_API_KEY", ""),
            cerebras_model=os.environ.get("CEREBRAS_MODEL", cls.cerebras_model),
            cerebras_tpm_limit=_int("CEREBRAS_TPM_LIMIT", cls.cerebras_tpm_limit),
            cerebras_rpm_limit=_int("CEREBRAS_RPM_LIMIT", cls.cerebras_rpm_limit),
            tool_timeout=_float("TOOL_TIMEOUT", cls.tool_timeout),
            tool_retries=_int("TOOL_RETRIES", cls.tool_retries),
            max_plan_steps=_int("MAX_PLAN_STEPS", cls.max_plan_steps),
            max_actions_per_step=_int("MAX_ACTIONS_PER_STEP", cls.max_actions_per_step),
        )
        return replace(s, **{k: v for k, v in overrides.items() if v is not None})
