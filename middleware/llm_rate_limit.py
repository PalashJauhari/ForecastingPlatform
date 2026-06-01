"""
Shared ``InMemoryRateLimiter`` for LangChain ``ChatOpenAI`` clients (OpenAI API throttling).

Configured from root ``.env`` ``MAIN_RATE_LIMIT_*`` variables. One process-wide instance so all
orchestrator / tool / summarisation LLM calls draw from the same bucket.
"""

from __future__ import annotations

import os

from langchain_core.rate_limiters import InMemoryRateLimiter

def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env flag with safe fallback."""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float) -> float:
    """Parse float env value; fallback when missing/invalid."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


if not _env_bool("MAIN_RATE_LIMIT_ENABLED", default=True):
    OPENAI_RATE_LIMITER: InMemoryRateLimiter | None = None
else:
    OPENAI_RATE_LIMITER = InMemoryRateLimiter(
        requests_per_second=_env_float("MAIN_RATE_LIMIT_RPS", default=1.0),
        check_every_n_seconds=_env_float("MAIN_RATE_LIMIT_CHECK_SECONDS", default=0.1),
        max_bucket_size=_env_float("MAIN_RATE_LIMIT_MAX_BUCKET", default=5.0),
    )
