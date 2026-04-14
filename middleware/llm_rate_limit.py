"""
Shared ``InMemoryRateLimiter`` for LangChain ``ChatOpenAI`` clients (OpenAI API throttling).

Configured from ``config.yaml`` → ``llm_rate_limit``. One process-wide instance so all
orchestrator / tool / summarisation LLM calls draw from the same bucket.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.rate_limiters import InMemoryRateLimiter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)
_sec = _cfg.get("llm_rate_limit")

if isinstance(_sec, dict) and _sec.get("enabled") is False:
    OPENAI_RATE_LIMITER: InMemoryRateLimiter | None = None
else:
    s = dict(_sec or {})
    OPENAI_RATE_LIMITER = InMemoryRateLimiter(
        requests_per_second=float(s.get("requests_per_second", 1.0)),
        check_every_n_seconds=float(s.get("check_every_n_seconds", 0.1)),
        max_bucket_size=float(s.get("max_bucket_size", 1.0)),
    )
