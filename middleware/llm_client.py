"""
Shared factory for LangChain ``ChatOpenAI`` clients used across the repo.

Keeps model defaults and optional structured-output wrapping in one place so
call sites do not repeat rate-limiter wiring.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from middleware.llm_rate_limit import OPENAI_RATE_LIMITER


def make_llm(
    *,
    model: str,
    temperature: float = 0,
    output_schema: Any | None = None,
) -> Any:
    """Build a ``ChatOpenAI`` client with shared rate limiting and optional schema output."""
    kw: dict[str, Any] = {"model": model, "temperature": temperature}
    if OPENAI_RATE_LIMITER is not None:
        kw["rate_limiter"] = OPENAI_RATE_LIMITER

    llm = ChatOpenAI(**kw)
    if output_schema is not None:
        return llm.with_structured_output(output_schema)
    return llm
