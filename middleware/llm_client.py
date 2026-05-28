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
    include_raw: bool = False,
) -> Any:
    """Build a ``ChatOpenAI`` client with shared rate limiting and optional schema output.

    Pass ``output_schema`` (Pydantic model) when the caller needs structured JSON
    instead of free-form assistant text (e.g. safety judges). Set ``include_raw=True``
    with a schema to receive ``{"parsed": ..., "raw": AIMessage}`` for Langfuse token counts.
    """
    kw: dict[str, Any] = {"model": model, "temperature": temperature}
    if OPENAI_RATE_LIMITER is not None:
        kw["rate_limiter"] = OPENAI_RATE_LIMITER

    llm = ChatOpenAI(**kw)
    if output_schema is not None:
        return llm.with_structured_output(output_schema, include_raw=include_raw)
    return llm
