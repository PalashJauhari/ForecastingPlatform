"""
Optional Langfuse tracing for LangChain / LangGraph invocations.

If ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set in the environment,
:func:`get_langfuse_callbacks` returns a LangChain callback list so LLM and tool calls
appear in Langfuse. Import failures or missing keys yield an empty list (no-op).
"""

from __future__ import annotations

import os
from typing import Any, List


def get_langfuse_callbacks() -> List[Any]:
    """
    Build Langfuse :class:`~langfuse.langchain.CallbackHandler` instances when configured.

    Returns
        A list with one handler, or ``[]`` if keys are missing or Langfuse cannot be imported.

    Environment
        ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` — required for tracing.
    """
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return []
    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        return [handler]
    except Exception:
        return []
