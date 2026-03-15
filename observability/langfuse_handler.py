"""
Langfuse callback handler for LangChain/LangGraph.
When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, returns a list of callbacks
so that graph invocations (LLM + tool calls) are traced in Langfuse.
"""
import os
from typing import List, Any


def get_langfuse_callbacks() -> List[Any]:
    """
    Return a list of callbacks for LangChain/LangGraph invoke config.
    When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, LLM and tool calls are traced in Langfuse.
    Otherwise returns an empty list (no-op).
    """
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        return []
    try:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()
        return [handler]
    except Exception:
        return []
