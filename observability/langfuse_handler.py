"""
Shared Langfuse tracing helpers for the Forecasting Platform.

This module intentionally does not expose LangChain callback handlers. The codebase
uses Langfuse's SDK primitives directly: ``@observe`` for workflow boundaries,
``propagate_attributes`` for request/session context, and manual observations for
LLM generations and multi-step pipelines.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langfuse import get_client

MAX_METADATA_VALUE_LEN = 200


def get_langfuse_client():
    """Return the singleton Langfuse client configured from environment variables."""
    return get_client()


def short_text(value: Any, max_len: int = MAX_METADATA_VALUE_LEN) -> str:
    """
    Convert a value to a small ASCII-only string suitable for propagated metadata.

    Langfuse enforces ASCII strings <= 200 characters for propagated metadata values.
    """
    text = str(value)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def build_request_metadata(*, endpoint: str, interface: str, query: str | None = None) -> dict[str, str]:
    """Create small, filterable request metadata for propagate_attributes()."""
    metadata = {
        "endpoint": short_text(endpoint),
        "interface": short_text(interface),
    }
    if query is not None:
        metadata["query_length"] = short_text(len(query))
        metadata["query_preview"] = short_text(query[:120])
    return metadata


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message into a Langfuse-friendly dict."""
    payload: dict[str, Any] = {"type": getattr(message, "type", type(message).__name__)}
    if isinstance(message, HumanMessage):
        payload["role"] = "human"
    elif isinstance(message, AIMessage):
        payload["role"] = "ai"
        if getattr(message, "tool_calls", None):
            payload["tool_calls"] = message.tool_calls
    elif isinstance(message, SystemMessage):
        payload["role"] = "system"
    elif isinstance(message, ToolMessage):
        payload["role"] = "tool"
        if getattr(message, "tool_call_id", None):
            payload["tool_call_id"] = message.tool_call_id
    else:
        payload["role"] = payload["type"]

    payload["content"] = normalize_content(getattr(message, "content", ""))

    response_metadata = getattr(message, "response_metadata", None)
    if response_metadata:
        payload["response_metadata"] = response_metadata

    usage_metadata = getattr(message, "usage_metadata", None)
    if usage_metadata:
        payload["usage_metadata"] = usage_metadata

    return payload


def serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Serialize a list of LangChain messages for Langfuse observation input/output."""
    return [serialize_message(message) for message in messages]


def normalize_content(content: Any) -> Any:
    """Recursively coerce LangChain content blocks into JSON-serializable structures."""
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    if isinstance(content, list):
        return [normalize_content(item) for item in content]
    if isinstance(content, dict):
        return {str(k): normalize_content(v) for k, v in content.items()}
    return str(content)


def extract_usage_details(response: Any) -> dict[str, int] | None:
    """Extract token usage from LangChain responses when available."""
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if not isinstance(token_usage, dict):
        usage_metadata = getattr(response, "usage_metadata", None) or {}
        if isinstance(usage_metadata, dict):
            token_usage = usage_metadata

    if not isinstance(token_usage, dict):
        return None

    usage: dict[str, int] = {}
    aliases = {
        "input": ("input_tokens", "prompt_tokens"),
        "output": ("output_tokens", "completion_tokens"),
        "total": ("total_tokens",),
    }
    for key, candidates in aliases.items():
        for candidate in candidates:
            value = token_usage.get(candidate)
            if isinstance(value, int):
                usage[key] = value
                break

    return usage or None
