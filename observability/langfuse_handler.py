"""Langfuse tracing for the Forecasting Platform agent graph.

One root span per ``run`` / ``stream_run`` / ``resume``; inline gated node spans and nested
``{node}-llm`` generation spans with token counts on ``update``. Gated by ``LANGFUSE_TRACING_ENABLED``.
No ``CallbackHandler``, no ``@observe``, no RunnableConfig metadata pins.

Nested coding sub-graph invokes must pass ``trace_context`` from ``trace_context_for_nested_invoke()``.
LangGraph steps often drop OTel context, so the run trace is also carried in
``RunnableConfig["configurable"]``. Spans without a parent are skipped instead of
being sent as noisy top-level traces.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langfuse import get_client
from langfuse.types import TraceContext

LANGFUSE_TRACE_CONTEXT_CONFIG_KEY = "langfuse_trace_context"

# Snapshot ``run`` / ``stream_run`` / ``resume`` trace id + root span id for LangGraph node/tool spans.
_run_trace_ctx: ContextVar[TraceContext | None] = ContextVar("_run_trace_ctx", default=None)


def safe_reset_contextvar(var: ContextVar[Any], token: object | None) -> None:
    """Reset a ContextVar token; ignore cross-context errors from streaming generators."""
    if token is None:
        return
    try:
        var.reset(token)
    except ValueError:
        pass


def is_tracing_enabled() -> bool:
    """Return whether Langfuse tracing is enabled via ``LANGFUSE_TRACING_ENABLED`` env."""
    raw = (os.getenv("LANGFUSE_TRACING_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def configure_langfuse_env() -> None:
    """Ensure env vars the Langfuse SDK reads are set from project ``.env``."""
    base = (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").strip().rstrip("/")
    if base:
        os.environ.setdefault("LANGFUSE_BASE_URL", base)
        os.environ.setdefault("LANGFUSE_HOST", base)
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "true" if is_tracing_enabled() else "false")


def get_langfuse_client() -> Any | None:
    """Return the Langfuse client when tracing is enabled; otherwise ``None``."""
    if not is_tracing_enabled():
        return None
    configure_langfuse_env()
    return get_client()


def flush_langfuse() -> None:
    """Flush buffered observations after a traced invoke; no-op when tracing is disabled."""
    if not is_tracing_enabled():
        return
    get_client().flush()


def llm_token_counts(raw: AIMessage | None) -> tuple[int | None, int | None]:
    """Return ``(input_tokens, output_tokens)`` from LangChain ``usage_metadata``."""
    if raw is None:
        return None, None
    usage = getattr(raw, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    in_val = int(input_tokens) if input_tokens is not None else None
    out_val = int(output_tokens) if output_tokens is not None else None
    return in_val, out_val


def update_llm_generation(observation: Any, *, model: str, raw: AIMessage | None) -> None:
    """Finish a Langfuse generation span with model id and token counts."""
    in_tok, out_tok = llm_token_counts(raw)
    observation.update(model=model, input=in_tok, output=out_tok)


def current_trace_context() -> TraceContext | None:
    """Build ``TraceContext`` from the active Langfuse observation, if any."""
    if not is_tracing_enabled():
        return None
    client = get_langfuse_client()
    if client is None:
        return None
    trace_id = client.get_current_trace_id()
    if not trace_id:
        return None
    obs_id = client.get_current_observation_id()
    if obs_id:
        return TraceContext(trace_id=str(trace_id), parent_span_id=str(obs_id))
    return TraceContext(trace_id=str(trace_id))


def trace_context_to_config_value(trace_context: TraceContext | None) -> dict[str, str] | None:
    """Convert a Langfuse ``TraceContext`` into a JSON-safe graph config value."""
    if trace_context is None:
        return None
    if isinstance(trace_context, dict):
        trace_id = str(trace_context.get("trace_id") or "").strip()
        parent_span_id = str(trace_context.get("parent_span_id") or "").strip()
    else:
        trace_id = str(getattr(trace_context, "trace_id", "") or "").strip()
        parent_span_id = str(getattr(trace_context, "parent_span_id", "") or "").strip()
    if not trace_id:
        return None
    value = {"trace_id": trace_id}
    if parent_span_id:
        value["parent_span_id"] = parent_span_id
    return value


def trace_context_from_config_value(value: Any) -> TraceContext | None:
    """Read a Langfuse ``TraceContext`` from a graph config value."""
    if not isinstance(value, dict):
        return None
    trace_id = str(value.get("trace_id") or "").strip()
    parent_span_id = str(value.get("parent_span_id") or "").strip()
    if not trace_id:
        return None
    if parent_span_id:
        return TraceContext(trace_id=trace_id, parent_span_id=parent_span_id)
    return TraceContext(trace_id=trace_id)


def trace_context_from_runnable_config(config: Any = None) -> TraceContext | None:
    """Return explicit Langfuse context carried by ``RunnableConfig.configurable``."""
    configurable = (config or {}).get("configurable", {}) if isinstance(config or {}, dict) else {}
    return trace_context_from_config_value(configurable.get(LANGFUSE_TRACE_CONTEXT_CONFIG_KEY))


def add_trace_context_to_config(config: dict[str, Any], trace_context: TraceContext | None) -> dict[str, Any]:
    """Return a copy of ``config`` with the Langfuse trace context stored under ``configurable``."""
    value = trace_context_to_config_value(trace_context)
    if value is None:
        return config
    updated = dict(config)
    configurable = dict(updated.get("configurable") or {})
    configurable[LANGFUSE_TRACE_CONTEXT_CONFIG_KEY] = value
    updated["configurable"] = configurable
    return updated


def trace_context_for_nested_invoke() -> TraceContext | None:
    """Snapshot active trace context at tool entry for nested sub-graph spans."""
    return current_trace_context() or _run_trace_ctx.get()


def _otel_has_active_observation() -> bool:
    """Return whether Langfuse OTel context is active in the current thread."""
    if not is_tracing_enabled():
        return False
    client = get_langfuse_client()
    if client is None:
        return False
    return bool(client.get_current_observation_id())


def _resolve_trace_context(explicit: TraceContext | None) -> TraceContext | None:
    """Use explicit context, else OTel parent, else run-root fallback."""
    if explicit is not None:
        return explicit
    if _otel_has_active_observation():
        return None
    return _run_trace_ctx.get()


def should_skip_orphan_observation(parent_ctx: TraceContext | None) -> bool:
    """Avoid creating standalone Langfuse traces when a node/tool lost its parent context."""
    return is_tracing_enabled() and parent_ctx is None and not _otel_has_active_observation()


@contextmanager
def traced_span(name: str, *, trace_context: TraceContext | None = None, **kwargs: Any) -> Iterator[Any]:
    """Gated ``start_as_current_observation(as_type=\"span\")``; yields ``None`` when tracing is off."""
    if not is_tracing_enabled():
        yield None
        return
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    parent_ctx = _resolve_trace_context(trace_context)
    if should_skip_orphan_observation(parent_ctx):
        yield None
        return
    if parent_ctx is not None:
        kwargs = {**kwargs, "trace_context": parent_ctx}
    with client.start_as_current_observation(as_type="span", name=name, **kwargs) as span:
        yield span


@contextmanager
def traced_generation(name: str, *, model: str, trace_context: TraceContext | None = None, **kwargs: Any) -> Iterator[Any]:
    """Gated ``start_as_current_observation(as_type=\"generation\")``; yields ``None`` when tracing is off."""
    if not is_tracing_enabled():
        yield None
        return
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    parent_ctx = _resolve_trace_context(trace_context)
    if should_skip_orphan_observation(parent_ctx):
        yield None
        return
    if parent_ctx is not None:
        kwargs = {**kwargs, "trace_context": parent_ctx}
    with client.start_as_current_observation(as_type="generation", name=name, model=model, **kwargs) as gen:
        yield gen


@contextmanager
def tracing_root(name: str, *, metadata: dict[str, Any] | None = None) -> Iterator[TraceContext | None]:
    """Outer ``run`` / ``stream_run`` / ``resume`` span; snapshots trace context for nested spans."""
    if not is_tracing_enabled():
        yield None
        return
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    kw: dict[str, Any] = {"as_type": "span", "name": name}
    if metadata:
        kw["metadata"] = metadata
    token: object | None = None
    with client.start_as_current_observation(**kw):
        root_context = current_trace_context()
        token = _run_trace_ctx.set(root_context)
        try:
            yield root_context
        finally:
            safe_reset_contextvar(_run_trace_ctx, token)


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
    """Serialize LangChain messages for node span output summaries."""
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
