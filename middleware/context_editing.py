"""
Context editing: token-aware truncation and running summarisation.

When the estimated token count of ``state["messages"]`` exceeds the
configured threshold, old messages are evicted at a **HumanMessage** boundary
and summarised into a running summary.

Uses ``RemoveMessage`` to work with the ``add_messages`` reducer.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from middleware.llm_client import make_llm
from observability.langfuse_handler import traced_generation, traced_span, update_llm_generation

# Prefer explicit summary model; fall back to orchestrator model when unset.
SUMMARY_MODEL = (
    (os.environ.get("MAIN_MODEL_MESSAGE_SUMMARISATION") or "").strip()
    or (os.environ.get("MAIN_MODEL_ORCHESTRATOR") or "").strip()
    or "gpt-5.4-mini"
)

SUMMARY_SYSTEM = """\
# Role
Conversation summariser for a data-analysis agent session.

# Goal
Produce an updated running summary that preserves everything needed to continue the session without the evicted messages.

# Must preserve
- User goals and decisions
- File basenames, column names, and data references
- Analysis results, model choices, and key numbers
- User preferences and open questions

# Decision rules
- If a previous summary is provided, **integrate** new information — do not repeat what is already captured.
- Focus on what is new in the evicted messages.
- Prefer concise factual statements over narrative padding.

# Output
Plain-text summary only — no markdown fences, no preamble.
"""


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: total characters of message content ÷ 4."""
    return sum(len(str(getattr(m, "content", "") or "")) for m in messages) // 4


def find_human_truncation_cut(messages: list, keep: int) -> int | None:
    """Cut index at ``user_input`` / ``user_input-{uuid}`` Human if possible, else first Human ahead of naive cut.

    ``AnalysisGraph`` tags each user turn with ``user_input-{uuid}`` ids so eviction
    never splits mid-turn tool chains.
    """
    candidate = max(0, len(messages) - keep)
    idx = candidate
    while idx < len(messages):
        m = messages[idx]
        if isinstance(m, HumanMessage):
            mid = getattr(m, "id", None)
            s = "" if mid is None else str(mid)
            if s == "user_input" or s.startswith("user_input-"):
                return idx
        idx += 1
    idx = candidate
    while idx < len(messages):
        if isinstance(messages[idx], HumanMessage):
            return idx
        idx += 1
    return None


async def summarize_evicted(
    previous_summary: str,
    messages_to_evict: list,
    *,
    runnable_config: Any | None = None,
) -> str:
    """LLM call: merge *previous_summary* with *messages_to_evict* into an updated summary."""
    del runnable_config

    llm = make_llm(model=SUMMARY_MODEL, temperature=0)

    prompt_messages = [SystemMessage(content=SUMMARY_SYSTEM)]
    if previous_summary:
        prompt_messages.append(HumanMessage(content=f"## Previous Summary\n{previous_summary}"))
    prompt_messages.extend(messages_to_evict)
    prompt_messages.append(HumanMessage(content="Update the running summary using the evicted messages above."))

    with traced_generation("context.summarize_evicted.llm", model=SUMMARY_MODEL) as generation:
        response = await llm.ainvoke(prompt_messages)
        if generation is not None:
            update_llm_generation(generation, model=SUMMARY_MODEL, raw=response)
        return response.content


async def truncate_and_summarize(
    messages: list,
    previous_summary: str,
    keep: int,
    token_threshold: int,
    *,
    runnable_config: Any | None = None,
) -> tuple[str, list, list[RemoveMessage]]:
    """
    Truncate *messages* and update the running summary if token budget is exceeded.

    Returns ``(updated_summary, kept_messages, remove_ops)``; ``remove_ops`` is empty when unchanged.
    """
    del runnable_config

    with traced_span("context.truncate_and_summarize", metadata={"token_threshold": token_threshold, "keep": keep}) as span:
        token_estimate = estimate_tokens(messages)
        if token_estimate <= token_threshold or len(messages) <= keep:
            if span is not None:
                span.update(metadata={"token_estimate": token_estimate, "truncated": "false"})
            return previous_summary, messages, []

        cut = find_human_truncation_cut(messages, keep)
        if cut is None:
            if span is not None:
                span.update(metadata={"token_estimate": token_estimate, "truncated": "false", "truncate_skip": "no_human_boundary"})
            return previous_summary, messages, []

        to_evict = messages[:cut]
        remove_ops = [RemoveMessage(id=m.id) for m in to_evict]
        with traced_span("context.summarize_evicted", metadata={"evicted_count": len(to_evict)}):
            updated_summary = await summarize_evicted(previous_summary, to_evict)
        if span is not None:
            span.update(metadata={"token_estimate": token_estimate, "truncated": "true", "evicted_count": len(to_evict)})
        kept_messages = messages[cut:]
        return updated_summary, kept_messages, remove_ops
