"""
Context editing: token-aware truncation and running summarisation.

When the estimated token count of ``state["messages"]`` exceeds the
configured threshold, old turns are evicted — keeping only the last N
**human turns** (a human turn is a ``HumanMessage`` plus every AI/Tool
message that follows it, up to the next ``HumanMessage``) — and summarised
into a running summary.

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
from observability.langfuse_handler import (
    serialize_messages,
    traced_generation,
    traced_span,
    update_llm_generation,
)

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


def count_human_messages(messages: list) -> int:
    """Count ``HumanMessage`` instances, i.e. the number of human turns in *messages*."""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def find_human_turn_cut(messages: list, keep_recent_human_messages: int) -> int | None:
    """Cut index that keeps the last *keep_recent_human_messages* human turns intact.

    Walks backward over *messages* counting ``HumanMessage`` instances. Once the
    Nth-from-last human message is reached, returns its index — everything from
    that index onward (the human message and every AI/Tool message that follows
    it) is kept; everything before it is evicted.

    Returns ``None`` when there are ``<= keep_recent_human_messages`` human turns
    in total (nothing safe to evict).
    """
    if keep_recent_human_messages <= 0:
        return None
    if count_human_messages(messages) <= keep_recent_human_messages:
        return None

    seen = 0
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            seen += 1
            if seen == keep_recent_human_messages:
                return idx
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

    ``keep`` is a count of **human turns** (not raw messages): the last ``keep``
    ``HumanMessage``s and everything that follows each of them are always retained.

    Returns ``(updated_summary, kept_messages, remove_ops)``; ``remove_ops`` is empty when unchanged.
    """
    del runnable_config

    with traced_span("context.truncate_and_summarize", metadata={"token_threshold": token_threshold, "keep": keep}) as span:
        if span is not None:
            span.update(input={"messages": serialize_messages(messages), "message_summary": previous_summary})
        token_estimate = estimate_tokens(messages)
        human_count = count_human_messages(messages)
        if token_estimate <= token_threshold or human_count <= keep:
            if span is not None:
                span.update(
                    metadata={"token_estimate": token_estimate, "human_count": human_count, "truncated": "false"},
                    output={"messages": serialize_messages(messages), "message_summary": previous_summary},
                )
            return previous_summary, messages, []

        cut = find_human_turn_cut(messages, keep)
        if cut is None:
            if span is not None:
                span.update(
                    metadata={
                        "token_estimate": token_estimate,
                        "human_count": human_count,
                        "truncated": "false",
                        "truncate_skip": "not_enough_human_turns",
                    },
                    output={"messages": serialize_messages(messages), "message_summary": previous_summary},
                )
            return previous_summary, messages, []

        to_evict = messages[:cut]
        remove_ops = [RemoveMessage(id=m.id) for m in to_evict]
        with traced_span("context.summarize_evicted", metadata={"evicted_count": len(to_evict)}) as evict_span:
            updated_summary = await summarize_evicted(previous_summary, to_evict)
            if evict_span is not None:
                evict_span.update(
                    input={"messages": serialize_messages(to_evict), "message_summary": previous_summary},
                    output={"message_summary": updated_summary},
                )
        kept_messages = messages[cut:]
        if span is not None:
            span.update(
                metadata={"token_estimate": token_estimate, "human_count": human_count, "truncated": "true", "evicted_count": len(to_evict)},
                output={"messages": serialize_messages(kept_messages), "message_summary": updated_summary},
            )
        return updated_summary, kept_messages, remove_ops
