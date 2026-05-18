"""
Context editing: token-aware truncation and running summarisation.

When the estimated token count of ``state["messages"]`` exceeds the
configured threshold, old messages are evicted at a **HumanMessage** boundary
and summarised into a running summary.

Uses ``RemoveMessage`` to work with the ``add_messages`` reducer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    extract_usage_details,
    get_langfuse_client,
    observation_parented_to_run,
    serialize_message,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
SUMMARY_MODEL = cfg["models"].get("message_summarisation", "gpt-4o-mini")
langfuse = get_langfuse_client()

SUMMARY_SYSTEM = """\
You are a conversation summariser. Produce a concise running summary that
preserves all important facts, data references, file paths, column names,
analysis results, and user preferences mentioned so far.

If a previous summary is provided, integrate the new messages into it —
do not repeat information already captured. Focus on what is new."""


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: total characters of message content ÷ 4."""
    return sum(len(str(getattr(m, "content", "") or "")) for m in messages) // 4


def find_human_truncation_cut(messages: list, keep: int) -> int | None:
    """Cut index at ``user_input`` / ``user_input-{uuid}`` Human if possible, else first Human ahead of naive cut."""
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

    llm = make_llm(model=SUMMARY_MODEL, temperature=0)

    prompt_messages = [SystemMessage(content=SUMMARY_SYSTEM)]
    if previous_summary:
        prompt_messages.append(HumanMessage(content=f"## Previous Summary\n{previous_summary}"))
    prompt_messages.extend(messages_to_evict)
    prompt_messages.append(HumanMessage(content="Update the running summary using the evicted messages above."))

    serial_in = [serialize_message(message) for message in prompt_messages]
    with observation_parented_to_run(
        langfuse,
        runnable_config,
        name="context.summarize_evicted",
        as_type="chain",
    ):
        with observation_parented_to_run(
            langfuse,
            runnable_config,
            name="context.summarize_evicted.llm",
            as_type="generation",
            model=SUMMARY_MODEL,
            input=serial_in,
        ) as generation:
            response = await llm.ainvoke(prompt_messages)
            generation.update(output=serialize_message(response), usage_details=extract_usage_details(response))
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
    with observation_parented_to_run(
        langfuse,
        runnable_config,
        name="context.truncate_and_summarize",
        as_type="chain",
    ):
        if estimate_tokens(messages) <= token_threshold or len(messages) <= keep:
            langfuse.update_current_span(metadata={"token_estimate": estimate_tokens(messages), "token_threshold": token_threshold, "truncated": "false"})
            return previous_summary, messages, []

        cut = find_human_truncation_cut(messages, keep)
        if cut is None:
            langfuse.update_current_span(
                metadata={
                    "token_estimate": estimate_tokens(messages),
                    "token_threshold": token_threshold,
                    "truncated": "false",
                    "truncate_skip": "no_human_boundary",
                },
            )
            return previous_summary, messages, []

        to_evict = messages[:cut]
        remove_ops = [RemoveMessage(id=m.id) for m in to_evict]
        updated_summary = await summarize_evicted(
            previous_summary,
            to_evict,
            runnable_config=runnable_config,
        )
        langfuse.update_current_span(
            metadata={
                "token_estimate": estimate_tokens(messages),
                "token_threshold": token_threshold,
                "truncated": "true",
                "evicted_count": len(to_evict),
            },
        )
        kept_messages = messages[cut:]
        return updated_summary, kept_messages, remove_ops
