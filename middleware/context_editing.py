"""
Context editing: token-aware truncation and running summarisation.

When the estimated token count of ``state["messages"]`` exceeds the
configured threshold, old messages are evicted at a safe turn boundary
and summarised into a running summary.  The orchestrator calls a single
function — ``truncate_and_summarize`` — which handles both steps.

Uses ``RemoveMessage`` to work with the ``add_messages`` reducer.
Truncation never orphans a ``ToolMessage`` from its parent ``AIMessage``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

_ROOT = Path(__file__).resolve().parent.parent
_cfg  = yaml.safe_load(open(_ROOT / "config.yaml"))
_SUMMARY_MODEL = _cfg["models"].get("summarization", "gpt-4o-mini")

_SUMMARY_SYSTEM = """\
You are a conversation summariser. Produce a concise running summary that
preserves all important facts, data references, file paths, column names,
analysis results, and user preferences mentioned so far.

If a previous summary is provided, integrate the new messages into it —
do not repeat information already captured. Focus on what is new."""


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: total characters of message content ÷ 4."""
    return sum(len(str(getattr(m, "content", "") or "")) for m in messages) // 4


def find_safe_truncation_point(messages: list, keep: int) -> int:
    """
    Index into *messages* at which truncation can safely begin.

    Walks forward from the naive cut point until a ``HumanMessage`` or a
    non-tool-calling ``AIMessage`` is found, so the kept window never
    starts with an orphaned ``ToolMessage``.
    """
    candidate = max(0, len(messages) - keep)
    while candidate < len(messages):
        msg = messages[candidate]
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            break
        candidate += 1
    return candidate


def summarize_evicted(previous_summary: str, messages_to_evict: list) -> str:
    """LLM call: merge *previous_summary* with *messages_to_evict* into an updated summary."""
    llm = ChatOpenAI(model=_SUMMARY_MODEL, temperature=0)

    conversation = "\n".join(
        f"{type(m).__name__}: {m.content}"
        for m in messages_to_evict
        if hasattr(m, "content") and m.content
    )

    user_content = ""
    if previous_summary:
        user_content += f"## Previous Summary\n{previous_summary}\n\n"
    user_content += f"## New Messages to Integrate\n{conversation}"

    response = llm.invoke([
        SystemMessage(content=_SUMMARY_SYSTEM),
        HumanMessage(content=user_content),
    ])
    return response.content


def truncate_and_summarize(
    messages: list,
    previous_summary: str,
    keep: int,
    token_threshold: int,
) -> tuple[str, list, list[RemoveMessage]]:
    """
    Truncate *messages* and update the running summary if token budget is exceeded.

    Parameters
        messages        — full message list from state.
        previous_summary — existing running summary (may be empty).
        keep            — number of recent messages to retain after truncation.
        token_threshold — trigger truncation when estimated tokens exceed this.

    Returns
        ``(updated_summary, kept_messages, remove_ops)``

        *updated_summary* — new running summary (unchanged if no truncation).
        *kept_messages*   — the message slice to send to the LLM.
        *remove_ops*      — ``RemoveMessage`` list for the ``add_messages`` reducer.

        When the token count is within budget, returns the inputs unchanged
        with an empty *remove_ops* list.
    """
    if estimate_tokens(messages) <= token_threshold or len(messages) <= keep:
        return previous_summary, messages, []

    cut = find_safe_truncation_point(messages, keep)
    to_evict = messages[:cut]
    remove_ops = [RemoveMessage(id=m.id) for m in to_evict]
    updated_summary = summarize_evicted(previous_summary, to_evict)
    kept_messages = messages[cut:]

    return updated_summary, kept_messages, remove_ops
