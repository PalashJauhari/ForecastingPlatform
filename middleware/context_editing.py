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
from langfuse import observe

from middleware.llm_client import make_llm
from observability.langfuse_handler import (
    extract_usage_details,
    get_langfuse_client,
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


@observe(name="context.summarize_evicted", capture_input=False, capture_output=False)
def summarize_evicted(previous_summary: str, messages_to_evict: list) -> str:
    """LLM call: merge *previous_summary* with *messages_to_evict* into an updated summary."""
    llm = make_llm(model=SUMMARY_MODEL, temperature=0)

    prompt_messages = [SystemMessage(content=SUMMARY_SYSTEM)]
    if previous_summary:
        prompt_messages.append(HumanMessage(content=f"## Previous Summary\n{previous_summary}"))
    prompt_messages.extend(messages_to_evict)
    prompt_messages.append(HumanMessage(content="Update the running summary using the evicted messages above."))
    with langfuse.start_as_current_observation(name="context.summarize_evicted.llm", as_type="generation", model=SUMMARY_MODEL, input=[serialize_message(message) for message in prompt_messages]) as generation:
        response = llm.invoke(prompt_messages)
        generation.update(output=serialize_message(response), usage_details=extract_usage_details(response))
    return response.content


@observe(name="context.truncate_and_summarize", capture_input=False, capture_output=False)
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
        langfuse.update_current_span(metadata={"token_estimate": estimate_tokens(messages), "token_threshold": token_threshold, "truncated": "false"})
        return previous_summary, messages, []

    cut = find_safe_truncation_point(messages, keep)
    to_evict = messages[:cut]
    remove_ops = [RemoveMessage(id=m.id) for m in to_evict]
    updated_summary = summarize_evicted(previous_summary, to_evict)
    kept_messages = messages[cut:]
    langfuse.update_current_span(metadata={"token_estimate": estimate_tokens(messages), "token_threshold": token_threshold, "truncated": "true", "evicted_count": len(to_evict)})

    return updated_summary, kept_messages, remove_ops
