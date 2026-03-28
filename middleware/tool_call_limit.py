"""
Tool call limit check.

Returns an ``AIMessage`` instructing the agent to stop when the
configured maximum number of individual tool invocations has been
reached. Parallel tool calls count individually (3 parallel = 3).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage


def check_tool_call_limit(number_of_tool_calls: int, max_calls: int) -> AIMessage | None:
    """
    Return an ``AIMessage`` if *number_of_tool_calls* >= *max_calls*, else ``None``.

    The returned message tells the user the agent has exhausted its
    tool call budget for this invocation.
    """
    if number_of_tool_calls >= max_calls:
        return AIMessage(
            content=(
                f"Tool call limit reached ({max_calls}). "
                "Summarising what I have so far and stopping."
            )
        )
    return None
