"""
Write one entry to session ``scratchpad`` state (list of strings).

Uses ``Command`` so ``ToolNode`` merges via the graph reducer and appends the
full ``note`` to state; ``ToolMessage`` content is the caller-supplied ``summary``.
"""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langfuse import observe
from langgraph.types import Command

from observability.langfuse_handler import get_langfuse_client
from output_validation.scratchpad import WriteScratchpadInput

langfuse = get_langfuse_client()


@observe(name="tool.write_scratchpad", as_type="tool")
def _write_scratchpad_impl(note: str, summary: str, runtime: ToolRuntime) -> Command:
    validated = WriteScratchpadInput(note=note, summary=summary)
    text = validated.note
    tool_content = validated.summary

    with langfuse.start_as_current_observation(
        name="tool.write_scratchpad.state_update",
        as_type="span",
        input={"note_length": len(text), "summary_length": len(tool_content)},
    ) as span:
        span.update(output={"tool_message": tool_content}, metadata={"note_chars": len(text)})

    return Command(
        update={
            "scratchpad": [text],
            "messages": [
                ToolMessage(
                    content=tool_content,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        },
    )


@tool(args_schema=WriteScratchpadInput)
def write_scratchpad(note: str, summary: str, runtime: ToolRuntime) -> Command:
    """Persist a **detailed note** to session scratchpad state; **summary** is the tool result text in chat.

    **Scratchpad** JSON (each turn) contains the full ``note`` strings in order. The ``ToolMessage`` content is
    **only** ``summary``: **1–2 short lines** (≤400 characters, at most one line break); put depth in ``note``.

    ## When to use
    - **Insights and findings** (detailed in ``note``; ``summary`` = 1–2 line takeaway for chat).
    - **Assumptions and decisions** the user or you made.
    - **Open questions / blockers** to resolve later.
    - **Context that may vanish** after summarisation—full detail in ``note``, gist in ``summary``.
    - **User preferences or constraints** (forecast horizon, output format, business rules).
    - **Links between steps** (paths written, intermediate artifacts).

    ## When not to use
    - **Trivial or redundant** entries.
    - **Dumping large tables**—use workspace files; reference paths in ``note``/``summary``.
    - **Structured step lists**—use **write_todos** instead.

    ## Behaviour
    - Each call **appends** ``note`` to state; earlier entries are preserved.
    - **summary** (1–2 lines max) is the tool return in the message history; **note** is under **Scratchpad** in orchestrator context.
    """
    return _write_scratchpad_impl(note=note, summary=summary, runtime=runtime)
