"""
LangChain tool: append to the agent scratchpad (``agent_filesystem/scratchpad/scratchpad.md``).
"""

from __future__ import annotations

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langfuse import observe
from pydantic import BaseModel, Field

from session_paths import ensure_session_dirs, resolve_agent_path, session_id_from_config

SCRATCHPAD_PATH = "agent_filesystem/scratchpad/scratchpad.md"


class WriteScratchpadInput(BaseModel):
    """Arguments for ``write_scratchpad``."""

    content: str = Field(
        description="Text to append to the scratchpad.",
    )


@observe(name="tool.write_scratchpad", as_type="tool")
def _write_scratchpad_impl(content: str, runtime: ToolRuntime) -> str:
    """
    Append text to the agent's scratchpad (agent_filesystem/scratchpad/scratchpad.md).

    Use this to save intermediate findings, working notes, or plans
    that may be useful in later steps of the analysis.

    Returns a JSON string with:
      - ``"status"`` — ``"ok"``
      - ``"path"``   — the scratchpad file path.
    """
    session_id = session_id_from_config(runtime.config)
    ensure_session_dirs(session_id)
    scratchpad = resolve_agent_path(session_id, SCRATCHPAD_PATH)
    with open(scratchpad, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    return json.dumps({"status": "ok", "path": SCRATCHPAD_PATH})


@tool(args_schema=WriteScratchpadInput)
def write_scratchpad(content: str, runtime: ToolRuntime) -> str:
    """LangChain wrapper for the traced scratchpad-write implementation."""
    return _write_scratchpad_impl(content=content, runtime=runtime)
