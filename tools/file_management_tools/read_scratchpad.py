"""
LangChain tool: read the agent scratchpad (``agent_filesystem/scratchpad/scratchpad.md``).
"""

from __future__ import annotations

import json

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langfuse import observe

from session_paths import resolve_agent_path, session_id_from_config

SCRATCHPAD_PATH = "agent_filesystem/scratchpad/scratchpad.md"


@observe(name="tool.read_scratchpad", as_type="tool")
def _read_scratchpad_impl(runtime: ToolRuntime) -> str:
    """
    Read the agent's scratchpad (agent_filesystem/scratchpad/scratchpad.md).

    Use this to review working notes, intermediate findings, or plans
    saved during earlier steps of the analysis.

    Returns a JSON string with:
      - ``"content"`` — full text of scratchpad.md (empty string if the
        file does not exist yet).
    """
    session_id = session_id_from_config(runtime.config)
    scratchpad = resolve_agent_path(session_id, SCRATCHPAD_PATH)
    if not scratchpad.exists():
        return json.dumps({"content": ""})
    return json.dumps({"content": scratchpad.read_text(encoding="utf-8")})


@tool
def read_scratchpad(runtime: ToolRuntime) -> str:
    """LangChain wrapper for the traced scratchpad-read implementation."""
    return _read_scratchpad_impl(runtime=runtime)
