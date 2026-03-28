"""
LangChain tool: append to the agent scratchpad (``agent_filesystem/scratchpad/scratchpad.md``).

The scratchpad is cleared automatically at the start of each new session
by the ``refresh_data_schema`` graph node.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent.parent
_cfg  = yaml.safe_load(open(_ROOT / "config.yaml"))
_SCRATCHPAD_DIR  = _ROOT / _cfg["paths"]["scratchpad"]
_SCRATCHPAD_FILE = _SCRATCHPAD_DIR / "scratchpad.md"


class WriteScratchpadInput(BaseModel):
    """Arguments for ``write_scratchpad``."""

    content: str = Field(
        description="Text to append to the scratchpad.",
    )


@tool(args_schema=WriteScratchpadInput)
def write_scratchpad(content: str) -> str:
    """
    Append text to the agent's scratchpad (agent_filesystem/scratchpad/scratchpad.md).

    Use this to save intermediate findings, working notes, or plans
    that may be useful in later steps of the analysis.

    Returns a JSON string with:
      - ``"status"`` — ``"ok"``
      - ``"path"``   — the scratchpad file path.
    """
    _SCRATCHPAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SCRATCHPAD_FILE, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    return json.dumps({"status": "ok", "path": str(_SCRATCHPAD_FILE)})
