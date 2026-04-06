"""
LangChain tool: read the agent scratchpad (``agent_filesystem/scratchpad/scratchpad.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.tools import tool
from langfuse import observe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
SCRATCHPAD = PROJECT_ROOT / cfg["paths"]["scratchpad"] / "scratchpad.md"


@observe(name="tool.read_scratchpad", as_type="tool")
def _read_scratchpad_impl() -> str:
    """
    Read the agent's scratchpad (agent_filesystem/scratchpad/scratchpad.md).

    Use this to review working notes, intermediate findings, or plans
    saved during earlier steps of the analysis.

    Returns a JSON string with:
      - ``"content"`` — full text of scratchpad.md (empty string if the
        file does not exist yet).
    """
    if not SCRATCHPAD.exists():
        return json.dumps({"content": ""})
    return json.dumps({"content": SCRATCHPAD.read_text(encoding="utf-8")})


@tool
def read_scratchpad() -> str:
    """LangChain wrapper for the traced scratchpad-read implementation."""
    return _read_scratchpad_impl()
