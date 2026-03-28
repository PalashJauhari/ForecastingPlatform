"""
LangChain tool: read the agent scratchpad (``agent_filesystem/scratchpad/scratchpad.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.tools import tool

_ROOT = Path(__file__).resolve().parent.parent.parent
_cfg  = yaml.safe_load(open(_ROOT / "config.yaml"))
_SCRATCHPAD = _ROOT / _cfg["paths"]["scratchpad"] / "scratchpad.md"


@tool
def read_scratchpad() -> str:
    """
    Read the agent's scratchpad (agent_filesystem/scratchpad/scratchpad.md).

    Use this to review working notes, intermediate findings, or plans
    saved during earlier steps of the analysis.

    Returns a JSON string with:
      - ``"content"`` — full text of scratchpad.md (empty string if the
        file does not exist yet).
    """
    if not _SCRATCHPAD.exists():
        return json.dumps({"content": ""})
    return json.dumps({"content": _SCRATCHPAD.read_text(encoding="utf-8")})
