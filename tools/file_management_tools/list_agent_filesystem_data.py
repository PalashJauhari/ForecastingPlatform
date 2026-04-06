"""
LangChain tool: list all .csv and .xlsx files inside agent_filesystem/.
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from langchain_core.tools import tool
from langfuse import observe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
AGENT_FILESYSTEM_ROOT = PROJECT_ROOT / cfg["paths"]["agent_filesystem"]


@observe(name="tool.list_agent_filesystem_data", as_type="tool")
def _list_agent_filesystem_data_impl() -> str:
    """
    Scan agent_filesystem/ recursively and return all .csv and .xlsx files found.

    Call this first to discover what data files are available. Each path in the
    result starts with 'agent_filesystem/' so it can be passed directly to
    read_agent_filesystem_data or used as an input path in ``code_pipeline``.

    Returns a JSON string with two keys:
      - "files": list of matching file paths, e.g.
            [
              "agent_filesystem/input/sales.csv",
              "agent_filesystem/processed/clean.xlsx"
            ]
      - "count": total number of files found (int).

    If agent_filesystem/ does not exist, returns {"error": "...", "files": [], "count": 0}.
    If no .csv or .xlsx files are found, returns {"files": [], "count": 0}.
    """
    base = AGENT_FILESYSTEM_ROOT.resolve()

    if not base.exists():
        return json.dumps({"error": "agent_filesystem/ does not exist.", "files": [], "count": 0})

    files = [
        f"agent_filesystem/{f.relative_to(base).as_posix()}"
        for f in sorted(base.rglob("*"))
        if f.is_file() and f.suffix.lower() in {".csv", ".xlsx"}
    ]

    return json.dumps({"files": files, "count": len(files)})


@tool
def list_agent_filesystem_data() -> str:
    """LangChain wrapper for the traced filesystem listing implementation."""
    return _list_agent_filesystem_data_impl()
