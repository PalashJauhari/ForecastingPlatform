"""
LangChain tool: read a .csv or .xlsx file from agent_filesystem/ and return schema + a preview of rows.
"""

from __future__ import annotations

import json
import pandas as pd
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langfuse import observe
from pydantic import BaseModel, Field

from session_paths import resolve_agent_path, session_id_from_config

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


class ReadAgentFilesystemDataInput(BaseModel):
    path: str = Field(
        description=(
            "Path to a .csv or .xlsx file starting with 'agent_filesystem/' "
            "(e.g. 'agent_filesystem/input/data.csv'). Use list_agent_filesystem_data to discover files."
        ),
    )
    n_rows: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "How many rows to return as a preview from the start of the file. "
            "Default 5; maximum 100 to limit context size."
        ),
    )


@observe(name="tool.read_agent_filesystem_data", as_type="tool")
def _read_agent_filesystem_data_impl(path: str, n_rows: int = 5, runtime: ToolRuntime | None = None) -> str:
    """
    Read a .csv or .xlsx file from agent_filesystem/ and return its schema and a sample of rows.

    Can be used any time you need to inspect or understand the contents of a data file —
    not just before ``code_pipeline``. Use it to answer questions about the data, check column
    names, verify row counts, or preview values.

    Using it before ``code_pipeline`` is recommended so you can pass the returned columns and
            sample rows as a data profile snippet, giving the code-generation model precise knowledge of the
    file structure.

    Args:
        path: Full path to the file, must start with 'agent_filesystem/'
              e.g. 'agent_filesystem/input/sales.csv'.
              Use list_agent_filesystem_data first to get valid paths.
        n_rows: Number of preview rows from the top of the file (default 5, max 100).

    Returns a JSON string with the following keys on success:
      - "path"            : the path you passed in (string).
      - "preview_n_rows"  : how many preview rows were returned (same as n_rows, capped 1–100).
      - "columns"         : list of column names, e.g. ["date", "revenue", "region"].
      - "rows"            : first n_rows rows as a list of dicts, e.g.
                            [{"date": "2024-01-01", "revenue": 1200.5, "region": "North"}, ...]
      - "total_rows"      : total number of rows in the file (int).

    Returns a JSON string with an "error" key on failure:
      - Path does not start with 'agent_filesystem/'
      - Path resolves outside agent_filesystem/ (path traversal rejected)
      - File not found
      - Path is a directory
      - File extension is not .csv or .xlsx
      - File could not be parsed by pandas
    """
    p = path.strip()
    session_id = session_id_from_config(runtime.config if runtime is not None else None)

    if not p.startswith("agent_filesystem/"):
        return json.dumps({"error": "Path must start with 'agent_filesystem/' (e.g. 'agent_filesystem/input/data.csv')."})

    try:
        target = resolve_agent_path(session_id, p)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if not target.exists():
        return json.dumps({"error": f"File not found: {p}"})

    if target.is_dir():
        return json.dumps({"error": "Path is a directory. Use list_agent_filesystem_data to browse files."})

    ext = target.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return json.dumps({"error": f"Only .csv and .xlsx files are supported. Got: '{ext}'"})

    try:
        df = pd.read_csv(target) if ext == ".csv" else pd.read_excel(target)
    except Exception as e:
        return json.dumps({"error": f"Failed to read file: {e}"})

    n = max(1, min(int(n_rows), 100))
    preview = df.head(n).to_dict(orient="records")

    return json.dumps({
        "path": p,
        "preview_n_rows": n,
        "columns": list(df.columns),
        "rows": preview,
        "total_rows": len(df),
    }, default=str)


@tool(args_schema=ReadAgentFilesystemDataInput)
def read_agent_filesystem_data(
    path: str,
    n_rows: int = 5,
    runtime: ToolRuntime | None = None,
) -> str:
    """LangChain wrapper for the traced data-read implementation."""
    return _read_agent_filesystem_data_impl(path=path, n_rows=n_rows, runtime=runtime)
