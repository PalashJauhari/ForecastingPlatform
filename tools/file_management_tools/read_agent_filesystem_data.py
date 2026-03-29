"""
LangChain tool: read a .csv or .xlsx file from agent_filesystem/ and return the top 5 rows.
"""

from __future__ import annotations

import json
import yaml
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
AGENT_FILESYSTEM_ROOT = (PROJECT_ROOT / cfg["paths"]["agent_filesystem"]).resolve()

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


class ReadAgentFilesystemDataInput(BaseModel):
    path: str = Field(
        description=(
            "Path to a .csv or .xlsx file starting with 'agent_filesystem/' "
            "(e.g. 'agent_filesystem/input/data.csv'). Use list_agent_filesystem_data to discover files."
        ),
    )


@tool(args_schema=ReadAgentFilesystemDataInput)
def read_agent_filesystem_data(path: str) -> str:
    """
    Read a .csv or .xlsx file from agent_filesystem/ and return its schema and a sample of rows.

    Can be used any time you need to inspect or understand the contents of a data file —
    not just before generate_code. Use it to answer questions about the data, check column
    names, verify row counts, or preview values.

    Using it before generate_code is recommended so you can pass the returned columns and
    sample rows as data_schema, giving the code-generation model precise knowledge of the
    file structure.

    Args:
        path: Full path to the file, must start with 'agent_filesystem/'
              e.g. 'agent_filesystem/input/sales.csv'.
              Use list_agent_filesystem_data first to get valid paths.

    Returns a JSON string with the following keys on success:
      - "path"       : the path you passed in (string).
      - "columns"    : list of column names, e.g. ["date", "revenue", "region"].
      - "rows"       : first 5 rows as a list of dicts, e.g.
                       [{"date": "2024-01-01", "revenue": 1200.5, "region": "North"}, ...]
      - "total_rows" : total number of rows in the file (int).

    Returns a JSON string with an "error" key on failure:
      - Path does not start with 'agent_filesystem/'
      - Path resolves outside agent_filesystem/ (path traversal rejected)
      - File not found
      - Path is a directory
      - File extension is not .csv or .xlsx
      - File could not be parsed by pandas
    """
    p = path.strip()

    if not p.startswith("agent_filesystem/"):
        return json.dumps({"error": "Path must start with 'agent_filesystem/' (e.g. 'agent_filesystem/input/data.csv')."})

    relative = p[len("agent_filesystem/"):]

    if not relative:
        return json.dumps({"error": "No file path provided after 'agent_filesystem/'."})

    rel_path = Path(relative)

    if rel_path.is_absolute():
        return json.dumps({"error": "Path must be inside agent_filesystem/."})

    target = (AGENT_FILESYSTEM_ROOT / rel_path).resolve()

    try:
        target.relative_to(AGENT_FILESYSTEM_ROOT)
    except ValueError:
        return json.dumps({"error": "Path must be inside agent_filesystem/."})

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

    return json.dumps({
        "path": p,
        "columns": list(df.columns),
        "rows": df.head(5).to_dict(orient="records"),
        "total_rows": len(df),
    }, default=str)
