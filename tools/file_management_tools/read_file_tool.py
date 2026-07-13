# NOTE: Do NOT add ``from __future__ import annotations`` here. LangChain's
# ``@tool`` decorator introspects function annotations to detect the
# ``ToolRuntime`` parameter for auto-injection by LangGraph.

"""
Read rows from a session CSV/XLSX file for orchestrator inspection.

Returns a JSON table preview capped by READ_FILE_MAX_ROWS (default 5000).
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from output_validation.read_file_tool import ReadFileToolInput
from session_paths import session_root, session_id_from_config


def read_file_max_rows() -> int:
    """Max rows returned by read_file_tool; configured via READ_FILE_MAX_ROWS."""
    raw = (os.environ.get("READ_FILE_MAX_ROWS") or "").strip()
    if not raw:
        return 5000
    try:
        value = int(raw)
    except ValueError:
        return 5000
    return max(1, value)


def to_json_value(value: Any) -> Any:
    """Convert pandas/numpy cell values to JSON-safe Python."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def read_tabular_file(path: Path) -> pd.DataFrame:
    """Load one supported tabular file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


@tool(args_schema=ReadFileToolInput)
def read_file_tool(
    file_name: str,
    columns: list[str],
    runtime: ToolRuntime,
) -> str:
    """Read rows from a session CSV/XLSX file; return JSON table preview.

    ## Purpose
    Return actual cell values from one uploaded file for inspection before forecasting
    or coding. Use when data_profile head (5 rows) is not enough.

    ## When to use
    - Preview values and columns before choosing forecast tool arguments.
    - Spot-check a subset of columns after a file was written by another tool.

    ## When NOT to use
    - Schema/stats only → use ``data_profile`` (dtypes, head, column_profiles).
    - Plots, transforms, aggregations, or stats → ``coding_tool``.
    - Forecasting → ``sarima_tool``, ``prophet_tool``, or ``holt_winters_tool``.

    ## How to call
    - ``file_name`` — basename from ``data_profile`` only (e.g. ``monthly_revenue.csv``).
    - ``columns`` — list of column names to return; ``[]`` returns all columns.

    ## Output (JSON string)
    On success: ``status``, ``file``, ``columns``, ``rows_total``, ``rows_returned``,
    ``truncated``, ``data`` (list of row dicts).
    On error: ``status`` ``error``, ``stage``, ``error`` with ``code`` and ``message``.
    """
    params = ReadFileToolInput(file_name=file_name, columns=columns)
    session_id = session_id_from_config(runtime.config)
    basename = Path(params.file_name).name
    path = session_root(session_id) / basename

    if not path.exists() or not path.is_file():
        return json.dumps(
            {
                "status": "error",
                "stage": "data_validation",
                "error": {
                    "code": "file_not_found",
                    "message": f"File '{basename}' was not found in the session workspace.",
                },
            },
            default=str,
        )

    try:
        df = read_tabular_file(path)
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "stage": "data_validation",
                "error": {"code": "read_failed", "message": str(exc)[:500]},
            },
            default=str,
        )

    requested = list(params.columns)
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            return json.dumps(
                {
                    "status": "error",
                    "stage": "data_validation",
                    "error": {
                        "code": "missing_required_columns",
                        "message": f"Columns not found in '{basename}': {missing}.",
                    },
                },
                default=str,
            )
        df = df[requested]
        out_columns = requested
    else:
        out_columns = [str(c) for c in df.columns]

    rows_total = int(len(df))
    max_rows = read_file_max_rows()
    truncated = rows_total > max_rows
    slice_df = df.head(max_rows)
    records = [
        {str(col): to_json_value(row[col]) for col in slice_df.columns}
        for row in slice_df.to_dict(orient="records")
    ]

    return json.dumps(
        {
            "status": "success",
            "file": basename,
            "columns": out_columns,
            "rows_total": rows_total,
            "rows_returned": len(records),
            "truncated": truncated,
            "data": records,
        },
        default=str,
    )
