"""
peek_csv: inspect the session CSV (schema, dtypes, nulls, sample rows).

Flow:
  1. Read csv_path from session context (set by SessionContextMiddleware).
  2. Load CSV (up to 1000 rows); build structured peek (columns, dtypes, null_counts, sample_rows).
  3. Return JSON string for orchestrator or downstream tools.

No arguments; path comes from context. Same peek shape as get_peek_data() in clean_csv/ask_csv.
"""
import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from graph.middleware.session_context import csv_path_var


@tool
def peek_csv() -> str:
    """Inspect the session CSV: schema, column types, null counts, and sample rows. Use this FIRST when the user uploads a new file or asks about the data shape. No arguments; uses the CSV path from the current session. Returns a JSON string with keys: columns, dtypes, null_counts, sample_rows, total_rows_preview. Call before clean_csv or ask_csv so you know the column names and data shape."""
    # --- Resolve path from session context ---
    csv_path = csv_path_var.get()
    if not csv_path or not Path(csv_path).exists():
        return json.dumps({"error": "No CSV loaded for this session or file not found."})

    # --- Load CSV and build structured peek (same shape as get_peek_data in clean_csv) ---
    try:
        df = pd.read_csv(csv_path, nrows=1000)
    except Exception as e:
        return json.dumps({"error": str(e)})

    columns = list(df.columns)
    dtypes = {c: str(df.dtypes[c]) for c in columns}
    null_counts = df.isnull().sum().to_dict()
    sample_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")

    # --- Return JSON for orchestrator / downstream tools ---
    return json.dumps({
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "sample_rows": sample_rows,
        "total_rows_preview": len(df),
    }, default=str)
