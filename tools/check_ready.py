"""
check_ready: validate that processed/clean.csv is ready for forecasting.

Flow:
  1. Read data_dir from session context; resolve clean_path = data_dir/processed/clean.csv.
  2. If file missing or unreadable, return ready=False.
  3. Check: infer date column, infer numeric target, no nulls in date/target.
  4. Return JSON with ready (bool), message (str), and optional date_col, target_col, rows.

Used as a gate before forecasting; orchestrator should call after clean_csv.
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from graph.middleware.session_context import data_dir_var


def _infer_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return first column that looks like date/datetime, else None."""
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "datetime" in dtype or "date" in dtype:
            return col
        try:
            pd.to_datetime(df[col].dropna().head(1))
            return col
        except Exception:
            continue
    return None


def _infer_numeric_target(df: pd.DataFrame, date_col: Optional[str]) -> Optional[str]:
    """Return first numeric column that is not the date column."""
    for col in df.columns:
        if col == date_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


@tool
def check_ready() -> str:
    """Check if processed/clean.csv is ready for forecasting. Call this AFTER clean_csv and before running any forecast. No arguments; uses session data path. Validates: file exists, has a date/datetime column, has a numeric target column, no nulls in those columns. Returns JSON: {ready: bool, message: str, date_col?: str, target_col?: str, rows?: int}. If ready is false, use the message to tell the user what is missing."""
    # --- Resolve path from session context ---
    data_dir = data_dir_var.get()
    clean_path = Path(data_dir) / "processed" / "clean.csv"
    if not clean_path.exists():
        return json.dumps({"ready": False, "message": "No cleaned file found. Run clean_csv first."})

    try:
        df = pd.read_csv(clean_path)
    except Exception as e:
        return json.dumps({"ready": False, "message": f"Could not read clean.csv: {e}"})

    # --- Infer date and numeric target columns ---
    date_col = _infer_date_column(df)
    if not date_col:
        return json.dumps({"ready": False, "message": "No date/datetime column detected."})

    target_col = _infer_numeric_target(df, date_col)
    if not target_col:
        return json.dumps({"ready": False, "message": "No numeric target column found."})

    # --- Require no nulls in date/target columns ---
    key_cols = [c for c in [date_col, target_col] if c]
    nulls = df[key_cols].isnull().sum()
    if nulls.any():
        bad = nulls[nulls > 0].to_dict()
        return json.dumps({"ready": False, "message": f"Nulls in key columns: {bad}"})

    # --- All checks passed ---
    return json.dumps({
        "ready": True,
        "message": "Data is ready for forecasting.",
        "date_col": date_col,
        "target_col": target_col,
        "rows": len(df),
    })
