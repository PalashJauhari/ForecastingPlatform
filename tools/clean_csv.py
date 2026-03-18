"""
clean_csv: generate and run pandas cleaning code via an internal LLM.

Flow:
  1. Read session csv_path/data_dir from context; resolve output to data_dir/processed/clean.csv.
  2. Build peek data (schema, dtypes, nulls, sample) and user_content for the LLM.
  3. Call LLM with CLEANING_CODE_GENERATOR_PROMPT; parse JSON response for "code" key.
  4. Run AST safety check; execute code in subprocess with preamble (imports + input_path/output_path).
  5. Verify output file exists and is valid CSV; return JSON result.

Model: CODING_MODEL env (default gpt-4o-mini). Response format: JSON with "code" key only.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graph.middleware.session_context import csv_path_var, data_dir_var
from prompts.cleaning_prompt import CLEANING_CODE_GENERATOR_PROMPT
from tools.safety import check_code_safety


class CleanCsvInput(BaseModel):
    """Arguments for clean_csv. Pass user instructions for how to clean; leave empty for automatic cleaning."""

    user_instructions: str = Field(
        default="",
        description="Optional instructions for cleaning. E.g. 'drop column X', 'fill nulls with 0', 'normalise dates'. Leave empty to let the tool infer cleaning from the data peek.",
    )


def get_peek_data(csv_path: str) -> dict:
    """
    Build peek dict for LLM context: columns, dtypes, null_counts, sample_rows, total_rows_preview.
    Used by both clean_csv and ask_csv so the model knows the data shape.
    """
    df = pd.read_csv(csv_path, nrows=1000)
    return {
        "columns": list(df.columns),
        "dtypes": {c: str(df.dtypes[c]) for c in df.columns},
        "null_counts": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).fillna("").astype(str).to_dict(orient="records"),
        "total_rows_preview": len(df),
    }


@tool(args_schema=CleanCsvInput)
def clean_csv(user_instructions: str = "") -> str:
    """Clean the session CSV and save to processed/clean.csv. Use when the user wants to fix or prepare the data (nulls, dtypes, column names, dates). Call after peek_csv so the tool has context. An internal LLM generates pandas code; do NOT use ask_csv for cleaning. Optional: pass user_instructions with specific requests (e.g. 'drop column X', 'fill nulls with 0'). Returns JSON: {cleaned: bool, path: str, error: str, snapshot?: {columns, dtypes, sample_rows, total_rows}} when successful."""
    # --- Resolve paths from session context ---
    csv_path = csv_path_var.get()
    data_dir = data_dir_var.get()
    if not csv_path or not Path(csv_path).exists():
        return json.dumps({"cleaned": False, "path": "", "error": "No CSV loaded or file not found."})

    out_dir = Path(data_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "clean.csv"

    # --- Build peek data and user_content (same pattern as ask_csv) ---
    try:
        peek = get_peek_data(csv_path)
    except Exception as e:
        return json.dumps({"cleaned": False, "path": "", "error": str(e)})

    user_content = (
        "User instructions: "
        + (user_instructions.strip() if user_instructions else "(none)")
        + "\n\nPeek data (schema/sample):\n"
        + json.dumps(peek, default=str)
    )

    # --- LLM call: enforce JSON response with "code" key ---
    coding_model = os.environ.get("CODING_MODEL", "gpt-4o-mini")
    model = ChatOpenAI(
        model=coding_model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    messages = [SystemMessage(content=CLEANING_CODE_GENERATOR_PROMPT), HumanMessage(content=user_content)]
    try:
        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        data = json.loads(content)
        code = (data.get("code") or "").strip()
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        return json.dumps({"cleaned": False, "path": "", "error": f"LLM did not return valid JSON with 'code' key: {e}"})

    if not code:
        return json.dumps({"cleaned": False, "path": "", "error": "No code generated."})

    # --- Safety check: only allowed modules (pandas, numpy, sklearn, scipy) and no file/exec ---
    is_safe, violations = check_code_safety(code)
    if not is_safe:
        return json.dumps({
            "cleaned": False,
            "path": "",
            "error": "Generated code failed safety check: " + "; ".join(violations),
        })

    # --- Preamble: same imports as ask_csv, then inject input_path/output_path for generated code ---
    preamble = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import sklearn\n"
        "import scipy\n"
        f"input_path = {repr(str(csv_path))}\n"
        f"output_path = {repr(str(output_path))}\n"
    )
    full_code = preamble + code

    # --- Run in subprocess; cwd = session data_dir ---
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        script_path = f.name
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(Path(data_dir).resolve()),
        )
        if result.returncode != 0:
            return json.dumps({
                "cleaned": False,
                "path": "",
                "error": (result.stderr or result.stdout or "Subprocess failed.").strip()[:500],
            })
    finally:
        Path(script_path).unlink(missing_ok=True)

    # --- Verify output: file exists and is valid CSV; build snapshot of cleaned data ---
    if not output_path.exists():
        return json.dumps({"cleaned": False, "path": "", "error": "Output file was not created."})
    try:
        df_out = pd.read_csv(output_path)
    except Exception as e:
        return json.dumps({"cleaned": False, "path": "", "error": f"Output CSV invalid: {e}"})

    snapshot = {
        "columns": list(df_out.columns),
        "dtypes": {c: str(df_out.dtypes[c]) for c in df_out.columns},
        "sample_rows": df_out.head(5).fillna("").astype(str).to_dict(orient="records"),
        "total_rows": len(df_out),
    }
    return json.dumps({
        "cleaned": True,
        "path": str(output_path),
        "error": "",
        "snapshot": snapshot,
    }, default=str)
