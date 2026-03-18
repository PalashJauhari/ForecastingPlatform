"""
ask_csv: answer a natural-language question about the session data via an internal LLM.

Flow:
  1. Resolve data path: processed/clean.csv if present, else session CSV from context.
  2. Build peek data and user_content (same pattern as clean_csv: label + peek JSON).
  3. Call LLM with ASK_CSV_CODE_GENERATOR_PROMPT; parse JSON response for "code" key.
  4. Run AST safety check; execute code in subprocess with preamble (imports + df = pd.read_csv(...)).
  5. Return stdout (or error string).

Model: CODING_MODEL env (default gpt-4o-mini). Response format: JSON with "code" key only.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graph.middleware.session_context import csv_path_var, data_dir_var
from prompts.ask_csv_prompt import ASK_CSV_CODE_GENERATOR_PROMPT
from tools.clean_csv import get_peek_data
from tools.safety import check_code_safety


class AskCsvInput(BaseModel):
    """Arguments for ask_csv. The query is a natural-language question about the data."""

    query: str = Field(
        description="Natural-language question about the session data. E.g. 'What is the mean of column X?', 'Show top 5 rows by column Y', 'How many nulls in column Z?'. The tool loads the data and generates pandas code to answer. Use for analysis and ad hoc questions only; do NOT use for cleaning (use clean_csv instead).",
    )


@tool(args_schema=AskCsvInput)
def ask_csv(query: str) -> str:
    """Answer a question about the session data in natural language. Use when the user asks for stats, aggregates, filters, or exploration (e.g. mean, top N rows, counts). Data is loaded from processed/clean.csv if present, else the raw session CSV. An internal LLM generates and runs pandas code; you only pass the question. Do NOT use for cleaning or transforming the file—use clean_csv for that. Returns the printed result as a string, or an error message."""
    # --- Resolve data path: prefer clean.csv, else session CSV ---
    csv_path = csv_path_var.get()
    data_dir = data_dir_var.get()
    clean_path = Path(data_dir) / "processed" / "clean.csv"
    data_path = str(clean_path) if clean_path.exists() else (csv_path or "")

    if not data_path or not Path(data_path).exists():
        return "Error: No CSV or clean.csv found for this session."

    # --- Build peek data and user_content (same pattern as clean_csv) ---
    try:
        peek = get_peek_data(data_path)
    except Exception as e:
        return f"Error: Could not read data: {e}"

    user_content = "User question: " + query.strip() + "\n\nPeek data (schema/sample):\n" + json.dumps(peek, default=str)

    # --- LLM call: enforce JSON response with "code" key ---
    coding_model = os.environ.get("CODING_MODEL", "gpt-4o-mini")
    model = ChatOpenAI(
        model=coding_model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    messages = [SystemMessage(content=ASK_CSV_CODE_GENERATOR_PROMPT), HumanMessage(content=user_content)]
    try:
        response = model.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        data = json.loads(content)
        code = (data.get("code") or "").strip()
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        return f"Error: LLM did not return valid JSON with 'code' key: {e}"

    if not code:
        return "Error: No code generated."

    # --- Safety check: only allowed modules (pandas, numpy, sklearn, scipy) and no file/exec ---
    is_safe, violations = check_code_safety(code)
    if not is_safe:
        return "Error: Generated code failed safety check: " + "; ".join(violations)

    # --- Preamble: same imports as clean_csv, then load CSV into df for generated code ---
    preamble = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import sklearn\n"
        "import scipy\n"
        f"df = pd.read_csv({repr(data_path)})\n"
    )
    full_code = preamble + code

    # --- Run in subprocess; return stdout ---
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        script_path = f.name
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(data_dir).resolve()),
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0:
            return f"Error: {err or 'Non-zero exit'}"
        return out or "(no output)"
    finally:
        Path(script_path).unlink(missing_ok=True)
