from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


from .code_scan.path_scan import format_path_scan_issues, run_path_scan
from .code_scan.semgrep_scan import format_semgrep_issues, run_semgrep_scan
from prompts.code_generation_prompt import CODE_GENERATION_SYSTEM_PROMPT

# Read codegen model from config.yaml once at import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
CODING_MODEL = cfg["models"].get("code_generation", "gpt-4o-mini")
CODE_DIR = Path(PROJECT_ROOT) / cfg["paths"]["code"]


class GenerateCodeInput(BaseModel):
    """
    Arguments the **orchestrator** passes into ``generate_code``.

    The orchestrator should embed read/write locations inside ``task`` (e.g. lines
    like **Input:** ... **Output:** ... under ``agent_filesystem``). ``data_schema`` is
    optional structured hints for the codegen model only.
    """

    task: str = Field(
        description=(
            "Full description of what the script must do. "
            "Specify every file the script reads using its complete agent_filesystem/... path. "
            "You can specify multiple input paths — use clear labels so paths are unambiguous. "
            "Output path is only required if the script needs to save a file. "
            "If the script just computes values or prints results, no output path is needed. "
            "If the script saves a file, you MUST specify the full output path under agent_filesystem/. "
            "Example — script that saves output: "
            "'Input: agent_filesystem/input/sales.csv | Output: agent_filesystem/output/forecast.csv | Fit a linear regression on date and revenue and save predictions.' "
            "Example — script that only prints: "
            "'Input: agent_filesystem/input/sales.csv | Compute summary statistics and print the results.' "
            "Example with multiple inputs: "
            "'Input 1: agent_filesystem/input/sales.csv | Input 2: agent_filesystem/input/targets.csv | Output: agent_filesystem/output/report.csv | Merge both files on date and compute variance.' "
            "Never use bare filenames, relative paths, or paths outside agent_filesystem/."
        ),
    )
    data_schema: str = Field(
        default="",
        description=(
            "Schema of all input files the script will read — column names, dtypes, date formats, nullability. "
            "If there are multiple input files, include the schema for each one. "
            "If you do not know the schema, call read_agent_filesystem_data with the file path first "
            "and pass its output (columns + sample rows) here. "
            "Leave empty only if the schema is truly unknown and read_agent_filesystem_data has not been called."
        ),
    )


@tool(args_schema=GenerateCodeInput)
def generate_code(task: str, data_schema: str = "") -> str:
    """
    Generate a Python script for a given data task, validate it, and save it to agent_filesystem/code/.

    Describe what you want the script to do in ``task``. The tool sends this to a dedicated
    codegen LLM which writes the Python source. The source is validated by two scans before
    being saved — if validation fails nothing is written.

    After a successful call, pass the returned ``filename`` to run_python_file to execute the script.

    Args:
        task:
            Full description of what the script must do. Specify every data file the Python
            code reads or writes using its complete agent_filesystem/... path. These are the
            paths to the data files (CSV, Excel, etc.) — not the path of the Python script
            itself (that is handled automatically).

            Input paths: data files the Python code will read. Multiple inputs are supported.
            Output path: data file the Python code will save. Only required if the script
            saves a result — if it just computes or prints values, no output path is needed.
            If saving, the output path MUST start with 'agent_filesystem/'.
            Never use bare filenames or relative paths.

            Example — script that saves a file:
                "Input: agent_filesystem/input/sales.csv
                 Output: agent_filesystem/output/forecast.csv
                 Fit a linear regression on date and revenue and save predictions."

            Example — script that only prints results:
                "Input: agent_filesystem/input/sales.csv
                 Compute and print summary statistics."

            Example — multiple input files:
                "Input 1: agent_filesystem/input/sales.csv
                 Input 2: agent_filesystem/input/targets.csv
                 Output: agent_filesystem/output/report.csv
                 Merge both files on date and compute variance."

        data_schema:
            Schema of all input files the script will read — column names, dtypes, date
            formats, nullability. If there are multiple input files, include schema for each.
            If you do not know the schema, call read_agent_filesystem_data first and pass
            its output (columns + sample rows) here. Leave empty only if schema is truly
            unknown and read_agent_filesystem_data has not been called.

    Returns a JSON string with the following keys:
      - "code"                  : generated Python source (empty string if validation failed).
      - "explanation"           : short summary of what the script does.
      - "filename"              : script filename, e.g. "forecast.py" — pass this to run_python_file.
      - "path"                  : absolute path to the saved .py file (empty if not saved).
      - "code_safety_evaluation": {"passed": true} on success;
                                  {"passed": false, "detail": "..."} on failure with reason.
    """
    # --- Model and prompt ---------------------------------------------------------
    model_name = CODING_MODEL

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    user_payload = (
        "## Task\n"
        + task.strip()
        + "\n\n## Data schema\n"
        + data_schema.strip()
    )

    resp = llm.invoke(
        [
            SystemMessage(content=CODE_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=user_payload),
        ]
    )
    data = json.loads(resp.content if hasattr(resp, "content") else str(resp))

    # --- Parse codegen JSON -----------------------------------------------------
    code = (data.get("code") or "").strip()
    explanation = (data.get("explanation") or "").strip()
    filename = (data.get("filename") or "generated.py").strip()

    # No code -> skip scans (empty source would pass checks and could write an empty file).
    failure = None
    if code:
        semgrep_report = run_semgrep_scan(code)
        path_report = run_path_scan(code)
        if not (semgrep_report["passed"] and path_report["passed"]):
            parts: list[str] = []
            if not semgrep_report["passed"]:
                parts.append(format_semgrep_issues(semgrep_report["violations"]))
            if not path_report["passed"]:
                parts.append(format_path_scan_issues(path_report["unsafe"]))
            failure = {"passed": False, "detail": "\n\n".join(parts)}
    else:
        failure = {"passed": False, "detail": "No code returned from model."}

    if failure is not None:
        return json.dumps(
            {
                "code": "",
                "explanation": explanation,
                "filename": filename,
                "path": "",
                "code_safety_evaluation": failure,
            },
            default=str,
        )

    # --- Persist ----------------------------------------------------------------
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CODE_DIR / filename
    out_path.write_text(code, encoding="utf-8")

    return json.dumps(
        {
            "code": code,
            "explanation": explanation,
            "filename": filename,
            "path": str(out_path.resolve()),
            "code_safety_evaluation": {"passed": True},
        },
        default=str,
    )
