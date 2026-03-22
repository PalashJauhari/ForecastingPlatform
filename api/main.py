"""
FastAPI HTTP API for the Forecasting Platform agent.

Endpoints:
    POST /run — multipart form: user ``query``, optional CSV/Excel ``file`` upload or ``csv_path``,
    ``session_id`` for namespacing uploads under ``agent_filesystem/input/<session>/``.

Loads ``.env`` from the project root for ``OPENAI_API_KEY`` and optional Langfuse keys.
"""

from __future__ import annotations

import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage

from graph import AnalysisGraph
_cfg   = yaml.safe_load(open(_ROOT / 'config.yaml'))
INPUT_DIR = _ROOT / _cfg['paths']['input']
from observability.langfuse_handler import get_langfuse_callbacks

ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = FastAPI(
    title="GaussianBlurr — Forecasting Platform",
    description="Agent API: upload CSV/Excel, send a natural-language query, receive summaries or clarifications.",
)
_graph = AnalysisGraph()


def _session_input_dir(session_id: str) -> Path:
    """
    Return (and create) ``agent_filesystem/input/<sanitized_session_id>/``.

    Only alphanumeric, underscore, and hyphen are kept in *session_id* to avoid path injection.
    """
    safe = re.sub(r"[^\w\-]", "", session_id) or "default"
    d = INPUT_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.post("/run")
async def run(
    query: str = Form(...),
    file: UploadFile = File(None),
    csv_path: str = Form(None),
    session_id: str = Form("default"),
):
    """
    Run the analysis agent on a user query with optional file context.

    Form fields
        query
            Natural-language instruction; appended with uploaded file path when ``file`` is sent.
        file
            Optional CSV/Excel upload (first message in a flow); stored under ``input/<session>/``.
        csv_path
            Optional path to an existing file (follow-up turns after first upload).
        session_id
            Correlates uploads and LangGraph ``thread_id`` for conversation memory.

    Returns
        JSON: ``session_id``, ``csv_path``, ``summary``, ``last_tool_result``.
    """
    resolved_csv_path = ""

    if file is not None:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            return JSONResponse(
                content={"error": f"Only CSV and Excel files are accepted. Got: {ext or '(no extension)'}"},
                status_code=400,
            )
        session_d = _session_input_dir(session_id)
        path = session_d / f"{uuid.uuid4().hex[:8]}_{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        resolved_csv_path = str(path.resolve())
        user_query = f"{query}\n\n[Uploaded file path: {resolved_csv_path}]"
    elif csv_path:
        resolved_csv_path = csv_path
        user_query = query
    else:
        user_query = query

    callbacks = get_langfuse_callbacks()
    invoke_cfg: Dict[str, Any] = {"callbacks": callbacks} if callbacks else {}

    result = _graph.run_graph(session_id, user_query, config=invoke_cfg)
    messages = result.get("messages", [])

    last_ai = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai = msg.content or ""
            if last_ai:
                break

    last_tool_result = None
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "tool":
            last_tool_result = getattr(msg, "content", None)
            break

    return {
        "session_id": session_id,
        "csv_path": resolved_csv_path or (csv_path or ""),
        "summary": last_ai or None,
        "last_tool_result": last_tool_result,
    }
