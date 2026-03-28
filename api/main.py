"""
FastAPI HTTP API for the Forecasting Platform agent.

Endpoints
    POST /run    — multipart form: user ``query``, optional CSV/Excel ``file``
                   upload or ``csv_path``, ``session_id`` for namespacing uploads
                   under ``agent_filesystem/input/<session>/``.
    POST /resume — resume a paused graph after an ``ask_user`` interrupt.

Loads ``.env`` from the project root for ``OPENAI_API_KEY`` and optional
Langfuse keys.
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
from observability.langfuse_handler import get_langfuse_callbacks

_cfg      = yaml.safe_load(open(_ROOT / "config.yaml"))
INPUT_DIR = _ROOT / _cfg["paths"]["input"]

ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = FastAPI(
    title="GaussianBlurr — Forecasting Platform",
    description=(
        "Agent API: upload CSV/Excel, send a natural-language query, "
        "receive summaries or clarifications."
    ),
)
_graph = AnalysisGraph()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_input_dir(session_id: str) -> Path:
    """
    Return (and create) ``agent_filesystem/input/<sanitized_session_id>/``.

    Only alphanumeric, underscore, and hyphen are kept in *session_id*
    to avoid path injection.
    """
    safe = re.sub(r"[^\w\-]", "", session_id) or "default"
    d = INPUT_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_response(result: Dict[str, Any]) -> tuple[str, Any]:
    """
    Pull the last ``AIMessage`` content and the last tool result from
    the graph result's message list.

    Returns
        ``(summary_text, last_tool_result)``
    """
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

    return last_ai, last_tool_result


def _check_interrupt(session_id: str) -> Dict[str, Any] | None:
    """
    If the graph is paused (``ask_user`` interrupt), return a response
    dict with the clarifying question.  Otherwise return ``None``.
    """
    state = _graph.get_state(session_id)
    if not state or not state.next:
        return None

    question = ""
    for task in (state.tasks or []):
        for intr in (task.interrupts or []):
            val = intr.value if hasattr(intr, "value") else intr
            if isinstance(val, dict):
                question = val.get("question", str(val))
            else:
                question = str(val)
            break
        if question:
            break

    return {
        "session_id": session_id,
        "interrupted": True,
        "question": question,
        "csv_path": "",
        "summary": None,
        "last_tool_result": None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
        query       — natural-language instruction.
        file        — optional CSV/Excel upload (first message in a flow).
        csv_path    — optional path to existing file (follow-up turns).
        session_id  — correlates uploads and ``thread_id`` for conversation memory.

    Returns
        JSON with ``session_id``, ``csv_path``, ``summary``,
        ``last_tool_result``.  If the agent asks a clarifying question,
        ``interrupted`` is ``true`` and ``question`` contains the text.
    """
    resolved_csv_path = ""

    if file is not None:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            return JSONResponse(
                content={
                    "error": (
                        f"Only CSV and Excel files are accepted. "
                        f"Got: {ext or '(no extension)'}"
                    )
                },
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

    result = _graph.run_graph(
        session_id, user_query, config={"callbacks": get_langfuse_callbacks()},
    )

    # Check for interrupt (ask_user tool)
    interrupt_resp = _check_interrupt(session_id)
    if interrupt_resp:
        interrupt_resp["csv_path"] = resolved_csv_path or (csv_path or "")
        return interrupt_resp

    summary, last_tool_result = _extract_response(result)

    return {
        "session_id": session_id,
        "csv_path": resolved_csv_path or (csv_path or ""),
        "summary": summary or None,
        "last_tool_result": last_tool_result,
    }


@app.post("/resume")
async def resume(
    resume_value: str = Form(...),
    session_id: str = Form("default"),
    csv_path: str = Form(None),
):
    """
    Resume a paused agent after an ``ask_user`` interrupt.

    Form fields
        resume_value — the user's answer to the clarifying question.
        session_id   — same session that was interrupted.
        csv_path     — carried forward from the original ``/run`` call.

    Returns
        Same shape as ``/run``.
    """
    result = _graph.resume(
        session_id, resume_value, config={"callbacks": get_langfuse_callbacks()},
    )

    # Check for another interrupt
    interrupt_resp = _check_interrupt(session_id)
    if interrupt_resp:
        interrupt_resp["csv_path"] = csv_path or ""
        return interrupt_resp

    summary, last_tool_result = _extract_response(result)

    return {
        "session_id": session_id,
        "csv_path": csv_path or "",
        "summary": summary or None,
        "last_tool_result": last_tool_result,
    }
