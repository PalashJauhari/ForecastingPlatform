"""
FastAPI HTTP API for the Forecasting Platform agent.

Endpoints
    POST /run         — form: ``query``, ``session_id`` (``thread_id`` for the graph).
    POST /resume      — resume after an ``ask_user`` interrupt.
    POST /upload-data — multipart: CSV/Excel files → ``agent_filesystem/input/``.

Loads ``.env`` from the project root for ``OPENAI_API_KEY`` and optional
Langfuse keys.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from dotenv import load_dotenv
from langfuse import observe, propagate_attributes

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage

from graph import AnalysisGraph
from observability.langfuse_handler import build_request_metadata, get_langfuse_client

cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
INPUT_DIR = PROJECT_ROOT / cfg["paths"]["input"]

ALLOWED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = FastAPI(
    title="GaussianBlurr — Forecasting Platform",
    description="Agent API: natural-language queries with optional human-in-the-loop.",
)
analysis_graph = AnalysisGraph()
langfuse = get_langfuse_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_api_response(session_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the JSON body from a graph ``invoke()`` return value.

    When ``interrupt()`` runs inside the graph (e.g. ``ask_user``), v1 ``invoke``
    includes ``__interrupt__`` on that dict — no separate ``get_state()`` call.
    See https://docs.langchain.com/oss/python/langgraph/interrupts
    """
    interrupts = result.get("__interrupt__") or []
    if interrupts:
        val = getattr(interrupts[0], "value", interrupts[0])
        if isinstance(val, dict):
            question = val.get("question", str(val))
        else:
            question = str(val)
        return {
            "session_id": session_id,
            "interrupted": True,
            "question": question,
            "summary": None,
            "last_tool_result": None,
        }

    messages = result.get("messages", [])
    last_ai = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    last_tool_result = next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "tool"),
        None,
    )
    return {
        "session_id": session_id,
        "interrupted": False,
        "summary": last_ai,
        "last_tool_result": last_tool_result,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/run")
@observe(name="api.run", as_type="chain")
async def run(
    query: str = Form(...),
    session_id: str = Form("default"),
):
    """
    Run the analysis agent on a user query.

    Form fields
        query       — natural-language instruction.
        session_id  — ``thread_id`` for conversation memory.

    Returns
        JSON with ``session_id``, ``summary``, ``last_tool_result``.
        If the agent asks a clarifying question, ``interrupted`` is ``true``
        and ``question`` contains the text.
    """
    with propagate_attributes(
        session_id=session_id,
        tags=["api", "run"],
        metadata=build_request_metadata(endpoint="/run", interface="fastapi", query=query),
    ):
        result = analysis_graph.run_graph(session_id, query, config={})
        response = get_api_response(session_id, result)
        langfuse.update_current_span(
            output=response,
            metadata={
                "interrupted": response["interrupted"],
                "has_last_tool_result": response["last_tool_result"] is not None,
            },
        )
        return response


@app.post("/resume")
@observe(name="api.resume", as_type="chain")
async def resume(
    resume_value: str = Form(...),
    session_id: str = Form("default"),
):
    """
    Resume a paused agent after an ``ask_user`` interrupt.

    Form fields
        resume_value — the user's answer to the clarifying question.
        session_id   — same session that was interrupted.

    Returns
        Same shape as ``/run``.
    """
    with propagate_attributes(
        session_id=session_id,
        tags=["api", "resume"],
        metadata=build_request_metadata(
            endpoint="/resume", interface="fastapi", query=resume_value,
        ),
    ):
        result = analysis_graph.resume(session_id, resume_value, config={})
        response = get_api_response(session_id, result)
        langfuse.update_current_span(
            output=response,
            metadata={
                "interrupted": response["interrupted"],
                "has_last_tool_result": response["last_tool_result"] is not None,
            },
        )
        return response


@app.post("/upload-data")
@observe(name="api.upload_data", as_type="tool")
async def upload_data(files: list[UploadFile] = File(...)):
    """Save uploaded CSV/Excel files to ``agent_filesystem/input/``."""

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for upload in files:
        name = Path(upload.filename or "").name or "data"
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_DATA_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Only .csv, .xlsx, .xls allowed. Got: {upload.filename or name}"
                    )
                },
            )
        dest = INPUT_DIR / name
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(f"agent_filesystem/input/{name}")

    response = {"saved": saved, "count": len(saved)}
    langfuse.update_current_span(
        output=response,
        metadata={"uploaded_count": len(saved)},
    )
    return response
