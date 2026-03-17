"""
FastAPI: single endpoint that invokes the forecast agent and returns its output.
File upload saves to data/<session_id>/source_files/.
Loads .env from project root so OPENAI_API_KEY (and optional LANGFUSE_* keys) are set.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import re
import uuid
import shutil
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from graph import AnalysisGraph
from observability.langfuse_handler import get_langfuse_callbacks

app = FastAPI(title="GaussianBlurr — Forecasting Platform")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Single shared agent instance; sessions isolated via thread_id + checkpointer
_graph = AnalysisGraph()


def _session_dir(session_id: str) -> Path:
    safe = re.sub(r"[^\w\-]", "", session_id) or "default"
    d = DATA_DIR / safe
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
    Resolve input (upload or csv_path), invoke the agent, return output.
    """
    if file is not None:
        session_d = _session_dir(session_id)
        source_dir = session_d / "source_files"
        source_dir.mkdir(parents=True, exist_ok=True)
        path = source_dir / f"{uuid.uuid4().hex[:8]}_{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        resolved_csv_path = str(path)
    elif csv_path:
        resolved_csv_path = csv_path
    else:
        return JSONResponse(
            content={"error": "Provide either file (first message) or csv_path (follow-up)."},
            status_code=400,
        )

    _graph.set_csv_path(session_id, resolved_csv_path)

    callbacks = get_langfuse_callbacks()
    config: Dict[str, Any] = {"callbacks": callbacks} if callbacks else {}

    output = _graph.run_graph(session_id, query, config=config)

    return {
        "session_id": output.get("session_id", session_id),
        "csv_path": output.get("csv_path", resolved_csv_path),
        "summary": output.get("summary"),
        "clarification_question": output.get("clarification_question"),
        "tool_output": output.get("tool_output") or {},
        "last_tool_result": output.get("last_tool_result"),
    }
