"""
Load coding sub-agent settings from root ``.env``.

Env matrix: ``CODING_*`` variables from root ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
E2B_API_KEY = (os.environ.get("CODING_E2B_API_KEY") or "").strip()
E2B_TEMPLATE_NAME = (os.environ.get("CODING_E2B_TEMPLATE_NAME") or "").strip()
E2B_TEMPLATE_ID = (os.environ.get("CODING_E2B_TEMPLATE_ID") or "").strip()
CODING_MODEL = (os.environ.get("CODING_MODEL") or "").strip() or "gpt-4o-mini"
# Stronger model for the last CodeGen only (``codegen_node`` when attempt count == MAX_CODEGEN_ATTEMPTS). Empty = always CODING_MODEL.
CODING_MODEL_LAST_ATTEMPT = (os.environ.get("CODING_MODEL_LAST_ATTEMPT") or "").strip()
CODE_JUDGE_MODEL = (os.environ.get("CODING_CODE_JUDGE_MODEL") or "").strip() or CODING_MODEL
IO_JUDGE_MODEL = (os.environ.get("CODING_IO_JUDGE_MODEL") or "").strip() or CODE_JUDGE_MODEL


def _parse_max_codegen_attempts() -> int:
    try:
        raw = int((os.environ.get("CODING_MAX_CODEGEN_ATTEMPTS") or "").strip() or "3")
    except ValueError:
        raw = 3
    if raw < 1:
        raise ValueError(f"CODING_MAX_CODEGEN_ATTEMPTS must be >= 1 (got {raw})")
    return raw


MAX_CODEGEN_ATTEMPTS = _parse_max_codegen_attempts()
# LangGraph backup cap only; retries stop at CodeGenLimitGate when codegen_count >= MAX.
_CODING_STEPS_PER_ATTEMPT = 8
CODING_GRAPH_RECURSION_LIMIT = MAX_CODEGEN_ATTEMPTS * _CODING_STEPS_PER_ATTEMPT + 4
try:
    E2B_SANDBOX_TIMEOUT_SECONDS = int((os.environ.get("CODING_E2B_SANDBOX_TIMEOUT_SECONDS") or "").strip() or "120")
except ValueError:
    E2B_SANDBOX_TIMEOUT_SECONDS = 120
try:
    E2B_EXECUTION_TIMEOUT_SECONDS = int((os.environ.get("CODING_E2B_EXECUTION_TIMEOUT_SECONDS") or "").strip() or "120")
except ValueError:
    E2B_EXECUTION_TIMEOUT_SECONDS = 120


# When true, ``e2b_execute_node`` calls ``sandbox.kill()`` after each run (production default).
# Set false to leave sandboxes in the E2B dashboard for debugging (clean up manually).
_kill_raw = (os.environ.get("CODING_E2B_KILL_SANDBOX") or "").strip().lower()
if not _kill_raw:
    E2B_KILL_SANDBOX = False
else:
    E2B_KILL_SANDBOX = _kill_raw in {"1", "true", "yes", "on"}

PLOT_FILE_EXTENSIONS = frozenset({".png", ".svg"})
TABULAR_OUTPUT_EXTENSIONS = frozenset({".csv", ".xlsx"})
