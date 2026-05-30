"""
Load coding sub-agent settings from ``sub_agents/coding_sub_agent/.env``.

Env matrix: ``CODING_MODEL``, judge models, E2B template name/key/timeouts, ``MAX_CODEGEN_ATTEMPTS``, ``CODING_RECURSION_LIMIT``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PACKAGE_ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
E2B_API_KEY = os.environ.get("E2B_API_KEY", "").strip()
E2B_TEMPLATE_NAME = os.environ.get("E2B_TEMPLATE_NAME", "").strip()
E2B_TEMPLATE_ID = os.environ.get("E2B_TEMPLATE_ID", "").strip()
CODING_MODEL = os.environ.get("CODING_MODEL", "gpt-4o-mini").strip()
CODE_JUDGE_MODEL = os.environ.get("CODE_JUDGE_MODEL", CODING_MODEL).strip()
IO_JUDGE_MODEL = os.environ.get("IO_JUDGE_MODEL", CODE_JUDGE_MODEL).strip()
MAX_CODEGEN_ATTEMPTS = int(os.environ.get("MAX_CODEGEN_ATTEMPTS", "3"))
# LangGraph step cap — safety net only; normal give-up uses MAX_CODEGEN_ATTEMPTS + CodeGenFailure.
CODING_RECURSION_LIMIT = int(os.environ.get("CODING_RECURSION_LIMIT", "24"))
E2B_SANDBOX_TIMEOUT_SECONDS = int(os.environ.get("E2B_SANDBOX_TIMEOUT_SECONDS", "120"))
E2B_EXECUTION_TIMEOUT_SECONDS = int(os.environ.get("E2B_EXECUTION_TIMEOUT_SECONDS", "120"))


def env_bool(name: str, *, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# When true, ``e2b_execute_node`` calls ``sandbox.kill()`` after each run (production default).
# Set false to leave sandboxes in the E2B dashboard for debugging (clean up manually).
E2B_KILL_SANDBOX = env_bool("E2B_KILL_SANDBOX", default=False)

PLOT_FILE_EXTENSIONS = frozenset({".png", ".svg"})
TABULAR_OUTPUT_EXTENSIONS = frozenset({".csv", ".xlsx"})
