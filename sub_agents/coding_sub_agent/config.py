"""
Load coding sub-agent settings from ``sub_agents/coding_sub_agent/.env``.
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
CODING_MODEL = os.environ.get("CODING_MODEL", "gpt-4o-mini").strip()
CODE_JUDGE_MODEL = os.environ.get("CODE_JUDGE_MODEL", CODING_MODEL).strip()
IO_JUDGE_MODEL = os.environ.get("IO_JUDGE_MODEL", CODE_JUDGE_MODEL).strip()
CODING_RECURSION_LIMIT = int(os.environ.get("CODING_RECURSION_LIMIT", "50"))
MAX_CODEGEN_ATTEMPTS = int(os.environ.get("MAX_CODEGEN_ATTEMPTS", "3"))
E2B_SANDBOX_TIMEOUT_SECONDS = int(os.environ.get("E2B_SANDBOX_TIMEOUT_SECONDS", "300"))
E2B_EXECUTION_TIMEOUT_SECONDS = int(os.environ.get("E2B_EXECUTION_TIMEOUT_SECONDS", "120"))

PLOT_FILE_EXTENSIONS = frozenset({".png", ".svg"})
TABULAR_OUTPUT_EXTENSIONS = frozenset({".csv", ".xlsx"})
