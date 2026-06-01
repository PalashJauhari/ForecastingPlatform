"""
Load planner sub-agent settings from root ``.env``.

``PLANNER_RECURSION_LIMIT`` / ``PLANNER_MAX_CONCURRENCY`` are unused today: the
mounted subgraph inherits the parent ``AnalysisGraph`` thread config.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
PLANNER_MODEL = (os.environ.get("PLANNER_MODEL") or "").strip() or "gpt-4o-mini"

# These are currently informational (planner runs mounted under parent graph config),
# but we keep them parsed here for clarity and future planner-local tuning.
try:
    PLANNER_RECURSION_LIMIT = int((os.environ.get("PLANNER_RECURSION_LIMIT") or "").strip() or "25")
except ValueError:
    PLANNER_RECURSION_LIMIT = 25

try:
    PLANNER_MAX_CONCURRENCY = int((os.environ.get("PLANNER_MAX_CONCURRENCY") or "").strip() or "2")
except ValueError:
    PLANNER_MAX_CONCURRENCY = 2
