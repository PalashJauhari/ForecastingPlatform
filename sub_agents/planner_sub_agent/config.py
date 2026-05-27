"""
Load planner sub-agent settings from ``sub_agents/planner_sub_agent/.env``.

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
load_dotenv(PACKAGE_ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-4o-mini").strip()
PLANNER_RECURSION_LIMIT = int(os.environ.get("PLANNER_RECURSION_LIMIT", "25"))
PLANNER_MAX_CONCURRENCY = int(os.environ.get("PLANNER_MAX_CONCURRENCY", "2"))
