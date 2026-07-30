"""
Load planner sub-agent settings from root ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PLANNER_MODEL = (os.environ.get("PLANNER_MODEL") or "").strip() or "gpt-4o-mini"
