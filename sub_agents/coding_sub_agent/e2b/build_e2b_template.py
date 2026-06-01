#!/usr/bin/env python3
"""
Build or replace the E2B sandbox template for the coding sub-agent.

Run manually (not on app startup) from the repo root or this package directory::

    source ForecastingPlatform_env_1/bin/activate
    pip install 'e2b>=2.3.0'
    python sub_agents/coding_sub_agent/e2b/build_e2b_template.py
    python sub_agents/coding_sub_agent/e2b/build_e2b_template.py --write-env

Requires ``CODING_E2B_API_KEY`` in root ``.env``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

E2B_DIR = Path(__file__).resolve().parent
CODING_AGENT_ROOT = E2B_DIR.parent
PROJECT_ROOT = CODING_AGENT_ROOT.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_TEMPLATE_NAME = "forecasting-platform-ds"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def delete_template_cli(name: str) -> None:
    """Best-effort delete via E2B CLI before rebuild."""
    try:
        subprocess.run(
            ["e2b", "template", "delete", "-y", name],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("Note: e2b CLI not found; skipping delete (Template.build may replace in place).")


def patch_env_file(template_name: str, template_id: str) -> None:
    lines: list[str] = []
    if ENV_PATH.is_file():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    updates = {
        "CODING_E2B_TEMPLATE_NAME": template_name,
        "CODING_E2B_TEMPLATE_ID": template_id,
    }
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        matched = False
        for key, value in updates.items():
            if line.startswith(f"{key}="):
                out.append(f"{key}={value}")
                seen.add(key)
                matched = True
                break
        if not matched:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Updated {ENV_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build E2B DS template for coding sub-agent.")
    parser.add_argument(
        "--write-env",
        action="store_true",
        help=f"Write CODING_E2B_TEMPLATE_NAME and CODING_E2B_TEMPLATE_ID into {ENV_PATH.name}",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=f"Template name (default: env CODING_E2B_TEMPLATE_NAME or {DEFAULT_TEMPLATE_NAME})",
    )
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("CODING_E2B_API_KEY", "").strip()
    if not api_key:
        print("CODING_E2B_API_KEY is missing. Set it in root .env", file=sys.stderr)
        return 1

    template_name = (
        args.name
        or os.environ.get("CODING_E2B_TEMPLATE_NAME")
        or DEFAULT_TEMPLATE_NAME
    ).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", template_name):
        print(
            "Template name must be lowercase letters, numbers, dashes, underscores.",
            file=sys.stderr,
        )
        return 1

    from e2b import Template, default_build_logger

    from sub_agents.coding_sub_agent.e2b.template_def import template

    print(f"Deleting existing template '{template_name}' if present...")
    delete_template_cli(template_name)

    print(f"Building template '{template_name}' (skip_cache=True)...")
    build_info = Template.build(
        template,
        template_name,
        cpu_count=2,
        memory_mb=2048,
        skip_cache=True,
        on_build_logs=default_build_logger(),
        api_key=api_key,
    )

    template_id = str(getattr(build_info, "template_id", "") or "")
    print("\nBuild complete.")
    print(f"  name:        {getattr(build_info, 'name', template_name)}")
    print(f"  template_id: {template_id}")
    print("\nAdd to root .env:")
    print(f"  CODING_E2B_TEMPLATE_NAME={template_name}")
    if template_id:
        print(f"  CODING_E2B_TEMPLATE_ID={template_id}")

    if args.write_env:
        patch_env_file(template_name, template_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
