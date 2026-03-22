"""
LangChain tool: execute (dummy) a ``.py`` script under ``agent_filesystem``.

Currently a **dummy executor** -- applies **runtime patch scan** (Layer 3) to
demonstrate the sandbox mechanism, but does **not** actually run the code.
Replace the dummy block with real ``exec()`` when ready for production.

Layer 3 (``tools.coding_tools.code_scan.runtime_patch_scan``) monkey-patches
``builtins.open``, ``pd.read_excel``, ``pd.read_csv``, ``df.to_excel``,
``df.to_csv`` so that even dynamically constructed paths are confined to
``agent_filesystem/``.
"""

from __future__ import annotations

import json
from pathlib import Path
import yaml

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent.parent
_cfg  = yaml.safe_load(open(_ROOT / 'config.yaml'))
AGENT_FILESYSTEM_ROOT = _ROOT / _cfg['paths']['agent_filesystem']
from .code_scan.runtime_patch_scan import apply_patches, remove_patches


def _agent_fs_root() -> Path:
    """Resolved absolute path to ``agent_filesystem`` (subprocess ``cwd``)."""
    return AGENT_FILESYSTEM_ROOT.resolve()


def _resolve_script(filename: str) -> Path:
    """
    Resolve *filename* to a ``.py`` path confined under ``agent_filesystem``.

    Raises
        ValueError -- path escapes workspace or suffix is not ``.py``.
    """
    base = _agent_fs_root()
    p = Path(filename)
    candidate = p.resolve() if p.is_absolute() else (base / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError("Path must be under agent_filesystem root.")
    if not candidate.suffix.lower() == ".py":
        raise ValueError("Only .py files can be executed.")
    return candidate


class RunPythonFileInput(BaseModel):
    """Schema for :func:`run_python_file`."""

    filename: str = Field(
        description="Path to a .py file relative to agent_filesystem (e.g. 'code/my_script.py') or absolute path under that root.",
    )


@tool(args_schema=RunPythonFileInput)
def run_python_file(filename: str) -> str:
    """
    Dummy executor: validate script location, apply runtime patch scan, print
    confirmation, and tear down patches.  No code is actually executed.

    Returns
        JSON with ``stdout``, ``stderr``, ``returncode``.
    """
    try:
        script = _resolve_script(filename.strip())
    except ValueError as e:
        return json.dumps({"error": str(e), "stdout": "", "stderr": "", "returncode": -1})

    if not script.exists():
        return json.dumps({"error": f"File not found: {script}", "stdout": "", "stderr": "", "returncode": -1})

    src = script.read_text(encoding="utf-8")
    line_count = len(src.splitlines())

    apply_patches()
    try:
        print(f"[SANDBOX] Runtime patch scan applied -- dummy mode, no real execution")
        print(f"[SANDBOX] Script : {script}")
        print(f"[SANDBOX] Lines  : {line_count}")
    finally:
        remove_patches()

    return json.dumps(
        {
            "stdout": (
                f"[SANDBOX] Dummy execution complete for {script.name}. "
                f"{line_count} lines validated. No code was actually run."
            ),
            "stderr": "",
            "returncode": 0,
        },
        default=str,
    )
