"""
LangChain tool: execute a ``.py`` script under ``agent_filesystem``.

Applies **runtime patch scan** (Layer 3) via same-process ``exec()`` so that
monkey-patched ``builtins.open``, ``pd.read_csv``, ``pd.read_excel``,
``df.to_csv``, ``df.to_excel`` are in effect during execution.

Layer 3 (``tools.coding_tools.code_scan.runtime_patch_scan``) confines all
file I/O to ``agent_filesystem/``.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import yaml

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent.parent
_cfg  = yaml.safe_load(open(_ROOT / 'config.yaml'))
AGENT_FILESYSTEM_ROOT = _ROOT / _cfg['paths']['agent_filesystem']
from .code_scan.runtime_patch_scan import apply_patches, remove_patches


def _agent_fs_root() -> Path:
    """Resolved absolute path to ``agent_filesystem``."""
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
    Execute a Python script from agent_filesystem/ with Layer 3 runtime sandbox patches applied.

    The script runs via exec() in the same process with monkey-patched I/O functions
    that confine all file operations to agent_filesystem/. stdout and stderr are captured
    and returned.

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

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    apply_patches()
    try:
        compiled = compile(src, str(script), "exec")
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compiled, {"__name__": "__main__", "__file__": str(script)})
        returncode = 0
    except Exception as e:
        stderr_buf.write(f"{type(e).__name__}: {e}\n")
        returncode = 1
    finally:
        remove_patches()

    return json.dumps({
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "returncode": returncode,
    }, default=str)
