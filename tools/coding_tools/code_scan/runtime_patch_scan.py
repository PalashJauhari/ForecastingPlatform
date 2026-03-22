"""
Layer 3 — **Runtime patch scan**: monkey-patching for LLM-generated code execution.

Lives under ``tools/coding_tools/`` alongside codegen and static scans.
Applied inside ``run_python_file`` just before execution and torn down in a
``finally`` block.

Patches ``builtins.open``, ``pd.read_excel``, ``pd.read_csv``,
``df.to_excel``, ``df.to_csv`` to enforce:

1. All file paths resolve inside ``agent_filesystem/``
2. Globally blocked extensions (``.key``, ``.pem``, ...) are rejected everywhere
3. Per-operation extension allow-lists
4. ``open()`` is blocked entirely -- LLM must use pandas helpers
"""

from __future__ import annotations

import builtins
from pathlib import Path
import yaml

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_cfg  = yaml.safe_load(open(_ROOT / 'config.yaml'))
AGENT_FILESYSTEM_ROOT = _ROOT / _cfg['paths']['agent_filesystem']

SANDBOX = AGENT_FILESYSTEM_ROOT.resolve()

GLOBALLY_BLOCKED_EXTENSIONS = {
    ".crt", ".json", ".pem", ".key", ".p12",
    ".pfx", ".der", ".cer", ".p8",
}


# -- Validators ---------------------------------------------------------------

def _is_safe_path(path: str) -> bool:
    """True when *path* resolves inside the sandbox (catches traversal)."""
    try:
        return str(Path(path).resolve()).startswith(str(SANDBOX))
    except Exception:
        return False


def _validate(path: str, operation: str, allowed_extensions: set[str]) -> None:
    ext = Path(path).suffix.lower()

    if ext in GLOBALLY_BLOCKED_EXTENSIONS:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : '{ext}' files are globally blocked -- no exceptions\n"
            f"  Blocked   : {GLOBALLY_BLOCKED_EXTENSIONS}\n"
        )
    if not _is_safe_path(path):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : path is outside agent_filesystem/\n"
            f"  Allowed   : {SANDBOX}\n"
        )
    if ext not in allowed_extensions:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : extension '{ext}' not allowed for {operation}\n"
            f"  Allowed   : {allowed_extensions}\n"
        )


# -- Originals (captured at import time) --------------------------------------

_original_open = builtins.open
_original_read_excel = pd.read_excel
_original_read_csv = pd.read_csv
_original_to_excel = pd.DataFrame.to_excel
_original_to_csv = pd.DataFrame.to_csv


# -- Patched replacements -----------------------------------------------------

def safe_open(path, mode="r", *args, **kwargs):  # noqa: A001
    """``open()`` is fully blocked -- checks globally blocked extensions first."""
    ext = Path(str(path)).suffix.lower()
    if ext in GLOBALLY_BLOCKED_EXTENSIONS:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : open()\n"
            f"  Path      : {path}\n"
            f"  Reason    : '{ext}' files are globally blocked -- no exceptions\n"
        )
    raise PermissionError(
        f"\n[SANDBOX VIOLATION]\n"
        f"  Operation : open()\n"
        f"  Path      : {path}\n"
        f"  Reason    : open() is not allowed\n"
        f"  Use       : pd.read_excel() or pd.read_csv() instead\n"
    )


def safe_read_excel(path, *args, **kwargs):
    _validate(str(path), "pd.read_excel()", {".xlsx", ".xls"})
    return _original_read_excel(path, *args, **kwargs)


def safe_read_csv(path, *args, **kwargs):
    _validate(str(path), "pd.read_csv()", {".csv"})
    return _original_read_csv(path, *args, **kwargs)


def safe_to_excel(self, path, *args, **kwargs):
    _validate(str(path), "df.to_excel()", {".xlsx", ".xls"})
    return _original_to_excel(self, path, *args, **kwargs)


def safe_to_csv(self, path=None, *args, **kwargs):
    if path is not None:
        _validate(str(path), "df.to_csv()", {".csv"})
    return _original_to_csv(self, path, *args, **kwargs)


# -- Public API ---------------------------------------------------------------

def apply_patches() -> None:
    """Swap real functions with sandboxed versions. Call just before ``exec(llm_code)``."""
    builtins.open = safe_open  # type: ignore[assignment]
    pd.read_excel = safe_read_excel  # type: ignore[assignment]
    pd.read_csv = safe_read_csv  # type: ignore[assignment]
    pd.DataFrame.to_excel = safe_to_excel  # type: ignore[assignment]
    pd.DataFrame.to_csv = safe_to_csv  # type: ignore[assignment]


def remove_patches() -> None:
    """Restore originals. **Always** call in a ``finally`` block."""
    builtins.open = _original_open  # type: ignore[assignment]
    pd.read_excel = _original_read_excel  # type: ignore[assignment]
    pd.read_csv = _original_read_csv  # type: ignore[assignment]
    pd.DataFrame.to_excel = _original_to_excel  # type: ignore[assignment]
    pd.DataFrame.to_csv = _original_to_csv  # type: ignore[assignment]
