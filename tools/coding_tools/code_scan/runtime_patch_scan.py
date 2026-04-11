"""
Layer 2 — **Runtime patch scan**: monkey-patching for LLM-generated code execution.

Lives under ``tools/coding_tools/`` alongside codegen and static scans.
Intended for callers that execute LLM code in-process with ``exec`` (apply before
``exec``, remove in ``finally``).

Patches ``builtins.open``, ``pd.read_excel``, ``pd.read_csv``,
``df.to_excel``, ``df.to_csv`` to enforce:

1. Paths use ``agent_filesystem/<session-folder>/input/...`` or ``.../output/...`` and match the sandbox session directory name
2. **Reads** may use **input** or **output**; **writes** (pandas saves + ``savefig``) may use **output** only
3. Globally blocked extensions (``.key``, ``.pem``, ...) are rejected everywhere
4. Per-operation extension allow-lists
5. ``open()`` is blocked entirely -- LLM must use pandas helpers
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
cfg = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
LOGICAL_AGENT_FS = str(cfg["paths"]["agent_filesystem"]).rstrip("/")

SANDBOX: Path | None = None

GLOBALLY_BLOCKED_EXTENSIONS = {
    ".crt", ".json", ".pem", ".key", ".p12",
    ".pfx", ".der", ".cer", ".p8",
}


# -- Validators ---------------------------------------------------------------

def _require_sandbox() -> Path:
    if SANDBOX is None:
        raise RuntimeError("Session sandbox is not configured.")
    return SANDBOX


def _resolve_session_path(
    path: str,
    operation: str,
    allowed_extensions: set[str],
    *,
    io_mode: Literal["read", "write"],
) -> Path:
    sandbox = _require_sandbox()
    ext = Path(path).suffix.lower()

    if ext in GLOBALLY_BLOCKED_EXTENSIONS:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : '{ext}' files are globally blocked -- no exceptions\n"
            f"  Blocked   : {GLOBALLY_BLOCKED_EXTENSIONS}\n"
        )
    if ext not in allowed_extensions:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : extension '{ext}' not allowed for {operation}\n"
            f"  Allowed   : {allowed_extensions}\n"
        )

    prefix = f"{LOGICAL_AGENT_FS}/"
    if not path.startswith(prefix):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : path must start with {LOGICAL_AGENT_FS}/ for the current session\n"
        )

    rel = path[len(prefix) :].strip("/")
    if "/" not in rel:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : path must be {LOGICAL_AGENT_FS}/<session>/input|output/...\n"
        )
    session_key, rest = rel.split("/", 1)
    if session_key != sandbox.name:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : path session folder {session_key!r} must match workspace {sandbox.name!r}\n"
        )

    if io_mode == "read":
        if not (rest.startswith("input/") or rest.startswith("output/")):
            raise PermissionError(
                f"\n[SANDBOX VIOLATION]\n"
                f"  Operation : {operation}\n"
                f"  Path      : {path}\n"
                f"  Reason    : reads must use {LOGICAL_AGENT_FS}/<session>/input/... or .../output/...\n"
            )
    else:
        if not rest.startswith("output/"):
            raise PermissionError(
                f"\n[SANDBOX VIOLATION]\n"
                f"  Operation : {operation}\n"
                f"  Path      : {path}\n"
                f"  Reason    : writes and plots must use {LOGICAL_AGENT_FS}/<session>/output/... (not input/)\n"
            )

    target = (sandbox / rest).resolve()
    try:
        target.relative_to(sandbox)
    except ValueError as exc:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : path escapes the current session workspace\n"
            f"  Allowed   : {sandbox}\n"
        ) from exc
    return target


# -- Originals (captured at import time) --------------------------------------

original_open = builtins.open
original_read_excel = pd.read_excel
original_read_csv = pd.read_csv
original_to_excel = pd.DataFrame.to_excel
original_to_csv = pd.DataFrame.to_csv
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - matplotlib is optional at runtime here.
    plt = None
    Figure = None
original_pyplot_savefig = plt.savefig if plt is not None else None
original_figure_savefig = Figure.savefig if Figure is not None else None


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
    resolved = _resolve_session_path(
        str(path), "pd.read_excel()", {".xlsx"}, io_mode="read",
    )
    return original_read_excel(resolved, *args, **kwargs)


def safe_read_csv(path, *args, **kwargs):
    resolved = _resolve_session_path(
        str(path), "pd.read_csv()", {".csv"}, io_mode="read",
    )
    return original_read_csv(resolved, *args, **kwargs)


def safe_to_excel(self, path, *args, **kwargs):
    resolved = _resolve_session_path(
        str(path), "df.to_excel()", {".xlsx"}, io_mode="write",
    )
    return original_to_excel(self, resolved, *args, **kwargs)


def safe_to_csv(self, path=None, *args, **kwargs):
    if path is not None:
        path = _resolve_session_path(
            str(path), "df.to_csv()", {".csv"}, io_mode="write",
        )
    return original_to_csv(self, path, *args, **kwargs)


def safe_pyplot_savefig(*args, **kwargs):
    if original_pyplot_savefig is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    if args:
        new_args = list(args)
        new_args[0] = _resolve_session_path(
            str(args[0]),
            "plt.savefig()",
            {".png", ".jpg", ".jpeg", ".pdf", ".svg"},
            io_mode="write",
        )
        return original_pyplot_savefig(*new_args, **kwargs)
    if "fname" in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs["fname"] = _resolve_session_path(
            str(kwargs["fname"]),
            "plt.savefig()",
            {".png", ".jpg", ".jpeg", ".pdf", ".svg"},
            io_mode="write",
        )
        return original_pyplot_savefig(**new_kwargs)
    return original_pyplot_savefig(*args, **kwargs)


def safe_figure_savefig(self, fname, *args, **kwargs):
    if original_figure_savefig is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    resolved = _resolve_session_path(
        str(fname),
        "Figure.savefig()",
        {".png", ".jpg", ".jpeg", ".pdf", ".svg"},
        io_mode="write",
    )
    return original_figure_savefig(self, resolved, *args, **kwargs)


# -- Public API ---------------------------------------------------------------

def apply_patches(session_root: str | Path) -> None:
    """Swap real functions with sandboxed versions for the current session workspace."""
    global SANDBOX
    SANDBOX = Path(session_root).resolve()
    builtins.open = safe_open  # type: ignore[assignment]
    pd.read_excel = safe_read_excel  # type: ignore[assignment]
    pd.read_csv = safe_read_csv  # type: ignore[assignment]
    pd.DataFrame.to_excel = safe_to_excel  # type: ignore[assignment]
    pd.DataFrame.to_csv = safe_to_csv  # type: ignore[assignment]
    if plt is not None and Figure is not None:
        plt.savefig = safe_pyplot_savefig  # type: ignore[assignment]
        Figure.savefig = safe_figure_savefig  # type: ignore[assignment]


def remove_patches() -> None:
    """Restore originals. **Always** call in a ``finally`` block."""
    global SANDBOX
    SANDBOX = None
    builtins.open = original_open  # type: ignore[assignment]
    pd.read_excel = original_read_excel  # type: ignore[assignment]
    pd.read_csv = original_read_csv  # type: ignore[assignment]
    pd.DataFrame.to_excel = original_to_excel  # type: ignore[assignment]
    pd.DataFrame.to_csv = original_to_csv  # type: ignore[assignment]
    if plt is not None and Figure is not None and original_pyplot_savefig is not None and original_figure_savefig is not None:
        plt.savefig = original_pyplot_savefig  # type: ignore[assignment]
        Figure.savefig = original_figure_savefig  # type: ignore[assignment]
