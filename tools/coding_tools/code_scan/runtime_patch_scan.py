"""
Layer 2 — **Runtime patch scan**: monkey-patching for LLM-generated code execution.

Lives under ``tools/coding_tools/`` alongside codegen and static scans.
Intended for callers that execute LLM code in-process with ``exec`` (apply before
``exec``, remove in ``finally``).

Patches ``builtins.open``, ``pd.read_excel``, ``pd.read_csv``,
``df.to_excel``, ``df.to_csv`` to enforce:

1. **Bare filename only** for every pandas path string (e.g. ``sales.csv``). No
   folders, no ``agent_filesystem/...`` prefixes, no silent rewriting.
2. Paths resolve only under the current session sandbox directory.
3. Only ``.csv`` and ``.xlsx`` file reads/writes are allowed.
4. Globally blocked extensions (``.key``, ``.pem``, ...) are rejected everywhere.
5. ``open()`` is blocked entirely; plot/image file outputs are blocked.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pandas as pd

_MISSING = object()

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


def _is_bare_session_filename(path: str) -> bool:
    """Single path segment only — no directories, traversal, or URL/drive tricks."""
    p = str(path).strip()
    if not p or ".." in p:
        return False
    if "/" in p or "\\" in p:
        return False
    if p.startswith("~"):
        return False
    if ":" in p:
        return False
    return Path(p).name == p


def _resolve_session_path(
    path: str,
    operation: str,
    allowed_extensions: set[str],
) -> Path:
    sandbox = _require_sandbox()
    raw = str(path).strip()

    if not _is_bare_session_filename(raw):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : only a bare filename is allowed (e.g. sales.csv) — no folders, prefixes, or paths\n"
        )

    ext = Path(raw).suffix.lower()

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

    if raw == "pipeline_run.py":
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : pipeline_run.py is the generated runner — do not read/write it as data\n"
        )

    target = (sandbox / raw).resolve()
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


def _require_bare_path_file(candidate: Any, operation: str, allowed_extensions: set[str]) -> Path:
    """Reject buffers, URLs, and non-path types; only str/Path with basename-only rules."""
    if not isinstance(candidate, (str, Path)):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Reason    : only a bare filename as str or pathlib.Path is allowed — not buffers, URLs, or file objects\n"
        )
    return _resolve_session_path(str(candidate), operation, allowed_extensions)


def _pop_path_target(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    kw_names: tuple[str, ...],
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """
    Pandas path target: first positional wins; else first matching keyword (popped).

    Removes duplicate keyword entries so the original function is not called with
    two sources for the same argument.
    """
    kwargs = dict(kwargs)
    if args:
        target = args[0]
        rest = args[1:]
        for name in kw_names:
            kwargs.pop(name, None)
        return target, rest, kwargs
    for name in kw_names:
        if name in kwargs:
            return kwargs.pop(name), (), kwargs
    return _MISSING, args, kwargs


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


def safe_read_excel(*args, **kwargs):
    target, rest, kw = _pop_path_target(args, kwargs, ("io",))
    if target is _MISSING:
        return original_read_excel(*args, **kwargs)
    resolved = _require_bare_path_file(target, "pd.read_excel()", {".xlsx"})
    return original_read_excel(resolved, *rest, **kw)


def safe_read_csv(*args, **kwargs):
    target, rest, kw = _pop_path_target(args, kwargs, ("filepath_or_buffer",))
    if target is _MISSING:
        return original_read_csv(*args, **kwargs)
    resolved = _require_bare_path_file(target, "pd.read_csv()", {".csv"})
    return original_read_csv(resolved, *rest, **kw)


def safe_to_excel(self, *args, **kwargs):
    target, rest, kw = _pop_path_target(args, kwargs, ("excel_writer",))
    if target is _MISSING:
        return original_to_excel(self, *args, **kwargs)
    resolved = _require_bare_path_file(target, "df.to_excel()", {".xlsx"})
    return original_to_excel(self, resolved, *rest, **kw)


def safe_to_csv(self, *args, **kwargs):
    target, rest, kw = _pop_path_target(args, kwargs, ("path_or_buf",))
    if target is _MISSING:
        return original_to_csv(self, *args, **kwargs)
    if target is None:
        return original_to_csv(self, None, *rest, **kw)
    resolved = _require_bare_path_file(target, "df.to_csv()", {".csv"})
    return original_to_csv(self, resolved, *rest, **kw)


def safe_pyplot_savefig(*args, **kwargs):
    raise PermissionError(
        "\n[SANDBOX VIOLATION]\n"
        "  Operation : plt.savefig()\n"
        "  Reason    : only .csv and .xlsx file outputs are allowed\n"
    )


def safe_figure_savefig(self, fname, *args, **kwargs):
    raise PermissionError(
        "\n[SANDBOX VIOLATION]\n"
        "  Operation : Figure.savefig()\n"
        "  Reason    : only .csv and .xlsx file outputs are allowed\n"
    )


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
