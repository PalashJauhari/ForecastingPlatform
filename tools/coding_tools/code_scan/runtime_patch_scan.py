"""
Runtime I/O sandbox for **LLM-generated pipeline code** (Layer 2).

Applied in the subprocess that runs ``pipeline_run.py`` (see ``run_pipeline_sandboxed.py``):
call ``apply_patches(session_root)`` before user code runs and ``remove_patches()`` in a
``finally`` block so the interpreter returns to normal behavior.

**Two enforcement layers**

1. **Pandas entry points** (``read_excel``, ``read_csv``, ``to_excel``, ``to_csv``): the model
   must pass a **bare filename** (e.g. ``sales.csv``). The path is joined with ``session_root``,
   resolved, and checked to stay inside that directory. Allowed extensions depend on the call
   (Excel vs CSV); see each ``safe_*`` wrapper.

2. **``builtins.open``** (used by pandas, openpyxl, and temp files): str/Path opens are allowed
   only for ``.csv``, ``.xlsx``, and ``.xls``, **anywhere** on the host, so engines
   can open temp copies and absolute paths. Sensitive extensions (``.pem``, ``.json``, …) are
   always rejected.

**Mimetype database passthrough**

On import, openpyxl constructs ``mimetypes.MimeTypes()``, which opens files listed in
``mimetypes.knownfiles`` (e.g. ``/etc/apache2/mime.types``). Those paths are not tabular; we
allow **read-only** access to exactly that CPython allowlist so Excel works without widening
arbitrary ``open()`` to other system files.

**Plots**

``plt.savefig`` / ``Figure.savefig`` are replaced with stubs that raise: tabular outputs only.
"""

from __future__ import annotations

import builtins
import mimetypes
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Sentinel: ``_pop_path_target`` returns this when no path argument is present.
_MISSING = object()

# Session workspace root (``agent_filesystem/<session>/``), set by ``apply_patches``.
SANDBOX: Path | None = None

# Rejected for every I/O path (credentials, keys, arbitrary JSON reads of secrets, etc.).
GLOBALLY_BLOCKED_EXTENSIONS = {
    ".crt", ".json", ".pem", ".key", ".p12",
    ".pfx", ".der", ".cer", ".p8",
}

# Allowed suffixes for patched ``open()`` on str/Path (not used for mimetype DB passthrough).
ALLOWED_OPEN_EXTENSIONS = frozenset({".csv", ".xlsx", ".xls"})


def _norm_open_path_key(path: str | Path) -> str:
    """Return a canonical string key so two spellings of the same file compare equal."""
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _mimetype_knownfile_path_keys() -> frozenset[str]:
    """Collect normalized paths from ``mimetypes.knownfiles`` plus ``realpath`` when resolvable."""
    keys: set[str] = set()
    for raw in getattr(mimetypes, "knownfiles", ()) or ():
        if not raw or not isinstance(raw, str):
            continue
        keys.add(_norm_open_path_key(raw))
        try:
            real = os.path.realpath(raw)
        except OSError:
            continue
        keys.add(_norm_open_path_key(real))
    return frozenset(keys)


# Frozen at import time; mirrors what ``mimetypes.init()`` iterates on Unix/macOS.
MIMETYPE_DB_PATH_KEYS: frozenset[str] = _mimetype_knownfile_path_keys()


def _open_mode_allows_write(mode: object) -> bool:
    """Return True if the mode string can write, append, create exclusively, or update via ``+``."""
    m = str(mode) if mode is not None else "r"
    return any(ch in m for ch in "wWaax+")


# -- Validators ---------------------------------------------------------------

def _require_sandbox() -> Path:
    """Return the active session root, or raise if ``apply_patches`` has not run."""
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
    """Map a bare filename to an absolute path under ``SANDBOX``; validate extension and traversal."""
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

def safe_open(file, mode="r", *args, **kwargs):  # noqa: A001
    """
    Restricted replacement for ``builtins.open`` while patches are active.

    Evaluation order:

    1. File descriptors ``0``, ``1``, ``2`` pass through (stdio).
    2. Paths in ``MIMETYPE_DB_PATH_KEYS`` pass through **read-only** (stdlib / openpyxl).
    3. Other str/Path targets: suffix must be in ``ALLOWED_OPEN_EXTENSIONS`` and not in
       ``GLOBALLY_BLOCKED_EXTENSIONS``.

    Session directory rules apply to **pandas** helpers, not to every ``open()`` call.
    """
    # Fail closed if patches were not installed (caller bug).
    _require_sandbox()

    if isinstance(file, int):
        if file in (0, 1, 2):
            return original_open(file, mode, *args, **kwargs)
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : open()\n"
            f"  Reason    : only file descriptors 0, 1, 2 are allowed\n"
        )

    if isinstance(file, bytes):
        try:
            file = os.fsdecode(file)
        except Exception as exc:
            raise PermissionError(
                f"\n[SANDBOX VIOLATION]\n"
                f"  Operation : open()\n"
                f"  Reason    : could not decode bytes path for sandbox check\n"
            ) from exc

    if not isinstance(file, (str, Path)):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : open()\n"
            f"  Reason    : only str, os.PathLike, or int (0–2) are allowed as the file target\n"
        )

    candidate = Path(file)
    path_key = _norm_open_path_key(candidate)
    # openpyxl → mimetypes reads fixed host paths; never allow writes there.
    if path_key in MIMETYPE_DB_PATH_KEYS:
        if _open_mode_allows_write(mode):
            raise PermissionError(
                f"\n[SANDBOX VIOLATION]\n"
                f"  Operation : open()\n"
                f"  Path      : {file}\n"
                f"  Reason    : system MIME database paths are read-only passthrough\n"
            )
        return original_open(file, mode, *args, **kwargs)

    ext = candidate.suffix.lower()
    if ext in GLOBALLY_BLOCKED_EXTENSIONS:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : open()\n"
            f"  Path      : {file}\n"
            f"  Reason    : '{ext}' files are globally blocked -- no exceptions\n"
            f"  Blocked   : {GLOBALLY_BLOCKED_EXTENSIONS}\n"
        )
    if ext not in ALLOWED_OPEN_EXTENSIONS:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : open()\n"
            f"  Path      : {file}\n"
            f"  Reason    : open() is only allowed for .csv, .xlsx, and .xls\n"
            f"  Allowed   : {ALLOWED_OPEN_EXTENSIONS}\n"
        )

    return original_open(file, mode, *args, **kwargs)


def safe_read_excel(*args, **kwargs):
    """``pd.read_excel`` with ``io`` restricted to a bare ``.xlsx`` / ``.xls`` under the session."""
    target, rest, kw = _pop_path_target(args, kwargs, ("io",))
    if target is _MISSING:
        return original_read_excel(*args, **kwargs)
    resolved = _require_bare_path_file(target, "pd.read_excel()", {".xlsx", ".xls"})
    return original_read_excel(resolved, *rest, **kw)


def safe_read_csv(*args, **kwargs):
    """``pd.read_csv`` with ``filepath_or_buffer`` restricted to a bare ``.csv`` under the session."""
    target, rest, kw = _pop_path_target(args, kwargs, ("filepath_or_buffer",))
    if target is _MISSING:
        return original_read_csv(*args, **kwargs)
    resolved = _require_bare_path_file(target, "pd.read_csv()", {".csv"})
    return original_read_csv(resolved, *rest, **kw)


def safe_to_excel(self, *args, **kwargs):
    """``DataFrame.to_excel`` with ``excel_writer`` restricted to a bare ``.xlsx`` under the session."""
    target, rest, kw = _pop_path_target(args, kwargs, ("excel_writer",))
    if target is _MISSING:
        return original_to_excel(self, *args, **kwargs)
    resolved = _require_bare_path_file(target, "df.to_excel()", {".xlsx"})
    return original_to_excel(self, resolved, *rest, **kw)


def safe_to_csv(self, *args, **kwargs):
    """``DataFrame.to_csv`` with ``path_or_buf`` restricted to a bare ``.csv`` under the session (or ``None``)."""
    target, rest, kw = _pop_path_target(args, kwargs, ("path_or_buf",))
    if target is _MISSING:
        return original_to_csv(self, *args, **kwargs)
    if target is None:
        return original_to_csv(self, None, *rest, **kw)
    resolved = _require_bare_path_file(target, "df.to_csv()", {".csv"})
    return original_to_csv(self, resolved, *rest, **kw)


def safe_pyplot_savefig(*args, **kwargs):
    """Block image export; use CSV/Excel writers for artifacts."""
    raise PermissionError(
        "\n[SANDBOX VIOLATION]\n"
        "  Operation : plt.savefig()\n"
        "  Reason    : image output is disabled; use DataFrame.to_csv / to_excel only\n"
    )


def safe_figure_savefig(self, fname, *args, **kwargs):
    """Block image export; use CSV/Excel writers for artifacts."""
    raise PermissionError(
        "\n[SANDBOX VIOLATION]\n"
        "  Operation : Figure.savefig()\n"
        "  Reason    : image output is disabled; use DataFrame.to_csv / to_excel only\n"
    )


# -- Public API ---------------------------------------------------------------

def apply_patches(session_root: str | Path) -> None:
    """
    Install sandbox wrappers for ``open``, selected pandas methods, and matplotlib savefig.

    ``session_root`` must be the absolute or resolvable directory for the current run
    (typically ``agent_filesystem/<session_id>/``).
    """
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
    """
    Undo ``apply_patches``; always run in a ``finally`` so a failed script does not leave
    patched globals in the subprocess interpreter.
    """
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
