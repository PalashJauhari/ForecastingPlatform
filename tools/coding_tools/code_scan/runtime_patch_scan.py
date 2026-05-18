"""
Runtime I/O sandbox for **LLM-generated pipeline code** (Layer 2).

Applied in the subprocess that runs ``pipeline_run.py`` (see ``run_pipeline_sandboxed.py``):
call ``apply_patches(session_root, run_root)`` before user code runs and ``remove_patches()``
in a ``finally`` block so the interpreter returns to normal behavior.

**Two write boundaries**

* ``session_root`` (``SANDBOX``) — long-lived per-session workspace. Holds uploaded data
  files and tabular outputs that may be reused by future questions in the same session.
* ``run_root`` (``SANDBOX_RUN``) — short-lived per-question subfolder, e.g.
  ``agent_filesystem/<session>/run_<run_id>/``. Holds **plot artifacts** for one tool call
  so the UI can show only the plots produced by the current message (no leakage between
  turns). Tabular outputs do **not** go here — they live at session root so the next
  question can read them back.

**Three enforcement layers**

1. **Pandas entry points** (``read_excel``, ``read_csv``, ``to_excel``, ``to_csv``): the model
   must pass a **bare filename** (e.g. ``sales.csv``). The path is joined with ``SANDBOX``,
   resolved, and checked to stay inside that directory. Allowed extensions depend on the call
   (Excel vs CSV); see each ``safe_*`` wrapper.

2. **Matplotlib savefig** (``plt.savefig`` / ``Figure.savefig``): the model must pass a
   **bare filename** ending in ``.png`` or ``.svg``. The path is joined with ``SANDBOX_RUN``
   so plots are isolated per question. Other extensions (e.g. ``.pdf``, ``.jpg``) are
   rejected to keep the artifact surface small and predictable in the UI.

3. **``builtins.open``** (used by pandas, openpyxl, matplotlib backends, PIL, etc.): str/Path
   opens are allowed only for ``.csv``, ``.xlsx``, ``.xls``, ``.png``, and ``.svg``, **anywhere**
   on the host, so engines can open temp copies and absolute paths the savefig wrappers have
   already validated. Sensitive extensions (``.pem``, ``.key``, ``.json``, …) are always rejected.

**Mimetype database passthrough**

On import, openpyxl constructs ``mimetypes.MimeTypes()``, which opens files listed in
``mimetypes.knownfiles`` (e.g. ``/etc/apache2/mime.types``). Those paths are not tabular; we
allow **read-only** access to exactly that CPython allowlist so Excel works without widening
arbitrary ``open()`` to other system files.
"""

from __future__ import annotations

import builtins
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Sentinel: ``_pop_path_target`` returns this when no path argument is present.
_MISSING = object()

# Session workspace root (``agent_filesystem/<session>/``), set by ``apply_patches``.
# Holds uploaded data + tabular outputs that may be reused across questions.
SANDBOX: Path | None = None

# When set by ``apply_patches`` (non-empty lists from tool task), pandas/plot basenames must be in these sets.
IO_ALLOWLIST_INPUT: frozenset[str] | None = None
IO_ALLOWLIST_OUTPUT: frozenset[str] | None = None


def enforce_io_allowlist(raw: str, operation: str) -> None:
    """
    If allowlists were configured, reject basenames outside the declared Task ``input`` / ``output`` lists.

    Reads (pandas) → ``input`` list when set. Tabular writes and plot saves → ``output`` list when set.
    """
    if IO_ALLOWLIST_INPUT is None and IO_ALLOWLIST_OUTPUT is None:
        return
    o = operation.lower()
    is_read = o.startswith("pd.read") or "read_csv" in o or "read_excel" in o
    is_tabular_write = "to_csv" in o or "to_excel" in o
    is_plot_save = "savefig" in o

    if is_read:
        if IO_ALLOWLIST_INPUT is not None and raw not in IO_ALLOWLIST_INPUT:
            raise PermissionError(
                f"\n[SANDBOX VIOLATION]\n"
                f"  Operation : {operation}\n"
                f"  Path      : {raw}\n"
                f"  Reason    : basename not in the Task ``input`` allowlist\n"
                f"  Allowed   : {sorted(IO_ALLOWLIST_INPUT)}\n"
            )
        return

    if is_tabular_write or is_plot_save:
        if IO_ALLOWLIST_OUTPUT is not None and raw not in IO_ALLOWLIST_OUTPUT:
            raise PermissionError(
                f"\n[SANDBOX VIOLATION]\n"
                f"  Operation : {operation}\n"
                f"  Path      : {raw}\n"
                f"  Reason    : basename not in the Task ``output`` allowlist\n"
                f"  Allowed   : {sorted(IO_ALLOWLIST_OUTPUT)}\n"
            )


# Per-question plot folder (``agent_filesystem/<session>/run_<run_id>/``), set by
# ``apply_patches``. Only ``plt.savefig`` / ``Figure.savefig`` writes are routed here so
# the UI can scope plot rendering to one message bubble.
SANDBOX_RUN: Path | None = None

# Rejected for every I/O path (credentials, keys, arbitrary JSON reads of secrets, etc.).
GLOBALLY_BLOCKED_EXTENSIONS = {
    ".crt", ".json", ".pem", ".key", ".p12",
    ".pfx", ".der", ".cer", ".p8",
}

# Allowed suffixes for patched ``open()`` on str/Path. Includes ``.png`` / ``.svg`` so the
# matplotlib backend (AGG → PIL for PNG, native XML writer for SVG) can write the file
# handle that ``safe_pyplot_savefig`` / ``safe_figure_savefig`` already redirected into
# ``SANDBOX_RUN``. Direct ``open()`` calls in user code are blocked at the Semgrep layer.
ALLOWED_OPEN_EXTENSIONS = frozenset({".csv", ".xlsx", ".xls", ".png", ".svg"})

# Allowed suffixes for the savefig wrappers (matplotlib only writes one file per call).
ALLOWED_PLOT_EXTENSIONS = frozenset({".png", ".svg"})


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


def _require_run_workspace() -> Path:
    """Return the active per-question plot folder; raise if ``apply_patches`` has not run."""
    if SANDBOX_RUN is None:
        raise RuntimeError("Plot run workspace is not configured.")
    return SANDBOX_RUN


def _resolve_run_plot_path(path: str, operation: str) -> Path:
    """
    Map a bare plot filename to an absolute path under ``SANDBOX_RUN``.

    Mirrors ``_resolve_session_path`` but routes savefig writes into the per-question
    run folder instead of the session root, so each user message owns its plots.
    """
    run_root = _require_run_workspace()
    raw = str(path).strip()

    if not _is_bare_session_filename(raw):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : only a bare filename is allowed (e.g. plot.png) — no folders, prefixes, or paths\n"
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
    if ext not in ALLOWED_PLOT_EXTENSIONS:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : extension '{ext}' not allowed for {operation}\n"
            f"  Allowed   : {ALLOWED_PLOT_EXTENSIONS}\n"
        )

    target = (run_root / raw).resolve()
    try:
        target.relative_to(run_root)
    except ValueError as exc:
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : {operation}\n"
            f"  Path      : {path}\n"
            f"  Reason    : path escapes the per-question plot workspace\n"
            f"  Allowed   : {run_root}\n"
        ) from exc
    enforce_io_allowlist(raw, operation)
    return target


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
    enforce_io_allowlist(raw, operation)
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

    Session directory rules apply to **pandas** helpers; per-question plot folder rules
    apply to **savefig** wrappers. Direct ``open()`` calls in user code are blocked at the
    Semgrep layer, so the only opens reaching this wrapper are from library internals
    (pandas/openpyxl reading temp copies, matplotlib backend writing PNG/SVG bytes).
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


def safe_pyplot_savefig(fname, *args, **kwargs):
    """
    ``plt.savefig`` with ``fname`` restricted to a bare ``.png`` / ``.svg`` under ``SANDBOX_RUN``.

    Implementation note: matplotlib's ``plt.savefig`` internally delegates to
    ``gcf().savefig(fname, ...)``. Because we also patch ``Figure.savefig``, naively
    calling ``original_pyplot_savefig(resolved, ...)`` would re-enter ``safe_figure_savefig``
    with an *absolute* path, which the bare-filename check then rejects. To avoid that
    re-entry, we resolve the path here and call the **unpatched** ``Figure.savefig``
    directly on the current figure.
    """
    if not isinstance(fname, (str, Path)):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : plt.savefig()\n"
            f"  Reason    : only str or pathlib.Path is allowed as fname (no buffers, no streams)\n"
        )
    resolved = _resolve_run_plot_path(str(fname), "plt.savefig()")
    if plt is None or original_figure_savefig is None:
        raise RuntimeError("matplotlib is not available; plt.savefig cannot be served.")
    return original_figure_savefig(plt.gcf(), resolved, *args, **kwargs)


def safe_figure_savefig(self, fname, *args, **kwargs):
    """
    ``Figure.savefig`` with ``fname`` restricted to a bare ``.png`` / ``.svg`` under ``SANDBOX_RUN``.

    Bound method form — first positional after ``self`` is the filename.
    """
    if not isinstance(fname, (str, Path)):
        raise PermissionError(
            f"\n[SANDBOX VIOLATION]\n"
            f"  Operation : Figure.savefig()\n"
            f"  Reason    : only str or pathlib.Path is allowed as fname (no buffers, no streams)\n"
        )
    resolved = _resolve_run_plot_path(str(fname), "Figure.savefig()")
    return original_figure_savefig(self, resolved, *args, **kwargs)


# -- Public API ---------------------------------------------------------------

def apply_patches(
    session_root: str | Path,
    run_root: str | Path,
    *,
    io_allowlist_path: str | Path | None = None,
) -> None:
    """
    Install sandbox wrappers for ``open``, selected pandas methods, and matplotlib savefig.

    Parameters
    ----------
    session_root :
        Long-lived per-session workspace (typically ``agent_filesystem/<session_id>/``).
        Pandas reads / writes resolve here so tabular outputs persist across questions.
    run_root :
        Per-question plot folder (typically ``agent_filesystem/<session_id>/run_<run_id>/``).
        Created if missing. ``plt.savefig`` / ``Figure.savefig`` resolve here so each
        message bubble in the UI shows only the plots from its own tool call.
    io_allowlist_path :
        Optional path to a JSON file ``{"input": ["a.csv"], "output": ["b.png"]}``. When present and
        non-empty, basenames outside the corresponding list are rejected at I/O time.
    """
    global SANDBOX, SANDBOX_RUN, IO_ALLOWLIST_INPUT, IO_ALLOWLIST_OUTPUT
    IO_ALLOWLIST_INPUT = None
    IO_ALLOWLIST_OUTPUT = None
    if io_allowlist_path:
        p = Path(io_allowlist_path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            ins = data.get("input") or []
            outs = data.get("output") or []
            if ins:
                IO_ALLOWLIST_INPUT = frozenset(str(x).strip() for x in ins if str(x).strip())
            if outs:
                IO_ALLOWLIST_OUTPUT = frozenset(str(x).strip() for x in outs if str(x).strip())
    SANDBOX = Path(session_root).resolve()
    SANDBOX_RUN = Path(run_root).resolve()
    SANDBOX_RUN.mkdir(parents=True, exist_ok=True)

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
    global SANDBOX, SANDBOX_RUN, IO_ALLOWLIST_INPUT, IO_ALLOWLIST_OUTPUT
    SANDBOX = None
    SANDBOX_RUN = None
    IO_ALLOWLIST_INPUT = None
    IO_ALLOWLIST_OUTPUT = None
    builtins.open = original_open  # type: ignore[assignment]
    pd.read_excel = original_read_excel  # type: ignore[assignment]
    pd.read_csv = original_read_csv  # type: ignore[assignment]
    pd.DataFrame.to_excel = original_to_excel  # type: ignore[assignment]
    pd.DataFrame.to_csv = original_to_csv  # type: ignore[assignment]
    if plt is not None and Figure is not None and original_pyplot_savefig is not None and original_figure_savefig is not None:
        plt.savefig = original_pyplot_savefig  # type: ignore[assignment]
        Figure.savefig = original_figure_savefig  # type: ignore[assignment]
