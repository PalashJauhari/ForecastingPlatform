"""
Execute ``pipeline_run.py`` with ``runtime_patch_scan`` I/O monkey-patches enabled.

``code_pipeline`` invokes this script as a subprocess (project root as cwd) instead of
``python pipeline_run.py`` directly, so ``builtins.open`` and pandas I/O are wrapped
before any user code runs.

Resource policy (applied in this file before pandas is imported):
    - **200 MiB** virtual address cap (``RLIMIT_AS``) on Unix where supported
    - **BLAS/OpenMP** single-thread env vars
    - **Linux:** pin this process to **CPU 0** via ``sched_setaffinity``
"""

from __future__ import annotations

import os

# Before numpy/pandas load (``runtime_patch_scan`` imports pandas).
for _key, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
):
    os.environ.setdefault(_key, _val)

import platform
import resource
import runpy
import sys
from pathlib import Path

# Hard cap for generated-script process (virtual address space).
MEMORY_LIMIT_BYTES = 200 * 1024 * 1024


def _apply_process_limits() -> None:
    """``RLIMIT_AS`` on Unix; CPU 0 only on Linux."""
    if hasattr(resource, "RLIMIT_AS"):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        except (ValueError, OSError):
            pass
    if platform.system() == "Linux":
        try:
            os.sched_setaffinity(0, {0})
        except (AttributeError, OSError):
            pass


_apply_process_limits()

from runtime_patch_scan import apply_patches, remove_patches


def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: run_pipeline_sandboxed.py <path_to_pipeline_run.py> <session_workspace>",
            file=sys.stderr,
        )
        sys.exit(2)
    target = Path(sys.argv[1]).resolve()
    session_workspace = Path(sys.argv[2]).resolve()
    if not target.is_file():
        print(f"Not a file: {target}", file=sys.stderr)
        sys.exit(2)

    apply_patches(session_workspace)
    try:
        # Mirrors ``python pipeline_run.py`` (``__name__ == "__main__"``, etc.).
        runpy.run_path(str(target), run_name="__main__")
    finally:
        remove_patches()


if __name__ == "__main__":
    main()
