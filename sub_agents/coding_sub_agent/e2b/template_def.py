"""
E2B template definition for the coding sandbox (DS libraries only).

Used only by ``build_e2b_template.py`` (requires ``e2b>=2.3.0``), not at app runtime.

Uses ``from_base_image()`` (e2bdev/base) so ``/home/user`` exists — vanilla
``python:3.12`` does not, which causes "failed to move files in sandbox".
"""

from __future__ import annotations

from pathlib import Path

from e2b import Template

E2B_DIR = Path(__file__).resolve().parent
REQ_DEST = "/home/user/requirements-sandbox.txt"

template = (
    Template(file_context_path=str(E2B_DIR))
    .from_base_image()
    .copy("requirements-sandbox.txt", REQ_DEST)
    .run_cmd(f"pip install --no-cache-dir -r {REQ_DEST}", user="root")
)
