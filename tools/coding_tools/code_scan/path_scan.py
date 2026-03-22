"""
AST path scan for generated Python (used by ``tools.generate_code``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_cfg  = yaml.safe_load(open(_ROOT / 'config.yaml'))
AGENT_FILESYSTEM_ROOT = _ROOT / _cfg['paths']['agent_filesystem']

SANDBOX = AGENT_FILESYSTEM_ROOT.resolve()


class Status(Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    DYNAMIC = "DYNAMIC"


@dataclass
class FoundPath:
    line: int
    path: str
    kind: str
    status: Status


def _looks_like_path(val: str) -> bool:
    return (
        val.startswith("/")
        or val.startswith("./")
        or val.startswith("../")
        or val.startswith("agent_filesystem")
        or (
            "/" in val
            and len(val) > 5
            and val.endswith((".xlsx", ".xls", ".csv", ".txt", ".json", ".py", ".parquet", ".tsv"))
        )
    )


def _check_status(path_str: str) -> Status:
    from pathlib import Path
import yaml

    if "dynamic" in path_str:
        prefix = path_str.replace("... (dynamic)", "").strip()
        if prefix.startswith("agent_filesystem"):
            return Status.DYNAMIC
        return Status.UNSAFE
    try:
        resolved = Path(path_str).resolve()
        if str(resolved).startswith(str(SANDBOX)):
            return Status.SAFE
    except Exception:
        pass
    return Status.UNSAFE


def _scan_code(code: str) -> list[FoundPath]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [FoundPath(line=0, path=str(e), kind="syntax-error", status=Status.UNSAFE)]

    found: list[FoundPath] = []
    seen: set[tuple[int, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if _looks_like_path(val):
                key = (node.lineno, val)
                if key not in seen:
                    seen.add(key)
                    found.append(FoundPath(line=node.lineno, path=val, kind="hardcoded", status=_check_status(val)))

        if isinstance(node, ast.JoinedStr):
            parts = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            prefix = parts[0] if parts else ""
            if _looks_like_path(prefix):
                path_repr = f"{prefix}... (dynamic)"
                key = (node.lineno, path_repr)
                if key not in seen:
                    seen.add(key)
                    found.append(FoundPath(line=node.lineno, path=path_repr, kind="fstring", status=_check_status(path_repr)))

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = node.left
            if isinstance(left, ast.Constant) and isinstance(left.value, str):
                val = left.value
                if _looks_like_path(val):
                    path_repr = f"{val}... (dynamic)"
                    key = (node.lineno, path_repr)
                    if key not in seen:
                        seen.add(key)
                        found.append(FoundPath(line=node.lineno, path=path_repr, kind="concatenation", status=_check_status(path_repr)))

    return sorted(found, key=lambda x: x.line)


def run_path_scan(code: str) -> dict:
    paths = _scan_code(code)
    unsafe = [p for p in paths if p.status == Status.UNSAFE]
    dynamic = [p for p in paths if p.status == Status.DYNAMIC]
    safe = [p for p in paths if p.status == Status.SAFE]
    return {"passed": len(unsafe) == 0, "paths": paths, "unsafe": unsafe, "dynamic": dynamic, "safe": safe}


def format_path_scan_issues(unsafe: list[FoundPath]) -> str:
    lines = [
        "Path scan rejected the generated code.",
        "Paths outside agent_filesystem/ or unsafe:",
        "",
    ]
    for p in unsafe:
        lines.append(f"  Line {p.line:<4} | {p.kind:<13} | {p.path}")
    lines += ["", "RULES: paths must resolve under agent_filesystem/."]
    return "\n".join(lines)


def dynamic_paths_for_report(report: dict) -> list[dict]:
    return [{"line": p.line, "path": p.path, "kind": p.kind} for p in report.get("dynamic", [])]
