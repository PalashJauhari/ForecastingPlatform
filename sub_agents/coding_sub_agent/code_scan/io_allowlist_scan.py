"""
Deterministic IO allowlist scan for generated Python in the coding sub-agent.

Compares pandas/matplotlib basename literals in code to declared input_files and output_files.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from sub_agents.coding_sub_agent.code_scan.safety_check import SafetyCheckResult

READ_METHODS = frozenset({"read_csv", "read_excel"})
WRITE_METHODS = frozenset({"to_csv", "to_excel"})
SAVE_METHODS = frozenset({"savefig"})


def format_io_issues(violations: list[dict[str, Any]]) -> str:
    """Human-readable block for tool payloads and codegen retry context."""
    lines = ["IO allowlist rejected the generated code.", "Violations:", ""]
    for violation in violations:
        lines.append(f"  Line {violation['line']:<4} | {violation['rule']}")
        lines.append(f"           {violation['message']}")
        lines.append("")
    return "\n".join(lines)


def path_literal_from_expr(node: ast.expr | None) -> str | None:
    """Return basename from a string literal path, or None if not a plain literal."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value).name
    return None


def call_path_and_kind(node: ast.Call) -> tuple[str | None, str | None, int]:
    """
    Extract (basename, kind, line) from a pandas/matplotlib IO call.

    kind is ``read``, ``write``, or ``plot``.
    """
    line = getattr(node, "lineno", 0) or 0
    func = node.func
    method_name: str | None = None
    receiver: str | None = None

    if isinstance(func, ast.Attribute):
        method_name = func.attr
        if isinstance(func.value, ast.Name):
            receiver = func.value.id
    elif isinstance(func, ast.Name):
        method_name = func.id

    if method_name in READ_METHODS and receiver in ("pd", "pandas"):
        for keyword in node.keywords:
            if keyword.arg in ("filepath_or_buffer", "io"):
                path = path_literal_from_expr(keyword.value)
                if path:
                    return path, "read", line
        if node.args:
            path = path_literal_from_expr(node.args[0])
            if path:
                return path, "read", line
        return None, "read", line

    if method_name in WRITE_METHODS:
        for keyword in node.keywords:
            if keyword.arg in ("path_or_buf", "excel_writer"):
                path = path_literal_from_expr(keyword.value)
                if path:
                    return path, "write", line
        if node.args:
            path = path_literal_from_expr(node.args[0])
            if path:
                return path, "write", line
        return None, "write", line

    if method_name in SAVE_METHODS:
        for keyword in node.keywords:
            if keyword.arg == "fname":
                path = path_literal_from_expr(keyword.value)
                if path:
                    return path, "plot", line
        if node.args:
            path = path_literal_from_expr(node.args[0])
            if path:
                return path, "plot", line
        return None, "plot", line

    return None, None, line


def run_io_allowlist_scan(
    code: str,
    input_files: list[str],
    output_files: list[str],
) -> SafetyCheckResult:
    """Check generated code against declared input/output basenames."""
    allowed_inputs = {Path(name).name for name in input_files if name}
    allowed_outputs = {Path(name).name for name in output_files if name}
    enforce_inputs = bool(allowed_inputs)
    enforce_outputs = bool(allowed_outputs)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        violations = [
            {
                "rule": "syntax-error",
                "line": exc.lineno or 0,
                "message": f"Generated code is not valid Python: {exc.msg}",
            }
        ]
        return SafetyCheckResult(
            passed=False,
            source="io",
            detail=format_io_issues(violations),
            violations=violations,
        )

    reads: list[tuple[str, int]] = []
    writes: list[tuple[str, int]] = []
    violations: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path, kind, line = call_path_and_kind(node)
        if kind is None:
            continue
        if path is None:
            label = "read" if kind == "read" else "write/plot"
            violations.append(
                {
                    "rule": "dynamic-path",
                    "line": line,
                    "message": (
                        f"IO {label} must use a plain basename string literal "
                        "(no variables, f-strings, or concatenation)."
                    ),
                }
            )
            continue
        if kind == "read":
            reads.append((path, line))
        else:
            writes.append((path, line))

    if enforce_inputs:
        for path, line in reads:
            if path not in allowed_inputs:
                violations.append(
                    {
                        "rule": "undeclared-read",
                        "line": line,
                        "message": (
                            f"Read '{path}' is not in input_files {sorted(allowed_inputs)}."
                        ),
                    }
                )
        for declared in sorted(allowed_inputs):
            if not any(path == declared for path, _ in reads):
                violations.append(
                    {
                        "rule": "missing-read",
                        "line": 0,
                        "message": (
                            f"Declared input '{declared}' is not read in the script."
                        ),
                    }
                )

    if enforce_outputs:
        for path, line in writes:
            if path not in allowed_outputs:
                violations.append(
                    {
                        "rule": "undeclared-write",
                        "line": line,
                        "message": (
                            f"Write/plot '{path}' is not in output_files {sorted(allowed_outputs)}."
                        ),
                    }
                )
        for declared in sorted(allowed_outputs):
            if not any(path == declared for path, _ in writes):
                violations.append(
                    {
                        "rule": "missing-write",
                        "line": 0,
                        "message": (
                            f"Declared output '{declared}' is not written or saved in the script."
                        ),
                    }
                )

    if violations:
        return SafetyCheckResult(
            passed=False,
            source="io",
            detail=format_io_issues(violations),
            violations=violations,
        )
    return SafetyCheckResult(passed=True, source="io")
