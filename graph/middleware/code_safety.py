"""
Custom middleware: inspects Python code before sandbox execution.

Allowed: pandas, numpy, sklearn, plotly, matplotlib, math, statistics, json, csv,
         io, datetime, collections, itertools, functools, re, typing.
Blocked: os, sys, subprocess, shutil, socket, http, urllib, requests, etc.
No network, no filesystem deletion, no raw file open.

If unsafe → auto-rewrite (strip bad imports/calls).
If still unsafe → reject with explanation.
"""
import ast
import re as re_mod
from typing import Any, Callable, Set

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

ALLOWED_MODULES: Set[str] = {
    "pandas", "pd",
    "numpy", "np",
    "sklearn",
    "plotly", "plotly.express", "plotly.graph_objects", "plotly.subplots",
    "matplotlib", "matplotlib.pyplot", "plt",
    "math", "statistics", "json", "csv", "io",
    "datetime", "collections", "itertools", "functools",
    "re", "typing", "dataclasses", "decimal", "fractions",
}

BLOCKED_MODULES: Set[str] = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "ftplib", "smtplib", "paramiko", "fabric",
    "ctypes", "cffi", "importlib", "signal",
    "multiprocessing", "threading", "asyncio",
    "pickle", "shelve", "dbm", "sqlite3",
}

BLOCKED_BUILTINS: Set[str] = {
    "exec", "eval", "compile", "__import__",
    "open", "exit", "quit", "breakpoint",
    "globals", "locals", "vars", "dir",
}

BLOCKED_METHODS: Set[str] = {
    "remove", "rmdir", "unlink", "rmtree",
    "rename", "replace", "chmod", "chown",
    "system", "popen", "spawn",
}


def check_code_safety(code: str) -> tuple:
    """
    Parse and inspect Python code. Returns (is_safe: bool, violations: list[str]).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    violations.append(f"Blocked import: {alias.name}")
                elif root not in ALLOWED_MODULES and alias.name.split(".")[0] not in ALLOWED_MODULES:
                    violations.append(f"Unrecognized import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BLOCKED_MODULES:
                violations.append(f"Blocked import: from {node.module}")
            elif root not in ALLOWED_MODULES:
                violations.append(f"Unrecognized import: from {node.module}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                violations.append(f"Blocked builtin: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_METHODS:
                violations.append(f"Blocked method: .{node.func.attr}()")

    return len(violations) == 0, violations


def auto_rewrite(code: str) -> str:
    """Strip blocked imports and known-dangerous calls from code."""
    lines = code.split("\n")
    safe_lines = []
    for line in lines:
        stripped = line.strip()
        skip = False
        for blocked in BLOCKED_MODULES:
            if re_mod.match(rf"^\s*(import|from)\s+{re_mod.escape(blocked)}\b", stripped):
                skip = True
                break
        for blocked in BLOCKED_BUILTINS:
            if re_mod.search(rf"\b{re_mod.escape(blocked)}\s*\(", stripped):
                skip = True
                break
        if not skip:
            safe_lines.append(line)
    return "\n".join(safe_lines)


class CodeSafetyMiddleware(AgentMiddleware):
    """
    Wraps shell/code-execution tool calls. Inspects Python code for safety.
    Attempts auto-rewrite if unsafe; rejects if still unsafe.
    """

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[..., Any],
    ) -> Any:
        tool_name = getattr(request, "tool_name", "") or ""
        if tool_name != "shell":
            return handler(request)

        tool_args = getattr(request, "tool_args", {}) or {}
        command = ""
        if isinstance(tool_args, dict):
            command = tool_args.get("command", "") or tool_args.get("input", "")
        elif isinstance(tool_args, str):
            command = tool_args

        is_python = any(p in command for p in ["python", ".py", "python3"])
        if not is_python:
            return handler(request)

        code_match = re_mod.search(r'python3?\s+-c\s+["\'](.+?)["\']', command, re_mod.DOTALL)
        if not code_match:
            return handler(request)

        code = code_match.group(1)
        is_safe, violations = check_code_safety(code)
        if is_safe:
            return handler(request)

        rewritten = auto_rewrite(code)
        is_safe_2, violations_2 = check_code_safety(rewritten)
        if is_safe_2:
            new_command = command.replace(code, rewritten)
            if hasattr(request, "tool_args") and isinstance(request.tool_args, dict):
                request.tool_args["command"] = new_command
            return handler(request)

        return ToolMessage(
            content=(
                f"Code execution blocked: {'; '.join(violations)}. "
                f"Only pandas, numpy, sklearn, plotly, matplotlib are allowed. "
                f"No network access, no file deletion. Please rewrite."
            ),
            tool_call_id=getattr(request, "tool_call_id", ""),
        )
