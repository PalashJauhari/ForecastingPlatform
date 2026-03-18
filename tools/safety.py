"""
safety: shared AST-based safety check for generated pandas code.

Flow:
  1. Parse code with ast; on syntax error return (False, [error message]).
  2. Walk AST: reject imports not in ALLOWED_MODULES; reject calls to BLOCKED_BUILTINS/BLOCKED_ATTRS.
  3. Return (is_safe: bool, violations: list of strings).

Used by: clean_csv (after LLM code gen), ask_csv (after LLM code gen).
Allowed: pandas, numpy, sklearn, scipy, plotly, matplotlib.
Blocked: open, exec, eval, os.system, path.remove, etc.
"""

import ast
from typing import List, Set, Tuple

# Modules the generated code may import (clean_csv and ask_csv)
ALLOWED_MODULES: Set[str] = {"pandas", "numpy", "sklearn", "scipy", "plotly", "matplotlib"}
# Builtins that must not be called (no exec, eval, open, etc.)
BLOCKED_BUILTINS: Set[str] = {"open", "exec", "eval", "compile", "__import__", "input", "file"}
# Method names that must not appear in calls (e.g. os.system, path.unlink)
BLOCKED_ATTRS: Set[str] = {"system", "popen", "remove", "unlink", "rmdir", "rmtree", "chmod", "chown"}


def check_code_safety(code: str) -> Tuple[bool, List[str]]:
    """
    Parse code with ast and check for disallowed imports/attributes.
    Returns (is_safe, list of violation messages). Caller runs code only if is_safe is True.
    """
    violations: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    # --- Walk AST: check every Import, ImportFrom, and Call ---
    for node in ast.walk(tree):
        # Only allow imports from ALLOWED_MODULES
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base not in ALLOWED_MODULES:
                    violations.append(f"Import not allowed: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            base = module.split(".")[0]
            if base not in ALLOWED_MODULES:
                violations.append(f"Import not allowed: from {module}")
        # Block dangerous builtins and attribute calls (e.g. open, exec, .remove, .unlink)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_BUILTINS:
                    violations.append(f"Call to blocked builtin: {node.func.id}")
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_ATTRS:
                    violations.append(f"Call to blocked attribute: {node.func.attr}")

    return len(violations) == 0, violations
