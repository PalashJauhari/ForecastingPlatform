"""
Semgrep static analysis for generated Python (used by ``tools.generate_code``).

Runs ``semgrep`` with ``tools/agent_sandbox.yaml``. Requires ``pip install semgrep``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

SEMGREP_CONFIG = Path(__file__).resolve().parent / "agent_sandbox.yaml"


def run_semgrep_scan(code: str) -> dict:
    """Run semgrep on *code*; returns ``{"passed": bool, "violations": list}``."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["semgrep", "--config", str(SEMGREP_CONFIG), "--json", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            output = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            return {
                "passed": False,
                "violations": [{
                    "rule": "semgrep-invalid-output",
                    "line": 0,
                    "message": "semgrep returned non-JSON output",
                    "code": (result.stdout or "")[:500],
                }],
            }
        findings = output.get("results", [])
        violations = []
        for f in findings:
            violations.append({
                "rule": f.get("check_id", "unknown"),
                "line": f.get("start", {}).get("line", "?"),
                "message": f.get("extra", {}).get("message", ""),
                "code": f.get("extra", {}).get("lines", "").strip(),
            })
        return {"passed": len(violations) == 0, "violations": violations}

    except FileNotFoundError:
        return {
            "passed": False,
            "violations": [{
                "rule": "semgrep-not-found",
                "line": 0,
                "message": "semgrep not installed — run: pip install semgrep",
                "code": "",
            }],
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "violations": [{
                "rule": "semgrep-timeout",
                "line": 0,
                "message": "semgrep scan timed out after 30 seconds",
                "code": "",
            }],
        }
    finally:
        os.unlink(tmp_path)


def format_semgrep_issues(violations: list[dict]) -> str:
    lines = ["Semgrep rejected the generated code.", "Violations:", ""]
    for v in violations:
        lines.append(f"  Line {v['line']:<4} | {v['rule']}")
        lines.append(f"           {v['message']}")
        if v.get("code"):
            lines.append(f"           Code: {v['code']}")
        lines.append("")
    lines.append("Rules: see tools/coding_tools/code_scan/agent_sandbox.yaml.")
    return "\n".join(lines)
