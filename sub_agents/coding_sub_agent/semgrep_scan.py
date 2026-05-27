"""
Semgrep static analysis for generated Python in the coding sub-agent.

Fail-closed when semgrep is missing or errors; rules live in ``codegen_scan_semgrep.yaml``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from sub_agents.coding_sub_agent.safety_check import SafetyCheckResult

SEMGREP_CONFIG = Path(__file__).resolve().parent / "codegen_scan_semgrep.yaml"


def format_semgrep_issues(violations: list[dict]) -> str:
    """Human-readable block for tool payloads and retry context."""
    lines = ["Semgrep rejected the generated code.", "Violations:", ""]
    for v in violations:
        lines.append(f"  Line {v['line']:<4} | {v['rule']}")
        lines.append(f"           {v['message']}")
        if v.get("code"):
            lines.append(f"           Code: {v['code']}")
        lines.append("")
    lines.append("Rules: see sub_agents/coding_sub_agent/codegen_scan_semgrep.yaml.")
    return "\n".join(lines)


def synthetic_failure(
    rule: str,
    message: str,
    *,
    stdout: str = "",
    stderr: str = "",
) -> SafetyCheckResult:
    """Fail-closed Semgrep gate with one synthetic violation row."""
    details: list[str] = []
    if stderr.strip():
        details.append(f"stderr: {stderr.strip()[:500]}")
    if stdout.strip():
        details.append(f"stdout: {stdout.strip()[:500]}")
    if details:
        message = f"{message} ({' | '.join(details)})"
    violations = [{"rule": rule, "line": 0, "message": message, "code": ""}]
    return SafetyCheckResult(
        passed=False,
        source="semgrep",
        detail=format_semgrep_issues(violations),
        violations=violations,
    )


def run_semgrep_scan(code: str) -> SafetyCheckResult:
    """Run semgrep on generated code; returns :class:`SafetyCheckResult`."""
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
        if result.returncode not in {0, 1}:
            return synthetic_failure(
                "semgrep-execution-failed",
                f"semgrep exited with status {result.returncode}",
                stdout=result.stdout,
                stderr=result.stderr,
            )
        if not result.stdout.strip():
            return synthetic_failure(
                "semgrep-empty-output",
                "semgrep produced no JSON output",
                stderr=result.stderr,
            )
        try:
            output = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError:
            return synthetic_failure(
                "semgrep-invalid-output",
                "semgrep returned invalid JSON output",
                stdout=result.stdout,
                stderr=result.stderr,
            )
        semgrep_errors = output.get("errors") or []
        if semgrep_errors:
            error_summary = " | ".join(
                str(err.get("message") or err)[:200] for err in semgrep_errors[:3]
            )
            return synthetic_failure(
                "semgrep-reported-errors",
                f"semgrep reported scanner/configuration errors: {error_summary}",
                stderr=result.stderr,
            )
        findings = output.get("results", [])
        violations: list[dict] = []
        for finding in findings:
            violations.append(
                {
                    "rule": finding.get("check_id", "unknown"),
                    "line": finding.get("start", {}).get("line", "?"),
                    "message": finding.get("extra", {}).get("message", ""),
                    "code": finding.get("extra", {}).get("lines", "").strip(),
                }
            )
        if violations:
            return SafetyCheckResult(
                passed=False,
                source="semgrep",
                detail=format_semgrep_issues(violations),
                violations=violations,
            )
        return SafetyCheckResult(passed=True, source="semgrep")

    except FileNotFoundError:
        return synthetic_failure(
            "semgrep-not-found",
            "semgrep not installed — run: pip install semgrep",
        )
    except subprocess.TimeoutExpired:
        return synthetic_failure(
            "semgrep-timeout",
            "semgrep scan timed out after 30 seconds",
        )
    finally:
        os.unlink(tmp_path)
