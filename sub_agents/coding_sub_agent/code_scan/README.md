# Code scan (coding sub-agent)

Static safety scanning before E2B execution. With LLM judges disabled, **Semgrep is the primary static gate** — it blocks OS APIs (`import os`, subprocess, `open()`), network clients (`socket`, `requests`, etc.), and non-pandas file I/O.

| File | Purpose |
|------|---------|
| `codegen_scan_semgrep.yaml` | Semgrep rules for generated Python (OS/network blocks, I/O allowlists, blocked imports) |
| `safety_check.py` | `SafetyCheckResult` type shared by Semgrep and judge gates |
| `semgrep_scan.py` | Runs Semgrep on generated code (`run_semgrep_scan`) |

Used from `SemgrepScan` in [`../graph.py`](../graph.py). Host app needs `semgrep>=1.50.0` (see root `requirements.txt`).
