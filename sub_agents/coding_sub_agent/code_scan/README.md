# Code scan (coding sub-agent)

Static safety scanning before E2B execution.

| File | Purpose |
|------|---------|
| `codegen_scan_semgrep.yaml` | Semgrep rules for generated Python (I/O allowlists, blocked imports) |
| `safety_check.py` | `SafetyCheckResult` type shared by Semgrep and judge gates |
| `semgrep_scan.py` | Runs Semgrep on generated code (`run_semgrep_scan`) |

Used from `SemgrepScan` in [`../graph.py`](../graph.py). Host app needs `semgrep>=1.50.0` (see root `requirements.txt`).
