"""Unit tests for deterministic IO allowlist scan."""

from sub_agents.coding_sub_agent.code_scan.io_allowlist_scan import run_io_allowlist_scan


def test_io_scan_passes_when_reads_and_writes_match_lists() -> None:
    code = """
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sales.csv")
df.to_csv("out.csv")
plt.savefig("plot.png")
"""
    result = run_io_allowlist_scan(
        code,
        input_files=["sales.csv"],
        output_files=["out.csv", "plot.png"],
    )
    assert result.passed


def test_io_scan_undeclared_read() -> None:
    code = 'import pandas as pd\ndf = pd.read_csv("other.csv")\n'
    result = run_io_allowlist_scan(code, input_files=["sales.csv"], output_files=[])
    assert not result.passed
    assert result.detail
    assert any(v.get("rule") == "undeclared-read" for v in (result.violations or []))


def test_io_scan_missing_declared_output() -> None:
    code = 'import pandas as pd\ndf = pd.read_csv("sales.csv")\ndf.to_csv("out.csv")\n'
    result = run_io_allowlist_scan(
        code,
        input_files=["sales.csv"],
        output_files=["out.csv", "plot.png"],
    )
    assert not result.passed
    assert any(v.get("rule") == "missing-write" for v in (result.violations or []))


def test_io_scan_empty_lists_skip_enforcement() -> None:
    code = 'import pandas as pd\ndf = pd.read_csv("anything.csv")\n'
    result = run_io_allowlist_scan(code, input_files=[], output_files=[])
    assert result.passed


def test_io_scan_dynamic_path_fails() -> None:
    code = """
import pandas as pd
name = "sales.csv"
df = pd.read_csv(name)
"""
    result = run_io_allowlist_scan(code, input_files=["sales.csv"], output_files=[])
    assert not result.passed
    assert any(v.get("rule") == "dynamic-path" for v in (result.violations or []))
