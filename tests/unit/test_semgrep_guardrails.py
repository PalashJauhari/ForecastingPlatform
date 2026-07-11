"""Smoke tests for Semgrep OS/network guardrails (primary static gate before E2B execution)."""

import shutil
from pathlib import Path

import pytest

from sub_agents.coding_sub_agent.code_scan.semgrep_scan import _semgrep_executable, run_semgrep_scan


def _semgrep_available() -> bool:
    exe = _semgrep_executable()
    if exe != "semgrep":
        return Path(exe).is_file()
    return shutil.which("semgrep") is not None


pytestmark = pytest.mark.skipif(
    not _semgrep_available(),
    reason="semgrep CLI not installed (pip install semgrep)",
)

VALID_PANDAS_SCRIPT = """
import pandas as pd

df = pd.read_csv("data.csv")
print(df.shape)
df.to_csv("output.csv", index=False)
"""

IMPORT_OS_SCRIPT = """
import os
import pandas as pd

df = pd.read_csv("data.csv")
print(os.getcwd())
"""

IMPORT_SOCKET_SCRIPT = """
import socket
import pandas as pd

df = pd.read_csv("data.csv")
print(df.shape)
"""


def test_semgrep_blocks_import_os():
    result = run_semgrep_scan(IMPORT_OS_SCRIPT)
    assert not result.passed
    assert result.source == "semgrep"


def test_semgrep_blocks_import_socket():
    result = run_semgrep_scan(IMPORT_SOCKET_SCRIPT)
    assert not result.passed
    assert result.source == "semgrep"


def test_semgrep_allows_basename_pandas_io():
    result = run_semgrep_scan(VALID_PANDAS_SCRIPT)
    assert result.passed
    assert result.source == "semgrep"
