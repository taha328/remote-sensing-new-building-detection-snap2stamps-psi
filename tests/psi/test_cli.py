from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from aoi_psi.cli import app


def test_python_module_cli_help_runs() -> None:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-m", "aoi_psi.cli", "--help"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "run-pipeline" in result.stdout


def test_cli_disables_pretty_exception_locals() -> None:
    assert app.pretty_exceptions_show_locals is False
