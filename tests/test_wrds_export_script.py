from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_wrds_export_script_dry_run_prints_manifest() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_wrds_us_equities.py",
            "--dry-run",
            "--datasets",
            "crsp_daily",
            "--limit",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["datasets"]["crsp_daily"]["table"] == "crsp.dsf"
    assert "limit 10" in payload["datasets"]["crsp_daily"]["sql"]
