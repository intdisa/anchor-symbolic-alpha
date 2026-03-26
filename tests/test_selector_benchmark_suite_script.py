from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_selector_benchmark_suite_script_runs_synthetic_smoke(tmp_path: Path) -> None:
    run_name = "synthetic_suite_smoke"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_selector_benchmark_suite.py",
            "--suite",
            "synthetic",
            "--output-root",
            str(tmp_path),
            "--run-name",
            run_name,
            "--seeds",
            "7,17",
            "--samples-per-env",
            "24",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    summary_json = tmp_path / run_name / "reports" / "synthetic_selector_benchmark_summary.json"
    leaderboard_csv = tmp_path / run_name / "reports" / "synthetic_selector_benchmark_leaderboard.csv"

    assert payload["task_report_count"] == 4
    assert summary_json.exists()
    assert leaderboard_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert any(item["baseline"] == "support_adjusted_cross_seed_consensus" for item in summary["leaderboard"])
