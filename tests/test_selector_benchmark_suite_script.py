from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from knowledge_guided_symbolic_alpha.benchmarks import synthetic_selector_scenarios


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
    stress_json = tmp_path / run_name / "reports" / "selector_benchmark_stress_summary.json"

    assert payload["task_report_count"] == len(synthetic_selector_scenarios())
    assert summary_json.exists()
    assert leaderboard_csv.exists()
    assert stress_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert any(item["baseline"] == "pareto_cross_seed_consensus" for item in summary["leaderboard"])
    assert any(item["worst_case_task_id"] == "adversarial_support_lockin" for item in summary["stress_summary"])
