from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_benchmark_pareto_plot_data_script(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "leaderboard": [
                    {
                        "baseline": "pareto_cross_seed_consensus",
                        "selection_accuracy": 0.91,
                        "selected_formula_stability": 1.0,
                    },
                    {
                        "baseline": "support_adjusted_cross_seed_consensus",
                        "selection_accuracy": 0.9,
                        "selected_formula_stability": 0.95,
                    },
                    {
                        "baseline": "naive_rank_ic",
                        "selection_accuracy": 0.93,
                        "selected_formula_stability": 0.5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_benchmark_pareto_plot_data.py",
            "--summary-json",
            str(summary_path),
            "--output-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    csv_path = Path(payload["csv"])
    frame = pd.read_csv(csv_path)

    assert csv_path.exists()
    assert "pareto_frontier" in frame.columns
    assert "pareto_cross_seed_consensus" in set(frame["baseline"])
    assert "support_adjusted_cross_seed_consensus" not in set(frame["baseline"])
