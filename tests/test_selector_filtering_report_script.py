from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_selector_filtering_report_writes_pseudoalpha_cases(tmp_path: Path) -> None:
    baseline_rows = pd.DataFrame(
        [
            {
                "universe": "liquid500",
                "baseline": "support_adjusted_cross_seed_consensus",
                "formula": "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD",
                "matches_consensus_formula": True,
                "gross_sharpe": 0.5694,
            },
            {
                "universe": "liquid500",
                "baseline": "lasso_formula_screening",
                "formula": "CASH_RATIO_Q RANK",
                "matches_consensus_formula": False,
                "gross_sharpe": 0.0755,
            },
        ]
    )
    baseline_rows.to_csv(tmp_path / "finance_walkforward_baselines.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/build_selector_filtering_report.py",
            "--output-root",
            str(tmp_path),
            "--skip-live-selection",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    pseudoalpha_path = tmp_path / "selector_pseudoalpha_cases.csv"
    assert pseudoalpha_path.exists()
    pseudoalpha = pd.read_csv(pseudoalpha_path)
    assert "baseline_misfire" in set(pseudoalpha["case_type"])
