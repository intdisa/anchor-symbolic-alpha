from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_split(frame: pd.DataFrame, root: Path, split_name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / f"{split_name}.csv.gz", index=False, compression="gzip")


def test_evaluate_formula_script_runs_on_csv_only_subset(tmp_path: Path) -> None:
    split_root = tmp_path / "data" / "processed" / "us_equities" / "subsets" / "liquid2_2020_2020"
    dates = pd.to_datetime(
        [
            "2020-01-02",
            "2020-01-02",
            "2020-01-03",
            "2020-01-03",
            "2020-01-06",
            "2020-01-06",
        ]
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "permno": [1, 2, 1, 2, 1, 2],
            "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "comnam": ["Alpha", "Beta", "Alpha", "Beta", "Alpha", "Beta"],
            "siccd": [3571, 3571, 3571, 3571, 3571, 3571],
            "RET_1": [0.1, -0.1, 0.05, -0.02, 0.03, -0.01],
            "TARGET_RET_1": [0.02, -0.02, 0.01, -0.01, 0.03, -0.03],
            "TARGET_XS_RET_1": [0.02, -0.02, 0.01, -0.01, 0.03, -0.03],
        }
    )
    _write_split(frame.iloc[:2], split_root, "train")
    _write_split(frame.iloc[2:], split_root, "valid")
    _write_split(frame.iloc[2:], split_root, "test")

    data_config = tmp_path / "us_equities_liquid2.yaml"
    data_config.write_text(
        "us_equities_subset:\n"
        f"  split_root: {split_root.as_posix()}\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs" / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_formula.py",
            "--formula",
            "RET_1 RANK",
            "--data-config",
            str(data_config),
            "--split",
            "valid",
            "--output-root",
            str(output_root),
            "--run-name",
            "formula-eval-smoke",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    report_path = output_root / "formula-eval-smoke" / "reports" / "formula_eval.json"
    assert payload["dataset"] == "us_equities"
    assert report_path.exists()
