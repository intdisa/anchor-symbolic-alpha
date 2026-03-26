from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.eval
def test_build_us_equities_subset_script_smoke(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")

    source_root = tmp_path / "data" / "processed" / "us_equities" / "splits"
    source_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "permno": [1, 2, 1, 2],
            "DOLLAR_VOLUME_20": [10.0, 20.0, 10.0, 20.0],
            "TARGET_RET_1": [0.01, -0.01, 0.02, -0.02],
            "TARGET_XS_RET_1": [0.01, -0.01, 0.02, -0.02],
        }
    )
    for split_name in ("train", "valid", "test"):
        frame.to_parquet(source_root / f"{split_name}.parquet", index=False)

    output_root = tmp_path / "data" / "processed" / "us_equities" / "subsets"
    summary_root = tmp_path / "outputs" / "reports"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_us_equities_subset.py",
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--summary-root",
            str(summary_root),
            "--name",
            "liquid2_2020_2020",
            "--max-permnos",
            "1",
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-12-31",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["name"] == "liquid2_2020_2020"
    assert (output_root / "liquid2_2020_2020" / "train.parquet").exists()
