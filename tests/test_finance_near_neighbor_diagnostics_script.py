from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_finance_near_neighbor_diagnostics_script(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_finance_near_neighbor_diagnostics.py",
            "--output-root",
            str(tmp_path),
            "--top-k",
            "3",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    frame = pd.read_csv(Path(payload["csv"]))

    assert "token_overlap" in frame.columns
    assert "signal_spearman_corr" in frame.columns
    assert frame["left_is_canonical"].any() or frame["right_is_canonical"].any()
