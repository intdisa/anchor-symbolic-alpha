from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_synthetic_selector_benchmark_script(tmp_path: Path) -> None:
    run_name = "synthetic_selector_test"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_selector_benchmark.py",
            "--output-root",
            str(tmp_path),
            "--run-name",
            run_name,
            "--seed",
            "11",
            "--dates",
            "48",
            "--entities",
            "48",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    report_path = tmp_path / run_name / "reports" / "synthetic_selector_benchmark.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["naive_formula"] == payload["spurious_formula"]
    assert payload["selector_records"][0] == payload["true_formula"]
