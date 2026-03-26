from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_standard_training_script_dry_run_builds_anchor_and_backtest_stages(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_standard_training.py",
            "--mode",
            "anchor",
            "--run-name",
            "anchor_plan",
            "--data-config",
            "configs/us_equities_smoke.yaml",
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["mode"] == "anchor"
    assert [stage["name"] for stage in payload["commands"]] == ["train_preflight", "anchor", "backtest"]
    assert payload["commands"][1]["run_name"] == "anchor_plan__anchor"
    assert payload["commands"][2]["run_name"] == "anchor_plan__backtest"
    assert (output_root / "anchor_plan" / "reports" / "workflow_summary.json").exists()


def test_standard_training_script_dry_run_skips_backtest_for_ablation(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_standard_training.py",
            "--mode",
            "ablation",
            "--run-name",
            "ablation_plan",
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert [stage["name"] for stage in payload["commands"]] == ["train_preflight", "ablation"]


def test_standard_training_script_dry_run_exposes_multiseed_canonical_artifact(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_standard_training.py",
            "--mode",
            "multiseed",
            "--run-name",
            "multiseed_plan",
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["artifacts"]["multiseed_report"].endswith("us_equities_multiseed.json")
    assert payload["artifacts"]["multiseed_canonical_report"].endswith("us_equities_multiseed_canonical.json")


def test_standard_training_script_dry_run_skips_backtest_in_smoke_mode(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "runs"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_standard_training.py",
            "--mode",
            "anchor",
            "--run-name",
            "anchor_smoke_plan",
            "--smoke",
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert [stage["name"] for stage in payload["commands"]] == ["train_preflight", "anchor"]
