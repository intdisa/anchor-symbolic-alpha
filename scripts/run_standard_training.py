#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.runtime import ensure_run_output_dirs


DEFAULT_OUTPUT_ROOT = Path("outputs/runs")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
DEFAULT_BACKTEST_CONFIG = Path("configs/backtest.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical U.S. equities training workflow.")
    parser.add_argument("--mode", choices=("anchor", "full", "ablation", "multiseed"), default="anchor")
    parser.add_argument("--run-name", type=str, default="us_equities_standard")
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--skills", type=str, default="")
    parser.add_argument("--variants", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--partition-mode", type=str, default="skill_hierarchy")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_data_config(mode: str, smoke: bool) -> Path:
    if smoke:
        return Path("configs/us_equities_smoke.yaml")
    return Path("configs/us_equities_liquid500.yaml")


def default_training_config(mode: str, smoke: bool) -> Path:
    if smoke and mode == "anchor":
        return Path("configs/training_anchor_smoke.yaml")
    return Path("configs/training.yaml")


def default_episodes(mode: str, smoke: bool) -> int:
    if smoke:
        return 1
    if mode in {"anchor", "full"}:
        return 20
    return 5


def stage_run_name(base: str, stage: str) -> str:
    return f"{base}__{stage}"


def factor_file_name(mode: str) -> str:
    return "anchor_champions.json" if mode == "anchor" else "us_equities_champions.json"


def train_script_path(mode: str) -> str:
    mapping = {
        "anchor": "experiments/run_anchor_train.py",
        "full": "experiments/run_train.py",
        "ablation": "experiments/run_ablation.py",
        "multiseed": "experiments/run_multiseed.py",
    }
    return mapping[mode]


def build_stage_commands(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_config = args.data_config or default_data_config(args.mode, args.smoke)
    training_config = args.training_config or default_training_config(args.mode, args.smoke)
    episodes = int(args.episodes or default_episodes(args.mode, args.smoke))

    commands: list[dict[str, Any]] = [
        {
            "name": "train_preflight",
            "kind": "preflight",
            "run_name": None,
            "command": [sys.executable, "scripts/run_preflight.py", "--profile", "train"],
        }
    ]

    train_stage_name = stage_run_name(args.run_name, args.mode)
    train_command = [
        sys.executable,
        train_script_path(args.mode),
        "--episodes",
        str(episodes),
        "--data-config",
        str(data_config),
        "--training-config",
        str(training_config),
        "--backtest-config",
        str(args.backtest_config),
        "--experiment-config",
        str(args.experiment_config),
        "--output-root",
        str(args.output_root),
        "--run-name",
        train_stage_name,
    ]
    if args.mode == "anchor" and args.skills:
        train_command.extend(["--skills", args.skills])
    if args.mode in {"ablation", "multiseed"} and args.variants:
        train_command.extend(["--variants", args.variants])
    if args.mode == "multiseed" and args.seeds:
        train_command.extend(["--seeds", args.seeds])
    if args.mode in {"full", "ablation", "multiseed"}:
        train_command.extend(["--partition-mode", args.partition_mode])

    commands.append(
        {
            "name": args.mode,
            "kind": "train",
            "run_name": train_stage_name,
            "command": train_command,
        }
    )

    if args.mode in {"anchor", "full"} and not args.skip_backtest and not args.smoke:
        backtest_stage_name = stage_run_name(args.run_name, "backtest")
        factor_file = args.output_root / train_stage_name / "factors" / factor_file_name(args.mode)
        commands.append(
            {
                "name": "backtest",
                "kind": "backtest",
                "run_name": backtest_stage_name,
                "command": [
                    sys.executable,
                    "experiments/run_backtest.py",
                    "--data-config",
                    str(data_config),
                    "--training-config",
                    str(training_config),
                    "--backtest-config",
                    str(args.backtest_config),
                    "--experiment-config",
                    str(args.experiment_config),
                    "--output-root",
                    str(args.output_root),
                    "--run-name",
                    backtest_stage_name,
                    "--formula-file",
                    str(factor_file),
                ],
            }
        )

    return commands


def expected_artifacts(args: argparse.Namespace) -> dict[str, str]:
    train_stage_name = stage_run_name(args.run_name, args.mode)
    artifacts: dict[str, str] = {
        "workflow_summary": str(args.output_root / args.run_name / "reports" / "workflow_summary.json"),
    }
    if args.mode == "anchor":
        artifacts["train_summary"] = str(args.output_root / train_stage_name / "reports" / "anchor_train_summary.json")
        if not args.skip_backtest and not args.smoke:
            artifacts["backtest_report"] = str(
                args.output_root / stage_run_name(args.run_name, "backtest") / "reports" / "us_equities_walk_forward.json"
            )
    elif args.mode == "full":
        artifacts["train_summary"] = str(args.output_root / train_stage_name / "reports" / "us_equities_train_summary.json")
        if not args.skip_backtest and not args.smoke:
            artifacts["backtest_report"] = str(
                args.output_root / stage_run_name(args.run_name, "backtest") / "reports" / "us_equities_walk_forward.json"
            )
    elif args.mode == "ablation":
        artifacts["ablation_report"] = str(args.output_root / train_stage_name / "reports" / "us_equities_ablation.json")
    elif args.mode == "multiseed":
        artifacts["multiseed_report"] = str(args.output_root / train_stage_name / "reports" / "us_equities_multiseed.json")
        artifacts["multiseed_canonical_report"] = str(
            args.output_root / train_stage_name / "reports" / "us_equities_multiseed_canonical.json"
        )
    return artifacts


def run_stage(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "returncode": completed.returncode,
        "duration_seconds": round(time.time() - started, 2),
        "log_path": str(log_path),
    }


def write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    workflow_dirs = ensure_run_output_dirs(args.output_root, args.run_name)
    summary_path = workflow_dirs["reports"] / "workflow_summary.json"
    commands = build_stage_commands(args)

    if args.dry_run:
        payload = {
            "mode": args.mode,
            "run_name": args.run_name,
            "dry_run": True,
            "commands": commands,
            "artifacts": expected_artifacts(args),
            "summary_path": str(summary_path),
        }
        write_summary(summary_path, payload)
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    stage_results: list[dict[str, Any]] = []
    for index, stage in enumerate(commands, start=1):
        log_path = workflow_dirs["logs"] / f"{index:02d}_{stage['name']}.log"
        result = run_stage(stage["command"], log_path)
        stage_results.append(
            {
                "name": stage["name"],
                "kind": stage["kind"],
                "run_name": stage["run_name"],
                "command": stage["command"],
                **result,
            }
        )
        print(f"[{index}/{len(commands)}] {stage['name']} returncode={result['returncode']} log={log_path}")
        if result["returncode"] != 0:
            payload = {
                "mode": args.mode,
                "run_name": args.run_name,
                "success": False,
                "stages": stage_results,
                "artifacts": expected_artifacts(args),
                "summary_path": str(summary_path),
            }
            write_summary(summary_path, payload)
            raise SystemExit(result["returncode"])

    payload = {
        "mode": args.mode,
        "run_name": args.run_name,
        "success": True,
        "stages": stage_results,
        "artifacts": expected_artifacts(args),
        "summary_path": str(summary_path),
    }
    write_summary(summary_path, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
