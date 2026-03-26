#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.runtime import ensure_preflight, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selector benchmark suites.")
    parser.add_argument("--suite", choices=("synthetic", "public", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-name", type=str, default="selector_benchmark_suite")
    parser.add_argument("--seeds", type=str, default="7,17,27,37,47")
    parser.add_argument("--samples-per-env", type=int, default=96)
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    preflight = ensure_preflight("core")

    from experiments.common import ensure_output_dirs, write_json
    from knowledge_guided_symbolic_alpha.benchmarks import (
        generate_public_symbolic_task,
        generate_synthetic_selector_task,
        public_symbolic_task_specs,
        run_task_benchmark,
        suite_leaderboard,
        synthetic_selector_scenarios,
    )

    output_dirs = ensure_output_dirs(args.output_root, args.run_name)
    manifest_path = write_run_manifest(
        output_dirs,
        script_name="scripts/run_selector_benchmark_suite.py",
        profile="core",
        preflight=preflight.to_dict(),
        config_paths={},
        dataset_name="selector_benchmark_suite",
        subset=args.suite,
        extra={"seeds": seeds, "samples_per_env": args.samples_per_env},
    )

    task_groups: list[list[Any]] = []
    if args.suite in {"synthetic", "all"}:
        for scenario in synthetic_selector_scenarios():
            task_groups.append(
                [
                    generate_synthetic_selector_task(scenario, seed=seed, samples_per_env=args.samples_per_env)
                    for seed in seeds
                ]
            )
    if args.suite in {"public", "all"}:
        for spec in public_symbolic_task_specs():
            task_groups.append(
                [
                    generate_public_symbolic_task(spec.task_id, seed=seed, samples_per_env=args.samples_per_env)
                    for seed in seeds
                ]
            )

    results = [run_task_benchmark(tasks) for tasks in task_groups]
    leaderboard = suite_leaderboard(results)
    task_payloads = [asdict(result) if is_dataclass(result) else result for result in results]
    suite_payload = {
        "benchmark_name": f"{args.suite}_selector_benchmark_suite",
        "suite": args.suite,
        "seeds": seeds,
        "samples_per_env": args.samples_per_env,
        "task_results": task_payloads,
        "leaderboard": leaderboard,
        "manifest": str(manifest_path),
    }
    task_root = output_dirs["reports"] / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    for task_payload in task_payloads:
        task_path = task_root / f"{task_payload['benchmark_name']}__{task_payload['task_id']}.json"
        write_json(task_path, task_payload)
    summary_json = output_dirs["reports"] / f"{args.suite}_selector_benchmark_summary.json"
    summary_md = output_dirs["reports"] / f"{args.suite}_selector_benchmark_summary.md"
    leaderboard_csv = output_dirs["reports"] / f"{args.suite}_selector_benchmark_leaderboard.csv"
    write_json(summary_json, suite_payload)
    summary_md.write_text(build_markdown(suite_payload) + "\n", encoding="utf-8")
    import pandas as pd

    pd.DataFrame(leaderboard).to_csv(leaderboard_csv, index=False)
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_markdown": str(summary_md),
                "leaderboard_csv": str(leaderboard_csv),
                "task_report_count": len(task_payloads),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def build_markdown(payload: dict[str, Any]) -> str:
    leaderboard = payload.get("leaderboard", [])
    lines = [f"# {payload['benchmark_name']}", "", "## Leaderboard", ""]
    if leaderboard:
        headers = ["Baseline", "SelAcc", "MisSel", "OracleRegret", "RankIC", "Sharpe", "Stability"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in leaderboard:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["baseline"]),
                        _fmt(row.get("selection_accuracy")),
                        _fmt(row.get("misselection_rate")),
                        _fmt(row.get("oracle_regret_rank_ic")),
                        _fmt(row.get("mean_test_rank_ic")),
                        _fmt(row.get("mean_test_sharpe")),
                        _fmt(row.get("selected_formula_stability")),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Tasks", ""])
    for task in payload.get("task_results", []):
        lines.append(
            f"- `{task['benchmark_name']}` / `{task['task_id']}` / `{task['scenario']}` -> true `{task['true_formula']}`"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
