from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
from knowledge_guided_symbolic_alpha.training import MultiAgentTrainer

from experiments.common import (
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRAINING_CONFIG,
    build_portfolio_config,
    build_signal_fusion_config,
    build_walk_forward_config,
    dataset_columns,
    dataset_diagnostics,
    dataset_input_frame,
    ensure_output_dirs,
    group_formula_summaries_by_role,
    load_dataset_bundle,
    load_experiment_name,
    load_yaml,
    select_evaluation_formulas,
    write_json,
)
from experiments.run_ablation import build_variant


HIGHER_IS_BETTER_METRICS = {"best_validation_pool_score", "sharpe", "annual_return", "mean_test_rank_ic", "max_drawdown"}
LOWER_IS_BETTER_METRICS = {"turnover"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed training and summarize robustness.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,17,27,37,47",
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="full,target_price_only,target_flow_only",
        help="Comma-separated variants supported by run_ablation.py.",
    )
    parser.add_argument(
        "--partition-mode",
        type=str,
        default="skill_hierarchy",
        choices=("skill_hierarchy", "competitive_three_way", "target_context", "macro_micro"),
    )
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def numeric_summary(values: list[float]) -> dict[str, float] | dict[str, int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def evaluate_formulas(
    backtester: WalkForwardBacktester,
    formulas: list[str],
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    return_column: str,
    walk_forward_config,
) -> dict[str, Any]:
    if not formulas:
        return {"formula_count": 0, "formulas": formulas, "aggregate_metrics": {}}
    report = backtester.run(
        formulas=formulas,
        frame=frame,
        feature_columns=feature_columns,
        target_column=target_column,
        return_column=return_column,
        config=walk_forward_config,
    )
    return {
        "formula_count": len(formulas),
        "formulas": formulas,
        "aggregate_metrics": report.aggregate_metrics,
    }


def aggregate_variant_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    numeric_fields = ("best_validation_pool_score", "final_pool_size", "accepted_episodes", "formula_count")
    for field in numeric_fields:
        aggregated[field] = numeric_summary([float(run[field]) for run in runs])
    metric_keys = sorted(
        {
            key
            for run in runs
            for key in run.get("walk_forward_metrics", {})
            if isinstance(run["walk_forward_metrics"].get(key), (int, float))
        }
    )
    aggregated["walk_forward_metrics"] = {
        key: numeric_summary([float(run["walk_forward_metrics"][key]) for run in runs if key in run["walk_forward_metrics"]])
        for key in metric_keys
    }
    aggregated["seed_count_with_formulas"] = int(sum(1 for run in runs if run["formula_count"] > 0))
    aggregated["context_formula_survived_count"] = int(sum(1 for run in runs if run["context_formula_survived"]))
    return aggregated


def pairwise_comparison(runs_by_variant: dict[str, list[dict[str, Any]]], left_variant: str, right_variant: str) -> dict[str, Any]:
    left_runs = {int(run["seed"]): run for run in runs_by_variant.get(left_variant, [])}
    right_runs = {int(run["seed"]): run for run in runs_by_variant.get(right_variant, [])}
    common_seeds = sorted(left_runs.keys() & right_runs.keys())
    comparisons: list[dict[str, Any]] = []
    win_counts: dict[str, int] = {metric: 0 for metric in HIGHER_IS_BETTER_METRICS | LOWER_IS_BETTER_METRICS}
    valid_counts: dict[str, int] = {metric: 0 for metric in HIGHER_IS_BETTER_METRICS | LOWER_IS_BETTER_METRICS}

    for seed in common_seeds:
        left = left_runs[seed]
        right = right_runs[seed]
        delta_payload: dict[str, float] = {
            "best_validation_pool_score": float(left["best_validation_pool_score"]) - float(right["best_validation_pool_score"])
        }
        left_metrics = left.get("walk_forward_metrics", {})
        right_metrics = right.get("walk_forward_metrics", {})
        for metric in ("sharpe", "annual_return", "mean_test_rank_ic", "max_drawdown", "turnover"):
            if metric in left_metrics and metric in right_metrics:
                delta = float(left_metrics[metric]) - float(right_metrics[metric])
                delta_payload[metric] = delta
        for metric, delta in delta_payload.items():
            if metric in HIGHER_IS_BETTER_METRICS:
                valid_counts[metric] += 1
                if delta > 0.0:
                    win_counts[metric] += 1
            elif metric in LOWER_IS_BETTER_METRICS:
                valid_counts[metric] += 1
                if delta < 0.0:
                    win_counts[metric] += 1
        comparisons.append(
            {
                "seed": seed,
                "left_variant": left_variant,
                "right_variant": right_variant,
                "deltas": delta_payload,
            }
        )
    return {
        "left_variant": left_variant,
        "right_variant": right_variant,
        "common_seeds": common_seeds,
        "seed_level_deltas": comparisons,
        "win_counts": {metric: count for metric, count in win_counts.items() if valid_counts[metric] > 0},
        "valid_counts": {metric: count for metric, count in valid_counts.items() if count > 0},
    }


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in parse_csv_arg(args.seeds)]
    variants = parse_csv_arg(args.variants)
    if not seeds:
        raise ValueError("At least one seed is required.")
    if not variants:
        raise ValueError("At least one variant is required.")

    training_config = load_yaml(args.training_config)
    backtest_config = load_yaml(args.backtest_config)
    dataset_name = load_experiment_name(args.experiment_config)
    bundle = load_dataset_bundle(args.data_config)
    target_column, return_column = dataset_columns(dataset_name)
    train_frame = dataset_input_frame(bundle.splits.train, bundle.feature_columns, dataset_name)
    valid_frame = dataset_input_frame(bundle.splits.valid, bundle.feature_columns, dataset_name)
    train_target = bundle.splits.train[target_column]
    valid_target = bundle.splits.valid[target_column]
    backtest_frame = pd.concat([bundle.splits.valid, bundle.splits.test], axis=0)

    output_dirs = ensure_output_dirs(args.output_root)
    backtester = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    )
    walk_forward_config = build_walk_forward_config(backtest_config)

    runs_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    for seed in seeds:
        for variant in variants:
            manager, curriculum = build_variant(
                variant,
                training_config,
                dataset_name,
                args.partition_mode,
                seed_override=seed,
            )
            trainer = MultiAgentTrainer(manager, curriculum=curriculum)
            summary = trainer.train(
                train_frame,
                train_target,
                episodes=args.episodes,
                validation_data=valid_frame,
                validation_target=valid_target,
            )
            formulas, evaluation_formula_source = select_evaluation_formulas(
                summary.champion_records,
                summary.final_records,
            )
            evaluation = evaluate_formulas(
                backtester,
                formulas,
                backtest_frame,
                bundle.feature_columns,
                target_column,
                return_column,
                walk_forward_config,
            )
            run_record = {
                "seed": seed,
                "variant": variant,
                "best_validation_pool_score": summary.best_validation_pool_score,
                "final_pool_size": summary.final_pool_size,
                "accepted_episodes": int(sum(1 for episode in summary.history if episode.accepted)),
                "formula_count": evaluation["formula_count"],
                "evaluation_formula_source": evaluation_formula_source,
                "champion_records": list(summary.champion_records),
                "final_records": list(summary.final_records),
                "records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
                "context_formula_survived": any(item.role == "context" for item in summary.final_record_summaries),
                "walk_forward_metrics": evaluation["aggregate_metrics"],
            }
            runs_by_variant[variant].append(run_record)
            print(
                f"seed={seed} variant={variant} "
                f"best_validation_pool_score={summary.best_validation_pool_score:.6f} "
                f"final_pool_size={summary.final_pool_size} "
                f"formula_count={evaluation['formula_count']}"
            )

    aggregates = {variant: aggregate_variant_runs(runs) for variant, runs in runs_by_variant.items()}
    comparisons: dict[str, Any] = {}
    if "full" in runs_by_variant and "target_price_only" in runs_by_variant:
        comparisons["full_vs_target_price_only"] = pairwise_comparison(
            runs_by_variant,
            "full",
            "target_price_only",
        )
    if "full" in runs_by_variant and "target_flow_only" in runs_by_variant:
        comparisons["full_vs_target_flow_only"] = pairwise_comparison(
            runs_by_variant,
            "full",
            "target_flow_only",
        )

    payload = {
        "dataset": dataset_name,
        "partition_mode": args.partition_mode,
        "episodes": args.episodes,
        "seeds": seeds,
        "variants": variants,
        "dataset_diagnostics": dataset_diagnostics(bundle),
        "runs_by_variant": runs_by_variant,
        "aggregates": aggregates,
        "comparisons": comparisons,
    }
    report_path = output_dirs["reports"] / f"{dataset_name}_multiseed.json"
    write_json(report_path, payload)
    print(f"multiseed_report={report_path}")


if __name__ == "__main__":
    main()
