from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.generation import FormulaCandidate
from knowledge_guided_symbolic_alpha.runtime import enable_torch_import, ensure_preflight, write_run_manifest

from experiments.run_ablation import build_variant


DEFAULT_DATA_CONFIG = Path("configs/us_equities_smoke.yaml")
DEFAULT_TRAINING_CONFIG = Path("configs/training.yaml")
DEFAULT_BACKTEST_CONFIG = Path("configs/backtest.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")

HIGHER_IS_BETTER_METRICS = {"best_validation_pool_score", "sharpe", "annual_return", "mean_test_rank_ic", "max_drawdown"}
LOWER_IS_BETTER_METRICS = {"turnover"}
CONSENSUS_CHAMPION_SUPPORT_WEIGHT = 0.03
CONSENSUS_SELECTOR_SUPPORT_WEIGHT = 0.01
CONSENSUS_CANDIDATE_SUPPORT_WEIGHT = 0.005
CONSENSUS_SELECTOR_RANK_PENALTY = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed training and summarize robustness.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="7,17,27,37,47", help="Comma-separated integer seeds.")
    parser.add_argument(
        "--variants",
        type=str,
        default="full,quality_solvency_only,short_horizon_flow_only",
        help="Comma-separated canonical variants supported by run_ablation.py.",
    )
    parser.add_argument("--partition-mode", type=str, default="skill_hierarchy", choices=("skill_hierarchy",))
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="multiseed_us_equities")
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
    backtester,
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


def consensus_metric_diagnostics(aggregated_variant: dict[str, Any]) -> dict[str, Any]:
    walk_forward_metrics = aggregated_variant.get("walk_forward_metrics", {})
    diagnostics: dict[str, Any] = {}
    for metric in ("sharpe", "annual_return", "mean_test_rank_ic", "turnover", "max_drawdown"):
        payload = walk_forward_metrics.get(metric)
        if isinstance(payload, dict):
            diagnostics[metric] = {
                "mean": payload.get("mean"),
                "std": payload.get("std"),
                "min": payload.get("min"),
                "max": payload.get("max"),
            }
    return diagnostics


def build_canonical_variant_result(
    variant: str,
    aggregated_variant: dict[str, Any],
    consensus_variant: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "result_kind": "cross_seed_consensus",
        "selector_records": list(consensus_variant.get("selector_records", [])),
        "evaluation_formula_source": consensus_variant.get("evaluation_formula_source", "none"),
        "walk_forward_metrics": dict(consensus_variant.get("walk_forward_metrics", {})),
        "seed_support": {
            "seed_count": int(consensus_variant.get("seed_count", 0)),
            "min_seed_support": int(consensus_variant.get("min_seed_support", 0)),
            "candidate_pool_size": int(consensus_variant.get("candidate_pool_size", 0)),
            "fallback_used": bool(consensus_variant.get("fallback_used", False)),
            "selector_fallback_used": bool(consensus_variant.get("selector_fallback_used", False)),
        },
        "raw_seed_diagnostics": consensus_metric_diagnostics(aggregated_variant),
        "support_adjusted_ranked_records": list(consensus_variant.get("support_adjusted_ranked_records", [])),
    }


def consensus_seed_support_threshold(seed_count: int) -> int:
    if seed_count <= 1:
        return 1
    return max(2, int(math.ceil(seed_count / 2.0)))


def summarize_formula_support(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidate_support: dict[str, set[int]] = defaultdict(set)
    selector_support: dict[str, set[int]] = defaultdict(set)
    champion_support: dict[str, set[int]] = defaultdict(set)
    mean_ranks: dict[str, list[float]] = defaultdict(list)
    robust_scores: dict[str, list[float]] = defaultdict(list)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for run in runs:
        seed = int(run["seed"])
        for formula in {str(item) for item in run.get("candidate_records", []) if str(item)}:
            candidate_support[formula].add(seed)
        for formula in {str(item) for item in run.get("selector_records", []) if str(item)}:
            selector_support[formula].add(seed)
        for formula in {str(item) for item in run.get("champion_records", []) if str(item)}:
            champion_support[formula].add(seed)
        for rank, record in enumerate(run.get("selector_ranked_records", []), start=1):
            formula = ""
            if isinstance(record, dict):
                formula = str(record.get("formula") or "")
                robust_score = record.get("robust_score")
                source = record.get("source")
                role = record.get("role")
            else:
                formula = str(getattr(record, "formula", "") or "")
                robust_score = getattr(record, "robust_score", None)
                source = getattr(record, "source", None)
                role = getattr(record, "role", None)
            if not formula:
                continue
            mean_ranks[formula].append(float(rank))
            if isinstance(robust_score, (int, float)):
                robust_scores[formula].append(float(robust_score))
            if source:
                source_counts[formula][str(source)] += 1
            if role:
                role_counts[formula][str(role)] += 1

    formulas = sorted(set(candidate_support) | set(selector_support) | set(champion_support))
    payload: dict[str, dict[str, Any]] = {}
    for formula in formulas:
        sources = source_counts.get(formula, Counter())
        roles = role_counts.get(formula, Counter())
        payload[formula] = {
            "candidate_seed_support": len(candidate_support.get(formula, set())),
            "selector_seed_support": len(selector_support.get(formula, set())),
            "champion_seed_support": len(champion_support.get(formula, set())),
            "mean_selector_rank": float(np.mean(mean_ranks[formula])) if mean_ranks.get(formula) else None,
            "mean_selector_robust_score": float(np.mean(robust_scores[formula])) if robust_scores.get(formula) else None,
            "primary_source": sources.most_common(1)[0][0] if sources else "multiseed_consensus",
            "primary_role": roles.most_common(1)[0][0] if roles else None,
        }
    return payload


def build_consensus_candidate_pool(
    runs: list[dict[str, Any]],
    min_seed_support: int | None = None,
) -> tuple[list[FormulaCandidate], dict[str, Any]]:
    support = summarize_formula_support(runs)
    seed_count = len({int(run["seed"]) for run in runs})
    threshold = min_seed_support or consensus_seed_support_threshold(seed_count)

    ranked_formulas = sorted(
        support,
        key=lambda formula: (
            support[formula]["candidate_seed_support"],
            support[formula]["selector_seed_support"],
            support[formula]["champion_seed_support"],
            -(support[formula]["mean_selector_rank"] or float("inf")),
            support[formula]["mean_selector_robust_score"] or float("-inf"),
        ),
        reverse=True,
    )
    selected_formulas = [
        formula
        for formula in ranked_formulas
        if int(support[formula]["candidate_seed_support"]) >= threshold
    ]
    fallback_used = False
    if not selected_formulas and ranked_formulas:
        fallback_used = True
        top_formula = ranked_formulas[0]
        selected_formulas = [top_formula]

    candidates = [
        FormulaCandidate(
            formula=formula,
            source=str(support[formula]["primary_source"]),
            role=support[formula]["primary_role"],
        )
        for formula in selected_formulas
    ]
    return candidates, {
        "seed_count": seed_count,
        "min_seed_support": threshold,
        "candidate_pool_size": len(candidates),
        "fallback_used": fallback_used,
        "formula_support": support,
    }


def support_adjusted_consensus_selection(
    selector_outcome,
    support_payload: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    seed_count = max(1, int(support_payload.get("seed_count", 0)))
    formula_support = support_payload.get("formula_support", {})
    ranked_records: list[dict[str, Any]] = []
    for record in selector_outcome.records:
        support = formula_support.get(record.formula, {})
        candidate_frac = float(support.get("candidate_seed_support", 0)) / seed_count
        selector_frac = float(support.get("selector_seed_support", 0)) / seed_count
        champion_frac = float(support.get("champion_seed_support", 0)) / seed_count
        mean_rank = support.get("mean_selector_rank")
        rank_penalty = 0.0 if mean_rank is None else CONSENSUS_SELECTOR_RANK_PENALTY * max(float(mean_rank) - 1.0, 0.0)
        consensus_score = (
            float(record.robust_score)
            + CONSENSUS_CHAMPION_SUPPORT_WEIGHT * champion_frac
            + CONSENSUS_SELECTOR_SUPPORT_WEIGHT * selector_frac
            + CONSENSUS_CANDIDATE_SUPPORT_WEIGHT * candidate_frac
            - rank_penalty
        )
        ranked_records.append(
            {
                "formula": record.formula,
                "support_adjusted_score": consensus_score,
                "candidate_seed_support": int(support.get("candidate_seed_support", 0)),
                "selector_seed_support": int(support.get("selector_seed_support", 0)),
                "champion_seed_support": int(support.get("champion_seed_support", 0)),
                "mean_selector_rank": mean_rank,
                "robust_score": float(record.robust_score),
            }
        )
    ranked_records.sort(
        key=lambda item: (
            item["support_adjusted_score"],
            item["champion_seed_support"],
            item["selector_seed_support"],
            item["candidate_seed_support"],
            -(item["mean_selector_rank"] or float("inf")),
        ),
        reverse=True,
    )
    selected_formulas = [ranked_records[0]["formula"]] if ranked_records else []
    return selected_formulas, ranked_records


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
                delta_payload[metric] = float(left_metrics[metric]) - float(right_metrics[metric])
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


def canonical_pairwise_comparison(
    canonical_by_variant: dict[str, dict[str, Any]],
    left_variant: str,
    right_variant: str,
) -> dict[str, Any]:
    left = canonical_by_variant.get(left_variant, {})
    right = canonical_by_variant.get(right_variant, {})
    left_metrics = left.get("walk_forward_metrics", {})
    right_metrics = right.get("walk_forward_metrics", {})
    deltas: dict[str, float] = {}
    for metric in ("sharpe", "annual_return", "mean_test_rank_ic", "max_drawdown", "turnover"):
        if metric in left_metrics and metric in right_metrics:
            deltas[metric] = float(left_metrics[metric]) - float(right_metrics[metric])
    return {
        "left_variant": left_variant,
        "right_variant": right_variant,
        "result_kind": "cross_seed_consensus",
        "left_selector_records": list(left.get("selector_records", [])),
        "right_selector_records": list(right.get("selector_records", [])),
        "metric_deltas": deltas,
    }


def main() -> None:
    args = parse_args()
    preflight = ensure_preflight("train")
    enable_torch_import()

    from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
    from knowledge_guided_symbolic_alpha.selection import CrossSeedConsensusSelector, RobustTemporalSelector
    from knowledge_guided_symbolic_alpha.training import MultiAgentTrainer

    from experiments.common import (
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
        run_validation_selector,
        select_evaluation_formulas,
        write_json,
    )

    seeds = [int(item) for item in parse_csv_arg(args.seeds)]
    variants = parse_csv_arg(args.variants)
    if not seeds:
        raise ValueError("At least one seed is required.")
    if not variants:
        raise ValueError("At least one variant is required.")

    training_config = load_yaml(args.training_config)
    backtest_config = load_yaml(args.backtest_config)
    data_config = load_yaml(args.data_config)
    dataset_name = load_experiment_name(args.experiment_config)
    subset_name = Path(data_config["us_equities_subset"]["split_root"]).name
    bundle = load_dataset_bundle(args.data_config)
    target_column, return_column = dataset_columns(dataset_name)
    train_frame = dataset_input_frame(bundle.splits.train, bundle.feature_columns, dataset_name)
    valid_frame = dataset_input_frame(bundle.splits.valid, bundle.feature_columns, dataset_name)
    train_target = bundle.splits.train[target_column]
    valid_target = bundle.splits.valid[target_column]
    backtest_frame = pd.concat([bundle.splits.valid, bundle.splits.test], axis=0)

    output_dirs = ensure_output_dirs(args.output_root, args.run_name)
    manifest_path = write_run_manifest(
        output_dirs,
        script_name="experiments/run_multiseed.py",
        profile="train",
        preflight=preflight.to_dict(),
        config_paths={
            "data_config": str(args.data_config),
            "training_config": str(args.training_config),
            "backtest_config": str(args.backtest_config),
            "experiment_config": str(args.experiment_config),
        },
        dataset_name=dataset_name,
        subset=subset_name,
        extra={"episodes": args.episodes, "seeds": seeds, "variants": variants, "partition_mode": args.partition_mode},
    )
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
            generation_summary, selector_outcome = run_validation_selector(summary, valid_frame, valid_target)
            formulas, evaluation_formula_source = select_evaluation_formulas(
                selector_outcome.selected_formulas,
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
                "candidate_records": list(generation_summary.candidate_records),
                "selector_records": list(selector_outcome.selected_formulas),
                "selector_fallback_used": selector_outcome.fallback_used,
                "selector_ranked_records": selector_outcome.records,
                "records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
                "context_formula_survived": any(item.role == "context" for item in summary.final_record_summaries),
                "walk_forward_metrics": evaluation["aggregate_metrics"],
            }
            runs_by_variant[variant].append(run_record)
            print(
                f"seed={seed} variant={variant} "
                f"best_validation_pool_score={summary.best_validation_pool_score:.6f} "
                f"formula_count={evaluation['formula_count']}"
            )

    aggregated = {variant: aggregate_variant_runs(runs) for variant, runs in runs_by_variant.items()}
    consensus_by_variant: dict[str, dict[str, Any]] = {}
    multiseed_selector = CrossSeedConsensusSelector(temporal_selector=RobustTemporalSelector())
    for variant, runs in runs_by_variant.items():
        consensus_candidates, consensus_support = build_consensus_candidate_pool(runs)
        if consensus_candidates:
            consensus_outcome = multiseed_selector.select(runs, valid_frame, valid_target, base_candidates=consensus_candidates)
            formulas, evaluation_formula_source = select_evaluation_formulas(
                consensus_outcome.selected_formulas,
                tuple(),
                tuple(),
            )
            consensus_evaluation = evaluate_formulas(
                backtester,
                formulas,
                backtest_frame,
                bundle.feature_columns,
                target_column,
                return_column,
                walk_forward_config,
            )
        else:
            consensus_outcome = None
            evaluation_formula_source = "none"
            consensus_evaluation = {"formula_count": 0, "formulas": [], "aggregate_metrics": {}}
        consensus_by_variant[variant] = {
            **(
                {
                    "seed_count": consensus_outcome.seed_count,
                    "min_seed_support": consensus_outcome.min_seed_support,
                    "candidate_pool_size": len(consensus_outcome.candidate_pool),
                    "fallback_used": consensus_outcome.fallback_used,
                    "formula_support": consensus_outcome.formula_support,
                }
                if consensus_outcome is not None
                else consensus_support
            ),
            "selector_records": formulas if consensus_candidates else [],
            "selector_fallback_used": bool(consensus_outcome.selector_fallback_used) if consensus_outcome is not None else False,
            "selector_ranked_records": list(consensus_outcome.temporal_records) if consensus_outcome is not None else [],
            "support_adjusted_ranked_records": [
                {
                    "formula": record.formula,
                    "support_adjusted_score": record.support_adjusted_score,
                    "candidate_seed_support": record.candidate_seed_support,
                    "selector_seed_support": record.selector_seed_support,
                    "champion_seed_support": record.champion_seed_support,
                    "mean_selector_rank": record.mean_selector_rank,
                    "mean_temporal_score": record.mean_temporal_score,
                }
                for record in (consensus_outcome.ranked_records if consensus_outcome is not None else [])
            ],
            "evaluation_formula_source": evaluation_formula_source,
            "walk_forward_metrics": consensus_evaluation["aggregate_metrics"],
        }
    comparisons = []
    if variants:
        anchor_variant = variants[0]
        for challenger in variants[1:]:
            comparisons.append(pairwise_comparison(runs_by_variant, anchor_variant, challenger))

    canonical_by_variant = {
        variant: build_canonical_variant_result(variant, aggregated.get(variant, {}), consensus_by_variant.get(variant, {}))
        for variant in variants
    }
    canonical_comparisons = []
    if variants:
        anchor_variant = variants[0]
        for challenger in variants[1:]:
            canonical_comparisons.append(canonical_pairwise_comparison(canonical_by_variant, anchor_variant, challenger))

    payload = {
        "dataset": dataset_name,
        "subset": subset_name,
        "partition_mode": args.partition_mode,
        "episodes": args.episodes,
        "seeds": seeds,
        "variants": variants,
        "runs_by_variant": runs_by_variant,
        "aggregated_by_variant": aggregated,
        "consensus_by_variant": consensus_by_variant,
        "canonical_result_kind": "cross_seed_consensus",
        "canonical_result_scope": "canonical_by_variant",
        "canonical_by_variant": canonical_by_variant,
        "comparisons": comparisons,
        "canonical_comparisons": canonical_comparisons,
        "dataset_diagnostics": dataset_diagnostics(bundle),
        "manifest": str(manifest_path),
    }
    report_path = output_dirs["reports"] / f"{dataset_name}_multiseed.json"
    write_json(report_path, payload)
    canonical_report_path = output_dirs["reports"] / f"{dataset_name}_multiseed_canonical.json"
    write_json(
        canonical_report_path,
        {
            "dataset": dataset_name,
            "subset": subset_name,
            "partition_mode": args.partition_mode,
            "episodes": args.episodes,
            "seeds": seeds,
            "variants": variants,
            "canonical_result_kind": "cross_seed_consensus",
            "canonical_by_variant": canonical_by_variant,
            "canonical_comparisons": canonical_comparisons,
            "full_report": str(report_path),
            "manifest": str(manifest_path),
        },
    )
    print(f"manifest={manifest_path}")
    print(f"multiseed_report={report_path}")
    print(f"multiseed_canonical_report={canonical_report_path}")


if __name__ == "__main__":
    main()
