from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.runtime import enable_torch_import, ensure_preflight, write_run_manifest


DEFAULT_DATA_CONFIG = Path("configs/us_equities_smoke.yaml")
DEFAULT_TRAINING_CONFIG = Path("configs/training.yaml")
DEFAULT_BACKTEST_CONFIG = Path("configs/backtest.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mainline ablations over anchor and challenger mechanisms.")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Optional comma-separated ablation list. When empty, canonical defaults are used.",
    )
    parser.add_argument("--partition-mode", type=str, default="skill_hierarchy", choices=("skill_hierarchy",))
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="ablation_us_equities")
    return parser.parse_args()


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def default_variants_for_mode(partition_mode: str) -> list[str]:
    if partition_mode != "skill_hierarchy":
        raise ValueError(f"Unsupported partition_mode {partition_mode!r}.")
    return [
        "full",
        "quality_solvency_only",
        "no_mcts",
        "no_memory",
        "no_seed_priority",
        "no_validation_backed",
        "no_second_upgrade",
        "no_flow_gate",
        "short_horizon_flow_only",
        "efficiency_growth_only",
        "valuation_size_only",
    ]


def build_variant(
    variant: str,
    training_config: dict,
    dataset_name: str,
    partition_mode: str,
    seed_override: int | None = None,
):
    from knowledge_guided_symbolic_alpha.training import FormulaCurriculum

    from experiments.common import build_curriculum, build_manager

    base_curriculum = build_curriculum(training_config)
    if variant == "full":
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            seed_override=seed_override,
        ), base_curriculum
    if variant == "no_seed_priority":
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            seed_override=seed_override,
            seed_priority_enabled=False,
        ), base_curriculum
    if variant == "no_validation_backed":
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            seed_override=seed_override,
            allow_validation_backed_bootstrap=False,
            allow_validation_backed_replacement=False,
            allow_validation_backed_upgrade=False,
        ), base_curriculum
    if variant == "no_second_upgrade":
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            seed_override=seed_override,
            allow_validation_backed_upgrade=False,
        ), base_curriculum
    if variant == "no_flow_gate":
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            seed_override=seed_override,
            enforce_flow_residual_gate=False,
        ), base_curriculum
    if variant == "no_mcts":
        if base_curriculum is None:
            ablated_curriculum = None
        else:
            ablated_curriculum = FormulaCurriculum(
                tuple(replace(stage, use_mcts=False) for stage in base_curriculum.stages)
            )
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            seed_override=seed_override,
        ), ablated_curriculum
    if variant == "no_memory":
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            no_memory=True,
            seed_override=seed_override,
        ), base_curriculum

    fixed_role_by_variant = {
        "short_horizon_flow_only": ("short_horizon_flow",),
        "quality_solvency_only": ("quality_solvency",),
        "efficiency_growth_only": ("efficiency_growth",),
        "valuation_size_only": ("valuation_size",),
    }
    fixed_agent_names = fixed_role_by_variant.get(variant)
    if fixed_agent_names is None:
        raise ValueError(f"Unsupported ablation variant {variant!r} for partition_mode {partition_mode!r}.")

    return build_manager(
        training_config,
        dataset_name=dataset_name,
        partition_mode=partition_mode,
        seed_override=seed_override,
        included_agent_names=fixed_agent_names,
        fixed_agent_name=fixed_agent_names[0],
    ), base_curriculum


def main() -> None:
    args = parse_args()
    preflight = ensure_preflight("train")
    enable_torch_import()

    from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
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
        summary_counters,
        training_seed,
        write_json,
    )

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
    variants = parse_csv_arg(args.variants) or default_variants_for_mode(args.partition_mode)
    output_dirs = ensure_output_dirs(args.output_root, args.run_name)

    manifest_path = write_run_manifest(
        output_dirs,
        script_name="experiments/run_ablation.py",
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
        seed=training_seed(training_config),
        extra={"episodes": args.episodes, "partition_mode": args.partition_mode, "variants": variants},
    )

    backtester = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    )
    mode_results = []
    for variant in variants:
        manager, curriculum = build_variant(variant, training_config, dataset_name, args.partition_mode)
        trainer = MultiAgentTrainer(manager, curriculum=curriculum)
        summary = trainer.train(
            train_frame,
            train_target,
            episodes=args.episodes,
            validation_data=valid_frame,
            validation_target=valid_target,
        )
        generation_summary, selector_outcome = run_validation_selector(summary, valid_frame, valid_target)
        record = {
            "variant": variant,
            "best_validation_pool_score": summary.best_validation_pool_score,
            "final_pool_size": summary.final_pool_size,
            "accepted_episodes": int(sum(1 for episode in summary.history if episode.accepted)),
            "decision_reason_counts": summary_counters(summary.history, "decision_reason"),
            "selected_agent_counts": summary_counters(summary.history, "selected_agent"),
            "champion_records": list(summary.champion_records),
            "final_records": list(summary.final_records),
            "candidate_records": list(generation_summary.candidate_records),
            "selector_records": list(selector_outcome.selected_formulas),
            "selector_fallback_used": selector_outcome.fallback_used,
            "selector_ranked_records": selector_outcome.records,
            "records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
            "context_formula_survived": any(item.role == "context" for item in summary.final_record_summaries),
        }
        formulas, evaluation_formula_source = select_evaluation_formulas(
            selector_outcome.selected_formulas,
            summary.champion_records,
            summary.final_records,
        )
        record["evaluation_formula_source"] = evaluation_formula_source
        if formulas:
            report = backtester.run(
                formulas=formulas,
                frame=backtest_frame,
                feature_columns=bundle.feature_columns,
                target_column=target_column,
                return_column=return_column,
                config=build_walk_forward_config(backtest_config),
            )
            record["walk_forward_metrics"] = report.aggregate_metrics
        else:
            record["walk_forward_metrics"] = {}
        mode_results.append(record)
        print(
            f"partition_mode={args.partition_mode} "
            f"variant={record['variant']} "
            f"best_validation_pool_score={summary.best_validation_pool_score:.6f} "
            f"final_pool_size={summary.final_pool_size} "
            f"formula_count={len(formulas)}"
        )

    payload = {
        "dataset": dataset_name,
        "subset": subset_name,
        "partition_modes": [args.partition_mode],
        "episodes": args.episodes,
        "results_by_partition_mode": {args.partition_mode: mode_results},
        "dataset_diagnostics": dataset_diagnostics(bundle),
        "manifest": str(manifest_path),
    }
    report_path = output_dirs["reports"] / f"{dataset_name}_ablation.json"
    write_json(report_path, payload)
    print(f"manifest={manifest_path}")
    print(f"ablation_report={report_path}")


if __name__ == "__main__":
    main()
