from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.runtime import (
    enable_torch_import,
    ensure_preflight,
    write_run_manifest,
)


DEFAULT_DATA_CONFIG = Path("configs/us_equities_smoke.yaml")
DEFAULT_TRAINING_CONFIG = Path("configs/training.yaml")
DEFAULT_BACKTEST_CONFIG = Path("configs/backtest.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symbolic alpha training with manager-guided skill selection.")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--partition-mode", type=str, default="skill_hierarchy", choices=("skill_hierarchy",))
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="train_us_equities")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = ensure_preflight("train")
    enable_torch_import()

    from knowledge_guided_symbolic_alpha.training import MultiAgentTrainer

    from experiments.common import (
        build_curriculum,
        build_manager,
        dataset_columns,
        dataset_diagnostics,
        dataset_input_frame,
        ensure_output_dirs,
        group_formula_summaries_by_role,
        load_dataset_bundle,
        load_experiment_name,
        load_yaml,
        run_validation_selector,
        summary_counters,
        training_seed,
        write_json,
    )

    training_config = load_yaml(args.training_config)
    data_config = load_yaml(args.data_config)
    dataset_name = load_experiment_name(args.experiment_config)
    subset_name = Path(data_config["us_equities_subset"]["split_root"]).name
    bundle = load_dataset_bundle(args.data_config)
    target_column, _ = dataset_columns(dataset_name)
    train_frame = dataset_input_frame(bundle.splits.train, bundle.feature_columns, dataset_name)
    valid_frame = dataset_input_frame(bundle.splits.valid, bundle.feature_columns, dataset_name)
    train_target = bundle.splits.train[target_column]
    valid_target = bundle.splits.valid[target_column]

    output_dirs = ensure_output_dirs(args.output_root, args.run_name)
    manifest_path = write_run_manifest(
        output_dirs,
        script_name="experiments/run_train.py",
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
        extra={"episodes": args.episodes, "partition_mode": args.partition_mode},
    )

    trainer = MultiAgentTrainer(
        build_manager(training_config, dataset_name=dataset_name, partition_mode=args.partition_mode),
        curriculum=build_curriculum(training_config),
    )
    summary = trainer.train(
        train_frame,
        train_target,
        episodes=args.episodes,
        validation_data=valid_frame,
        validation_target=valid_target,
    )
    generation_summary, selector_outcome = run_validation_selector(summary, valid_frame, valid_target)

    report_path = output_dirs["reports"] / f"{dataset_name}_train_summary.json"
    factors_path = output_dirs["factors"] / f"{dataset_name}_champions.json"
    payload = {
        "dataset": dataset_name,
        "subset": subset_name,
        "partition_mode": args.partition_mode,
        "episodes": args.episodes,
        "best_validation_pool_score": summary.best_validation_pool_score,
        "best_validation_selection_score": summary.best_validation_selection_score,
        "final_pool_size": summary.final_pool_size,
        "champion_records": list(summary.champion_records),
        "final_records": list(summary.final_records),
        "candidate_records": list(generation_summary.candidate_records),
        "selector_records": list(selector_outcome.selected_formulas),
        "selector_fallback_used": selector_outcome.fallback_used,
        "selector_ranked_records": selector_outcome.records,
        "decision_reason_counts": summary_counters(summary.history, "decision_reason"),
        "selected_agent_counts": summary_counters(summary.history, "selected_agent"),
        "accepted_counts_by_role": {
            role: count
            for role, count in summary_counters(
                [episode for episode in summary.history if episode.accepted],
                "selected_agent",
            ).items()
        },
        "champion_records_by_role": group_formula_summaries_by_role(summary.champion_record_summaries),
        "final_records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
        "context_formula_survived": any(item.role == "context" for item in summary.final_record_summaries),
        "accepted_episodes": int(sum(1 for episode in summary.history if episode.accepted)),
        "dataset_diagnostics": dataset_diagnostics(bundle),
        "history_tail": summary.history[-10:],
        "manifest": str(manifest_path),
    }
    write_json(report_path, payload)
    write_json(
        factors_path,
        {
            "dataset": dataset_name,
            "subset": subset_name,
            "champion_records": list(summary.champion_records),
            "final_records": list(summary.final_records),
            "candidate_records": list(generation_summary.candidate_records),
            "selector_records": list(selector_outcome.selected_formulas),
            "champion_records_by_role": group_formula_summaries_by_role(summary.champion_record_summaries),
            "final_records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
            "manifest": str(manifest_path),
        },
    )

    print(f"dataset={dataset_name}")
    print(f"subset={subset_name}")
    print(f"partition_mode={args.partition_mode}")
    print(f"episodes={args.episodes}")
    print(f"best_validation_pool_score={summary.best_validation_pool_score:.6f}")
    print(f"best_validation_selection_score={summary.best_validation_selection_score:.6f}")
    print(f"final_pool_size={summary.final_pool_size}")
    print(f"selector_records={list(selector_outcome.selected_formulas)}")
    print(f"champion_records={list(summary.champion_records)}")
    print(f"final_records={list(summary.final_records)}")
    print(f"context_formula_survived={any(item.role == 'context' for item in summary.final_record_summaries)}")
    print(f"manifest={manifest_path}")
    print(f"train_report={report_path}")
    print(f"factors_file={factors_path}")


if __name__ == "__main__":
    main()
