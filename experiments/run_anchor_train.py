from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.runtime import enable_torch_import, ensure_preflight, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anchor-agent training on the U.S. equities cross-sectional subset.")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--data-config", type=Path, default=Path("configs/us_equities_smoke.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training_anchor_smoke.yaml"))
    parser.add_argument("--backtest-config", type=Path, default=Path("configs/backtest.yaml"))
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/us_equities_anchor.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-name", type=str, default="anchor_train_us_equities")
    parser.add_argument(
        "--skills",
        type=str,
        default="",
        help="Comma-separated skill whitelist. Empty means anchor-only (`quality_solvency`).",
    )
    return parser.parse_args()


def parse_skill_list(raw: str) -> tuple[str, ...] | None:
    skills = tuple(item.strip() for item in raw.split(",") if item.strip())
    return skills or ("quality_solvency",)


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
        load_yaml,
        run_validation_selector,
        summary_counters,
        training_seed,
        write_json,
    )

    training_config = load_yaml(args.training_config)
    data_config = load_yaml(args.data_config)
    bundle = load_dataset_bundle(args.data_config)
    dataset_name = "us_equities"
    subset_name = Path(data_config["us_equities_subset"]["split_root"]).name
    target_column, _ = dataset_columns(dataset_name)
    train_frame = dataset_input_frame(bundle.splits.train, bundle.feature_columns, dataset_name)
    valid_frame = dataset_input_frame(bundle.splits.valid, bundle.feature_columns, dataset_name)
    train_target = bundle.splits.train[target_column]
    valid_target = bundle.splits.valid[target_column]

    output_dirs = ensure_output_dirs(args.output_root, args.run_name)
    manifest_path = write_run_manifest(
        output_dirs,
        script_name="experiments/run_anchor_train.py",
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
        extra={"episodes": args.episodes, "skills": list(parse_skill_list(args.skills) or ())},
    )

    trainer = MultiAgentTrainer(
        build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode="skill_hierarchy",
            included_agent_names=parse_skill_list(args.skills),
        ),
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

    report_path = output_dirs["reports"] / "anchor_train_summary.json"
    factors_path = output_dirs["factors"] / "anchor_champions.json"
    payload = {
        "dataset": dataset_name,
        "subset": subset_name,
        "episodes": args.episodes,
        "skills": list(parse_skill_list(args.skills) or ()),
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
    print(f"episodes={args.episodes}")
    print(f"skills={list(parse_skill_list(args.skills) or ())}")
    print(f"best_validation_pool_score={summary.best_validation_pool_score:.6f}")
    print(f"best_validation_selection_score={summary.best_validation_selection_score:.6f}")
    print(f"final_pool_size={summary.final_pool_size}")
    print(f"selector_records={list(selector_outcome.selected_formulas)}")
    print(f"champion_records={list(summary.champion_records)}")
    print(f"final_records={list(summary.final_records)}")
    print(f"manifest={manifest_path}")
    print(f"train_report={report_path}")
    print(f"factors_file={factors_path}")


if __name__ == "__main__":
    main()
