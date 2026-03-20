from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.training import MultiAgentTrainer

from experiments.common import (
    DEFAULT_OUTPUT_ROOT,
    build_curriculum,
    build_manager,
    dataset_columns,
    dataset_diagnostics,
    dataset_input_frame,
    ensure_output_dirs,
    group_formula_summaries_by_role,
    load_dataset_bundle,
    load_yaml,
    resolve_us_equities_skill_aliases,
    summary_counters,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anchor-agent training on the U.S. equities cross-sectional subset.")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--data-config", type=Path, default=Path("configs/us_equities_smoke.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training_anchor_smoke.yaml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--skills",
        type=str,
        default="",
        help="Comma-separated skill whitelist. Empty means anchor-only (`quality_solvency`).",
    )
    return parser.parse_args()


def parse_skill_list(raw: str) -> tuple[str, ...] | None:
    skills = tuple(item.strip() for item in raw.split(",") if item.strip())
    return resolve_us_equities_skill_aliases(
        skills
        or (
            "quality_solvency",
        )
    )


def main() -> None:
    args = parse_args()
    training_config = load_yaml(args.training_config)
    bundle = load_dataset_bundle(args.data_config)
    dataset_name = "us_equities"
    target_column, _ = dataset_columns(dataset_name)
    train_frame = dataset_input_frame(bundle.splits.train, bundle.feature_columns, dataset_name)
    valid_frame = dataset_input_frame(bundle.splits.valid, bundle.feature_columns, dataset_name)
    train_target = bundle.splits.train[target_column]
    valid_target = bundle.splits.valid[target_column]

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

    output_dirs = ensure_output_dirs(args.output_root)
    report_path = output_dirs["reports"] / "anchor_train_summary.json"
    factors_path = output_dirs["factors"] / "anchor_champions.json"
    payload = {
        "dataset": dataset_name,
        "episodes": args.episodes,
        "skills": list(parse_skill_list(args.skills) or ()),
        "best_validation_pool_score": summary.best_validation_pool_score,
        "best_validation_selection_score": summary.best_validation_selection_score,
        "final_pool_size": summary.final_pool_size,
        "champion_records": list(summary.champion_records),
        "final_records": list(summary.final_records),
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
    }
    write_json(report_path, payload)
    write_json(
        factors_path,
        {
            "dataset": dataset_name,
            "champion_records": list(summary.champion_records),
            "final_records": list(summary.final_records),
            "champion_records_by_role": group_formula_summaries_by_role(summary.champion_record_summaries),
            "final_records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
        },
    )

    print(f"dataset={dataset_name}")
    print(f"episodes={args.episodes}")
    print(f"skills={list(parse_skill_list(args.skills) or ())}")
    print(f"best_validation_pool_score={summary.best_validation_pool_score:.6f}")
    print(f"best_validation_selection_score={summary.best_validation_selection_score:.6f}")
    print(f"final_pool_size={summary.final_pool_size}")
    print(f"champion_records={list(summary.champion_records)}")
    print(f"final_records={list(summary.final_records)}")
    print(f"train_report={report_path}")
    print(f"factors_file={factors_path}")


if __name__ == "__main__":
    main()
