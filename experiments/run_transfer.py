from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.agents import ManagerAgent
from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
from knowledge_guided_symbolic_alpha.training import MultiAgentTrainer

from experiments.common import (
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TRAINING_CONFIG,
    build_curriculum,
    build_manager,
    build_portfolio_config,
    build_signal_fusion_config,
    build_walk_forward_config,
    dataset_columns,
    dataset_diagnostics,
    ensure_output_dirs,
    group_formula_summaries_by_role,
    load_dataset_bundle,
    load_experiment_name,
    load_yaml,
    select_evaluation_formulas,
    write_json,
)


class FixedRoleManager(ManagerAgent):
    def __init__(self, fixed_agent: str, **kwargs) -> None:
        self.fixed_agent = fixed_agent
        super().__init__(**kwargs)

    def select_agent(self, frame: pd.DataFrame, pool):
        del pool
        regime_context = self.regime_controller.infer(frame)
        first_name, second_name = self.agent_names
        if self.fixed_agent == first_name:
            return first_name, regime_context.regime, 1.0, 0.0
        return second_name, regime_context.regime, 0.0, 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on source market and evaluate transfer markets.")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--targets", type=str, default="gold,crude_oil,sp500")
    parser.add_argument(
        "--baseline-modes",
        type=str,
        default="from_scratch_full,from_scratch_target_price_only",
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


def fixed_agent_for_baseline(baseline_mode: str, partition_mode: str) -> str | None:
    if baseline_mode == "from_scratch_full":
        return None
    if baseline_mode != "from_scratch_target_price_only":
        raise ValueError(f"Unsupported baseline_mode {baseline_mode!r}.")
    mapping = {
        "skill_hierarchy": "price_structure",
        "competitive_three_way": "target_price",
        "target_context": "target",
        "macro_micro": "micro",
    }
    return mapping[partition_mode]


def train_summary(bundle, training_config: dict, dataset_name: str, episodes: int, partition_mode: str, baseline_mode: str):
    target_column, _ = dataset_columns(dataset_name)
    fixed_agent = fixed_agent_for_baseline(baseline_mode, partition_mode)
    manager = build_manager(
        training_config,
        dataset_name=dataset_name,
        partition_mode=partition_mode,
        fixed_agent_name=fixed_agent if partition_mode in {"competitive_three_way", "skill_hierarchy"} else None,
        included_agent_names=(fixed_agent,) if fixed_agent is not None and partition_mode in {"competitive_three_way", "skill_hierarchy"} else None,
    )
    if fixed_agent is not None and partition_mode not in {"competitive_three_way", "skill_hierarchy"}:
        if not isinstance(manager, ManagerAgent):
            raise TypeError("Expected ManagerAgent for non-competitive fixed-role transfer baseline.")
        manager = FixedRoleManager(
            fixed_agent=fixed_agent,
            macro_agent=manager.macro_agent,
            micro_agent=manager.micro_agent,
            reviewer_agent=manager.reviewer_agent,
            gating_net=manager.gating_net,
            regime_controller=manager.regime_controller,
            state_encoder=manager.state_encoder,
            reward_shaper=manager.reward_shaper,
            experience_memory=manager.experience_memory,
            selection_mode="greedy",
            seed=int(training_config.get("training", {}).get("seed", 7)),
            agent_names=manager.agent_names,
        )
    trainer = MultiAgentTrainer(
        manager,
        curriculum=build_curriculum(training_config),
    )
    return trainer.train(
        bundle.splits.train[list(bundle.feature_columns)],
        bundle.splits.train[target_column],
        episodes=episodes,
        validation_data=bundle.splits.valid[list(bundle.feature_columns)],
        validation_target=bundle.splits.valid[target_column],
    )


def evaluate_protocol(
    backtester: WalkForwardBacktester,
    bundle,
    dataset_name: str,
    formulas: list[str],
    walk_forward_config,
) -> dict:
    target_column, return_column = dataset_columns(dataset_name)
    metrics = {}
    if formulas:
        report = backtester.run(
            formulas=formulas,
            frame=pd.concat([bundle.splits.valid, bundle.splits.test], axis=0),
            feature_columns=bundle.feature_columns,
            target_column=target_column,
            return_column=return_column,
            config=walk_forward_config,
        )
        metrics = report.aggregate_metrics
    return {
        "formula_count": len(formulas),
        "formulas": formulas,
        "aggregate_metrics": metrics,
    }


def metric_deltas(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, value in candidate.items():
        if key in reference and isinstance(value, (int, float)) and isinstance(reference[key], (int, float)):
            deltas[key] = float(value) - float(reference[key])
    return deltas


def main() -> None:
    args = parse_args()
    training_config = load_yaml(args.training_config)
    backtest_config = load_yaml(args.backtest_config)
    source_dataset = load_experiment_name(args.experiment_config)
    target_datasets = parse_csv_arg(args.targets)
    baseline_modes = parse_csv_arg(args.baseline_modes)
    bundle = load_dataset_bundle(args.data_config)

    source_summary = train_summary(
        bundle,
        training_config,
        source_dataset,
        args.episodes,
        args.partition_mode,
        baseline_mode="from_scratch_full",
    )
    zero_shot_formulas, zero_shot_formula_source = select_evaluation_formulas(
        source_summary.champion_records,
        source_summary.final_records,
    )

    output_dirs = ensure_output_dirs(args.output_root)
    backtester = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    )
    walk_forward_config = build_walk_forward_config(backtest_config)

    transfer_results = []
    baseline_cache: dict[tuple[str, str], dict] = {}
    for dataset_name in target_datasets:
        zero_shot = evaluate_protocol(backtester, bundle, dataset_name, zero_shot_formulas, walk_forward_config)
        baselines: dict[str, dict] = {}
        for baseline_mode in baseline_modes:
            cache_key = (dataset_name, baseline_mode)
            if cache_key not in baseline_cache:
                summary = train_summary(
                    bundle,
                    training_config,
                    dataset_name,
                    args.episodes,
                    args.partition_mode,
                    baseline_mode=baseline_mode,
                )
                formulas, evaluation_formula_source = select_evaluation_formulas(
                    summary.champion_records,
                    summary.final_records,
                )
                baseline_cache[cache_key] = {
                    "best_validation_pool_score": summary.best_validation_pool_score,
                    "final_pool_size": summary.final_pool_size,
                    "champion_records": list(summary.champion_records),
                    "final_records": list(summary.final_records),
                    "evaluation_formula_source": evaluation_formula_source,
                    "records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
                    "evaluation": evaluate_protocol(backtester, bundle, dataset_name, formulas, walk_forward_config),
                }
            baselines[baseline_mode] = baseline_cache[cache_key]
        transfer_results.append(
            {
                "dataset": dataset_name,
                "zero_shot": {
                    "best_validation_pool_score": source_summary.best_validation_pool_score,
                    "evaluation_formula_source": zero_shot_formula_source,
                    "records_by_role": group_formula_summaries_by_role(source_summary.final_record_summaries),
                    **zero_shot,
                },
                "baselines": baselines,
                "metric_deltas_vs_zero_shot": {
                    baseline_mode: metric_deltas(
                        zero_shot["aggregate_metrics"],
                        baselines[baseline_mode]["evaluation"]["aggregate_metrics"],
                    )
                    for baseline_mode in baselines
                },
            }
        )
        print(f"dataset={dataset_name} zero_shot_metrics={zero_shot['aggregate_metrics']}")
        for baseline_mode, payload in baselines.items():
            print(
                f"dataset={dataset_name} baseline={baseline_mode} "
                f"metrics={payload['evaluation']['aggregate_metrics']}"
            )

    payload = {
        "source_dataset": source_dataset,
        "partition_mode": args.partition_mode,
        "episodes": args.episodes,
        "baseline_modes": baseline_modes,
        "source_training": {
            "best_validation_pool_score": source_summary.best_validation_pool_score,
            "final_pool_size": source_summary.final_pool_size,
            "champion_records": list(source_summary.champion_records),
            "final_records": list(source_summary.final_records),
            "records_by_role": group_formula_summaries_by_role(source_summary.final_record_summaries),
        },
        "dataset_diagnostics": dataset_diagnostics(bundle),
        "transfer_results": transfer_results,
    }
    report_path = output_dirs["reports"] / f"{source_dataset}_transfer.json"
    write_json(report_path, payload)
    print(f"transfer_report={report_path}")


if __name__ == "__main__":
    main()
