from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.agents import CompetitiveManagerAgent, HierarchicalManagerAgent, ManagerAgent
from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
from knowledge_guided_symbolic_alpha.training import FormulaCurriculum, MultiAgentTrainer

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
    dataset_input_frame,
    ensure_output_dirs,
    group_formula_summaries_by_role,
    load_dataset_bundle,
    load_experiment_name,
    load_yaml,
    select_evaluation_formulas,
    summary_counters,
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
    parser = argparse.ArgumentParser(description="Run Route B ablations over anchor and challenger mechanisms.")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Optional comma-separated ablation list. When empty, mode-specific defaults are used.",
    )
    parser.add_argument(
        "--partition-modes",
        type=str,
        default="skill_hierarchy,competitive_three_way,target_context",
        help="Comma-separated partition modes to evaluate.",
    )
    parser.add_argument(
        "--partition-mode",
        type=str,
        default="",
        choices=("", "skill_hierarchy", "competitive_three_way", "target_context", "macro_micro"),
        help="Backward-compatible single partition mode alias.",
    )
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def default_variants_for_mode(partition_mode: str) -> list[str]:
    if partition_mode == "skill_hierarchy":
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
            "context_only",
        ]
    if partition_mode == "competitive_three_way":
        return ["full", "no_mcts", "no_memory", "context_only", "target_price_only", "target_flow_only"]
    if partition_mode == "target_context":
        return ["full", "no_mcts", "no_memory", "context_only", "target_only"]
    if partition_mode == "macro_micro":
        return ["full", "no_mcts", "no_memory", "macro_only", "micro_only"]
    raise ValueError(f"Unsupported partition_mode {partition_mode!r}.")


def normalize_variant(variant: str, partition_mode: str) -> str:
    alias_map = {
        "skill_hierarchy": {
            "target_price_only": "valuation_size_only",
            "target_flow_only": "short_horizon_flow_only",
            "reversal_gap_only": "short_horizon_flow_only",
            "volatility_liquidity_only": "short_horizon_flow_only",
            "intraday_imbalance_only": "valuation_size_only",
            "price_structure_only": "valuation_size_only",
            "context_only": "context_only",
        },
        "target_context": {
            "target_price_only": "target_only",
            "context_only": "context_only",
        },
        "macro_micro": {
            "target_price_only": "micro_only",
            "context_only": "macro_only",
        },
    }
    return alias_map.get(partition_mode, {}).get(variant, variant)


def build_variant(
    variant: str,
    training_config: dict,
    dataset_name: str,
    partition_mode: str,
    seed_override: int | None = None,
) -> tuple[ManagerAgent | CompetitiveManagerAgent | HierarchicalManagerAgent, FormulaCurriculum | None]:
    training = training_config.get("training", {})
    seed = int(seed_override if seed_override is not None else training.get("seed", 7))
    base_curriculum = build_curriculum(training_config)
    variant = normalize_variant(variant, partition_mode)
    if variant == "full":
        return (
            build_manager(
                training_config,
                dataset_name=dataset_name,
                partition_mode=partition_mode,
                seed_override=seed_override,
            ),
            base_curriculum,
        )
    if variant == "no_seed_priority":
        return (
            build_manager(
                training_config,
                dataset_name=dataset_name,
                partition_mode=partition_mode,
                seed_override=seed_override,
                seed_priority_enabled=False,
            ),
            base_curriculum,
        )
    if variant == "no_validation_backed":
        return (
            build_manager(
                training_config,
                dataset_name=dataset_name,
                partition_mode=partition_mode,
                seed_override=seed_override,
                allow_validation_backed_bootstrap=False,
                allow_validation_backed_replacement=False,
                allow_validation_backed_upgrade=False,
            ),
            base_curriculum,
        )
    if variant == "no_second_upgrade":
        return (
            build_manager(
                training_config,
                dataset_name=dataset_name,
                partition_mode=partition_mode,
                seed_override=seed_override,
                allow_validation_backed_upgrade=False,
            ),
            base_curriculum,
        )
    if variant == "no_flow_gate":
        return (
            build_manager(
                training_config,
                dataset_name=dataset_name,
                partition_mode=partition_mode,
                seed_override=seed_override,
                enforce_flow_residual_gate=False,
            ),
            base_curriculum,
        )
    if variant == "no_mcts":
        if base_curriculum is None:
            no_mcts_curriculum = None
        else:
            no_mcts_curriculum = FormulaCurriculum(tuple(replace(stage, use_mcts=False) for stage in base_curriculum.stages))
        return (
            build_manager(
                training_config,
                dataset_name=dataset_name,
                partition_mode=partition_mode,
                seed_override=seed_override,
            ),
            no_mcts_curriculum,
        )
    if variant == "no_memory":
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            no_memory=True,
            seed_override=seed_override,
        ), base_curriculum

    fixed_role_by_variant = {
        "skill_hierarchy": {
            "short_horizon_flow_only": ("short_horizon_flow",),
            "quality_solvency_only": ("quality_solvency",),
            "efficiency_growth_only": ("efficiency_growth",),
            "valuation_size_only": ("valuation_size",),
            "price_structure_only": ("valuation_size",),
            "trend_structure_only": ("quality_solvency", "efficiency_growth", "valuation_size"),
            "context_only": ("cross_asset_context", "regime_filter"),
        },
        "competitive_three_way": {
            "context_only": ("context",),
            "target_price_only": ("target_price",),
            "target_flow_only": ("target_flow_vol", "target_flow_gap"),
        },
        "target_context": {
            "context_only": ("context",),
            "target_only": ("target",),
        },
        "macro_micro": {
            "macro_only": ("macro",),
            "micro_only": ("micro",),
        },
    }
    fixed_agent_names = fixed_role_by_variant.get(partition_mode, {}).get(variant)
    if fixed_agent_names is None:
        raise ValueError(f"Unsupported ablation variant {variant!r} for partition_mode {partition_mode!r}.")

    if partition_mode in {"skill_hierarchy", "competitive_three_way"}:
        return build_manager(
            training_config,
            dataset_name=dataset_name,
            partition_mode=partition_mode,
            seed_override=seed_override,
            included_agent_names=fixed_agent_names,
            fixed_agent_name=fixed_agent_names[0] if len(fixed_agent_names) == 1 else None,
        ), base_curriculum

    fixed_agent = fixed_agent_names[0]
    manager = build_manager(
        training_config,
        dataset_name=dataset_name,
        partition_mode=partition_mode,
        seed_override=seed_override,
    )
    if not isinstance(manager, ManagerAgent):
        raise TypeError("Expected a two-agent ManagerAgent for non-competitive fixed-role ablations.")
    return FixedRoleManager(
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
        seed=seed,
        agent_names=manager.agent_names,
    ), base_curriculum


def main() -> None:
    args = parse_args()
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
    partition_modes = [args.partition_mode] if args.partition_mode else parse_csv_arg(args.partition_modes)
    explicit_variants = parse_csv_arg(args.variants)
    output_dirs = ensure_output_dirs(args.output_root)

    backtester = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    )
    results_by_partition_mode: dict[str, list[dict]] = {}
    for partition_mode in partition_modes:
        variants = explicit_variants or default_variants_for_mode(partition_mode)
        mode_results = []
        for variant in variants:
            manager, curriculum = build_variant(variant, training_config, dataset_name, partition_mode)
            trainer = MultiAgentTrainer(manager, curriculum=curriculum)
            summary = trainer.train(
                train_frame,
                train_target,
                episodes=args.episodes,
                validation_data=valid_frame,
                validation_target=valid_target,
            )
            record = {
                "variant": normalize_variant(variant, partition_mode),
                "best_validation_pool_score": summary.best_validation_pool_score,
                "final_pool_size": summary.final_pool_size,
                "accepted_episodes": int(sum(1 for episode in summary.history if episode.accepted)),
                "decision_reason_counts": summary_counters(summary.history, "decision_reason"),
                "selected_agent_counts": summary_counters(summary.history, "selected_agent"),
                "champion_records": list(summary.champion_records),
                "final_records": list(summary.final_records),
                "records_by_role": group_formula_summaries_by_role(summary.final_record_summaries),
                "context_formula_survived": any(item.role == "context" for item in summary.final_record_summaries),
            }
            formulas, evaluation_formula_source = select_evaluation_formulas(
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
                f"partition_mode={partition_mode} "
                f"variant={record['variant']} "
                f"best_validation_pool_score={summary.best_validation_pool_score:.6f} "
                f"final_pool_size={summary.final_pool_size} "
                f"formula_count={len(formulas)}"
            )
        results_by_partition_mode[partition_mode] = mode_results

    payload = {
        "dataset": dataset_name,
        "partition_modes": partition_modes,
        "episodes": args.episodes,
        "results_by_partition_mode": results_by_partition_mode,
        "dataset_diagnostics": dataset_diagnostics(bundle),
    }
    report_path = output_dirs["reports"] / f"{dataset_name}_ablation.json"
    write_json(report_path, payload)
    print(f"ablation_report={report_path}")


if __name__ == "__main__":
    main()
