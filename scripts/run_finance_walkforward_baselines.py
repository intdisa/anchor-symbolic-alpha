#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (
    build_portfolio_config,
    build_signal_fusion_config,
    build_walk_forward_config,
    dataset_columns,
    load_dataset_bundle,
    load_experiment_name,
    load_yaml,
)
from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
from knowledge_guided_symbolic_alpha.benchmarks.task_protocol import (
    formula_complexity,
    select_best_formula_by_mean_slice_rank_ic,
    select_best_formula_by_metric,
    select_formula_by_lasso_screening,
    select_formula_by_pareto_front,
)
from knowledge_guided_symbolic_alpha.selection import (
    CrossSeedConsensusConfig,
    CrossSeedConsensusSelector,
    FormulaEvaluationCache,
    RobustSelectorConfig,
    TemporalRobustSelector,
)
from knowledge_guided_symbolic_alpha.evaluation.finance_reporting import (
    compute_significance_metrics,
    cost_adjusted_returns,
    load_fama_french_factors,
    summarize_returns,
)
from knowledge_guided_symbolic_alpha.generation import FormulaCandidate
from scripts.run_finance_rolling_meta_validation import (
    BASE_CONFIG,
    _consensus_config,
    _selector_config,
    derive_seed_runs,
    load_universe_inputs,
    load_universe_scale_stats,
)

BACKTEST_CONFIG = Path("configs/backtest.yaml")
EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
FORCED_REFERENCE_FORMULAS = {
    "liquid500": {"cash_only_reference": "CASH_RATIO_Q RANK"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward finance selection baselines with unified reporting.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def load_valid_split(universe: str) -> tuple[pd.DataFrame, pd.Series]:
    _, _, valid_frame, valid_target, _, _ = load_universe_inputs(universe)
    return valid_frame, valid_target


def load_candidates(universe: str) -> list[FormulaCandidate]:
    _, candidates, _, _, _, _ = load_universe_inputs(universe)
    return candidates


def _consensus_formula(universe: str, *, selection_mode: str) -> str:
    raw_runs, candidates, valid_frame, valid_target, _, _ = load_universe_inputs(universe)
    cache = FormulaEvaluationCache()
    scale_stats = load_universe_scale_stats(universe, candidates, cache)
    selector_cfg = dict(BASE_CONFIG)
    selector_cfg["temporal_selection_mode"] = selection_mode
    selector_cfg["cross_seed_selection_mode"] = selection_mode
    temporal_selector = TemporalRobustSelector(
        _selector_config(selector_cfg, scale_stats),
        evaluation_cache=cache,
    )
    consensus_selector = CrossSeedConsensusSelector(
        temporal_selector=temporal_selector,
        config=_consensus_config(selector_cfg),
    )
    seed_runs = derive_seed_runs(
        raw_runs,
        valid_frame,
        valid_target,
        temporal_selector,
        evaluation_context=f"{universe}:walkforward_baselines:valid",
    )
    outcome = consensus_selector.select(
        seed_runs,
        valid_frame,
        valid_target,
        base_candidates=candidates,
        evaluation_context=f"{universe}:walkforward_baselines:valid",
    )
    return outcome.selected_formulas[0] if outcome.selected_formulas else ""


def select_baseline_formulas(universe: str) -> tuple[dict[str, str], str]:
    candidates = load_candidates(universe)
    valid_frame, valid_target = load_valid_split(universe)
    pareto_consensus_formula = _consensus_formula(universe, selection_mode="pareto")
    pareto_discrete_formula = _consensus_formula(universe, selection_mode="pareto_discrete_legacy")
    legacy_formula = _consensus_formula(universe, selection_mode="legacy_linear")
    formulas = {
        "pareto_cross_seed_consensus": pareto_consensus_formula,
        "pareto_discrete_legacy": pareto_discrete_formula,
        "legacy_linear_selector": legacy_formula,
        "support_adjusted_cross_seed_consensus": legacy_formula,
        "naive_rank_ic": select_best_formula_by_metric(candidates, valid_frame, valid_target, "rank_ic"),
        "best_validation_mean_rank_ic": select_best_formula_by_mean_slice_rank_ic(candidates, valid_frame, valid_target),
        "pareto_front_selector": select_formula_by_pareto_front(candidates, valid_frame, valid_target),
        "lasso_formula_screening": select_formula_by_lasso_screening(candidates, valid_frame, valid_target),
    }
    for baseline, formula in FORCED_REFERENCE_FORMULAS.get(universe, {}).items():
        formulas[baseline] = formula
    return formulas, pareto_consensus_formula


def run_walk_forward(universe: str, formula: str) -> tuple[dict[str, float], pd.Series]:
    bundle = load_dataset_bundle(Path(f"configs/us_equities_{universe}.yaml"))
    backtest_frame = pd.concat([bundle.splits.valid, bundle.splits.test], axis=0)
    dataset_name = load_experiment_name(EXPERIMENT_CONFIG)
    target_column, return_column = dataset_columns(dataset_name)
    backtest_config = load_yaml(BACKTEST_CONFIG)
    report = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    ).run(
        formulas=[formula],
        frame=backtest_frame,
        feature_columns=bundle.feature_columns,
        target_column=target_column,
        return_column=return_column,
        config=build_walk_forward_config(backtest_config),
    )
    metrics = {key: float(value) for key, value in report.aggregate_metrics.items() if isinstance(value, (int, float))}
    returns = pd.Series(report.returns, copy=True)
    returns.index = pd.to_datetime(returns.index)
    return metrics, returns


def build_rows() -> list[dict[str, object]]:
    factors = load_fama_french_factors()
    rows: list[dict[str, object]] = []
    for universe in ("liquid500", "liquid1000"):
        baseline_formulas, consensus_formula = select_baseline_formulas(universe)
        formula_cache: dict[str, tuple[dict[str, float], pd.Series]] = {}
        for baseline, formula in baseline_formulas.items():
            if formula not in formula_cache:
                formula_cache[formula] = run_walk_forward(universe, formula)
            metrics, returns = formula_cache[formula]
            turnover = float(metrics.get("turnover") or 0.0)
            net_returns = cost_adjusted_returns(returns, turnover, cost_bps=15.0)
            gross_significance = compute_significance_metrics(returns, factors)
            net_significance = compute_significance_metrics(net_returns, factors)
            net_metrics = summarize_returns(net_returns)
            rows.append(
                {
                    "universe": universe,
                    "baseline": baseline,
                    "formula": formula,
                    "complexity": formula_complexity(formula),
                    "matches_consensus_formula": formula == consensus_formula,
                    "gross_sharpe": round(float(metrics.get("sharpe") or 0.0), 4),
                    "net_sharpe_15bps": round(float(net_metrics["sharpe"]), 4),
                    "gross_annual_return": round(float(metrics.get("annual_return") or 0.0), 4),
                    "net_annual_return_15bps": round(float(net_metrics["annual_return"]), 4),
                    "gross_max_drawdown": round(float(metrics.get("max_drawdown") or 0.0), 4),
                    "net_max_drawdown_15bps": round(float(net_metrics["max_drawdown"]), 4),
                    "mean_test_rank_ic": round(float(metrics.get("mean_test_rank_ic") or 0.0), 4),
                    "turnover": round(turnover, 4),
                    "gross_nw_t": round(float(gross_significance["nw_t"]), 4),
                    "gross_nw_p": round(float(gross_significance["nw_p"]), 4),
                    "gross_ff5_alpha_ann": round(float(gross_significance["ff5_alpha_ann"]), 4),
                    "gross_ff5_alpha_t": round(float(gross_significance["ff5_alpha_t"]), 4),
                    "gross_ff5_alpha_p": round(float(gross_significance["ff5_alpha_p"]), 4),
                    "net_nw_t_15bps": round(float(net_significance["nw_t"]), 4),
                    "net_nw_p_15bps": round(float(net_significance["nw_p"]), 4),
                    "net_ff5_alpha_ann_15bps": round(float(net_significance["ff5_alpha_ann"]), 4),
                    "net_ff5_alpha_t_15bps": round(float(net_significance["ff5_alpha_t"]), 4),
                    "net_ff5_alpha_p_15bps": round(float(net_significance["ff5_alpha_p"]), 4),
                }
            )
    return rows


def build_markdown(rows: list[dict[str, object]]) -> str:
    lines = ["# Finance Walk-Forward Baselines", ""]
    for universe in sorted({row["universe"] for row in rows}):
        lines.extend([f"## {universe}", ""])
        lines.append("| Baseline | Formula | Gross Sharpe | Net Sharpe(15bps) | Rank-IC | NW t | FF5 alpha_ann | Consensus? |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in [item for item in rows if item["universe"] == universe]:
            lines.append(
                f"| {row['baseline']} | {row['formula']} | {row['gross_sharpe']:.4f} | {row['net_sharpe_15bps']:.4f} | {row['mean_test_rank_ic']:.4f} | {row['gross_nw_t']:.4f} | {row['gross_ff5_alpha_ann']:.4f} | {row['matches_consensus_formula']} |"
            )
        lines.append("")
    return "\n".join(lines)


def load_external_rows(output_root: Path) -> list[dict[str, object]]:
    external_rows: list[dict[str, object]] = []
    for name in ("finance_pysr_baseline.csv", "finance_shap_baseline.csv"):
        path = output_root / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        external_rows.extend(frame.to_dict(orient="records"))
    return external_rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    base_frame = pd.DataFrame(rows)
    extended_rows = rows + load_external_rows(args.output_root)
    extended_frame = pd.DataFrame(extended_rows)
    csv_path = args.output_root / "finance_walkforward_baselines.csv"
    json_path = args.output_root / "finance_walkforward_baselines.json"
    md_path = args.output_root / "finance_walkforward_baselines.md"
    extended_csv_path = args.output_root / "finance_walkforward_baselines_extended.csv"
    extended_json_path = args.output_root / "finance_walkforward_baselines_extended.json"
    extended_md_path = args.output_root / "finance_walkforward_baselines_extended.md"
    pareto_csv_path = args.output_root / "finance_walkforward_baselines_pareto.csv"
    pareto_json_path = args.output_root / "finance_walkforward_baselines_pareto.json"
    pareto_md_path = args.output_root / "finance_walkforward_baselines_pareto.md"
    base_frame.to_csv(csv_path, index=False)
    json_path.write_text(base_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows) + "\n", encoding="utf-8")
    extended_frame.to_csv(extended_csv_path, index=False)
    extended_json_path.write_text(extended_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    extended_md_path.write_text(build_markdown(extended_rows) + "\n", encoding="utf-8")
    extended_frame.to_csv(pareto_csv_path, index=False)
    pareto_json_path.write_text(extended_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    pareto_md_path.write_text(build_markdown(extended_rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "json": str(json_path),
                "markdown": str(md_path),
                "extended_csv": str(extended_csv_path),
                "extended_json": str(extended_json_path),
                "extended_markdown": str(extended_md_path),
                "pareto_csv": str(pareto_csv_path),
                "pareto_json": str(pareto_json_path),
                "pareto_markdown": str(pareto_md_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
