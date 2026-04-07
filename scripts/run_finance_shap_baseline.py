#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

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
from knowledge_guided_symbolic_alpha.benchmarks.task_protocol import formula_complexity
from knowledge_guided_symbolic_alpha.evaluation.finance_baselines import (
    build_candidate_pool_from_runs,
    formula_features,
    load_json,
    rank_transform_frame,
)
from knowledge_guided_symbolic_alpha.evaluation.finance_reporting import (
    compute_significance_metrics,
    cost_adjusted_returns,
    load_fama_french_factors,
    summarize_returns,
)


UNIVERSE_SOURCES = {
    "liquid500": {
        "data_config": Path("configs/us_equities_liquid500.yaml"),
        "canonical": Path("outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json"),
        "multiseed": Path("outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed.json"),
    },
    "liquid1000": {
        "data_config": Path("configs/us_equities_liquid1000.yaml"),
        "canonical": Path("outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json"),
        "multiseed": Path("outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed.json"),
    },
}
BACKTEST_CONFIG = Path("configs/backtest.yaml")
EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP-ranked finance baseline and evaluate it with walk-forward backtests.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--universes", type=str, default="liquid500,liquid1000")
    parser.add_argument("--train-sample-size", type=int, default=60000)
    parser.add_argument("--shap-sample-size", type=int, default=10000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    return parser.parse_args()


def parse_universes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_valid_split(universe: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_parquet(f"data/processed/us_equities/subsets/{universe}_2010_2025/valid.parquet")
    return frame, frame["TARGET_XS_RET_1"].copy()


def load_consensus_formula(universe: str) -> str:
    payload = load_json(UNIVERSE_SOURCES[universe]["canonical"])
    return str(payload["canonical_by_variant"]["full"]["selector_records"][0])


def load_candidate_pool(universe: str) -> list[str]:
    payload = load_json(UNIVERSE_SOURCES[universe]["multiseed"])
    raw_runs = payload["runs_by_variant"]["full"]
    return [candidate.formula for candidate in build_candidate_pool_from_runs(raw_runs)]


def load_ranked_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    ranked, variable_map = rank_transform_frame(frame)
    raw_map = {column: mapping[0] for column, mapping in variable_map.items()}
    return ranked, raw_map


def shap_feature_importance(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    train_sample_size: int,
    shap_sample_size: int,
    n_estimators: int,
    max_depth: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    import shap

    ranked, variable_map = load_ranked_features(frame)
    aligned_target = target.loc[ranked.index].astype(float)
    if len(ranked) > train_sample_size:
        train_index = ranked.sample(n=train_sample_size, random_state=0).index
    else:
        train_index = ranked.index
    x_train = ranked.loc[train_index]
    y_train = aligned_target.loc[train_index]
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=0,
        n_jobs=1,
    )
    model.fit(x_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    if len(x_train) > shap_sample_size:
        shap_index = x_train.sample(n=shap_sample_size, random_state=1).index
        x_shap = x_train.loc[shap_index]
    else:
        x_shap = x_train
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_shap.to_numpy(dtype=float))
    values = np.asarray(values, dtype=float)
    raw_scores: dict[str, list[float]] = {}
    for column, contribution in zip(x_shap.columns, np.abs(values).mean(axis=0, dtype=float)):
        raw_scores.setdefault(variable_map[column], []).append(float(contribution))
    importance = {feature: float(np.mean(scores)) for feature, scores in raw_scores.items()}
    diagnostics = {
        "train_rows": int(len(x_train)),
        "shap_rows": int(len(x_shap)),
        "top_features": sorted(importance.items(), key=lambda item: item[1], reverse=True)[:8],
    }
    return importance, diagnostics


def score_formula(formula: str, feature_importance: dict[str, float]) -> float:
    features = tuple(dict.fromkeys(formula_features(formula)))
    if not features:
        return float("-inf")
    importance = float(np.mean([feature_importance.get(feature, 0.0) for feature in features]))
    complexity_penalty = 0.0015 * formula_complexity(formula)
    return importance - complexity_penalty


def select_formula(universe: str, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    valid_frame, valid_target = load_valid_split(universe)
    feature_importance, diagnostics = shap_feature_importance(
        valid_frame,
        valid_target,
        train_sample_size=args.train_sample_size,
        shap_sample_size=args.shap_sample_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    candidate_formulas = load_candidate_pool(universe)
    scored_rows = []
    for formula in candidate_formulas:
        scored_rows.append(
            {
                "formula": formula,
                "shap_score": round(score_formula(formula, feature_importance), 6),
                "features": list(dict.fromkeys(formula_features(formula))),
            }
        )
    scored_rows.sort(key=lambda item: (item["shap_score"], -formula_complexity(item["formula"])), reverse=True)
    if not scored_rows:
        raise RuntimeError(f"No candidate formulas available for {universe}.")
    diagnostics["candidate_rows"] = scored_rows[:10]
    return str(scored_rows[0]["formula"]), diagnostics


def build_walk_forward(universe: str, formula: str) -> tuple[dict[str, float], pd.Series]:
    bundle = load_dataset_bundle(UNIVERSE_SOURCES[universe]["data_config"])
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


def run_universe(args: argparse.Namespace, universe: str) -> dict[str, Any]:
    formula, diagnostics = select_formula(universe, args)
    walk_metrics, returns = build_walk_forward(universe, formula)
    factors = load_fama_french_factors()
    turnover = float(walk_metrics.get("turnover") or 0.0)
    net_returns = cost_adjusted_returns(returns, turnover, cost_bps=15.0)
    gross_sig = compute_significance_metrics(returns, factors)
    net_sig = compute_significance_metrics(net_returns, factors)
    net_metrics = summarize_returns(net_returns)
    consensus_formula = load_consensus_formula(universe)
    return {
        "universe": universe,
        "baseline": "shap_ranked_formula_screening",
        "formula": formula,
        "complexity": formula_complexity(formula),
        "matches_consensus_formula": formula == consensus_formula,
        "gross_sharpe": round(float(walk_metrics.get("sharpe") or 0.0), 4),
        "net_sharpe_15bps": round(float(net_metrics["sharpe"]), 4),
        "gross_annual_return": round(float(walk_metrics.get("annual_return") or 0.0), 4),
        "net_annual_return_15bps": round(float(net_metrics["annual_return"]), 4),
        "gross_max_drawdown": round(float(walk_metrics.get("max_drawdown") or 0.0), 4),
        "net_max_drawdown_15bps": round(float(net_metrics["max_drawdown"]), 4),
        "mean_test_rank_ic": round(float(walk_metrics.get("mean_test_rank_ic") or 0.0), 4),
        "turnover": round(turnover, 4),
        "gross_nw_t": round(float(gross_sig["nw_t"]), 4),
        "gross_nw_p": round(float(gross_sig["nw_p"]), 4),
        "gross_ff5_alpha_ann": round(float(gross_sig["ff5_alpha_ann"]), 4),
        "gross_ff5_alpha_t": round(float(gross_sig["ff5_alpha_t"]), 4),
        "gross_ff5_alpha_p": round(float(gross_sig["ff5_alpha_p"]), 4),
        "net_nw_t_15bps": round(float(net_sig["nw_t"]), 4),
        "net_nw_p_15bps": round(float(net_sig["nw_p"]), 4),
        "net_ff5_alpha_ann_15bps": round(float(net_sig["ff5_alpha_ann"]), 4),
        "net_ff5_alpha_t_15bps": round(float(net_sig["ff5_alpha_t"]), 4),
        "net_ff5_alpha_p_15bps": round(float(net_sig["ff5_alpha_p"]), 4),
        "diagnostics": diagnostics,
    }


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Finance SHAP Baseline",
        "",
        "| Universe | Formula | Gross Sharpe | Net Sharpe(15bps) | Rank-IC | NW t | FF5 alpha_ann | Consensus? |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['universe']} | {row['formula']} | {row['gross_sharpe']:.4f} | {row['net_sharpe_15bps']:.4f} | "
            f"{row['mean_test_rank_ic']:.4f} | {row['gross_nw_t']:.4f} | {row['gross_ff5_alpha_ann']:.4f} | {row['matches_consensus_formula']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [run_universe(args, universe) for universe in parse_universes(args.universes)]
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "diagnostics"} for row in rows])
    csv_path = args.output_root / "finance_shap_baseline.csv"
    json_path = args.output_root / "finance_shap_baseline.json"
    md_path = args.output_root / "finance_shap_baseline.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)},
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
