#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


UNIVERSE_CONFIGS = {
    "liquid500": Path("configs/us_equities_liquid500.yaml"),
    "liquid1000": Path("configs/us_equities_liquid1000.yaml"),
}
BACKTEST_CONFIG = Path("configs/backtest.yaml")
EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
CONSENSUS_SOURCES = {
    "liquid500": Path("outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json"),
    "liquid1000": Path("outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PySR-based finance baseline and evaluate it with walk-forward backtests.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--universes", type=str, default="liquid500,liquid1000")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--niterations", type=int, default=24)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--maxsize", type=int, default=8)
    return parser.parse_args()


def parse_universes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def configure_pysr_environment() -> None:
    shared_root = Path("/opt/anaconda3/julia_env")
    shared_exe = shared_root / "pyjuliapkg/install/bin/julia"
    if shared_exe.exists():
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", str(shared_exe))
        os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(shared_root))
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")


def load_consensus_formula(universe: str) -> str:
    payload = json.loads(CONSENSUS_SOURCES[universe].read_text(encoding="utf-8"))
    return str(payload["canonical_by_variant"]["full"]["selector_records"][0])


def load_valid_split(universe: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_parquet(f"data/processed/us_equities/subsets/{universe}_2010_2025/valid.parquet")
    return frame, frame["TARGET_XS_RET_1"].copy()


def load_ranked_features(
    universe: str,
    *,
    sample_size: int,
    max_features: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, tuple[str, ...]], list[str]]:
    from knowledge_guided_symbolic_alpha.evaluation.finance_baselines import rank_transform_frame

    frame, target = load_valid_split(universe)
    ranked, variable_map = rank_transform_frame(frame)
    target_aligned = target.loc[ranked.index].astype(float)
    score_rows: list[tuple[str, float]] = []
    for column in ranked.columns:
        values = ranked[column].astype(float)
        corr = np.corrcoef(values.to_numpy(dtype=float), target_aligned.to_numpy(dtype=float))[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        score_rows.append((column, abs(float(corr))))
    top_features = [name for name, _ in sorted(score_rows, key=lambda item: item[1], reverse=True)[:max_features]]
    ranked = ranked[top_features]
    if len(ranked) > sample_size:
        sampled_index = (
            ranked.assign(_target=target_aligned)
            .sample(n=sample_size, random_state=0)
            .index
        )
        ranked = ranked.loc[sampled_index].copy()
        target_aligned = target_aligned.loc[sampled_index].copy()
    return ranked, target_aligned, variable_map, top_features


def load_pysr_model(args: argparse.Namespace):
    configure_pysr_environment()
    from pysr import PySRRegressor

    population_size = int(args.population_size)
    return PySRRegressor(
        niterations=int(args.niterations),
        population_size=population_size,
        maxsize=int(args.maxsize),
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        model_selection="best",
        should_optimize_constants=False,
        parallelism="serial",
        random_state=0,
        deterministic=True,
        bumper=False,
        verbosity=0,
        update=False,
        progress=False,
        tournament_selection_n=max(3, min(population_size - 1, 8)),
    )


def sympy_expr_to_rpn(expr: Any, variable_map: dict[str, tuple[str, ...]]) -> list[str] | None:
    import sympy as sp

    if expr.is_Symbol:
        mapped = variable_map.get(str(expr))
        return list(mapped) if mapped else None
    if expr.is_Number:
        return None
    if expr.is_negative:
        positive = sp.simplify(-expr)
        tokens = sympy_expr_to_rpn(positive, variable_map)
        return None if tokens is None else tokens + ["NEG"]
    numer, denom = expr.as_numer_denom()
    if denom != 1:
        left = sympy_expr_to_rpn(numer, variable_map)
        right = sympy_expr_to_rpn(denom, variable_map)
        if left is None or right is None:
            return None
        return left + right + ["DIV"]
    if expr.func is sp.Add:
        args = [arg for arg in expr.args if not arg.is_Number]
        if not args:
            return None
        tokens = sympy_expr_to_rpn(args[0], variable_map)
        if tokens is None:
            return None
        for arg in args[1:]:
            right = sympy_expr_to_rpn(arg, variable_map)
            if right is None:
                return None
            tokens = tokens + right + ["ADD"]
        return tokens
    if expr.func is sp.Mul:
        sign = 1.0
        args = []
        for arg in expr.args:
            if arg.is_Number:
                try:
                    sign *= float(arg)
                except Exception:
                    return None
                continue
            args.append(arg)
        if not args:
            return None
        tokens = sympy_expr_to_rpn(args[0], variable_map)
        if tokens is None:
            return None
        for arg in args[1:]:
            right = sympy_expr_to_rpn(arg, variable_map)
            if right is None:
                return None
            tokens = tokens + right + ["MUL"]
        if sign < 0:
            tokens = tokens + ["NEG"]
        return tokens
    return None


def translate_equation(raw_expr: Any, variable_map: dict[str, tuple[str, ...]]) -> str | None:
    import sympy as sp

    expr = sp.sympify(raw_expr)
    tokens = sympy_expr_to_rpn(expr, variable_map)
    if not tokens:
        return None
    return " ".join(tokens)


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, np.floating) else value
    try:
        return float(value)
    except Exception:
        return str(value)


def build_walk_forward(universe: str, formula: str) -> tuple[dict[str, float], pd.Series]:
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

    bundle = load_dataset_bundle(UNIVERSE_CONFIGS[universe])
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
    from knowledge_guided_symbolic_alpha.benchmarks.task_protocol import (
        formula_complexity,
        select_formula_by_pareto_front,
    )
    from knowledge_guided_symbolic_alpha.evaluation.finance_reporting import (
        compute_significance_metrics,
        cost_adjusted_returns,
        load_fama_french_factors,
        summarize_returns,
    )
    from knowledge_guided_symbolic_alpha.generation import FormulaCandidate

    ranked_frame, target, variable_map, selected_features = load_ranked_features(
        universe,
        sample_size=args.sample_size,
        max_features=args.max_features,
    )
    model = load_pysr_model(args)
    model.fit(ranked_frame.to_numpy(dtype=float), target.to_numpy(dtype=float), variable_names=ranked_frame.columns.tolist())
    equations = model.equations_.copy()
    translated_rows: list[dict[str, Any]] = []
    candidates: list[FormulaCandidate] = []
    seen: set[str] = set()
    for row in equations.to_dict(orient="records"):
        formula = translate_equation(row.get("sympy_format") or row.get("equation"), variable_map)
        row["translated_formula"] = formula
        translated_rows.append({key: normalize_json_value(value) for key, value in row.items()})
        if not formula or formula in seen:
            continue
        seen.add(formula)
        candidates.append(FormulaCandidate(formula=formula, source="pysr", role="pysr"))
    if not candidates:
        raise RuntimeError(f"PySR did not produce any translatable formulas for {universe}.")

    valid_frame, valid_target = load_valid_split(universe)
    selected_formula = select_formula_by_pareto_front(candidates, valid_frame, valid_target)
    if not selected_formula:
        raise RuntimeError(f"PySR Pareto selection failed for {universe}.")

    walk_metrics, returns = build_walk_forward(universe, selected_formula)
    factors = load_fama_french_factors()
    turnover = float(walk_metrics.get("turnover") or 0.0)
    net_returns = cost_adjusted_returns(returns, turnover, cost_bps=15.0)
    gross_sig = compute_significance_metrics(returns, factors)
    net_sig = compute_significance_metrics(net_returns, factors)
    net_metrics = summarize_returns(net_returns)
    consensus_formula = load_consensus_formula(universe)
    return {
        "universe": universe,
        "baseline": "pysr_pareto_selection",
        "formula": selected_formula,
        "formula_complexity": formula_complexity(selected_formula),
        "matches_consensus_formula": selected_formula == consensus_formula,
        "ranked_features": selected_features,
        "translated_equation_count": len(candidates),
        "sample_size": int(len(ranked_frame)),
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
        "equations": translated_rows,
    }


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Finance PySR Baseline", "", "| Universe | Formula | Gross Sharpe | Net Sharpe(15bps) | Rank-IC | NW t | FF5 alpha_ann |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['universe']} | {row['formula']} | {row['gross_sharpe']:.4f} | {row['net_sharpe_15bps']:.4f} | "
            f"{row['mean_test_rank_ic']:.4f} | {row['gross_nw_t']:.4f} | {row['gross_ff5_alpha_ann']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [run_universe(args, universe) for universe in parse_universes(args.universes)]
    csv_path = args.output_root / "finance_pysr_baseline.csv"
    json_path = args.output_root / "finance_pysr_baseline.json"
    md_path = args.output_root / "finance_pysr_baseline.md"
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "equations"} for row in rows])
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
