#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import dataset_columns, load_dataset_bundle, load_experiment_name
from knowledge_guided_symbolic_alpha.evaluation.finance_reporting import (
    return_concentration_metrics,
    signal_coverage_metrics,
    summarize_returns,
)
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_metrics import cross_sectional_long_short_returns
from scripts.run_finance_rolling_meta_validation import load_universe_inputs
from scripts.run_finance_rolling_meta_validation import EXPERIMENT_CONFIG

BASELINE_FILES = (
    "finance_walkforward_baselines_pareto.csv",
    "finance_walkforward_baselines_extended.csv",
    "finance_walkforward_baselines.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sparse-signal diagnostics for finance baselines.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def load_baseline_frame(output_root: Path) -> pd.DataFrame:
    for name in BASELINE_FILES:
        path = output_root / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No finance baseline report found.")


def build_rows(output_root: Path) -> list[dict[str, Any]]:
    baseline_frame = load_baseline_frame(output_root)
    rows: list[dict[str, Any]] = []
    eval_cache: dict[tuple[str, str], dict[str, float]] = {}
    return_cache: dict[tuple[str, str], tuple[pd.Series, pd.Series]] = {}
    evaluator = CrossSectionalFormulaEvaluator()

    combined_frames: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for universe in sorted(baseline_frame["universe"].unique()):
        _, _, _, _, test_frame, test_target = load_universe_inputs(str(universe))
        combined_frames[str(universe)] = (test_frame, test_target)

    for row in baseline_frame.to_dict(orient="records"):
        universe = str(row["universe"])
        formula = str(row["formula"])
        cache_key = (universe, formula)
        if cache_key not in eval_cache:
            frame, target = combined_frames[universe]
            evaluated = evaluator.evaluate(formula, frame)
            signal = pd.Series(evaluated.signal, index=frame.index, dtype=float)
            coverage = signal_coverage_metrics(signal, frame["date"])
            eval_cache[cache_key] = coverage
        if cache_key not in return_cache:
            return_cache[cache_key] = test_formula_returns(universe, formula)
        returns, weights = return_cache[cache_key]
        tradable_days = weights.groupby(pd.to_datetime(combined_frames[universe][0].loc[weights.index, "date"]), sort=True).apply(
            lambda series: float(series.abs().sum()) > 0.0
        )
        summary = summarize_returns(returns)
        concentration = return_concentration_metrics(returns)
        rows.append(
            {
                **row,
                "gross_sharpe": round(float(summary["sharpe"]), 6),
                **{key: round(float(value), 6) for key, value in eval_cache[cache_key].items()},
                "tradable_date_fraction": round(float(tradable_days.mean()) if len(tradable_days) else 0.0, 6),
                **{key: round(float(value), 6) for key, value in concentration.items()},
            }
        )
    rows.sort(key=lambda item: (item["universe"], item["baseline"]))
    return rows


def test_formula_returns(universe: str, formula: str) -> tuple[pd.Series, pd.Series]:
    bundle = load_dataset_bundle(Path(f"configs/us_equities_{universe}.yaml"))
    frame = bundle.splits.test.copy()
    dataset_name = load_experiment_name(EXPERIMENT_CONFIG)
    _, return_column = dataset_columns(dataset_name)
    evaluator = CrossSectionalFormulaEvaluator()
    evaluated = evaluator.evaluate(formula, frame)
    signal = pd.Series(evaluated.signal, index=evaluated.signal.index, dtype=float)
    aligned = frame.loc[signal.index]
    returns, weights = cross_sectional_long_short_returns(
        signal,
        aligned[return_column],
        aligned["date"],
        aligned["permno"],
        quantile=0.2,
        weight_scheme="equal",
    )
    return returns, weights


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Finance Sparse Signal Diagnostics",
        "",
        "| Universe | Baseline | Signal | Non-null Fraction | Active Date Fraction | Tradable Date Fraction | Median Active Names | Top 1% PnL Share | Gross Sharpe |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['universe']} | {row['baseline']} | {row['formula']} | {float(row['signal_non_null_fraction']):.4f} | "
            f"{float(row['active_date_fraction']):.4f} | {float(row['tradable_date_fraction']):.4f} | {float(row['median_active_names_per_day']):.1f} | "
            f"{float(row['pnl_top_1pct_day_share']):.4f} | {float(row['gross_sharpe']):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.output_root)
    frame = pd.DataFrame(rows)
    csv_path = args.output_root / "finance_sparse_signal_diagnostics.csv"
    json_path = args.output_root / "finance_sparse_signal_diagnostics.json"
    md_path = args.output_root / "finance_sparse_signal_diagnostics.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
