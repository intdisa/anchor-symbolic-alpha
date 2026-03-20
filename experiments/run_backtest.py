from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester

from experiments.common import (
    DEFAULT_BACKTEST_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    build_portfolio_config,
    build_signal_fusion_config,
    build_walk_forward_config,
    dataset_columns,
    ensure_output_dirs,
    load_dataset_bundle,
    load_experiment_name,
    load_yaml,
    select_evaluation_formulas,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest for discovered formulas.")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--formula-file", type=Path, default=None)
    parser.add_argument("--formula", action="append", default=None)
    return parser.parse_args()


def load_formulas(formula_file: Path | None, formulas: list[str] | None) -> list[str]:
    if formulas:
        return [item for item in formulas if item]
    if formula_file is None or not formula_file.exists():
        raise ValueError("No formulas supplied. Pass --formula or a valid --formula-file.")
    payload = load_yaml(formula_file) if formula_file.suffix in {".yaml", ".yml"} else None
    if payload is not None:
        formulas, _ = select_evaluation_formulas(
            payload.get("champion_records", []),
            payload.get("final_records", []),
        )
        return formulas
    import json

    with formula_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    formulas, _ = select_evaluation_formulas(
        data.get("champion_records", []),
        data.get("final_records", []),
    )
    return formulas


def main() -> None:
    args = parse_args()
    dataset_name = load_experiment_name(args.experiment_config)
    bundle = load_dataset_bundle(args.data_config)
    target_column, return_column = dataset_columns(dataset_name)
    output_dirs = ensure_output_dirs(args.output_root)
    formula_file = args.formula_file or (output_dirs["factors"] / f"{dataset_name}_champions.json")
    formulas = load_formulas(formula_file, args.formula)
    if not formulas:
        raise ValueError("Backtest formulas are empty.")

    backtest_config = load_yaml(args.backtest_config)
    frame = pd.concat([bundle.splits.valid, bundle.splits.test], axis=0)
    backtester = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    )
    report = backtester.run(
        formulas=formulas,
        frame=frame,
        feature_columns=bundle.feature_columns,
        target_column=target_column,
        return_column=return_column,
        config=build_walk_forward_config(backtest_config),
    )

    report_path = output_dirs["reports"] / f"{dataset_name}_walk_forward.json"
    returns_path = output_dirs["reports"] / f"{dataset_name}_walk_forward_returns.csv"
    payload = {
        "dataset": dataset_name,
        "formula_count": len(formulas),
        "formulas": formulas,
        "aggregate_metrics": report.aggregate_metrics,
        "folds": report.folds,
    }
    write_json(report_path, payload)
    report.returns.to_frame(name="strategy_returns").to_csv(returns_path, index_label="Date")

    print(f"dataset={dataset_name}")
    print(f"formula_count={len(formulas)}")
    print(f"aggregate_metrics={report.aggregate_metrics}")
    print(f"walk_forward_report={report_path}")
    print(f"returns_file={returns_path}")


if __name__ == "__main__":
    main()
