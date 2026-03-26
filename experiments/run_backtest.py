from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.runtime import ensure_preflight, write_run_manifest


DEFAULT_DATA_CONFIG = Path("configs/us_equities_smoke.yaml")
DEFAULT_TRAINING_CONFIG = Path("configs/training.yaml")
DEFAULT_BACKTEST_CONFIG = Path("configs/backtest.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest for discovered formulas.")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--backtest-config", type=Path, default=DEFAULT_BACKTEST_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="backtest_us_equities")
    parser.add_argument("--formula-file", type=Path, default=None)
    parser.add_argument("--formula", action="append", default=None)
    return parser.parse_args()


def load_formulas(formula_file: Path | None, formulas: list[str] | None):
    from experiments.common import load_yaml, select_evaluation_formulas

    if formulas:
        return [item for item in formulas if item]
    if formula_file is None or not formula_file.exists():
        raise ValueError("No formulas supplied. Pass --formula or a valid --formula-file.")
    payload = load_yaml(formula_file) if formula_file.suffix in {".yaml", ".yml"} else None
    if payload is not None:
        formula_list, _ = select_evaluation_formulas(
            payload.get("selector_records", []),
            payload.get("champion_records", []),
            payload.get("final_records", []),
        )
        return formula_list
    import json

    with formula_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    formula_list, _ = select_evaluation_formulas(
        data.get("selector_records", []),
        data.get("champion_records", []),
        data.get("final_records", []),
    )
    return formula_list


def main() -> None:
    args = parse_args()
    preflight = ensure_preflight("core")

    from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester

    from experiments.common import (
        build_portfolio_config,
        build_signal_fusion_config,
        build_walk_forward_config,
        dataset_columns,
        ensure_output_dirs,
        load_dataset_bundle,
        load_experiment_name,
        load_yaml,
        write_json,
    )

    dataset_name = load_experiment_name(args.experiment_config)
    data_config = load_yaml(args.data_config)
    subset_name = Path(data_config["us_equities_subset"]["split_root"]).name
    bundle = load_dataset_bundle(args.data_config)
    target_column, return_column = dataset_columns(dataset_name)
    output_dirs = ensure_output_dirs(args.output_root, args.run_name)
    formula_file = args.formula_file or (output_dirs["factors"] / f"{dataset_name}_champions.json")
    formulas = load_formulas(formula_file, args.formula)
    if not formulas:
        raise ValueError("Backtest formulas are empty.")

    manifest_path = write_run_manifest(
        output_dirs,
        script_name="experiments/run_backtest.py",
        profile="core",
        preflight=preflight.to_dict(),
        config_paths={
            "data_config": str(args.data_config),
            "training_config": str(args.training_config),
            "backtest_config": str(args.backtest_config),
            "experiment_config": str(args.experiment_config),
        },
        dataset_name=dataset_name,
        subset=subset_name,
        extra={"formula_count": len(formulas), "formula_file": None if formula_file is None else str(formula_file)},
    )

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
        "subset": subset_name,
        "formula_count": len(formulas),
        "formulas": formulas,
        "aggregate_metrics": report.aggregate_metrics,
        "folds": report.folds,
        "manifest": str(manifest_path),
    }
    write_json(report_path, payload)
    report.returns.to_frame(name="strategy_returns").to_csv(returns_path, index_label="Date")

    print(f"dataset={dataset_name}")
    print(f"subset={subset_name}")
    print(f"formula_count={len(formulas)}")
    print(f"aggregate_metrics={report.aggregate_metrics}")
    print(f"manifest={manifest_path}")
    print(f"walk_forward_report={report_path}")
    print(f"returns_file={returns_path}")


if __name__ == "__main__":
    main()
