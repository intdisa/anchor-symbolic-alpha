#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester

from experiments.common import (
    build_portfolio_config,
    build_signal_fusion_config,
    build_walk_forward_config,
    dataset_columns,
    load_dataset_bundle,
    load_yaml,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Route B direct-formula walk-forward baselines.")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--backtest-config", type=Path, default=Path("configs/backtest.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formula", action="append", dest="formulas", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = "route_b"
    bundle = load_dataset_bundle(args.data_config)
    _, return_column = dataset_columns(dataset_name)
    backtest_config = load_yaml(args.backtest_config)
    frame = pd.concat([bundle.splits.valid, bundle.splits.test], axis=0)
    backtester = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    )

    payload: dict[str, dict[str, float]] = {}
    for formula in args.formulas:
        report = backtester.run(
            formulas=[formula],
            frame=frame,
            feature_columns=bundle.feature_columns,
            target_column="TARGET_XS_RET_1",
            return_column=return_column,
            config=build_walk_forward_config(backtest_config),
        )
        payload[formula] = dict(report.aggregate_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
