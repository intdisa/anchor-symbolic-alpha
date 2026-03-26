#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.runtime import ensure_preflight, ensure_run_output_dirs, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a U.S. equities formula on processed split data.")
    parser.add_argument("--formula", required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/us_equities_liquid500.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--backtest-config", type=Path, default=Path("configs/backtest.yaml"))
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiments/us_equities_anchor.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-name", type=str, default="formula_eval_us_equities")
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="valid")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def load_subset_config(path: str | Path) -> dict[str, str]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return dict(payload["us_equities_subset"])


def load_split_frame(split_path: Path, memory_limit: str, threads: int) -> pd.DataFrame:
    csv_path = split_path.with_suffix(".csv.gz")
    try:
        import duckdb

        con = duckdb.connect()
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET threads={threads}")
        return con.execute(f"SELECT * FROM read_parquet('{split_path.as_posix()}')").df()
    except ModuleNotFoundError:
        pass

    if split_path.exists():
        try:
            return pd.read_parquet(split_path)
        except ImportError:
            pass

    if csv_path.exists():
        return pd.read_csv(
            csv_path,
            parse_dates=["date"],
            dtype={
                "permno": "int64",
                "ticker": "string",
                "comnam": "string",
                "siccd": "Int64",
            },
        )

    raise FileNotFoundError(
        f"Missing processed split for {split_path.stem}: expected {split_path} or {csv_path}. "
        "Rebuild the canonical `data/processed/us_equities/...` subset first."
    )


def main() -> None:
    args = parse_args()
    preflight = ensure_preflight("eval")
    output_dirs = ensure_run_output_dirs(args.output_root, args.run_name)
    subset_config = load_subset_config(args.data_config)
    split_root = Path(subset_config["split_root"])
    subset_name = split_root.name

    manifest_path = write_run_manifest(
        output_dirs,
        script_name="scripts/evaluate_formula.py",
        profile="eval",
        preflight=preflight.to_dict(),
        config_paths={
            "data_config": str(args.data_config),
            "training_config": str(args.training_config),
            "backtest_config": str(args.backtest_config),
            "experiment_config": str(args.experiment_config),
        },
        dataset_name="us_equities",
        subset=subset_name,
        extra={"formula": args.formula, "split": args.split},
    )

    from knowledge_guided_symbolic_alpha.evaluation import (
        CrossSectionalFormulaEvaluator,
        cross_sectional_ic_summary,
        cross_sectional_risk_summary,
    )

    split_names = ("train", "valid", "test") if args.split == "all" else (args.split,)
    evaluator = CrossSectionalFormulaEvaluator()
    report: dict[str, dict[str, float | str | int]] = {}
    for split_name in split_names:
        split_path = split_root / f"{split_name}.parquet"
        frame = load_split_frame(split_path, memory_limit=args.memory_limit, threads=args.threads)
        evaluated = evaluator.evaluate(args.formula, frame)
        ic_metrics = cross_sectional_ic_summary(evaluated.signal, frame["TARGET_XS_RET_1"], frame["date"])
        risk_metrics = cross_sectional_risk_summary(
            evaluated.signal,
            frame["TARGET_XS_RET_1"],
            frame["date"],
            frame["permno"],
        )
        report[split_name] = {
            "rows": int(len(frame)),
            "formula": args.formula,
            **ic_metrics,
            **risk_metrics,
        }

    report_path = output_dirs["reports"] / "formula_eval.json"
    payload = {
        "dataset": "us_equities",
        "subset": subset_name,
        "split": args.split,
        "formula": args.formula,
        "metrics": report,
        "manifest": str(manifest_path),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
