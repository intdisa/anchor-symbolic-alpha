#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.evaluation import (
    CrossSectionalFormulaEvaluator,
    cross_sectional_ic_summary,
    cross_sectional_risk_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a U.S. equities formula on processed split data.")
    parser.add_argument("--formula", required=True)
    parser.add_argument("--subset-config", default="configs/us_equities_liquid500.yaml")
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="valid")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def load_subset_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload.get("us_equities_subset", payload["route_b_subset"])


def load_split_frame(split_path: Path, memory_limit: str, threads: int) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    return con.execute(f"SELECT * FROM read_parquet('{split_path.as_posix()}')").df()


def main() -> None:
    args = parse_args()
    subset_config = load_subset_config(args.subset_config)
    split_root = Path(subset_config["split_root"])
    split_names = ("train", "valid", "test") if args.split == "all" else (args.split,)
    evaluator = CrossSectionalFormulaEvaluator()

    report: dict[str, dict[str, float | str]] = {}
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

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
