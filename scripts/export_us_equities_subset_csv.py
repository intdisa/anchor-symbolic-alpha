#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.runtime import ensure_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a processed U.S. equities subset to csv.gz for environments without parquet support.")
    parser.add_argument("--subset-root", default="data/processed/us_equities/subsets/liquid500_2010_2025")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    ensure_preflight("eval")
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("`duckdb` is required for csv export. Install the `eval` dependency tier first.") from exc

    subset_root = Path(args.subset_root)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET threads={args.threads}")
    for split_name in ("train", "valid", "test"):
        parquet_path = subset_root / f"{split_name}.parquet"
        csv_path = subset_root / f"{split_name}.csv.gz"
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM read_parquet('{sql_path(parquet_path)}')
            ) TO '{sql_path(csv_path)}' (HEADER, COMPRESSION GZIP)
            """
        )
        print(f"wrote={csv_path}")


if __name__ == "__main__":
    main()
