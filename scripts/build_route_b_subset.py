#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


DEFAULT_SOURCE_ROOT = Path("data/processed/us_equities/splits")
DEFAULT_OUTPUT_ROOT = Path("data/processed/us_equities/subsets")
DEFAULT_TEMP_DIR = Path("/tmp/duckdb_us_equities")
DEFAULT_SUMMARY_ROOT = Path("outputs/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a memory-bounded U.S. equities subset from processed split data.")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--summary-root", default=DEFAULT_SUMMARY_ROOT.as_posix())
    parser.add_argument("--name", default="liquid500_2010_2025")
    parser.add_argument("--max-permnos", type=int, default=500)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp-dir", default=DEFAULT_TEMP_DIR.as_posix())
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root) / args.name
    summary_root = Path(args.summary_root)
    summary_path = summary_root / f"us_equities_subset_{args.name}.json"
    temp_dir = Path(args.temp_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")

    train_path = source_root / "train.parquet"
    valid_path = source_root / "valid.parquet"
    test_path = source_root / "test.parquet"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE top_permnos AS
        SELECT permno
        FROM read_parquet('{sql_path(train_path)}')
        WHERE date BETWEEN DATE '{args.start_date}' AND DATE '{args.end_date}'
        GROUP BY permno
        ORDER BY AVG(DOLLAR_VOLUME_20) DESC
        LIMIT {args.max_permnos}
        """
    )

    split_stats: dict[str, dict[str, int | str | None]] = {}
    for split_name, source_path in [("train", train_path), ("valid", valid_path), ("test", test_path)]:
        output_path = output_root / f"{split_name}.parquet"
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM read_parquet('{sql_path(source_path)}')
              WHERE permno IN (SELECT permno FROM top_permnos)
                AND date BETWEEN DATE '{args.start_date}' AND DATE '{args.end_date}'
            ) TO '{sql_path(output_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        rows, date_min, date_max, permnos = con.execute(
            f"""
            SELECT
              COUNT(*) AS rows,
              MIN(date) AS date_min,
              MAX(date) AS date_max,
              COUNT(DISTINCT permno) AS permnos
            FROM read_parquet('{sql_path(output_path)}')
            """
        ).fetchone()
        split_stats[split_name] = {
            "rows": int(rows or 0),
            "date_min": None if date_min is None else str(date_min),
            "date_max": None if date_max is None else str(date_max),
            "permnos": int(permnos or 0),
            "path": str(output_path),
        }

    summary = {
        "name": args.name,
        "max_permnos": args.max_permnos,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "output_root": str(output_root),
        "splits": split_stats,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
