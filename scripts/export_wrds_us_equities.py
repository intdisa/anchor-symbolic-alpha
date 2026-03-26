#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import ssl
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.dataio import WRDS_REQUIRED_DATASETS, build_wrds_query, load_us_equities_config
from knowledge_guided_symbolic_alpha.runtime import ensure_preflight


DEFAULT_CONFIG = Path("configs/us_equities_data.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export WRDS datasets for the U.S. equities cross-sectional mainline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", type=str, default=",".join(WRDS_REQUIRED_DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def apply_limit(sql: str, limit: int | None) -> str:
    if limit is None:
        return sql
    return f"{sql}\nlimit {int(limit)}"


def connect_wrds():
    try:
        import pg8000.dbapi as pg8000  # type: ignore
    except Exception as exc:
        raise RuntimeError("`pg8000` is required. Install the `wrds` dependency tier first.") from exc

    username = os.environ.get("WRDS_USERNAME", "")
    password = os.environ.get("WRDS_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("WRDS_USERNAME and WRDS_PASSWORD must be set in the local shell.")

    return pg8000.connect(
        user=username,
        password=password,
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        database="wrds",
        ssl_context=ssl.create_default_context(),
        timeout=10,
    )


def export_query_to_file(connection: Any, sql: str, target_path: Path, chunk_size: int = 50000) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        headers = [column[0] for column in cursor.description]
        with gzip.open(target_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                writer.writerows(rows)
    finally:
        cursor.close()


def main() -> None:
    args = parse_args()
    ensure_preflight("core" if args.dry_run else "wrds")
    config = load_us_equities_config(args.config)
    output_root = args.output_root or config.output_root
    dataset_names = parse_csv(args.datasets)
    spec_by_name = {spec.dataset_name: spec for spec in config.wrds_specs}
    selected = [(name, spec_by_name[name]) for name in dataset_names]

    manifest: dict[str, Any] = {
        "config": str(args.config),
        "output_root": str(output_root),
        "datasets": {},
    }
    for name, spec in selected:
        out = output_root / "wrds" / str(spec.output_file or f"{name}.csv.gz")
        manifest["datasets"][name] = {
            "table": spec.qualified_table,
            "output": str(out),
            "sql": apply_limit(build_wrds_query(spec), args.limit),
        }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    connection = connect_wrds()
    try:
        for name, spec in selected:
            sql = manifest["datasets"][name]["sql"]
            out = Path(manifest["datasets"][name]["output"])
            export_query_to_file(connection, sql, out)
            manifest["datasets"][name]["write_mode"] = "csv.gz"
            print(f"dataset={name} table={spec.qualified_table} output={out} mode=csv.gz")
    finally:
        connection.close()

    manifest_path = output_root / "wrds_export_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
