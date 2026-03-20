#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import ssl
from pathlib import Path
from typing import Any

WRDS_REQUIRED_DATASETS = (
    "crsp_daily",
    "crsp_names",
    "ccm_link",
    "compustat_quarterly",
    "compustat_annual",
)

DEFAULT_OUTPUT_ROOT = Path("data/raw/route_b")

WRDS_SPECS: dict[str, dict[str, Any]] = {
    "crsp_daily": {
        "table": "crsp.dsf",
        "columns": [
            "permno", "permco", "date", "ret", "retx", "dlret", "prc", "vol",
            "shrout", "bidlo", "askhi", "cfacpr", "cfacshr", "exchcd", "shrcd",
        ],
        "date_column": "date",
        "start_date": "2000-01-01",
        "end_date": "2025-12-31",
        "filters": ["shrcd in (10, 11)", "exchcd in (1, 2, 3)"],
        "order_by": ["permno", "date"],
        "output_file": "crsp_daily.csv.gz",
    },
    "crsp_names": {
        "table": "crsp.dsenames",
        "columns": ["permno", "ticker", "ncusip", "comnam", "namedt", "nameendt", "exchcd", "shrcd", "siccd"],
        "date_column": None,
        "filters": ["shrcd in (10, 11)", "exchcd in (1, 2, 3)"],
        "order_by": ["permno", "namedt"],
        "output_file": "crsp_names.csv.gz",
    },
    "ccm_link": {
        "table": "crsp.ccmxpf_linktable",
        "columns": ["gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"],
        "date_column": None,
        "filters": ["linktype in ('LC', 'LU', 'LS')", "linkprim in ('P', 'C')"],
        "order_by": ["gvkey", "lpermno", "linkdt"],
        "output_file": "ccm_link.csv.gz",
    },
    "compustat_quarterly": {
        "table": "comp.fundq",
        "columns": [
            "gvkey", "datadate", "rdq", "fyearq", "fqtr", "atq", "ltq", "ceqq", "seq", "saleq", "niq",
            "oiadpq", "cheq", "dlcq", "dlttq", "actq", "lctq", "rectq", "invtq", "cogsq", "xsgaq",
        ],
        "date_column": "datadate",
        "start_date": "2000-01-01",
        "end_date": "2025-12-31",
        "filters": ["indfmt = 'INDL'", "datafmt = 'STD'", "popsrc = 'D'", "consol = 'C'"],
        "order_by": ["gvkey", "datadate"],
        "output_file": "compustat_quarterly.csv.gz",
    },
    "compustat_annual": {
        "table": "comp.funda",
        "columns": [
            "gvkey", "datadate", "fyear", "at", "lt", "ceq", "seq", "sale", "ni", "oiadp",
            "capx", "txditc", "pstkrv", "pstkl", "pstk",
        ],
        "date_column": "datadate",
        "start_date": "2000-01-01",
        "end_date": "2025-12-31",
        "filters": ["indfmt = 'INDL'", "datafmt = 'STD'", "popsrc = 'D'", "consol = 'C'"],
        "order_by": ["gvkey", "datadate"],
        "output_file": "compustat_annual.csv.gz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export route-B WRDS datasets for U.S. equities cross-sectional discovery.")
    parser.add_argument("--config", type=Path, default=Path("configs/route_b_data.yaml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", type=str, default=",".join(WRDS_REQUIRED_DATASETS))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_sql(spec: dict[str, Any], limit: int | None) -> str:
    select_clause = ",\n       ".join(spec["columns"])
    where_clauses = list(spec.get("filters", []))
    date_column = spec.get("date_column")
    if date_column and spec.get("start_date"):
        where_clauses.append(f"{date_column} >= '{spec['start_date']}'")
    if date_column and spec.get("end_date"):
        where_clauses.append(f"{date_column} <= '{spec['end_date']}'")
    sql = [
        "select",
        f"       {select_clause}",
        f"from {spec['table']}",
    ]
    if where_clauses:
        sql.append("where " + "\n  and ".join(where_clauses))
    if spec.get("order_by"):
        sql.append("order by " + ", ".join(spec["order_by"]))
    if limit is not None:
        sql.append(f"limit {int(limit)}")
    return "\n".join(sql)


def connect_wrds():
    try:
        import pg8000.dbapi as pg8000  # type: ignore
    except Exception as exc:
        raise RuntimeError("`pg8000` is required. Activate `.venv-wrds` and install it first.") from exc

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
    dataset_names = parse_csv(args.datasets)
    selected = [(name, WRDS_SPECS[name]) for name in dataset_names]
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "output_root": str(args.output_root),
        "datasets": {},
    }
    for name, spec in selected:
        out = args.output_root / "wrds" / spec["output_file"]
        manifest["datasets"][name] = {
            "table": spec["table"],
            "output": str(out),
            "sql": build_sql(spec, args.limit),
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
            print(f"dataset={name} table={spec['table']} output={out} mode=csv.gz")
    finally:
        connection.close()

    manifest_path = args.output_root / "wrds_export_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
