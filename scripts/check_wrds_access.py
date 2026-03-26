#!/usr/bin/env python3
from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.runtime import ensure_preflight

REQUIRED_TABLES = (
    "crsp.dsf",
    "crsp.dsenames",
    "crsp.ccmxpf_linktable",
    "comp.fundq",
    "comp.funda",
)

FALLBACK_COMPUSTAT_TABLES = (
    "comp.secd",
    "comp.security",
    "comp.company",
    "comp.sec_history",
    "comp.sec_idhist",
)


def connect_wrds():
    import pg8000.dbapi as pg8000  # type: ignore

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


def fetch_rows(cursor, query: str) -> list[tuple]:
    cursor.execute(query)
    return list(cursor.fetchall())


def check_tables(connection, tables: Iterable[str]) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for table in tables:
        cursor = connection.cursor()
        try:
            cursor.execute(f"select * from {table} limit 1")
            cursor.fetchall()
            results.append((table, "OK"))
            connection.commit()
        except Exception as exc:
            connection.rollback()
            results.append((table, f"ERROR: {exc}"))
        finally:
            cursor.close()
    return results


def main() -> None:
    ensure_preflight("wrds")
    connection = connect_wrds()
    cursor = connection.cursor()
    try:
        schemas = fetch_rows(
            cursor,
            """
            select nspname
            from pg_namespace
            where has_schema_privilege(nspname, 'USAGE') = true
              and nspname !~ '(^pg_)|(_old$)|(_new$)|(information_schema)'
            order by 1
            """,
        )
        print("Accessible schemas:")
        for (schema,) in schemas:
            print(f"  - {schema}")

        print("\nU.S. equities table checks:")
        for table, status in check_tables(connection, REQUIRED_TABLES):
            print(f"  - {table}: {status}")

        print("\nCompustat-only fallback checks:")
        for table, status in check_tables(connection, FALLBACK_COMPUSTAT_TABLES):
            print(f"  - {table}: {status}")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
