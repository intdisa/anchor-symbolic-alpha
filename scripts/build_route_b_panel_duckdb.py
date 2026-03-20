#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import yaml


DEFAULT_CONFIG_PATH = Path("configs/us_equities_panel.yaml")
DEFAULT_RAW_ROOT = Path("data/raw/us_equities")
DEFAULT_INTERIM_PATH = Path("data/interim/us_equities/daily_feature_panel.parquet")
DEFAULT_PANEL_PATH = Path("data/processed/us_equities/us_equities_panel.parquet")
DEFAULT_SPLIT_ROOT = Path("data/processed/us_equities/splits")
DEFAULT_SUMMARY_PATH = Path("outputs/reports/us_equities_panel_summary.json")
DEFAULT_TEMP_DIR = Path("/tmp/duckdb_us_equities")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the U.S. equities panel with DuckDB using bounded memory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH.as_posix())
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT.as_posix())
    parser.add_argument("--interim-path", default=DEFAULT_INTERIM_PATH.as_posix())
    parser.add_argument("--panel-path", default=DEFAULT_PANEL_PATH.as_posix())
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT.as_posix())
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH.as_posix())
    parser.add_argument("--temp-dir", default=DEFAULT_TEMP_DIR.as_posix())
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-interim", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def load_panel_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload.get("us_equities_panel", payload["route_b_panel"])


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    config = load_panel_config(args.config_path)
    raw_root = Path(args.raw_root)
    interim_path = Path(args.interim_path)
    panel_path = Path(args.panel_path)
    split_root = Path(args.split_root)
    summary_path = Path(args.summary_path)
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    filters = config["filters"]
    min_price = float(filters.get("min_price", 5.0))
    min_dollar_volume_20 = float(filters.get("min_dollar_volume_20", 1_000_000.0))
    min_history_days = int(filters.get("min_history_days", 60))
    min_cross_section = int(filters.get("min_cross_section", 50))
    quarterly_lag_days = int(filters.get("quarterly_lag_days", 45))
    annual_lag_days = int(filters.get("annual_lag_days", 90))

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory='{sql_path(temp_dir)}'")
    con.execute("SET preserve_insertion_order=false")

    crsp_daily = raw_root / "wrds" / "crsp_daily.csv.gz"
    crsp_names = raw_root / "wrds" / "crsp_names.csv.gz"
    ccm_link = raw_root / "wrds" / "ccm_link.csv.gz"
    comp_q = raw_root / "wrds" / "compustat_quarterly.csv.gz"
    comp_a = raw_root / "wrds" / "compustat_annual.csv.gz"

    if not args.skip_interim and (args.force_rebuild or not interim_path.exists()):
        print("[us-equities] build daily interim", flush=True)
        con.execute(
            f"""
            COPY (
              WITH daily AS (
                SELECT
                  CAST(permno AS BIGINT) AS permno,
                  CAST(permco AS BIGINT) AS permco,
                  CAST(date AS DATE) AS date,
                  CAST(COALESCE(retx, ret) AS DOUBLE) AS ret_1,
                  ABS(CAST(prc AS DOUBLE)) AS close,
                  CAST(vol AS DOUBLE) AS vol,
                  CAST(shrout AS DOUBLE) AS shares_out
                FROM read_csv_auto('{sql_path(crsp_daily)}', header=true)
              ),
              enriched AS (
                SELECT
                  *,
                  close * shares_out AS market_cap,
                  close * vol AS dollar_volume,
                  vol / NULLIF(shares_out, 0.0) AS turnover,
                  CASE
                    WHEN COALESCE(ret_1, 0.0) <= -0.999999 THEN NULL
                    ELSE LN(1.0 + COALESCE(ret_1, 0.0))
                  END AS safe_log_ret,
                  LEAD(ret_1) OVER w AS target_ret_1_raw,
                  ROW_NUMBER() OVER w AS history_count,
                  EXP(SUM(CASE WHEN COALESCE(ret_1, 0.0) <= -0.999999 THEN NULL ELSE LN(1.0 + COALESCE(ret_1, 0.0)) END) OVER w5) - 1.0 AS ret_5,
                  EXP(SUM(CASE WHEN COALESCE(ret_1, 0.0) <= -0.999999 THEN NULL ELSE LN(1.0 + COALESCE(ret_1, 0.0)) END) OVER w20) - 1.0 AS ret_20,
                  STDDEV_POP(ret_1) OVER w20 AS volatility_20,
                  AVG(vol / NULLIF(shares_out, 0.0)) OVER w20 AS turnover_20,
                  AVG(close * vol) OVER w20 AS dollar_volume_20,
                  AVG(ABS(ret_1) / NULLIF(close * vol, 0.0)) OVER w20 AS amihud_20,
                  close / MAX(close) OVER w252 - 1.0 AS price_to_252_high
                FROM daily
                WINDOW
                  w AS (PARTITION BY permno ORDER BY date),
                  w5 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                  w20 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
                  w252 AS (PARTITION BY permno ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
              )
              SELECT
                permno,
                permco,
                date,
                ret_1,
                ret_5,
                ret_20,
                volatility_20,
                turnover_20,
                dollar_volume_20,
                amihud_20,
                price_to_252_high,
                market_cap,
                target_ret_1_raw
              FROM enriched
              WHERE close >= {min_price}
                AND dollar_volume_20 >= {min_dollar_volume_20}
                AND history_count >= {min_history_days}
                AND target_ret_1_raw IS NOT NULL
            ) TO '{sql_path(interim_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    source_for_daily = f"read_parquet('{sql_path(interim_path)}')" if not args.skip_interim else f"read_csv_auto('{sql_path(crsp_daily)}', header=true)"
    if args.skip_interim:
        raise ValueError("--skip-interim is not supported for the final build in this script.")

    print("[route-b] build final panel", flush=True)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW names_v AS
        SELECT
          CAST(permno AS BIGINT) AS permno,
          CAST(namedt AS DATE) AS namedt,
          COALESCE(CAST(nameendt AS DATE), DATE '2100-01-01') AS nameendt,
          ticker,
          comnam,
          CAST(siccd AS INTEGER) AS siccd
        FROM read_csv_auto('{sql_path(crsp_names)}', header=true);
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW ccm_v AS
        SELECT
          CAST(gvkey AS VARCHAR) AS gvkey,
          CAST(lpermno AS BIGINT) AS permno,
          CAST(linkdt AS DATE) AS linkdt,
          COALESCE(CAST(linkenddt AS DATE), DATE '2100-01-01') AS linkenddt
        FROM read_csv_auto('{sql_path(ccm_link)}', header=true);
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW quarterly_linked_v AS
        SELECT
          c.permno,
          COALESCE(CAST(q.rdq AS DATE), CAST(q.datadate AS DATE) + INTERVAL {quarterly_lag_days} DAY) AS effective_date,
          COALESCE(CAST(q.ceqq AS DOUBLE), CAST(q.seq AS DOUBLE)) AS quarterly_book_equity,
          CAST(q.oiadpq AS DOUBLE) / NULLIF(CAST(q.atq AS DOUBLE), 0.0) AS quarterly_profitability,
          CAST(q.ltq AS DOUBLE) / NULLIF(CAST(q.atq AS DOUBLE), 0.0) AS quarterly_leverage,
          CAST(q.cheq AS DOUBLE) / NULLIF(CAST(q.atq AS DOUBLE), 0.0) AS quarterly_cash_ratio,
          CAST(q.saleq AS DOUBLE) / NULLIF(CAST(q.atq AS DOUBLE), 0.0) AS quarterly_sales_to_assets
        FROM read_csv_auto('{sql_path(comp_q)}', header=true) q
        JOIN ccm_v c
          ON CAST(q.gvkey AS VARCHAR) = c.gvkey
        WHERE COALESCE(CAST(q.rdq AS DATE), CAST(q.datadate AS DATE) + INTERVAL {quarterly_lag_days} DAY)
          BETWEEN c.linkdt AND c.linkenddt;
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW annual_linked_v AS
        WITH annual_base AS (
          SELECT
            CAST(gvkey AS VARCHAR) AS gvkey,
            CAST(datadate AS DATE) + INTERVAL {annual_lag_days} DAY AS effective_date,
            COALESCE(CAST(ceq AS DOUBLE), CAST(seq AS DOUBLE)) AS annual_book_equity,
            CAST(oiadp AS DOUBLE) / NULLIF(CAST("at" AS DOUBLE), 0.0) AS annual_profitability,
            CAST("at" AS DOUBLE) AS at_value,
            LAG(CAST("at" AS DOUBLE)) OVER (PARTITION BY CAST(gvkey AS VARCHAR) ORDER BY CAST(datadate AS DATE)) AS at_prev,
            CAST("lt" AS DOUBLE) / NULLIF(CAST("at" AS DOUBLE), 0.0) AS annual_leverage
          FROM read_csv_auto('{sql_path(comp_a)}', header=true)
        )
        SELECT
          c.permno,
          a.effective_date,
          a.annual_book_equity,
          a.annual_profitability,
          CASE
            WHEN a.at_prev IS NULL OR a.at_prev = 0.0 THEN NULL
            ELSE a.at_value / a.at_prev - 1.0
          END AS annual_asset_growth,
          a.annual_leverage
        FROM annual_base a
        JOIN ccm_v c
          ON a.gvkey = c.gvkey
        WHERE a.effective_date BETWEEN c.linkdt AND c.linkenddt;
        """
    )
    con.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT * FROM {source_for_daily}
          ),
          with_names AS (
            SELECT
              b.*,
              n.ticker,
              n.comnam,
              n.siccd
            FROM base b
            ASOF LEFT JOIN names_v n
              ON b.permno = n.permno AND b.date >= n.namedt
            WHERE n.nameendt IS NULL OR b.date <= n.nameendt
          ),
          with_quarterly AS (
            SELECT
              b.*,
              q.quarterly_book_equity,
              q.quarterly_profitability,
              q.quarterly_leverage,
              q.quarterly_cash_ratio,
              q.quarterly_sales_to_assets
            FROM with_names b
            ASOF LEFT JOIN quarterly_linked_v q
              ON b.permno = q.permno AND b.date >= q.effective_date
          ),
          with_annual AS (
            SELECT
              b.*,
              a.annual_book_equity,
              a.annual_profitability,
              a.annual_asset_growth,
              a.annual_leverage
            FROM with_quarterly b
            ASOF LEFT JOIN annual_linked_v a
              ON b.permno = a.permno AND b.date >= a.effective_date
          ),
          finalized AS (
            SELECT
              date,
              permno,
              ticker,
              comnam,
              siccd,
              ret_1 AS RET_1,
              ret_5 AS RET_5,
              ret_20 AS RET_20,
              volatility_20 AS VOLATILITY_20,
              turnover_20 AS TURNOVER_20,
              dollar_volume_20 AS DOLLAR_VOLUME_20,
              amihud_20 AS AMIHUD_20,
              price_to_252_high AS PRICE_TO_252_HIGH,
              LN(NULLIF(market_cap, 0.0)) AS SIZE_LOG_MCAP,
              quarterly_book_equity / NULLIF(market_cap, 0.0) AS BOOK_TO_MARKET_Q,
              annual_book_equity / NULLIF(market_cap, 0.0) AS BOOK_TO_MARKET_A,
              quarterly_profitability AS PROFITABILITY_Q,
              annual_profitability AS PROFITABILITY_A,
              annual_asset_growth AS ASSET_GROWTH_A,
              quarterly_leverage AS LEVERAGE_Q,
              annual_leverage AS LEVERAGE_A,
              quarterly_cash_ratio AS CASH_RATIO_Q,
              quarterly_sales_to_assets AS SALES_TO_ASSETS_Q,
              target_ret_1_raw AS TARGET_RET_1,
              target_ret_1_raw - AVG(target_ret_1_raw) OVER (PARTITION BY date) AS TARGET_XS_RET_1,
              COUNT(*) OVER (PARTITION BY date) AS cross_section_size
            FROM with_annual
          )
          SELECT
            date,
            permno,
            ticker,
            comnam,
            siccd,
            RET_1,
            RET_5,
            RET_20,
            VOLATILITY_20,
            TURNOVER_20,
            DOLLAR_VOLUME_20,
            AMIHUD_20,
            PRICE_TO_252_HIGH,
            SIZE_LOG_MCAP,
            BOOK_TO_MARKET_Q,
            BOOK_TO_MARKET_A,
            PROFITABILITY_Q,
            PROFITABILITY_A,
            ASSET_GROWTH_A,
            LEVERAGE_Q,
            LEVERAGE_A,
            CASH_RATIO_Q,
            SALES_TO_ASSETS_Q,
            TARGET_RET_1,
            TARGET_XS_RET_1
          FROM finalized
          WHERE cross_section_size >= {min_cross_section}
            AND RET_1 IS NOT NULL
            AND RET_5 IS NOT NULL
            AND RET_20 IS NOT NULL
            AND VOLATILITY_20 IS NOT NULL
            AND TURNOVER_20 IS NOT NULL
            AND DOLLAR_VOLUME_20 IS NOT NULL
            AND AMIHUD_20 IS NOT NULL
            AND PRICE_TO_252_HIGH IS NOT NULL
            AND SIZE_LOG_MCAP IS NOT NULL
            AND TARGET_RET_1 IS NOT NULL
            AND TARGET_XS_RET_1 IS NOT NULL
        ) TO '{sql_path(panel_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    print("[route-b] summarize panel", flush=True)
    panel_summary = con.execute(
        f"""
        WITH panel AS (
          SELECT * FROM read_parquet('{sql_path(panel_path)}')
        )
        SELECT
          COUNT(*) AS panel_rows,
          MIN(date) AS date_min,
          MAX(date) AS date_max,
          COUNT(DISTINCT permno) AS permno_count
        FROM panel
        """
    ).fetchone()

    split_summaries = {}
    for split_name, bounds in config["splits"].items():
        split_rows, split_min, split_max = con.execute(
            f"""
            SELECT
              COUNT(*) AS rows,
              MIN(date) AS date_min,
              MAX(date) AS date_max
            FROM read_parquet('{sql_path(panel_path)}')
            WHERE date BETWEEN DATE '{bounds["start"]}' AND DATE '{bounds["end"]}'
            """
        ).fetchone()
        split_path = split_root / f"{split_name}.parquet"
        con.execute(
            f"""
            COPY (
              SELECT *
              FROM read_parquet('{sql_path(panel_path)}')
              WHERE date BETWEEN DATE '{bounds["start"]}' AND DATE '{bounds["end"]}'
            ) TO '{sql_path(split_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        split_summaries[split_name] = {
            "rows": int(split_rows or 0),
            "date_min": None if split_min is None else str(split_min),
            "date_max": None if split_max is None else str(split_max),
            "path": str(split_path),
        }

    summary = {
        "panel_rows": int(panel_summary[0]),
        "date_min": str(panel_summary[1]),
        "date_max": str(panel_summary[2]),
        "permno_count": int(panel_summary[3]),
        "interim_path": None if args.skip_interim else str(interim_path),
        "panel_path": str(panel_path),
        "filters": filters,
        "splits": {
            name: stats
            for name, stats in split_summaries.items()
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
