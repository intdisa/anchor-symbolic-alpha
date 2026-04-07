from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULT_US_EQUITIES_PANEL_CONFIG = Path("configs/us_equities_panel.yaml")
DEFAULT_US_EQUITIES_RAW_ROOT = Path("data/raw/us_equities")
DEFAULT_US_EQUITIES_SPLIT_ROOT = Path("data/processed/us_equities/splits")
TARGET_COLUMNS = ("TARGET_RET_1", "TARGET_XS_RET_1")
US_EQUITIES_FEATURE_COLUMNS = (
    "RET_1",
    "RET_5",
    "RET_20",
    "VOLATILITY_20",
    "TURNOVER_20",
    "DOLLAR_VOLUME_20",
    "AMIHUD_20",
    "PRICE_TO_252_HIGH",
    "SIZE_LOG_MCAP",
    "BOOK_TO_MARKET_Q",
    "BOOK_TO_MARKET_A",
    "PROFITABILITY_Q",
    "PROFITABILITY_A",
    "ASSET_GROWTH_A",
    "LEVERAGE_Q",
    "LEVERAGE_A",
    "CASH_RATIO_Q",
    "SALES_TO_ASSETS_Q",
)


@dataclass(frozen=True)
class USEquitiesSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class USEquitiesPanelBundle:
    panel: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    splits: USEquitiesSplits


@dataclass(frozen=True)
class USEquitiesProcessedBundle:
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    splits: USEquitiesSplits


@dataclass(frozen=True)
class USEquitiesPanelConfig:
    raw_root: Path
    splits: dict[str, dict[str, str]]
    filters: dict[str, Any]


def load_us_equities_panel_config(path: str | Path = DEFAULT_US_EQUITIES_PANEL_CONFIG) -> USEquitiesPanelConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "us_equities_panel" in payload:
        config = payload["us_equities_panel"]
    else:
        config = payload["us_equities_panel"]
    return USEquitiesPanelConfig(
        raw_root=Path(config.get("raw_root", DEFAULT_US_EQUITIES_RAW_ROOT)),
        splits={key: dict(value) for key, value in config["splits"].items()},
        filters=dict(config.get("filters", {})),
    )


def load_us_equities_raw_tables(raw_root: str | Path = DEFAULT_US_EQUITIES_RAW_ROOT) -> dict[str, pd.DataFrame]:
    root = Path(raw_root)
    _ensure_required_files(
        root,
        {
            "crsp_daily": root / "wrds" / "crsp_daily.csv.gz",
            "crsp_names": root / "wrds" / "crsp_names.csv.gz",
            "ccm_link": root / "wrds" / "ccm_link.csv.gz",
            "compustat_quarterly": root / "wrds" / "compustat_quarterly.csv.gz",
            "compustat_annual": root / "wrds" / "compustat_annual.csv.gz",
        },
        hint="Run `python scripts/export_wrds_us_equities.py` and place the extracted files under `data/raw/us_equities/wrds/`.",
    )
    return {
        "crsp_daily": pd.read_csv(
            root / "wrds" / "crsp_daily.csv.gz",
            parse_dates=["date"],
            dtype={"permno": "int64", "permco": "int64", "exchcd": "int64", "shrcd": "int64"},
        ),
        "crsp_names": pd.read_csv(
            root / "wrds" / "crsp_names.csv.gz",
            parse_dates=["namedt", "nameendt"],
            dtype={"permno": "int64", "ticker": "string", "ncusip": "string", "comnam": "string", "exchcd": "int64", "shrcd": "int64", "siccd": "Int64"},
        ),
        "ccm_link": pd.read_csv(
            root / "wrds" / "ccm_link.csv.gz",
            parse_dates=["linkdt", "linkenddt"],
            dtype={"gvkey": "string", "lpermno": "int64", "linktype": "string", "linkprim": "string"},
        ),
        "compustat_quarterly": pd.read_csv(
            root / "wrds" / "compustat_quarterly.csv.gz",
            parse_dates=["datadate", "rdq"],
            dtype={"gvkey": "string"},
        ),
        "compustat_annual": pd.read_csv(
            root / "wrds" / "compustat_annual.csv.gz",
            parse_dates=["datadate"],
            dtype={"gvkey": "string"},
        ),
        "crsp_delisting": _read_optional_delisting_table(root / "wrds" / "crsp_delisting.csv.gz"),
    }


def build_us_equities_panel(
    raw_root: str | Path | None = None,
    config_path: str | Path = DEFAULT_US_EQUITIES_PANEL_CONFIG,
) -> USEquitiesPanelBundle:
    config = load_us_equities_panel_config(config_path)
    tables = load_us_equities_raw_tables(raw_root or config.raw_root)
    panel = build_panel_from_tables(tables, config.filters)
    splits = split_panel_by_dates(panel, config.splits)
    feature_columns = tuple(
        column
        for column in panel.columns
        if column not in {"date", "permno", "ticker", "comnam", "siccd", *TARGET_COLUMNS}
    )
    return USEquitiesPanelBundle(
        panel=panel,
        feature_columns=feature_columns,
        target_columns=TARGET_COLUMNS,
        splits=splits,
    )


def load_processed_us_equities_splits(split_root: str | Path = DEFAULT_US_EQUITIES_SPLIT_ROOT) -> USEquitiesProcessedBundle:
    root = Path(split_root)
    if not root.exists():
        raise FileNotFoundError(
            "Processed split root is missing: "
            f"{root}. Build the panel first with `python scripts/build_us_equities_panel.py` "
            "or create a subset with `python scripts/build_us_equities_subset.py`."
        )
    train = _read_processed_split(root, "train")
    valid = _read_processed_split(root, "valid")
    test = _read_processed_split(root, "test")
    feature_columns = tuple(
        column
        for column in train.columns
        if column not in {"date", "permno", "ticker", "comnam", "siccd", *TARGET_COLUMNS}
    )
    return USEquitiesProcessedBundle(
        feature_columns=feature_columns,
        target_columns=TARGET_COLUMNS,
        splits=USEquitiesSplits(train=train, valid=valid, test=test),
    )


def _read_processed_split(root: Path, split_name: str) -> pd.DataFrame:
    parquet_path = root / f"{split_name}.parquet"
    csv_path = root / f"{split_name}.csv.gz"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            if not csv_path.exists():
                raise
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
        f"Missing split file for `{split_name}` under {root}. "
        f"Expected one of {parquet_path.name} or {csv_path.name}. "
        "Rebuild the processed split directory from the canonical `data/processed/us_equities/...` layout."
    )


def _ensure_required_files(root: Path, required: dict[str, Path], *, hint: str) -> None:
    missing = {name: path for name, path in required.items() if not path.exists()}
    if not missing:
        return
    details = ", ".join(f"{name}={path}" for name, path in missing.items())
    raise FileNotFoundError(f"Missing required U.S. equities raw files under {root}: {details}. {hint}")


def build_panel_from_tables(tables: dict[str, pd.DataFrame], filters: dict[str, Any] | None = None) -> pd.DataFrame:
    filters = filters or {}
    daily = _prepare_daily_panel(tables["crsp_daily"], filters, tables.get("crsp_delisting"))
    names = _prepare_names_table(tables["crsp_names"])
    ccm = _prepare_ccm_table(tables["ccm_link"])
    quarterly = _prepare_quarterly_fundamentals(tables["compustat_quarterly"], filters)
    annual = _prepare_annual_fundamentals(tables["compustat_annual"], filters)

    panel = _merge_names(daily, names)
    panel = _merge_fundamentals(panel, quarterly, ccm, effective_date_column="quarterly_effective_date")
    panel = _merge_fundamentals(panel, annual, ccm, effective_date_column="annual_effective_date")
    panel = _finalize_panel(panel, filters)
    return panel


def split_panel_by_dates(panel: pd.DataFrame, split_config: dict[str, dict[str, str]]) -> USEquitiesSplits:
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, bounds in split_config.items():
        start = pd.Timestamp(bounds["start"])
        end = pd.Timestamp(bounds["end"])
        split = panel.loc[(panel["date"] >= start) & (panel["date"] <= end)].copy()
        if split.empty:
            raise ValueError(f"Split {split_name!r} is empty for bounds {bounds!r}.")
        split_frames[split_name] = split
    return USEquitiesSplits(
        train=split_frames["train"],
        valid=split_frames["valid"],
        test=split_frames["test"],
    )


def _read_optional_delisting_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["permno", "dlstdt", "dlret", "dlstcd"])
    return pd.read_csv(
        path,
        parse_dates=["dlstdt"],
        dtype={"permno": "int64", "dlstcd": "Int64"},
    )


def _prepare_daily_panel(frame: pd.DataFrame, filters: dict[str, Any], delisting_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["permno"] = daily["permno"].astype("int64")
    if "dlret" in daily.columns:
        daily["dlret"] = pd.to_numeric(daily["dlret"], errors="coerce")
    else:
        daily["dlret"] = np.nan
    if delisting_frame is not None and not delisting_frame.empty:
        delisting = _prepare_delisting_table(delisting_frame)
        daily = daily.merge(delisting, on=["permno", "date"], how="left", suffixes=("", "_from_delist"))
        if "dlret_from_delist" in daily.columns:
            daily["dlret"] = daily["dlret"].where(daily["dlret"].notna(), daily["dlret_from_delist"])
            daily = daily.drop(columns=["dlret_from_delist"])
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)

    daily["close"] = daily["prc"].abs().replace(0.0, np.nan)
    daily["shares_out"] = daily["shrout"].replace(0.0, np.nan)
    daily["market_cap"] = daily["close"] * daily["shares_out"]
    daily["dollar_volume"] = daily["close"] * daily["vol"].fillna(0.0)
    daily["ret_1"] = daily["retx"].fillna(daily["ret"]).astype(float)
    daily["ret_total_1"] = _combine_delisting_return(daily["ret"], daily["dlret"])

    grouped = daily.groupby("permno", group_keys=False)
    daily["ret_5"] = grouped["ret_1"].transform(lambda s: (1.0 + s.fillna(0.0)).rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0)
    daily["ret_20"] = grouped["ret_1"].transform(lambda s: (1.0 + s.fillna(0.0)).rolling(20, min_periods=20).apply(np.prod, raw=True) - 1.0)
    daily["volatility_20"] = grouped["ret_1"].transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    daily["turnover"] = daily["vol"] / daily["shares_out"]
    daily["turnover_20"] = grouped["turnover"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    daily["dollar_volume_20"] = grouped["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    daily["amihud_20"] = (
        daily.groupby("permno", group_keys=False)[["ret_1", "dollar_volume"]]
        .apply(_rolling_amihud)
        .reset_index(level=0, drop=True)
    )
    daily["price_to_252_high"] = grouped["close"].transform(lambda s: s / s.rolling(252, min_periods=20).max() - 1.0)
    daily["target_ret_1_raw"] = grouped["ret_total_1"].shift(-1)

    min_price = float(filters.get("min_price", 5.0))
    min_dollar_volume = float(filters.get("min_dollar_volume_20", 1_000_000.0))
    min_history_days = int(filters.get("min_history_days", 60))
    history_count = grouped.cumcount() + 1
    daily = daily.loc[
        (daily["close"] >= min_price)
        & (daily["dollar_volume_20"] >= min_dollar_volume)
        & (history_count >= min_history_days)
    ].copy()
    return daily


def _combine_delisting_return(ret: pd.Series, dlret: pd.Series) -> pd.Series:
    ret_series = pd.to_numeric(ret, errors="coerce")
    dlret_series = pd.to_numeric(dlret, errors="coerce")
    has_observation = ret_series.notna() | dlret_series.notna()
    combined = (1.0 + ret_series.fillna(0.0)) * (1.0 + dlret_series.fillna(0.0)) - 1.0
    return combined.where(has_observation)


def _rolling_amihud(series: pd.DataFrame) -> pd.Series:
    abs_ret = series["ret_1"].abs()
    dollar_volume = series["dollar_volume"].replace(0.0, np.nan)
    ratio = abs_ret / dollar_volume
    return ratio.rolling(20, min_periods=20).mean()


def _prepare_delisting_table(frame: pd.DataFrame) -> pd.DataFrame:
    delisting = frame.copy()
    delisting["permno"] = pd.to_numeric(delisting["permno"], errors="coerce").astype("Int64")
    delisting["date"] = pd.to_datetime(delisting["dlstdt"], errors="coerce")
    delisting["dlret"] = pd.to_numeric(delisting["dlret"], errors="coerce")
    delisting = delisting.dropna(subset=["permno", "date"]).copy()
    delisting["permno"] = delisting["permno"].astype("int64")
    return delisting[["permno", "date", "dlret"]].drop_duplicates(subset=["permno", "date"], keep="last")


def _prepare_names_table(frame: pd.DataFrame) -> pd.DataFrame:
    names = frame.copy()
    names["permno"] = names["permno"].astype("int64")
    names["namedt"] = pd.to_datetime(names["namedt"])
    names["nameendt"] = pd.to_datetime(names["nameendt"]).fillna(pd.Timestamp("2100-01-01"))
    return names.sort_values(["permno", "namedt"]).reset_index(drop=True)


def _prepare_ccm_table(frame: pd.DataFrame) -> pd.DataFrame:
    ccm = frame.copy()
    ccm = ccm.dropna(subset=["gvkey", "lpermno"])
    ccm["permno"] = ccm["lpermno"].astype("int64")
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"]).fillna(pd.Timestamp("1900-01-01"))
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"]).fillna(pd.Timestamp("2100-01-01"))
    return ccm[["gvkey", "permno", "linkdt", "linkenddt", "linktype", "linkprim"]].sort_values(["permno", "linkdt"])


def _prepare_quarterly_fundamentals(frame: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    quarterly = frame.copy()
    quarterly["datadate"] = pd.to_datetime(quarterly["datadate"])
    quarterly["rdq"] = pd.to_datetime(quarterly["rdq"])
    lag_days = int(filters.get("quarterly_lag_days", 45))
    quarterly["quarterly_effective_date"] = quarterly["rdq"].fillna(quarterly["datadate"] + pd.to_timedelta(lag_days, unit="D"))
    quarterly["quarterly_book_equity"] = quarterly["ceqq"].fillna(quarterly["seq"])
    quarterly["quarterly_profitability"] = quarterly["oiadpq"] / quarterly["atq"].replace(0.0, np.nan)
    quarterly["quarterly_leverage"] = quarterly["ltq"] / quarterly["atq"].replace(0.0, np.nan)
    quarterly["quarterly_cash_ratio"] = quarterly["cheq"] / quarterly["atq"].replace(0.0, np.nan)
    quarterly["quarterly_sales_to_assets"] = quarterly["saleq"] / quarterly["atq"].replace(0.0, np.nan)
    keep = [
        "gvkey",
        "quarterly_effective_date",
        "quarterly_book_equity",
        "quarterly_profitability",
        "quarterly_leverage",
        "quarterly_cash_ratio",
        "quarterly_sales_to_assets",
    ]
    return quarterly[keep].sort_values(["gvkey", "quarterly_effective_date"])


def _prepare_annual_fundamentals(frame: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    annual = frame.copy()
    annual["datadate"] = pd.to_datetime(annual["datadate"])
    lag_days = int(filters.get("annual_lag_days", 90))
    annual["annual_effective_date"] = annual["datadate"] + pd.to_timedelta(lag_days, unit="D")
    annual = annual.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    annual["annual_book_equity"] = annual["ceq"].fillna(annual["seq"])
    annual["annual_profitability"] = annual["oiadp"] / annual["at"].replace(0.0, np.nan)
    annual["annual_asset_growth"] = annual.groupby("gvkey")["at"].pct_change()
    annual["annual_leverage"] = annual["lt"] / annual["at"].replace(0.0, np.nan)
    keep = [
        "gvkey",
        "annual_effective_date",
        "annual_book_equity",
        "annual_profitability",
        "annual_asset_growth",
        "annual_leverage",
    ]
    return annual[keep].sort_values(["gvkey", "annual_effective_date"])


def _merge_names(panel: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        panel.sort_values(["date", "permno"]),
        names.sort_values(["namedt", "permno"]),
        left_on="date",
        right_on="namedt",
        by="permno",
        direction="backward",
    )
    merged = merged.loc[merged["date"] <= merged["nameendt"]].copy()
    return merged


def _merge_fundamentals(
    panel: pd.DataFrame,
    fundamentals: pd.DataFrame,
    ccm: pd.DataFrame,
    effective_date_column: str,
) -> pd.DataFrame:
    merged_fundamentals = fundamentals.merge(ccm, on="gvkey", how="inner")
    merged_fundamentals = merged_fundamentals.loc[
        (merged_fundamentals[effective_date_column] >= merged_fundamentals["linkdt"])
        & (merged_fundamentals[effective_date_column] <= merged_fundamentals["linkenddt"])
    ].copy()
    merged_fundamentals = merged_fundamentals.sort_values([effective_date_column, "permno"])

    panel = pd.merge_asof(
        panel.sort_values(["date", "permno"]),
        merged_fundamentals,
        left_on="date",
        right_on=effective_date_column,
        by="permno",
        direction="backward",
    )
    return panel


def _finalize_panel(panel: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    result = panel.copy()
    result["RET_1"] = result["ret_1"]
    result["RET_5"] = result["ret_5"]
    result["RET_20"] = result["ret_20"]
    result["VOLATILITY_20"] = result["volatility_20"]
    result["TURNOVER_20"] = result["turnover_20"]
    result["DOLLAR_VOLUME_20"] = result["dollar_volume_20"]
    result["AMIHUD_20"] = result["amihud_20"]
    result["PRICE_TO_252_HIGH"] = result["price_to_252_high"]
    result["SIZE_LOG_MCAP"] = np.log(result["market_cap"].replace(0.0, np.nan))
    result["BOOK_TO_MARKET_Q"] = result["quarterly_book_equity"] / result["market_cap"].replace(0.0, np.nan)
    result["BOOK_TO_MARKET_A"] = result["annual_book_equity"] / result["market_cap"].replace(0.0, np.nan)
    result["PROFITABILITY_Q"] = result["quarterly_profitability"]
    result["PROFITABILITY_A"] = result["annual_profitability"]
    result["ASSET_GROWTH_A"] = result["annual_asset_growth"]
    result["LEVERAGE_Q"] = result["quarterly_leverage"]
    result["LEVERAGE_A"] = result["annual_leverage"]
    result["CASH_RATIO_Q"] = result["quarterly_cash_ratio"]
    result["SALES_TO_ASSETS_Q"] = result["quarterly_sales_to_assets"]

    result["TARGET_RET_1"] = result["target_ret_1_raw"]
    result["TARGET_XS_RET_1"] = result["TARGET_RET_1"] - result.groupby("date")["TARGET_RET_1"].transform("mean")

    min_cross_section = int(filters.get("min_cross_section", 50))
    cross_section_size = result.groupby("date")["permno"].transform("count")
    result = result.loc[cross_section_size >= min_cross_section].copy()

    keep = [
        "date",
        "permno",
        "ticker",
        "comnam",
        "siccd",
        *US_EQUITIES_FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]
    result = result[keep].sort_values(["date", "permno"]).reset_index(drop=True)
    required_columns = [
        "RET_1",
        "RET_5",
        "RET_20",
        "VOLATILITY_20",
        "TURNOVER_20",
        "DOLLAR_VOLUME_20",
        "AMIHUD_20",
        "PRICE_TO_252_HIGH",
        "SIZE_LOG_MCAP",
        *TARGET_COLUMNS,
    ]
    result = result.dropna(subset=required_columns)
    return result
