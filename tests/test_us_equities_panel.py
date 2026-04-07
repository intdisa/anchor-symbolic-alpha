from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knowledge_guided_symbolic_alpha.dataio.us_equities_panel import (
    build_panel_from_tables,
    build_us_equities_panel,
    load_processed_us_equities_splits,
    load_us_equities_panel_config,
    load_us_equities_raw_tables,
)


def _write_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def _make_us_equities_tables() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=80)
    rows = []
    for permno, base_price, gvkey in [(10001, 20.0, "001000"), (10002, 35.0, "002000")]:
        for i, date in enumerate(dates):
            ret = 0.001 * np.sin(i / 5.0 + permno / 10000.0)
            rows.append(
                {
                    "permno": permno,
                    "permco": permno,
                    "date": date,
                    "ret": ret,
                    "retx": ret,
                    "dlret": np.nan,
                    "prc": base_price + 0.1 * i,
                    "vol": 250000 + 1000 * i,
                    "shrout": 1000000,
                    "bidlo": base_price - 0.5,
                    "askhi": base_price + 0.5,
                    "cfacpr": 1.0,
                    "cfacshr": 1.0,
                    "exchcd": 1,
                    "shrcd": 10,
                }
            )
    crsp_daily = pd.DataFrame(rows)
    crsp_names = pd.DataFrame(
        [
            {"permno": 10001, "ticker": "AAA", "ncusip": "000001", "comnam": "Alpha A", "namedt": pd.Timestamp("2010-01-01"), "nameendt": pd.NaT, "exchcd": 1, "shrcd": 10, "siccd": 3571},
            {"permno": 10002, "ticker": "BBB", "ncusip": "000002", "comnam": "Beta B", "namedt": pd.Timestamp("2010-01-01"), "nameendt": pd.NaT, "exchcd": 1, "shrcd": 10, "siccd": 3571},
        ]
    )
    ccm_link = pd.DataFrame(
        [
            {"gvkey": "001000", "lpermno": 10001, "linkdt": pd.Timestamp("2010-01-01"), "linkenddt": pd.NaT, "linktype": "LC", "linkprim": "P"},
            {"gvkey": "002000", "lpermno": 10002, "linkdt": pd.Timestamp("2010-01-01"), "linkenddt": pd.NaT, "linktype": "LC", "linkprim": "P"},
        ]
    )
    crsp_delisting = pd.DataFrame(
        [
            {"permno": 10001, "dlstdt": dates[40], "dlret": -0.5, "dlstcd": 500},
        ]
    )
    compustat_quarterly = pd.DataFrame(
        [
            {"gvkey": "001000", "datadate": pd.Timestamp("2019-12-31"), "rdq": pd.Timestamp("2020-02-15"), "fyearq": 2019, "fqtr": 4, "atq": 500.0, "ltq": 200.0, "ceqq": 300.0, "seq": 300.0, "saleq": 150.0, "niq": 15.0, "oiadpq": 20.0, "cheq": 60.0, "dlcq": 20.0, "dlttq": 40.0, "actq": 80.0, "lctq": 50.0, "rectq": 25.0, "invtq": 20.0, "cogsq": 70.0, "xsgaq": 15.0},
            {"gvkey": "002000", "datadate": pd.Timestamp("2019-12-31"), "rdq": pd.Timestamp("2020-02-15"), "fyearq": 2019, "fqtr": 4, "atq": 800.0, "ltq": 300.0, "ceqq": 500.0, "seq": 500.0, "saleq": 250.0, "niq": 30.0, "oiadpq": 35.0, "cheq": 70.0, "dlcq": 25.0, "dlttq": 55.0, "actq": 90.0, "lctq": 60.0, "rectq": 35.0, "invtq": 22.0, "cogsq": 90.0, "xsgaq": 18.0},
        ]
    )
    compustat_annual = pd.DataFrame(
        [
            {"gvkey": "001000", "datadate": pd.Timestamp("2019-12-31"), "fyear": 2019, "at": 480.0, "lt": 190.0, "ceq": 290.0, "seq": 290.0, "sale": 580.0, "ni": 48.0, "oiadp": 55.0, "capx": 20.0, "txditc": 5.0, "pstkrv": 0.0, "pstkl": 0.0, "pstk": 0.0},
            {"gvkey": "002000", "datadate": pd.Timestamp("2019-12-31"), "fyear": 2019, "at": 760.0, "lt": 280.0, "ceq": 480.0, "seq": 480.0, "sale": 790.0, "ni": 66.0, "oiadp": 72.0, "capx": 25.0, "txditc": 6.0, "pstkrv": 0.0, "pstkl": 0.0, "pstk": 0.0},
        ]
    )
    return {
        "crsp_daily": crsp_daily,
        "crsp_names": crsp_names,
        "ccm_link": ccm_link,
        "compustat_quarterly": compustat_quarterly,
        "compustat_annual": compustat_annual,
        "crsp_delisting": crsp_delisting,
    }


def test_build_panel_from_tables_generates_cross_sectional_targets() -> None:
    tables = _make_us_equities_tables()
    panel = build_panel_from_tables(
        tables,
        {
            "min_price": 1.0,
            "min_dollar_volume_20": 1.0,
            "min_history_days": 20,
            "min_cross_section": 2,
            "quarterly_lag_days": 0,
            "annual_lag_days": 0,
        },
    )

    assert not panel.empty
    assert {"RET_20", "BOOK_TO_MARKET_Q", "PROFITABILITY_A", "TARGET_RET_1", "TARGET_XS_RET_1"}.issubset(panel.columns)
    by_date_mean = panel.groupby("date")["TARGET_XS_RET_1"].mean().abs().max()
    assert by_date_mean < 1e-10
    trigger_date = pd.bdate_range("2020-01-01", periods=80)[40]
    previous_date = pd.bdate_range("2020-01-01", periods=80)[39]
    trigger_ret = tables["crsp_daily"].loc[
        (tables["crsp_daily"]["permno"] == 10001) & (tables["crsp_daily"]["date"] == trigger_date),
        "ret",
    ].iloc[0]
    trigger_dlret = tables["crsp_delisting"].loc[
        (tables["crsp_delisting"]["permno"] == 10001) & (tables["crsp_delisting"]["dlstdt"] == trigger_date),
        "dlret",
    ].iloc[0]
    expected_total = (1.0 + trigger_ret) * (1.0 + trigger_dlret) - 1.0
    realized_target = panel.loc[
        (panel["permno"] == 10001) & (panel["date"] == previous_date),
        "TARGET_RET_1",
    ].iloc[0]
    assert realized_target == pytest.approx(expected_total)


def test_us_equities_raw_table_loader_reads_expected_files(tmp_path: Path) -> None:
    tables = _make_us_equities_tables()
    root = tmp_path / "us_equities"
    _write_csv_gz(tables["crsp_daily"], root / "wrds" / "crsp_daily.csv.gz")
    _write_csv_gz(tables["crsp_names"], root / "wrds" / "crsp_names.csv.gz")
    _write_csv_gz(tables["ccm_link"], root / "wrds" / "ccm_link.csv.gz")
    _write_csv_gz(tables["compustat_quarterly"], root / "wrds" / "compustat_quarterly.csv.gz")
    _write_csv_gz(tables["compustat_annual"], root / "wrds" / "compustat_annual.csv.gz")
    _write_csv_gz(tables["crsp_delisting"], root / "wrds" / "crsp_delisting.csv.gz")

    loaded = load_us_equities_raw_tables(root)

    assert set(loaded) == set(tables)
    assert len(loaded["crsp_daily"]) == len(tables["crsp_daily"])


def test_us_equities_raw_table_loader_raises_clear_error_for_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required U.S. equities raw files"):
        load_us_equities_raw_tables(tmp_path / "us_equities")


def test_build_us_equities_panel_creates_non_empty_splits(tmp_path: Path) -> None:
    tables = _make_us_equities_tables()
    root = tmp_path / "us_equities"
    _write_csv_gz(tables["crsp_daily"], root / "wrds" / "crsp_daily.csv.gz")
    _write_csv_gz(tables["crsp_names"], root / "wrds" / "crsp_names.csv.gz")
    _write_csv_gz(tables["ccm_link"], root / "wrds" / "ccm_link.csv.gz")
    _write_csv_gz(tables["compustat_quarterly"], root / "wrds" / "compustat_quarterly.csv.gz")
    _write_csv_gz(tables["compustat_annual"], root / "wrds" / "compustat_annual.csv.gz")
    _write_csv_gz(tables["crsp_delisting"], root / "wrds" / "crsp_delisting.csv.gz")

    config_path = tmp_path / "us_equities_panel.yaml"
    config_path.write_text(
        """
us_equities_panel:
  raw_root: {raw_root}
  splits:
    train:
      start: "2020-02-01"
      end: "2020-03-10"
    valid:
      start: "2020-03-11"
      end: "2020-03-31"
    test:
      start: "2020-04-01"
      end: "2020-04-30"
  filters:
    min_price: 1.0
    min_dollar_volume_20: 1.0
    min_history_days: 20
    min_cross_section: 2
    quarterly_lag_days: 0
    annual_lag_days: 0
""".format(raw_root=root.as_posix()),
        encoding="utf-8",
    )

    config = load_us_equities_panel_config(config_path)
    bundle = build_us_equities_panel(raw_root=config.raw_root, config_path=config_path)

    assert bundle.feature_columns
    assert not bundle.splits.train.empty
    assert not bundle.splits.valid.empty
    assert not bundle.splits.test.empty


@pytest.mark.eval
def test_load_processed_us_equities_splits_reads_split_parquets(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    tables = _make_us_equities_tables()
    panel = build_panel_from_tables(
        tables,
        {
            "min_price": 1.0,
            "min_dollar_volume_20": 1.0,
            "min_history_days": 20,
            "min_cross_section": 2,
            "quarterly_lag_days": 0,
            "annual_lag_days": 0,
        },
    )
    split_root = tmp_path / "splits"
    split_root.mkdir(parents=True)
    panel.loc[panel["date"] <= pd.Timestamp("2020-03-10")].to_parquet(split_root / "train.parquet", index=False)
    panel.loc[(panel["date"] > pd.Timestamp("2020-03-10")) & (panel["date"] <= pd.Timestamp("2020-03-31"))].to_parquet(split_root / "valid.parquet", index=False)
    panel.loc[panel["date"] > pd.Timestamp("2020-03-31")].to_parquet(split_root / "test.parquet", index=False)

    bundle = load_processed_us_equities_splits(split_root)

    assert bundle.feature_columns
    assert not bundle.splits.train.empty
    assert not bundle.splits.valid.empty
    assert not bundle.splits.test.empty


def test_load_processed_us_equities_splits_reads_csv_fallback(tmp_path: Path) -> None:
    tables = _make_us_equities_tables()
    panel = build_panel_from_tables(
        tables,
        {
            "min_price": 1.0,
            "min_dollar_volume_20": 1.0,
            "min_history_days": 20,
            "min_cross_section": 2,
            "quarterly_lag_days": 0,
            "annual_lag_days": 0,
        },
    )
    split_root = tmp_path / "splits_csv"
    split_root.mkdir(parents=True)
    panel.loc[panel["date"] <= pd.Timestamp("2020-03-10")].to_csv(split_root / "train.csv.gz", index=False, compression="gzip")
    panel.loc[(panel["date"] > pd.Timestamp("2020-03-10")) & (panel["date"] <= pd.Timestamp("2020-03-31"))].to_csv(split_root / "valid.csv.gz", index=False, compression="gzip")
    panel.loc[panel["date"] > pd.Timestamp("2020-03-31")].to_csv(split_root / "test.csv.gz", index=False, compression="gzip")

    bundle = load_processed_us_equities_splits(split_root)

    assert bundle.feature_columns
    assert not bundle.splits.train.empty
    assert not bundle.splits.valid.empty
    assert not bundle.splits.test.empty


def test_load_processed_us_equities_splits_raises_clear_error_for_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Processed split root is missing"):
        load_processed_us_equities_splits(tmp_path / "us_equities" / "subsets" / "liquid150_2010_2025")


def test_load_processed_us_equities_splits_raises_clear_error_for_missing_split(tmp_path: Path) -> None:
    split_root = tmp_path / "splits"
    split_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "permno": [1],
            "ticker": ["AAA"],
            "comnam": ["Alpha"],
            "siccd": [3571],
            "TARGET_RET_1": [0.01],
            "TARGET_XS_RET_1": [0.01],
        }
    ).to_csv(split_root / "train.csv.gz", index=False, compression="gzip")
    with pytest.raises(FileNotFoundError, match="Missing split file for `valid`"):
        load_processed_us_equities_splits(split_root)
