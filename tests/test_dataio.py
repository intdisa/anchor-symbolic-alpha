from pathlib import Path

import pandas as pd

from knowledge_guided_symbolic_alpha.dataio import (
    build_gold_dataset,
    load_fred_series_csv,
    load_yahoo_ohlcv_csv,
)


def test_load_yahoo_ohlcv_csv_parses_export_format(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(
        "\n".join(
            [
                "Price,Adj Close,Close,High,Low,Open,Volume",
                "Ticker,GC=F,GC=F,GC=F,GC=F,GC=F,GC=F",
                "Date,,,,,,",
                "2024-01-02,100,100,101,99,100,1000",
                "2024-01-03,101,101,102,100,100.5,1100",
            ]
        ),
        encoding="utf-8",
    )
    frame = load_yahoo_ohlcv_csv(path)
    assert list(frame.columns) == ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    assert frame.index[0] == pd.Timestamp("2024-01-02")
    assert float(frame.loc[pd.Timestamp("2024-01-03"), "Close"]) == 101.0


def test_load_fred_series_csv_parses_series(tmp_path: Path) -> None:
    path = tmp_path / "fred.csv"
    path.write_text(
        "\n".join(
            [
                "observation_date,CPIAUCSL",
                "2024-01-01,100.0",
                "2024-02-01,101.5",
            ]
        ),
        encoding="utf-8",
    )
    series = load_fred_series_csv(path)
    assert series.name == "CPIAUCSL"
    assert series.index[-1] == pd.Timestamp("2024-02-01")
    assert float(series.iloc[-1]) == 101.5


def test_build_gold_dataset_smoke() -> None:
    bundle = build_gold_dataset()
    assert set(bundle.feature_columns).issubset(bundle.frame.columns)
    assert bundle.target_column in bundle.frame.columns
    assert len(bundle.feature_columns) > 6
    assert {
        "GOLD_OPEN",
        "GOLD_HL_SPREAD",
        "CRUDE_OIL_OC_RET",
        "SP500_REALIZED_VOL_20",
    }.issubset(bundle.frame.columns)
    assert bundle.splits.train.index.min() >= pd.Timestamp("2000-01-01")
    assert bundle.splits.test.index.max() <= pd.Timestamp("2025-12-31")
    assert not bundle.splits.train.empty
    assert not bundle.splits.valid.empty
    assert not bundle.splits.test.empty
    assert bundle.splits.train["GOLD_REALIZED_VOL_20"].notna().any()
    assert bundle.splits.train["CRUDE_OIL_VOLUME_ZSCORE_20"].notna().any()
