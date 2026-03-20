from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_yahoo_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=[1, 2], parse_dates=["Price"])
    frame = frame.rename(columns={"Price": "Date"}).set_index("Date").sort_index()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index.name = "Date"
    return frame


def load_fred_series_csv(
    path: str | Path,
    date_column: str = "observation_date",
) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=[date_column]).sort_values(date_column)
    value_columns = [column for column in frame.columns if column != date_column]
    if len(value_columns) != 1:
        raise ValueError(f"Expected exactly one value column in {path!s}, got {value_columns!r}.")
    value_column = value_columns[0]
    series = pd.to_numeric(frame[value_column], errors="coerce")
    series.index = pd.DatetimeIndex(frame[date_column], name="Date")
    series.name = value_column
    return series.sort_index()
