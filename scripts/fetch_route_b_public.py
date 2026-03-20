#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_ROOT = Path("data/raw/us_equities/public")
FF5_DAILY_ZIP_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FRED_SERIES = ("VIXCLS", "DGS2", "DGS10", "DFII10", "DTWEXBGS")
FRED_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public benchmark and macro datasets for the U.S. equities mainline.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-fama-french", action="store_true")
    parser.add_argument("--skip-fred", action="store_true")
    return parser.parse_args()


def fetch_url_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_fama_french_daily_from_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        first_name = archive.namelist()[0]
        raw_text = archive.read(first_name).decode("latin-1")

    rows: list[list[str]] = []
    for line in raw_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and re.fullmatch(r"\d{8}", parts[0]):
            rows.append(parts)

    if not rows:
        raise ValueError("No daily Fama/French rows found in archive.")

    column_count = len(rows[0])
    expected_columns = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    if column_count != len(expected_columns):
        raise ValueError(f"Unexpected Fama/French daily column count: {column_count}")

    frame = pd.DataFrame(rows, columns=expected_columns)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    for column in expected_columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0
    return frame.sort_values("date").reset_index(drop=True)


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    payload = fetch_url_bytes(FRED_URL_TEMPLATE.format(series_id=series_id))
    frame = pd.read_csv(io.BytesIO(payload))
    date_column = "DATE" if "DATE" in frame.columns else "observation_date"
    frame = frame.rename(columns={date_column: "date", series_id: series_id})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    return frame[["date", series_id]]


def fetch_fred_macro_daily(series_ids: tuple[str, ...] = FRED_SERIES) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for series_id in series_ids:
        frame = fetch_fred_series(series_id)
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
    if merged is None:
        raise ValueError("No FRED series requested.")
    return merged.sort_values("date").reset_index(drop=True)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def main() -> None:
    args = parse_args()
    if not args.skip_fama_french:
        ff5 = parse_fama_french_daily_from_zip(fetch_url_bytes(FF5_DAILY_ZIP_URL))
        ff_path = args.output_root / "fama_french_daily.csv.gz"
        write_frame(ff5, ff_path)
        print(f"wrote={ff_path} rows={len(ff5)}")
    if not args.skip_fred:
        fred = fetch_fred_macro_daily()
        fred_path = args.output_root / "fred_macro_daily.csv.gz"
        write_frame(fred, fred_path)
        print(f"wrote={fred_path} rows={len(fred)}")


if __name__ == "__main__":
    main()
