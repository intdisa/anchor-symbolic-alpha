from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..domain.feature_registry import FEATURE_REGISTRY
from .loaders import load_fred_series_csv, load_yahoo_ohlcv_csv


DEFAULT_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_RAW_ROOT = Path("knowledge_guided_symbolic_alpha/data")
FEATURE_COLUMNS = tuple(
    name
    for name in FEATURE_REGISTRY
    if name in {"CPI", "TNX", "VIX", "DXY"}
    or name.startswith("GOLD_")
    or name.startswith("CRUDE_OIL_")
    or name.startswith("SP500_")
)
TARGET_COLUMN = "TARGET_GOLD_FWD_RET_1"


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    splits: DatasetSplits


def load_data_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload["data"]


def build_feature_frame(
    raw_root: str | Path | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    config = config or load_data_config()
    raw_root = Path(raw_root or config.get("raw_root", DEFAULT_RAW_ROOT))
    files = config["files"]

    gold = load_yahoo_ohlcv_csv(raw_root / files["gold"])
    crude = load_yahoo_ohlcv_csv(raw_root / files["crude_oil"])
    tnx = load_yahoo_ohlcv_csv(raw_root / files["tnx"])
    vix = load_yahoo_ohlcv_csv(raw_root / files["vix"])
    dxy = load_yahoo_ohlcv_csv(raw_root / files["dxy"])
    sp500 = load_yahoo_ohlcv_csv(raw_root / files["sp500"])
    cpi = load_fred_series_csv(raw_root / files["cpi"])

    frame = pd.DataFrame(index=gold.index)
    _add_asset_features(frame, "GOLD", gold)
    _add_asset_features(frame, "CRUDE_OIL", crude)
    _add_asset_features(frame, "SP500", sp500)

    frame["CPI"] = cpi.reindex(frame.index, method="ffill")
    frame["TNX"] = tnx["Close"].reindex(frame.index).ffill()
    frame["VIX"] = vix["Close"].reindex(frame.index).ffill()
    frame["DXY"] = dxy["Close"].reindex(frame.index).ffill()

    frame["GOLD_RET_1"] = frame["GOLD_CLOSE"].pct_change()
    frame["CRUDE_OIL_RET_1"] = frame["CRUDE_OIL_CLOSE"].pct_change()
    frame["SP500_RET_1"] = frame["SP500_CLOSE"].pct_change()
    frame[TARGET_COLUMN] = frame["GOLD_RET_1"].shift(-1)
    frame["TARGET_CRUDE_OIL_FWD_RET_1"] = frame["CRUDE_OIL_RET_1"].shift(-1)
    frame["TARGET_SP500_FWD_RET_1"] = frame["SP500_RET_1"].shift(-1)

    required_columns = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    frame = frame.dropna(subset=required_columns).sort_index()
    frame.index.name = "Date"
    return frame


def _add_asset_features(frame: pd.DataFrame, prefix: str, asset_frame: pd.DataFrame) -> None:
    close = asset_frame["Close"].reindex(frame.index).ffill()
    open_ = asset_frame["Open"].reindex(frame.index).ffill()
    high = asset_frame["High"].reindex(frame.index).ffill()
    low = asset_frame["Low"].reindex(frame.index).ffill()
    volume = asset_frame["Volume"].reindex(frame.index).ffill()

    frame[f"{prefix}_CLOSE"] = close
    frame[f"{prefix}_OPEN"] = open_
    frame[f"{prefix}_HIGH"] = high
    frame[f"{prefix}_LOW"] = low
    frame[f"{prefix}_VOLUME"] = volume

    close_safe = close.replace(0.0, np.nan)
    open_safe = open_.replace(0.0, np.nan)
    previous_close = close.shift(1).replace(0.0, np.nan)
    returns = close.pct_change()
    volume_mean_20 = volume.rolling(window=20, min_periods=20).mean()
    volume_std_20 = volume.rolling(window=20, min_periods=20).std(ddof=0).replace(0.0, np.nan)

    frame[f"{prefix}_HL_SPREAD"] = (high - low) / close_safe
    frame[f"{prefix}_OC_RET"] = (close - open_) / open_safe
    frame[f"{prefix}_GAP_RET"] = open_ / previous_close - 1.0
    frame[f"{prefix}_REALIZED_VOL_5"] = returns.rolling(window=5, min_periods=5).std(ddof=0)
    frame[f"{prefix}_REALIZED_VOL_20"] = returns.rolling(window=20, min_periods=20).std(ddof=0)
    frame[f"{prefix}_VOLUME_ZSCORE_20"] = (volume - volume_mean_20) / volume_std_20


def split_frame_by_dates(frame: pd.DataFrame, config: dict | None = None) -> DatasetSplits:
    config = config or load_data_config()
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, bounds in config["splits"].items():
        start = pd.Timestamp(bounds["start"])
        end = pd.Timestamp(bounds["end"])
        split = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
        if split.empty:
            raise ValueError(f"Split {split_name!r} is empty for bounds {bounds!r}.")
        split_frames[split_name] = split
    return DatasetSplits(
        train=split_frames["train"],
        valid=split_frames["valid"],
        test=split_frames["test"],
    )


def build_gold_dataset(
    raw_root: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> DatasetBundle:
    config = load_data_config(config_path)
    frame = build_feature_frame(raw_root=raw_root, config=config)
    splits = split_frame_by_dates(frame, config=config)
    return DatasetBundle(
        frame=frame,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        splits=splits,
    )
