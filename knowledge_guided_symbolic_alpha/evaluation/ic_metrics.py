from __future__ import annotations

import numpy as np
import pandas as pd


def _corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if method == "spearman":
        left = left.rank()
        right = right.rank()
        method = "pearson"
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return float("nan")
    value = left.corr(right, method=method)
    return float(value)


def _align(signal: pd.Series, target: pd.Series) -> pd.DataFrame:
    frame = (
        pd.concat({"signal": signal, "target": target}, axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if frame.empty:
        raise ValueError("Signal and target have no overlapping non-NaN observations.")
    return frame


def pearson_ic(signal: pd.Series, target: pd.Series) -> float:
    frame = _align(signal, target)
    return _corr(frame["signal"], frame["target"], method="pearson")


def rank_ic(signal: pd.Series, target: pd.Series) -> float:
    frame = _align(signal, target)
    return _corr(frame["signal"], frame["target"], method="spearman")


def _rolling_corr(frame: pd.DataFrame, window: int, method: str) -> pd.Series:
    left = frame["signal"]
    right = frame["target"]
    if method == "spearman":
        left = left.rank()
        right = right.rank()
    return left.rolling(window=window, min_periods=window).corr(right).replace([np.inf, -np.inf], np.nan)


def ic_summary(signal: pd.Series, target: pd.Series, window: int = 5) -> dict[str, float]:
    frame = _align(signal, target)
    rolling_pearson = _rolling_corr(frame, window=window, method="pearson").dropna()
    rolling_spearman = _rolling_corr(frame, window=window, method="spearman").dropna()
    pearson = _corr(frame["signal"], frame["target"], method="pearson")
    spearman = _corr(frame["signal"], frame["target"], method="spearman")
    pearson_icir = np.nan
    if not rolling_pearson.empty and float(rolling_pearson.std(ddof=0)) != 0.0:
        pearson_icir = float(rolling_pearson.mean() / rolling_pearson.std(ddof=0))
    spearman_icir = np.nan
    if not rolling_spearman.empty and float(rolling_spearman.std(ddof=0)) != 0.0:
        spearman_icir = float(rolling_spearman.mean() / rolling_spearman.std(ddof=0))
    return {
        "ic": pearson,
        "rank_ic": spearman,
        "icir": float(pearson_icir),
        "rank_icir": float(spearman_icir),
    }
