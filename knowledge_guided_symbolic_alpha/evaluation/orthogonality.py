from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_correlation(signal: pd.Series, other: pd.Series, method: str = "spearman") -> float:
    frame = pd.concat({"left": signal, "right": other}, axis=1).dropna()
    if frame.empty:
        return float("nan")
    left = frame["left"]
    right = frame["right"]
    if method == "spearman":
        left = left.rank()
        right = right.rank()
        method = "pearson"
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return float("nan")
    return float(left.corr(right, method=method))


def max_abs_correlation(
    signal: pd.Series,
    others: pd.DataFrame,
    method: str = "spearman",
) -> float:
    if others.empty:
        return 0.0
    values = [pairwise_correlation(signal, others[column], method=method) for column in others.columns]
    finite = [abs(value) for value in values if np.isfinite(value)]
    if not finite:
        return 0.0
    return float(max(finite))
