from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FoldReport:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_formulas: tuple[str, ...]
    weights: dict[str, float]
    train_rank_ic: float
    test_rank_ic: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class WalkForwardReport:
    folds: list[FoldReport]
    aggregate_metrics: dict[str, float]
    returns: pd.Series
