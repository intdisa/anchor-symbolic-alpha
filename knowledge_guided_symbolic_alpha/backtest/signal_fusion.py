from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalFusionConfig:
    normalize: bool = True
    clip_zscore: float = 5.0


def _standardize(signal: pd.Series, clip_zscore: float) -> pd.Series:
    clean = signal.replace([np.inf, -np.inf], np.nan)
    mean = clean.mean()
    std = clean.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=signal.index)
    standardized = (clean - mean) / std
    return standardized.clip(-clip_zscore, clip_zscore).fillna(0.0)


def fuse_signals(
    signal_frame: pd.DataFrame,
    weights: dict[str, float] | None = None,
    config: SignalFusionConfig | None = None,
) -> pd.Series:
    config = config or SignalFusionConfig()
    if signal_frame.empty:
        return pd.Series(dtype=float)
    processed = signal_frame.copy()
    if config.normalize:
        for column in processed.columns:
            processed[column] = _standardize(processed[column], config.clip_zscore)
    if weights is None:
        weights = {column: 1.0 for column in processed.columns}
    weight_vector = np.array([weights.get(column, 0.0) for column in processed.columns], dtype=float)
    if not np.isfinite(weight_vector).all() or np.allclose(weight_vector, 0.0):
        weight_vector = np.ones(len(processed.columns), dtype=float)
    weight_vector = weight_vector / np.sum(np.abs(weight_vector))
    fused = processed.to_numpy() @ weight_vector
    return pd.Series(fused, index=processed.index, name="fused_signal")
