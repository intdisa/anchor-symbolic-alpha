from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeContext:
    regime: str
    diagnostics: dict[str, float]


class RegimeController:
    def __init__(self, lookback: int = 63) -> None:
        self.lookback = lookback

    def infer(self, frame: pd.DataFrame) -> RegimeContext:
        if frame.empty:
            return RegimeContext(regime="BALANCED", diagnostics={})
        recent = frame.tail(self.lookback)
        latest = recent.iloc[-1]

        latest_vix = float(latest["VIX"]) if "VIX" in latest else float("nan")
        vix_threshold = self._volatility_threshold(recent["VIX"]) if "VIX" in recent else float("inf")
        tnx_momentum = self._momentum(recent["TNX"]) if "TNX" in recent else 0.0
        cpi_momentum = self._momentum(recent["CPI"]) if "CPI" in recent else 0.0
        dxy_momentum = self._momentum(recent["DXY"]) if "DXY" in recent else 0.0

        if latest_vix >= vix_threshold:
            regime = "HIGH_VOLATILITY"
        elif tnx_momentum >= 0.25:
            regime = "RATE_HIKING"
        elif cpi_momentum >= 0.25:
            regime = "INFLATION_SHOCK"
        elif dxy_momentum >= 1.0:
            regime = "USD_STRENGTH"
        else:
            regime = "BALANCED"
        diagnostics = {
            "vix_threshold": float(vix_threshold),
            "tnx_momentum": float(tnx_momentum),
            "cpi_momentum": float(cpi_momentum),
            "dxy_momentum": float(dxy_momentum),
            "latest_vix": latest_vix,
        }
        return RegimeContext(regime=regime, diagnostics=diagnostics)

    def _percentile(self, series: pd.Series) -> float:
        ranked = series.rank(pct=True)
        return float(ranked.iloc[-1])

    def _volatility_threshold(self, series: pd.Series) -> float:
        clean = series.dropna()
        if clean.empty:
            return float("inf")
        mean = float(clean.mean())
        std = float(clean.std(ddof=0))
        return max(25.0, mean + std)

    def _momentum(self, series: pd.Series) -> float:
        clean = series.dropna()
        if len(clean) < 2:
            return 0.0
        start = float(clean.iloc[0])
        end = float(clean.iloc[-1])
        if not np.isfinite(start) or not np.isfinite(end):
            return 0.0
        return end - start
