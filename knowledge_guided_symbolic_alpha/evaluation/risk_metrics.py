from __future__ import annotations

import numpy as np
import pandas as pd


def _positions_from_signal(signal: pd.Series) -> pd.Series:
    positions = np.sign(signal).astype(float)
    return positions.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def strategy_returns(signal: pd.Series, asset_returns: pd.Series) -> pd.Series:
    positions = _positions_from_signal(signal).shift(1).fillna(0.0)
    return positions * asset_returns.fillna(0.0)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    if clean.empty or float(clean.std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * clean.mean() / clean.std(ddof=0))


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    gross_returns = 1.0 + clean
    if (gross_returns <= 0.0).any():
        return float(clean.mean() * periods_per_year)
    equity = (1.0 + clean).cumprod()
    years = max(len(clean) / periods_per_year, 1.0 / periods_per_year)
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    equity = (1.0 + clean).cumprod()
    peaks = equity.cummax()
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min())


def turnover(signal: pd.Series) -> float:
    positions = _positions_from_signal(signal)
    return float(positions.diff().abs().fillna(0.0).mean())


def risk_summary(signal: pd.Series, asset_returns: pd.Series) -> dict[str, float]:
    pnl = strategy_returns(signal, asset_returns)
    return {
        "sharpe": sharpe_ratio(pnl),
        "max_drawdown": max_drawdown(pnl),
        "annual_return": annual_return(pnl),
        "turnover": turnover(signal),
    }
