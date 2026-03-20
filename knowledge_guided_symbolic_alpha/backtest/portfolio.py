from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation.risk_metrics import annual_return, max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class PortfolioConfig:
    transaction_cost_bps: float = 5.0
    signal_threshold: float = 0.0
    leverage: float = 1.0


def generate_positions(signal: pd.Series, config: PortfolioConfig | None = None) -> pd.Series:
    config = config or PortfolioConfig()
    clean = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positions = np.where(clean > config.signal_threshold, 1.0, 0.0)
    positions = np.where(clean < -config.signal_threshold, -1.0, positions)
    return pd.Series(config.leverage * positions, index=signal.index, name="positions")


def portfolio_returns(
    signal: pd.Series,
    asset_returns: pd.Series,
    config: PortfolioConfig | None = None,
) -> pd.Series:
    config = config or PortfolioConfig()
    positions = generate_positions(signal, config=config)
    shifted = positions.shift(1).fillna(0.0)
    turnover = positions.diff().abs().fillna(0.0)
    costs = turnover * (config.transaction_cost_bps / 10_000.0)
    returns = shifted * asset_returns.fillna(0.0) - costs
    returns.name = "strategy_returns"
    return returns


def portfolio_summary(
    signal: pd.Series,
    asset_returns: pd.Series,
    config: PortfolioConfig | None = None,
) -> dict[str, float]:
    config = config or PortfolioConfig()
    returns = portfolio_returns(signal, asset_returns, config=config)
    positions = generate_positions(signal, config=config)
    turnover = float(positions.diff().abs().fillna(0.0).mean())
    return {
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "annual_return": annual_return(returns),
        "turnover": turnover,
    }
