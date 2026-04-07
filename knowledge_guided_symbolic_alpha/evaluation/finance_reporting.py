from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .risk_metrics import annual_return, max_drawdown, sharpe_ratio


DEFAULT_FF5_PATH = Path("data/raw/us_equities/public/fama_french_daily.csv.gz")


def load_fama_french_factors(path: Path | str = DEFAULT_FF5_PATH) -> pd.DataFrame:
    factors = pd.read_csv(path)
    factors["date"] = pd.to_datetime(factors["date"])
    return factors.sort_values("date").reset_index(drop=True)


def returns_to_frame(returns: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"strategy_returns": pd.Series(returns, copy=True)})
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    else:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
        frame["date"] = pd.to_datetime(frame["date"])
    return frame[["date", "strategy_returns"]]


def merge_returns_with_factors(returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    returns_frame = returns_to_frame(returns)
    merged = returns_frame.merge(factors, on="date", how="left")
    return merged.dropna(subset=["strategy_returns", "RF", "MKT_RF", "SMB", "HML", "RMW", "CMA"]).reset_index(drop=True)


def compute_significance_metrics(returns: pd.Series, factors: pd.DataFrame, *, maxlags: int = 5) -> dict[str, float]:
    merged = merge_returns_with_factors(returns, factors)
    excess_returns = merged["strategy_returns"] - merged["RF"]
    mean_design = pd.DataFrame({"const": np.ones(len(excess_returns))}, index=excess_returns.index)
    mean_fit = sm.OLS(excess_returns, mean_design).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    ff5_design = sm.add_constant(merged[["MKT_RF", "SMB", "HML", "RMW", "CMA"]], has_constant="add")
    ff5_fit = sm.OLS(excess_returns, ff5_design).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "nw_t": float(mean_fit.tvalues["const"]),
        "nw_p": float(mean_fit.pvalues["const"]),
        "ff5_alpha_ann": float(ff5_fit.params["const"] * 252.0),
        "ff5_alpha_t": float(ff5_fit.tvalues["const"]),
        "ff5_alpha_p": float(ff5_fit.pvalues["const"]),
    }


def cost_adjusted_returns(returns: pd.Series, turnover: float, cost_bps: float) -> pd.Series:
    return pd.Series(returns, copy=True) - (float(cost_bps) / 10000.0) * float(turnover)


def summarize_returns(returns: pd.Series) -> dict[str, float]:
    series = pd.Series(returns, copy=True)
    return {
        "sharpe": float(sharpe_ratio(series)),
        "annual_return": float(annual_return(series)),
        "max_drawdown": float(max_drawdown(series)),
    }


def signal_coverage_metrics(signal: pd.Series, dates: pd.Series) -> dict[str, float]:
    series = pd.Series(signal, copy=True)
    grouped_dates = pd.to_datetime(pd.Series(dates, index=series.index))
    non_null = series.notna()
    active_per_day = non_null.groupby(grouped_dates).sum()
    active_fraction = float((active_per_day > 0).mean()) if len(active_per_day) else 0.0
    active_counts = active_per_day[active_per_day > 0]
    return {
        "signal_non_null_fraction": float(non_null.mean()) if len(non_null) else 0.0,
        "active_date_fraction": active_fraction,
        "median_active_names_per_day": float(active_counts.median()) if len(active_counts) else 0.0,
    }


def return_concentration_metrics(returns: pd.Series, *, top_fraction: float = 0.01) -> dict[str, float]:
    series = pd.Series(returns, copy=True).dropna().astype(float)
    if series.empty:
        return {"pnl_top_1pct_day_share": 0.0}
    count = max(1, int(np.ceil(len(series) * float(top_fraction))))
    abs_values = series.abs().sort_values(ascending=False)
    numerator = float(abs_values.iloc[:count].sum())
    denominator = float(abs_values.sum())
    share = 0.0 if denominator <= 1e-12 else numerator / denominator
    return {"pnl_top_1pct_day_share": share}
