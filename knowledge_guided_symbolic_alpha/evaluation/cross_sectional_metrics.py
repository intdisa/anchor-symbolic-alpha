from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_metrics import annual_return, max_drawdown, sharpe_ratio


def _align_panel(
    signal: pd.Series,
    target: pd.Series,
    dates: pd.Series,
    entities: pd.Series | None = None,
) -> pd.DataFrame:
    payload = {"signal": signal, "target": target, "date": dates}
    if entities is not None:
        payload["entity"] = entities
    frame = pd.concat(payload, axis=1).replace([np.inf, -np.inf], np.nan).dropna(subset=["signal", "target", "date"])
    if frame.empty:
        raise ValueError("Signal and target have no overlapping non-NaN panel observations.")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _datewise_corr(frame: pd.DataFrame, method: str) -> pd.Series:
    def corr_for_group(group: pd.DataFrame) -> float:
        if group["signal"].nunique(dropna=True) <= 1 or group["target"].nunique(dropna=True) <= 1:
            return float("nan")
        left = group["signal"]
        right = group["target"]
        if method == "spearman":
            left = left.rank()
            right = right.rank()
        return float(left.corr(right, method="pearson"))

    return frame.groupby("date", sort=True).apply(corr_for_group)


def cross_sectional_rank_ic(signal: pd.Series, target: pd.Series, dates: pd.Series) -> float:
    frame = _align_panel(signal, target, dates)
    by_date = _datewise_corr(frame, method="spearman").dropna()
    if by_date.empty:
        return float("nan")
    return float(by_date.mean())


def cross_sectional_ic(signal: pd.Series, target: pd.Series, dates: pd.Series) -> float:
    frame = _align_panel(signal, target, dates)
    by_date = _datewise_corr(frame, method="pearson").dropna()
    if by_date.empty:
        return float("nan")
    return float(by_date.mean())


def cross_sectional_ic_summary(signal: pd.Series, target: pd.Series, dates: pd.Series) -> dict[str, float]:
    frame = _align_panel(signal, target, dates)
    ic_by_date = _datewise_corr(frame, method="pearson").dropna()
    rank_ic_by_date = _datewise_corr(frame, method="spearman").dropna()
    ic_mean = float(ic_by_date.mean()) if not ic_by_date.empty else float("nan")
    rank_ic_mean = float(rank_ic_by_date.mean()) if not rank_ic_by_date.empty else float("nan")
    icir = float("nan")
    rank_icir = float("nan")
    if not ic_by_date.empty and float(ic_by_date.std(ddof=0)) != 0.0:
        icir = float(ic_by_date.mean() / ic_by_date.std(ddof=0))
    if not rank_ic_by_date.empty and float(rank_ic_by_date.std(ddof=0)) != 0.0:
        rank_icir = float(rank_ic_by_date.mean() / rank_ic_by_date.std(ddof=0))
    return {
        "ic": ic_mean,
        "rank_ic": rank_ic_mean,
        "icir": icir,
        "rank_icir": rank_icir,
    }


def cross_sectional_weights(
    signal: pd.Series,
    dates: pd.Series,
    entities: pd.Series,
    *,
    quantile: float = 0.2,
) -> pd.Series:
    frame = _align_panel(signal, signal, dates, entities=entities)

    def build_weights(group: pd.DataFrame) -> pd.Series:
        rank = group["signal"].rank(pct=True, method="average")
        long_mask = rank >= (1.0 - quantile)
        short_mask = rank <= quantile
        weights = pd.Series(0.0, index=group.index)
        long_count = int(long_mask.sum())
        short_count = int(short_mask.sum())
        if long_count > 0:
            weights.loc[long_mask] = 0.5 / long_count
        if short_count > 0:
            weights.loc[short_mask] = -0.5 / short_count
        return weights

    weights = frame.groupby("date", sort=True, group_keys=False).apply(build_weights)
    weights = weights.reindex(signal.index).fillna(0.0)
    return weights


def cross_sectional_long_short_returns(
    signal: pd.Series,
    target: pd.Series,
    dates: pd.Series,
    entities: pd.Series,
    *,
    quantile: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    weights = cross_sectional_weights(signal, dates, entities, quantile=quantile)
    frame = _align_panel(signal, target, dates, entities=entities)
    frame["weight"] = weights.loc[frame.index]
    daily_returns = (frame["weight"] * frame["target"]).groupby(frame["date"], sort=True).sum()
    return daily_returns, weights


def cross_sectional_turnover(weights: pd.Series, dates: pd.Series, entities: pd.Series) -> float:
    frame = pd.concat({"weight": weights, "date": dates, "entity": entities}, axis=1).dropna(subset=["date", "entity"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["entity", "date"]).reset_index(drop=True)
    frame["prev_weight"] = frame.groupby("entity", sort=False)["weight"].shift(1).fillna(0.0)
    frame["turn"] = (frame["weight"] - frame["prev_weight"]).abs()
    daily_turnover = frame.groupby("date", sort=True)["turn"].sum() * 0.5
    if daily_turnover.empty:
        return float("nan")
    return float(daily_turnover.mean())


def cross_sectional_risk_summary(
    signal: pd.Series,
    target: pd.Series,
    dates: pd.Series,
    entities: pd.Series,
    *,
    quantile: float = 0.2,
) -> dict[str, float]:
    returns, weights = cross_sectional_long_short_returns(signal, target, dates, entities, quantile=quantile)
    return {
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "annual_return": annual_return(returns),
        "turnover": cross_sectional_turnover(weights, dates, entities),
    }


def cross_sectional_stability_summary(
    signal: pd.Series,
    target: pd.Series,
    dates: pd.Series,
    entities: pd.Series,
    *,
    quantile: float = 0.2,
    window_count: int = 4,
) -> dict[str, float]:
    frame = _align_panel(signal, target, dates, entities=entities)
    rank_ic_by_date = _datewise_corr(frame, method="spearman").dropna()
    daily_returns, _ = cross_sectional_long_short_returns(signal, target, dates, entities, quantile=quantile)
    daily_returns = daily_returns.dropna()
    aligned_dates = pd.Index(sorted(set(rank_ic_by_date.index).intersection(set(daily_returns.index))))
    if aligned_dates.empty:
        return {
            "rank_ic_window_min": float("nan"),
            "rank_ic_window_std": float("nan"),
            "ls_return_window_min": float("nan"),
            "ls_return_window_std": float("nan"),
            "rank_ic_window_positive_frac": float("nan"),
            "ls_return_window_positive_frac": float("nan"),
            "stability_score": float("nan"),
        }
    windows = [pd.Index(chunk) for chunk in np.array_split(aligned_dates.to_numpy(), min(window_count, len(aligned_dates))) if len(chunk)]
    rank_window_means = np.asarray([float(rank_ic_by_date.loc[idx].mean()) for idx in windows], dtype=float)
    return_window_means = np.asarray([float(daily_returns.loc[idx].mean()) for idx in windows], dtype=float)
    rank_std = float(rank_window_means.std(ddof=0)) if rank_window_means.size else float("nan")
    return_std = float(return_window_means.std(ddof=0)) if return_window_means.size else float("nan")
    rank_min = float(rank_window_means.min()) if rank_window_means.size else float("nan")
    return_min = float(return_window_means.min()) if return_window_means.size else float("nan")
    stability_score = float(
        4.0 * rank_min
        + 20.0 * return_min
        - 0.5 * (0.0 if not np.isfinite(rank_std) else rank_std)
        - 5.0 * (0.0 if not np.isfinite(return_std) else return_std)
    )
    return {
        "rank_ic_window_min": rank_min,
        "rank_ic_window_std": rank_std,
        "ls_return_window_min": return_min,
        "ls_return_window_std": return_std,
        "rank_ic_window_positive_frac": float(np.mean(rank_window_means > 0.0)),
        "ls_return_window_positive_frac": float(np.mean(return_window_means > 0.0)),
        "stability_score": stability_score,
    }
