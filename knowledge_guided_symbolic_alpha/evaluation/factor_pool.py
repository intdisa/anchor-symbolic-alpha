from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .role_profiles import normalize_role, resolve_role_profile


@dataclass
class FactorRecord:
    tokens: tuple[str, ...]
    canonical: str
    signal: pd.Series
    metrics: dict[str, float]
    role: str | None = None


class FactorPool:
    def __init__(self, max_size: int = 16) -> None:
        self.max_size = max_size
        self.records: list[FactorRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def canonicals(self) -> set[str]:
        return {record.canonical for record in self.records}

    def signals_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.concat({record.canonical: record.signal for record in self.records}, axis=1)

    def copy(self) -> "FactorPool":
        clone = FactorPool(max_size=self.max_size)
        clone.records = list(self.records)
        return clone

    def pool_score(self) -> float:
        return self._score_records(self.records)

    def trade_proxy_score(self) -> float:
        return self._trade_proxy_records(self.records)

    def weakest_index(self) -> int:
        if not self.records:
            raise IndexError("Cannot select the weakest factor from an empty pool.")
        return min(range(len(self.records)), key=lambda index: self.records[index].metrics["rank_ic"])

    def score_with(self, candidate: FactorRecord, replace_index: int | None = None) -> float:
        records = list(self.records)
        if replace_index is None:
            records.append(candidate)
        else:
            records[replace_index] = candidate
        return self._score_records(records)

    def trade_proxy_with(self, candidate: FactorRecord, replace_index: int | None = None) -> float:
        records = list(self.records)
        if replace_index is None:
            records.append(candidate)
        else:
            records[replace_index] = candidate
        return self._trade_proxy_records(records)

    def baseline_replacement_gain(self, candidate: FactorRecord, replace_index: int) -> float:
        replaced = self.records[replace_index]
        return self._baseline_replacement_utility(candidate) - self._baseline_replacement_utility(replaced)

    def add(self, record: FactorRecord) -> None:
        if len(self.records) >= self.max_size:
            raise ValueError("Factor pool is full.")
        self.records.append(record)

    def replace(self, index: int, record: FactorRecord) -> None:
        self.records[index] = record

    def _score_records(self, records: list[FactorRecord]) -> float:
        if not records:
            return 0.0
        rank_ic_values = [float(record.metrics["rank_ic"]) for record in records]
        finite_rank_ic = [value for value in rank_ic_values if np.isfinite(value)]
        if not finite_rank_ic:
            return float("-inf")
        rank_ic = float(np.mean(finite_rank_ic))
        complexity_penalty = float(
            np.mean(
                [
                    resolve_role_profile(record.role).pool_complexity_penalty_scale * len(record.tokens)
                    for record in records
                ]
            )
        )
        corr_penalty = self._corr_penalty(records)
        diversity_bonus = self._role_diversity_bonus(records, corr_penalty)
        return rank_ic - 0.1 * corr_penalty - complexity_penalty + diversity_bonus

    def _trade_proxy_records(self, records: list[FactorRecord]) -> float:
        if not records:
            return 0.0
        utilities = [self._record_trade_utility(record) for record in records]
        finite_utilities = [value for value in utilities if np.isfinite(value)]
        if not finite_utilities:
            return float("-inf")
        corr_penalty = self._corr_penalty(records)
        diversity_bonus = 0.5 * self._role_diversity_bonus(records, corr_penalty)
        return float(np.mean(finite_utilities) - 0.025 * corr_penalty + diversity_bonus)

    def _corr_penalty(self, records: list[FactorRecord]) -> float:
        signals = pd.concat({record.canonical: record.signal for record in records}, axis=1)
        corr_penalty = 0.0
        if signals.shape[1] > 1:
            ranked = signals.rank()
            ranked = ranked.loc[:, ranked.nunique(dropna=True) > 1]
            if ranked.shape[1] > 1:
                corr = ranked.corr(method="pearson").abs().to_numpy()
                off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
                finite_off_diag = off_diag[np.isfinite(off_diag)]
                if finite_off_diag.size:
                    corr_penalty = float(np.mean(finite_off_diag))
        return corr_penalty

    def _record_trade_utility(self, record: FactorRecord) -> float:
        metrics = record.metrics
        sharpe = float(metrics.get("sharpe", 0.0))
        annual_return = float(metrics.get("annual_return", 0.0))
        rank_ic = abs(float(metrics.get("rank_ic", 0.0)))
        turnover = max(0.0, float(metrics.get("turnover", 0.0)))
        drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        stability = float(metrics.get("stability_score", 0.0))
        if not np.isfinite(sharpe):
            sharpe = 0.0
        if not np.isfinite(annual_return):
            annual_return = 0.0
        if not np.isfinite(rank_ic):
            rank_ic = 0.0
        if not np.isfinite(turnover):
            turnover = 0.0
        if not np.isfinite(drawdown):
            drawdown = 0.0
        if not np.isfinite(stability):
            stability = 0.0
        normalized_role = normalize_role(record.role)
        profile = resolve_role_profile(record.role)

        utility = (
            0.04 * np.tanh(sharpe / 2.0)
            + 0.03 * np.tanh(annual_return / 0.20)
            + 0.20 * min(rank_ic, 0.15)
            + 0.20 * max(min(stability, 0.05), -0.05)
            - 0.015 * turnover
            - 0.030 * drawdown
            - 0.25 * profile.pool_complexity_penalty_scale * max(0, len(record.tokens) - 2)
        )
        if normalized_role == "target_flow" and turnover < 0.80 and drawdown < 0.35:
            utility += 0.004
        if normalized_role == "context":
            utility -= 0.004 * sum(token in {"MUL", "DIV"} for token in record.tokens)
        return float(utility)

    def _baseline_replacement_utility(self, record: FactorRecord) -> float:
        metrics = record.metrics
        sharpe = float(metrics.get("sharpe", 0.0))
        annual_return = float(metrics.get("annual_return", 0.0))
        rank_ic = abs(float(metrics.get("rank_ic", 0.0)))
        turnover = max(0.0, float(metrics.get("turnover", 0.0)))
        drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        stability = float(metrics.get("stability_score", 0.0))
        if not np.isfinite(sharpe):
            sharpe = 0.0
        if not np.isfinite(annual_return):
            annual_return = 0.0
        if not np.isfinite(rank_ic):
            rank_ic = 0.0
        if not np.isfinite(turnover):
            turnover = 0.0
        if not np.isfinite(drawdown):
            drawdown = 0.0
        if not np.isfinite(stability):
            stability = 0.0
        return float(
            0.08 * np.tanh(sharpe / 1.5)
            + 0.06 * np.tanh(annual_return / 0.15)
            + 0.03 * min(rank_ic, 0.05)
            + 0.04 * max(min(stability, 0.02), -0.02)
            - 0.010 * turnover
            - 0.025 * drawdown
        )

    def _role_diversity_bonus(self, records: list[FactorRecord], corr_penalty: float) -> float:
        target_roles = {
            normalized
            for record in records
            if (normalized := normalize_role(record.role)) in {"target_price", "target_flow"}
        }
        if len(target_roles) <= 1:
            return 0.0
        orthogonality = max(0.0, 1.0 - corr_penalty)
        return 0.008 * float(len(target_roles) - 1) * orthogonality
