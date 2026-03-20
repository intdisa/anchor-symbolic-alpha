from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.portfolio import PortfolioConfig, portfolio_summary
from ..backtest.signal_fusion import SignalFusionConfig, fuse_signals
from .panel_dispatch import is_cross_sectional_frame, score_signal_metrics
from ..training.reward_shaping import PoolRewardShaper
from .factor_pool import FactorPool, FactorRecord
from .pool_scoring import CandidatePoolPreview, preview_candidate_on_dataset, rescore_pool_on_dataset
from .role_profiles import adapt_role_profile, resolve_role_profile


@dataclass(frozen=True)
class DistributionalCriticEstimate:
    expected_gain: float
    risk_adjusted_gain: float
    quantiles: tuple[float, float, float]
    uncertainty: float
    accepted: bool
    reason: str
    train_reward: float
    validation_gain: float
    trade_proxy_gain: float
    walk_forward_proxy_gain: float
    baseline_walk_forward_proxy: float
    new_walk_forward_proxy: float
    preview: CandidatePoolPreview | None


class DistributionalCollectionCritic:
    def __init__(
        self,
        reward_shaper: PoolRewardShaper | None = None,
        signal_fusion_config: SignalFusionConfig | None = None,
        portfolio_config: PortfolioConfig | None = None,
    ) -> None:
        self.reward_shaper = reward_shaper or PoolRewardShaper()
        self.signal_fusion_config = signal_fusion_config or SignalFusionConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()

    def estimate(
        self,
        formula: str | list[str] | tuple[str, ...],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        role: str | None = None,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> DistributionalCriticEstimate:
        train_outcome = self.reward_shaper.shape(
            formula,
            data,
            target,
            pool,
            commit=False,
            role=role,
        )
        profile = adapt_role_profile(
            resolve_role_profile(role),
            role,
            cross_sectional=is_cross_sectional_frame(validation_data if validation_data is not None else data),
        )
        scenario_values = [float(train_outcome.clipped_reward)]
        preview = None
        validation_gain = 0.0
        trade_proxy_gain = float(train_outcome.decision.trade_proxy_gain)
        baseline_walk_forward_proxy = 0.0
        new_walk_forward_proxy = 0.0
        walk_forward_proxy_gain = 0.0
        accepted = train_outcome.decision.accepted
        reason = train_outcome.decision.reason
        if validation_data is not None and validation_target is not None:
            validation_pool = rescore_pool_on_dataset(pool, validation_data, validation_target)
            preview = preview_candidate_on_dataset(
                formula,
                pool,
                validation_data,
                validation_target,
                role=role,
                min_abs_rank_ic=profile.resolved_preview_min_abs_rank_ic,
                max_correlation=profile.resolved_preview_max_correlation,
                replacement_margin=profile.replacement_margin,
                min_validation_marginal_gain=profile.preview_min_validation_marginal_gain,
                min_trade_proxy_gain=profile.resolved_preview_min_trade_proxy_gain,
            )
            validation_gain = float(
                preview.marginal_gain + profile.reward_trade_proxy_scale * preview.trade_proxy_gain
            )
            baseline_walk_forward_proxy = self._library_walk_forward_proxy(
                validation_pool.records,
                validation_data,
                validation_target,
            )
            if is_cross_sectional_frame(validation_data):
                baseline_walk_forward_proxy = self._windowed_walk_forward_proxy(
                    validation_pool.records,
                    validation_data,
                    validation_target,
                    fallback=baseline_walk_forward_proxy,
                )
            if preview.record is not None:
                candidate_records = list(validation_pool.records)
                if preview.replaced_canonical is not None:
                    candidate_records = [
                        preview.record if record.canonical == preview.replaced_canonical else record
                        for record in candidate_records
                    ]
                else:
                    candidate_records.append(preview.record)
                new_walk_forward_proxy = self._library_walk_forward_proxy(
                    candidate_records,
                    validation_data,
                    validation_target,
                )
                if is_cross_sectional_frame(validation_data):
                    new_walk_forward_proxy = self._windowed_walk_forward_proxy(
                        candidate_records,
                        validation_data,
                        validation_target,
                        fallback=new_walk_forward_proxy,
                    )
                walk_forward_proxy_gain = new_walk_forward_proxy - baseline_walk_forward_proxy
            scenario_values.append(validation_gain)
            scenario_values.append(walk_forward_proxy_gain)
            scenario_values.append(0.6 * float(train_outcome.clipped_reward) + 0.4 * validation_gain)
            scenario_values.append(
                0.35 * float(train_outcome.clipped_reward)
                + 0.30 * validation_gain
                + 0.35 * walk_forward_proxy_gain
            )
            trade_proxy_gain = float(0.5 * (trade_proxy_gain + preview.trade_proxy_gain))
            accepted = accepted and preview.accepted
            if not preview.accepted:
                reason = preview.reason
                validation_gain -= 0.05
                walk_forward_proxy_gain -= 0.05

        quantiles = tuple(float(value) for value in np.quantile(np.asarray(scenario_values), [0.1, 0.5, 0.9]))
        uncertainty = float(quantiles[2] - quantiles[0])
        risk_adjusted_gain = float(quantiles[1] - 0.35 * uncertainty)
        return DistributionalCriticEstimate(
            expected_gain=float(np.mean(scenario_values)),
            risk_adjusted_gain=risk_adjusted_gain,
            quantiles=quantiles,
            uncertainty=uncertainty,
            accepted=accepted,
            reason=reason,
            train_reward=float(train_outcome.clipped_reward),
            validation_gain=validation_gain,
            trade_proxy_gain=trade_proxy_gain,
            walk_forward_proxy_gain=walk_forward_proxy_gain,
            baseline_walk_forward_proxy=baseline_walk_forward_proxy,
            new_walk_forward_proxy=new_walk_forward_proxy,
            preview=preview,
        )

    def _windowed_walk_forward_proxy(
        self,
        records,
        data: pd.DataFrame,
        asset_returns: pd.Series,
        *,
        fallback: float,
        windows: int = 4,
    ) -> float:
        if not records or not is_cross_sectional_frame(data):
            return fallback
        unique_dates = pd.Index(sorted(pd.Index(data["date"]).drop_duplicates()))
        if len(unique_dates) < 120:
            return fallback
        window_dates = [chunk for chunk in np.array_split(unique_dates.to_numpy(), min(windows, len(unique_dates))) if len(chunk) > 0]
        proxies: list[float] = []
        for chunk in window_dates:
            window_frame = data[data["date"].isin(chunk)]
            if window_frame.empty:
                continue
            window_returns = asset_returns.reindex(window_frame.index).fillna(0.0)
            window_records = [
                FactorRecord(
                    tokens=record.tokens,
                    canonical=record.canonical,
                    signal=record.signal.reindex(window_frame.index),
                    metrics=record.metrics,
                    role=record.role,
                )
                for record in records
            ]
            proxies.append(self._library_walk_forward_proxy(window_records, window_frame, window_returns))
        if not proxies:
            return fallback
        proxy_array = np.asarray(proxies, dtype=float)
        mean_proxy = float(np.mean(proxy_array))
        stable_proxy = mean_proxy - 0.25 * float(np.std(proxy_array))
        return float(0.35 * fallback + 0.65 * stable_proxy)

    def _library_walk_forward_proxy(
        self,
        records,
        data: pd.DataFrame,
        asset_returns: pd.Series,
    ) -> float:
        if not records:
            return 0.0
        signal_frame = pd.concat({record.canonical: record.signal for record in records}, axis=1)
        signal_frame = signal_frame.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if signal_frame.empty:
            return 0.0
        aligned_returns = asset_returns.reindex(signal_frame.index).fillna(0.0)
        aligned_data = data.loc[signal_frame.index]
        weights = self._signal_weights(records)
        fused_signal = fuse_signals(signal_frame, weights=weights, config=self.signal_fusion_config)
        if is_cross_sectional_frame(aligned_data):
            metrics = score_signal_metrics(fused_signal, aligned_data, aligned_returns)
        else:
            metrics = portfolio_summary(fused_signal, aligned_returns, config=self.portfolio_config)
            metrics.update(score_signal_metrics(fused_signal, aligned_data, aligned_returns))
        sharpe = float(metrics.get("sharpe", 0.0))
        annual = float(metrics.get("annual_return", 0.0))
        turnover = max(0.0, float(metrics.get("turnover", 0.0)))
        drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        rank_ic = abs(float(metrics.get("rank_ic", 0.0)))
        if not np.isfinite(sharpe):
            sharpe = 0.0
        if not np.isfinite(annual):
            annual = 0.0
        if not np.isfinite(turnover):
            turnover = 0.0
        if not np.isfinite(drawdown):
            drawdown = 0.0
        if not np.isfinite(rank_ic):
            rank_ic = 0.0
        return float(
            0.05 * np.tanh(sharpe / 2.0)
            + 0.03 * np.tanh(annual / 0.20)
            + 0.18 * min(rank_ic, 0.15)
            - 0.020 * turnover
            - 0.040 * drawdown
        )

    def _signal_weights(self, records) -> dict[str, float]:
        weights: dict[str, float] = {}
        for record in records:
            weight = float(record.metrics.get("rank_ic", 0.0))
            if not np.isfinite(weight):
                weight = 0.0
            weights[record.canonical] = weight
        if not any(abs(weight) > 0.0 for weight in weights.values()):
            return {record.canonical: 1.0 for record in records}
        return weights
