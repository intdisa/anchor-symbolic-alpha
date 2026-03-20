from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RoleProfile:
    commit_min_abs_rank_ic: float
    commit_max_correlation: float
    replacement_margin: float
    reviewer_min_abs_rank_ic: float
    reviewer_max_corr: float
    reviewer_max_turnover: float
    commit_min_trade_proxy_gain: float = 0.0
    pool_complexity_penalty_scale: float = 0.01
    reward_complexity_penalty_scale: float = 0.01
    search_complexity_penalty_scale: float = 0.02
    preview_min_abs_rank_ic: float | None = None
    preview_max_correlation: float | None = None
    preview_min_trade_proxy_gain: float | None = None
    preview_min_validation_marginal_gain: float = 0.0
    reward_trade_proxy_scale: float = 0.0

    @property
    def resolved_preview_min_abs_rank_ic(self) -> float:
        if self.preview_min_abs_rank_ic is not None:
            return self.preview_min_abs_rank_ic
        return self.commit_min_abs_rank_ic

    @property
    def resolved_preview_max_correlation(self) -> float:
        if self.preview_max_correlation is not None:
            return self.preview_max_correlation
        return self.commit_max_correlation

    @property
    def resolved_preview_min_trade_proxy_gain(self) -> float:
        if self.preview_min_trade_proxy_gain is not None:
            return self.preview_min_trade_proxy_gain
        return self.commit_min_trade_proxy_gain


ROLE_ALIASES = {
    "macro": "context",
    "micro": "target_price",
    "target": "target_price",
    "target_flow_vol": "target_flow",
    "target_flow_gap": "target_flow",
    "short_horizon_flow": "target_flow",
    "quality_solvency": "target_price",
    "efficiency_growth": "target_price",
    "valuation_size": "target_price",
    "price_structure": "target_price",
    "reversal_gap": "target_flow",
    "intraday_imbalance": "target_price",
    "volatility_liquidity": "target_flow",
    "trend_structure": "target_price",
    "cross_asset_context": "context",
    "regime_filter": "context",
}


def normalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    return ROLE_ALIASES.get(role, role)


def resolve_role_profile(
    role: str | None,
    *,
    default_commit_min_abs_rank_ic: float = 0.05,
    default_commit_max_correlation: float = 0.90,
    default_replacement_margin: float = 1e-4,
    default_reviewer_min_abs_rank_ic: float = 0.03,
    default_reviewer_max_corr: float = 0.80,
    default_reviewer_max_turnover: float = 1.20,
) -> RoleProfile:
    normalized_role = normalize_role(role)
    if normalized_role == "context":
        return RoleProfile(
            commit_min_abs_rank_ic=0.07,
            commit_max_correlation=0.75,
            replacement_margin=0.003,
            commit_min_trade_proxy_gain=0.002,
            reviewer_min_abs_rank_ic=0.05,
            reviewer_max_corr=0.70,
            reviewer_max_turnover=0.90,
            pool_complexity_penalty_scale=0.012,
            reward_complexity_penalty_scale=0.012,
            search_complexity_penalty_scale=0.025,
            preview_min_abs_rank_ic=0.07,
            preview_max_correlation=0.75,
            preview_min_trade_proxy_gain=0.001,
            preview_min_validation_marginal_gain=0.002,
            reward_trade_proxy_scale=1.0,
        )
    if normalized_role == "target_flow":
        return RoleProfile(
            commit_min_abs_rank_ic=0.025,
            commit_max_correlation=0.90,
            replacement_margin=0.0,
            commit_min_trade_proxy_gain=5e-4,
            reviewer_min_abs_rank_ic=0.025,
            reviewer_max_corr=0.80,
            reviewer_max_turnover=1.20,
            pool_complexity_penalty_scale=0.004,
            reward_complexity_penalty_scale=0.005,
            search_complexity_penalty_scale=0.010,
            preview_min_abs_rank_ic=0.010,
            preview_max_correlation=0.85,
            preview_min_trade_proxy_gain=0.0,
            reward_trade_proxy_scale=2.5,
        )
    if normalized_role == "target_price":
        return RoleProfile(
            commit_min_abs_rank_ic=0.05,
            commit_max_correlation=0.90,
            replacement_margin=1e-4,
            commit_min_trade_proxy_gain=0.0,
            reviewer_min_abs_rank_ic=0.03,
            reviewer_max_corr=0.80,
            reviewer_max_turnover=1.20,
            pool_complexity_penalty_scale=0.01,
            reward_complexity_penalty_scale=0.01,
            search_complexity_penalty_scale=0.02,
            preview_min_abs_rank_ic=0.025,
            preview_max_correlation=0.90,
            preview_min_trade_proxy_gain=0.0,
            reward_trade_proxy_scale=0.0,
        )
    return RoleProfile(
        commit_min_abs_rank_ic=default_commit_min_abs_rank_ic,
        commit_max_correlation=default_commit_max_correlation,
        replacement_margin=default_replacement_margin,
        commit_min_trade_proxy_gain=0.0,
        reviewer_min_abs_rank_ic=default_reviewer_min_abs_rank_ic,
        reviewer_max_corr=default_reviewer_max_corr,
        reviewer_max_turnover=default_reviewer_max_turnover,
        pool_complexity_penalty_scale=0.01,
        reward_complexity_penalty_scale=0.01,
        search_complexity_penalty_scale=0.02,
        preview_min_abs_rank_ic=default_commit_min_abs_rank_ic,
        preview_max_correlation=default_commit_max_correlation,
        preview_min_trade_proxy_gain=0.0,
        reward_trade_proxy_scale=0.0,
    )


def adapt_role_profile(
    profile: RoleProfile,
    role: str | None,
    *,
    cross_sectional: bool = False,
) -> RoleProfile:
    if not cross_sectional:
        return profile
    normalized_role = normalize_role(role)
    if normalized_role == "target_flow":
        return replace(
            profile,
            commit_min_abs_rank_ic=min(profile.commit_min_abs_rank_ic, 0.010),
            reviewer_min_abs_rank_ic=min(profile.reviewer_min_abs_rank_ic, 0.008),
            preview_min_abs_rank_ic=min(profile.resolved_preview_min_abs_rank_ic, 0.008),
            replacement_margin=min(profile.replacement_margin, 0.0),
        )
    if normalized_role == "target_price":
        return replace(
            profile,
            commit_min_abs_rank_ic=min(profile.commit_min_abs_rank_ic, 0.010),
            reviewer_min_abs_rank_ic=min(profile.reviewer_min_abs_rank_ic, 0.008),
            preview_min_abs_rank_ic=min(profile.resolved_preview_min_abs_rank_ic, 0.008),
            replacement_margin=min(profile.replacement_margin, 0.0),
        )
    if normalized_role == "context":
        return replace(
            profile,
            commit_min_abs_rank_ic=min(profile.commit_min_abs_rank_ic, 0.015),
            reviewer_min_abs_rank_ic=min(profile.reviewer_min_abs_rank_ic, 0.010),
            preview_min_abs_rank_ic=min(profile.resolved_preview_min_abs_rank_ic, 0.012),
            commit_min_trade_proxy_gain=min(profile.commit_min_trade_proxy_gain, 0.0),
            preview_min_trade_proxy_gain=min(profile.resolved_preview_min_trade_proxy_gain, 0.0),
            preview_min_validation_marginal_gain=min(profile.preview_min_validation_marginal_gain, 0.0),
        )
    return profile
