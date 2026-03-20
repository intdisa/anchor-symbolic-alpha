from __future__ import annotations

from dataclasses import dataclass

from ..evaluation import AdmissionDecision, FactorPool
from ..evaluation.role_profiles import adapt_role_profile, normalize_role, resolve_role_profile


@dataclass(frozen=True)
class ReviewOutcome:
    approved: bool
    reason: str


class ReviewerAgent:
    def __init__(
        self,
        max_corr: float = 0.80,
        max_turnover: float = 1.20,
        min_abs_rank_ic: float = 0.03,
    ) -> None:
        self.max_corr = max_corr
        self.max_turnover = max_turnover
        self.min_abs_rank_ic = min_abs_rank_ic

    def review(self, decision: AdmissionDecision, pool: FactorPool, role: str | None = None) -> ReviewOutcome:
        del pool
        profile = resolve_role_profile(
            role if role is not None else getattr(decision.candidate, "role", None),
            default_reviewer_min_abs_rank_ic=self.min_abs_rank_ic,
            default_reviewer_max_corr=self.max_corr,
            default_reviewer_max_turnover=self.max_turnover,
        )
        if decision.candidate is None:
            return ReviewOutcome(False, decision.reason)
        if not decision.accepted:
            return ReviewOutcome(False, decision.reason)
        metrics = decision.candidate.metrics
        profile = adapt_role_profile(
            profile,
            role if role is not None else decision.candidate.role,
            cross_sectional=bool(metrics.get("cross_sectional", 0.0)),
        )
        normalized_role = normalize_role(role if role is not None else decision.candidate.role)
        cross_sectional = bool(metrics.get("cross_sectional", 0.0))
        max_turnover = profile.reviewer_max_turnover
        if normalized_role == "target_flow" and float(getattr(decision, "trade_proxy_gain", 0.0)) > 0.003:
            max_turnover += 0.15
        if (
            abs(float(metrics.get("rank_ic", 0.0))) < profile.reviewer_min_abs_rank_ic
            and not self._allow_cross_sectional_baseline_replacement(decision, normalized_role, cross_sectional)
        ):
            return ReviewOutcome(False, "review_low_rank_ic")
        if float(metrics.get("max_corr", 0.0)) >= profile.reviewer_max_corr:
            return ReviewOutcome(False, "review_high_corr")
        if float(metrics.get("turnover", 0.0)) >= max_turnover:
            return ReviewOutcome(False, "review_high_turnover")
        return ReviewOutcome(True, "review_accept")

    def _allow_cross_sectional_baseline_replacement(
        self,
        decision: AdmissionDecision,
        normalized_role: str | None,
        cross_sectional: bool,
    ) -> bool:
        if not cross_sectional or normalized_role != "target_price":
            return False
        if decision.reason not in {"replaced_baseline", "replaced"}:
            return False
        metrics = decision.candidate.metrics
        sharpe = float(metrics.get("sharpe", 0.0))
        annual_return = float(metrics.get("annual_return", 0.0))
        turnover = float(metrics.get("turnover", float("inf")))
        drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        trade_proxy_gain = float(getattr(decision, "trade_proxy_gain", 0.0))
        if sharpe <= 0.0 or annual_return <= 0.0:
            return False
        if turnover >= 0.10 or drawdown >= 0.35:
            return False
        return trade_proxy_gain > -0.01
