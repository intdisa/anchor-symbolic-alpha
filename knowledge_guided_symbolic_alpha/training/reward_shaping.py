from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..domain.feature_registry import FEATURE_REGISTRY
from ..domain.priors import PRIOR_RULES
from ..evaluation import AdmissionDecision, AdmissionPolicy, FactorPool
from ..evaluation.role_profiles import resolve_role_profile


@dataclass(frozen=True)
class RewardOutcome:
    reward: float
    clipped_reward: float
    decision: AdmissionDecision
    components: dict[str, float]


class PoolRewardShaper:
    def __init__(
        self,
        admission_policy: AdmissionPolicy | None = None,
        clip_range: tuple[float, float] = (-5.0, 5.0),
        invalid_penalty: float = 1.0,
        complexity_penalty_scale: float = 0.01,
        novelty_bonus_scale: float = 0.05,
        knowledge_bonus_scale: float = 0.05,
        redundancy_penalty_scale: float = 0.05,
        turnover_penalty_scale: float = 0.02,
        drawdown_penalty_scale: float = 0.05,
    ) -> None:
        self.admission_policy = admission_policy or AdmissionPolicy()
        self.clip_range = clip_range
        self.invalid_penalty = invalid_penalty
        self.complexity_penalty_scale = complexity_penalty_scale
        self.novelty_bonus_scale = novelty_bonus_scale
        self.knowledge_bonus_scale = knowledge_bonus_scale
        self.redundancy_penalty_scale = redundancy_penalty_scale
        self.turnover_penalty_scale = turnover_penalty_scale
        self.drawdown_penalty_scale = drawdown_penalty_scale

    def shape(
        self,
        formula: str | list[str] | tuple[str, ...],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        commit: bool = True,
        role: str | None = None,
    ) -> RewardOutcome:
        working_pool = pool if commit else pool.copy()
        decision = self.admission_policy.screen(formula, data, target, working_pool, role=role)

        components = self._build_components(decision)
        reward = (
            components["delta_pool_score"]
            + components["trade_proxy_bonus"]
            + components["novelty_bonus"]
            + components["knowledge_bonus"]
            - components["invalid_penalty"]
            - components["complexity_penalty"]
            - components["redundancy_penalty"]
            - components["risk_penalty"]
        )
        clipped_reward = float(np.clip(reward, self.clip_range[0], self.clip_range[1]))
        return RewardOutcome(
            reward=float(reward),
            clipped_reward=clipped_reward,
            decision=decision,
            components=components,
        )

    def _build_components(self, decision: AdmissionDecision) -> dict[str, float]:
        if decision.candidate is None:
            return {
                "delta_pool_score": 0.0,
                "trade_proxy_bonus": 0.0,
                "novelty_bonus": 0.0,
                "knowledge_bonus": 0.0,
                "invalid_penalty": self.invalid_penalty,
                "complexity_penalty": 0.0,
                "redundancy_penalty": 0.0,
                "risk_penalty": 0.0,
            }

        metrics = decision.candidate.metrics
        tokens = decision.candidate.tokens
        profile = resolve_role_profile(decision.candidate.role)
        max_corr = float(metrics.get("max_corr", 0.0))
        novelty_bonus = 0.0
        if decision.reason != "duplicate_canonical":
            novelty_bonus = self.novelty_bonus_scale * max(0.0, 1.0 - max_corr)
        knowledge_bonus = self.knowledge_bonus_scale * self._knowledge_alignment(tokens)
        complexity_penalty = profile.reward_complexity_penalty_scale * max(0, len(tokens) - 3)
        redundancy_penalty = self.redundancy_penalty_scale * max_corr
        risk_penalty = self._risk_penalty(metrics)
        trade_proxy_bonus = profile.reward_trade_proxy_scale * float(decision.trade_proxy_gain)
        invalid_penalty = 0.0 if decision.reason not in {"forced_evaluation_failure"} else self.invalid_penalty
        return {
            "delta_pool_score": float(decision.marginal_gain),
            "trade_proxy_bonus": float(trade_proxy_bonus),
            "novelty_bonus": float(novelty_bonus),
            "knowledge_bonus": float(knowledge_bonus),
            "invalid_penalty": float(invalid_penalty),
            "complexity_penalty": float(complexity_penalty),
            "redundancy_penalty": float(redundancy_penalty),
            "risk_penalty": float(risk_penalty),
        }

    def _knowledge_alignment(self, tokens: tuple[str, ...]) -> float:
        features = {token for token in tokens if token in FEATURE_REGISTRY}
        matched = 0
        for rule in PRIOR_RULES:
            if set(rule.features).issubset(features):
                matched += 1
        if not PRIOR_RULES:
            return 0.0
        return matched / len(PRIOR_RULES)

    def _risk_penalty(self, metrics: dict[str, float]) -> float:
        turnover = float(metrics.get("turnover", 0.0))
        max_drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        return (
            self.turnover_penalty_scale * turnover
            + self.drawdown_penalty_scale * max_drawdown
        )
