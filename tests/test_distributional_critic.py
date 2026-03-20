from dataclasses import dataclass

import numpy as np
import pandas as pd

from knowledge_guided_symbolic_alpha.agents import AgentProposal, HierarchicalManagerAgent
from knowledge_guided_symbolic_alpha.evaluation.admission import AdmissionDecision
from knowledge_guided_symbolic_alpha.evaluation.distributional_critic import (
    DistributionalCollectionCritic,
    DistributionalCriticEstimate,
)
from knowledge_guided_symbolic_alpha.evaluation.factor_pool import FactorPool, FactorRecord
from knowledge_guided_symbolic_alpha.evaluation.pool_scoring import CandidatePoolPreview, evaluate_formula_record
from knowledge_guided_symbolic_alpha.models.controllers import PlannerDecision
from knowledge_guided_symbolic_alpha.training.reward_shaping import RewardOutcome


def make_frame() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2021-01-01", periods=80, freq="D")
    gap = pd.Series(np.sin(np.arange(len(index)) / 4.0), index=index)
    oc = pd.Series(np.cos(np.arange(len(index)) / 3.0) * 0.7, index=index)
    hl = pd.Series(0.5 + 0.1 * np.sin(np.arange(len(index)) / 5.0), index=index)
    frame = pd.DataFrame(
        {
            "GOLD_GAP_RET": gap,
            "GOLD_OC_RET": oc,
            "GOLD_HL_SPREAD": hl.abs() + 0.2,
        }
    )
    target = gap.shift(-1).fillna(0.0)
    return frame, target


def test_distributional_critic_penalizes_harmful_second_factor(monkeypatch) -> None:
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(evaluate_formula_record(("GOLD_GAP_RET", "NEG"), frame, target, role="reversal_gap"))
    critic = DistributionalCollectionCritic()
    proxy_scores = iter([0.10, 0.05])
    monkeypatch.setattr(
        critic,
        "_library_walk_forward_proxy",
        lambda records, data, asset_returns: next(proxy_scores),
    )

    estimate = critic.estimate(
        ("GOLD_OC_RET", "NEG", "GOLD_HL_SPREAD", "DIV"),
        frame,
        target,
        pool,
        role="volatility_liquidity",
        validation_data=frame,
        validation_target=target,
    )

    assert estimate.walk_forward_proxy_gain < 0.0


def test_distributional_critic_uses_cross_sectional_role_profile_for_preview(monkeypatch) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02",
                ]
            ),
            "permno": [1, 2, 1, 2],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "TARGET_XS_RET_1": [0.01, 0.02, 0.015, 0.025],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    critic = DistributionalCollectionCritic()
    captured: dict[str, float] = {}

    def fake_preview(
        formula,
        pool,
        data,
        target,
        evaluator=None,
        role=None,
        min_abs_rank_ic=None,
        max_correlation=None,
        replacement_margin=None,
        min_validation_marginal_gain=None,
        min_trade_proxy_gain=None,
    ):
        del formula, pool, data, target, evaluator, role, max_correlation, replacement_margin, min_validation_marginal_gain, min_trade_proxy_gain
        captured["min_abs_rank_ic"] = float(min_abs_rank_ic)
        record = FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="PROFITABILITY_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.1, 2.1], index=panel.index),
            metrics={"rank_ic": 0.02, "sharpe": 0.3, "annual_return": 0.01, "turnover": 0.01, "max_drawdown": -0.02},
            role="trend_structure",
        )
        return CandidatePoolPreview(
            accepted=True,
            reason="accepted",
            record=record,
            marginal_gain=0.01,
            baseline_score=0.0,
            new_score=0.01,
            trade_proxy_gain=0.01,
            baseline_trade_proxy=0.0,
            new_trade_proxy=0.01,
        )

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.distributional_critic.preview_candidate_on_dataset",
        fake_preview,
    )

    estimate = critic.estimate(
        ("PROFITABILITY_Q", "RANK"),
        panel,
        target,
        FactorPool(),
        role="trend_structure",
        validation_data=panel,
        validation_target=target,
    )

    assert estimate.preview is not None
    assert estimate.preview.accepted
    assert captured["min_abs_rank_ic"] == 0.008


def test_distributional_critic_blends_windowed_proxy_for_cross_sectional_validation(monkeypatch) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=160, freq="D").repeat(2),
            "permno": [1, 2] * 160,
            "PROFITABILITY_Q": np.tile([1.0, 2.0], 160),
            "TARGET_XS_RET_1": np.tile([0.01, 0.02], 160),
        }
    )
    target = panel["TARGET_XS_RET_1"]
    critic = DistributionalCollectionCritic()

    def fake_preview(*args, **kwargs):
        del args, kwargs
        record = FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="PROFITABILITY_Q RANK",
            signal=pd.Series(np.tile([1.0, 2.0], 160), index=panel.index),
            metrics={"rank_ic": 0.02, "sharpe": 0.3, "annual_return": 0.01, "turnover": 0.01, "max_drawdown": -0.02},
            role="trend_structure",
        )
        return CandidatePoolPreview(
            accepted=True,
            reason="accepted",
            record=record,
            marginal_gain=0.01,
            baseline_score=0.0,
            new_score=0.01,
            trade_proxy_gain=0.01,
            baseline_trade_proxy=0.0,
            new_trade_proxy=0.01,
        )

    call_counter = {"count": 0}

    def fake_library_proxy(records, data, asset_returns):
        del records, data, asset_returns
        call_counter["count"] += 1
        return 0.01 if call_counter["count"] == 1 else 0.02

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.distributional_critic.preview_candidate_on_dataset",
        fake_preview,
    )
    monkeypatch.setattr(critic, "_library_walk_forward_proxy", fake_library_proxy)
    monkeypatch.setattr(
        critic,
        "_windowed_walk_forward_proxy",
        lambda records, data, asset_returns, fallback, windows=4: fallback + 0.03,
    )

    estimate = critic.estimate(
        ("PROFITABILITY_Q", "RANK"),
        panel,
        target,
        FactorPool(),
        role="trend_structure",
        validation_data=panel,
        validation_target=target,
    )

    assert estimate.baseline_walk_forward_proxy > 0.01
    assert estimate.new_walk_forward_proxy > 0.02
    assert estimate.walk_forward_proxy_gain > 0.01


@dataclass
class DummySkillAgent:
    role: str
    tokens: tuple[str, ...]
    experience_memory: object | None = None

    def apply_curriculum(self, stage) -> None:
        del stage

    def set_context(self, regime: str) -> None:
        del regime

    def propose(self, **kwargs) -> AgentProposal:
        del kwargs
        return AgentProposal(
            role=self.role,
            body_tokens=self.tokens,
            logprob=0.0,
            valid=True,
            terminal_error=None,
        )

    def observe(self, tokens: tuple[str, ...], reward: float, accepted: bool) -> None:
        del tokens, reward, accepted


class DummyPlanner:
    def __init__(self, ordered_skills: tuple[str, ...] = ("reversal_gap",)) -> None:
        self.ordered_skills = ordered_skills

    def plan(self, state) -> PlannerDecision:
        del state
        return PlannerDecision(
            ordered_skills=self.ordered_skills,
            selected_slot=0,
            budget_multiplier=1.0,
            stop=False,
            rationale="test",
            skill_scores={skill: float(len(self.ordered_skills) - index) for index, skill in enumerate(self.ordered_skills)},
        )


class DummyRewardShaper:
    def shape(self, formula, data, target, pool, commit=True, role=None) -> RewardOutcome:
        del data, target, pool, commit
        record = FactorRecord(
            tokens=tuple(formula),
            canonical=" ".join(formula),
            signal=pd.Series([1.0, 2.0, 3.0]),
            metrics={"rank_ic": 0.10, "max_corr": 0.0, "turnover": 0.0},
            role=role,
        )
        decision = AdmissionDecision(True, "accepted", record, 0.05, trade_proxy_gain=0.01)
        return RewardOutcome(reward=0.2, clipped_reward=0.2, decision=decision, components={})


class PairAwareRewardShaper(DummyRewardShaper):
    def shape(self, formula, data, target, pool, commit=True, role=None) -> RewardOutcome:
        del data, target
        record = FactorRecord(
            tokens=tuple(formula),
            canonical=" ".join(formula),
            signal=pd.Series([1.0, 2.0, 3.0]),
            metrics={"rank_ic": 0.10, "max_corr": 0.0, "turnover": 0.0},
            role=role,
        )
        if commit:
            pool.add(record)
        decision = AdmissionDecision(True, "accepted", record, 0.05, trade_proxy_gain=0.01)
        return RewardOutcome(reward=0.2, clipped_reward=0.2, decision=decision, components={})


class DummyCritic:
    def estimate(self, *args, **kwargs) -> DistributionalCriticEstimate:
        del args, kwargs
        return DistributionalCriticEstimate(
            expected_gain=0.01,
            risk_adjusted_gain=0.05,
            quantiles=(-0.1, 0.05, 0.1),
            uncertainty=0.2,
            accepted=True,
            reason="accepted",
            train_reward=0.2,
            validation_gain=0.02,
            trade_proxy_gain=0.01,
            walk_forward_proxy_gain=-0.03,
            baseline_walk_forward_proxy=0.01,
            new_walk_forward_proxy=-0.02,
            preview=None,
        )


def test_hierarchical_manager_blocks_negative_walk_forward_proxy_commit() -> None:
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("GOLD_GAP_RET", "NEG"),
            canonical="GOLD_GAP_RET NEG",
            signal=frame["GOLD_GAP_RET"],
            metrics={"rank_ic": 0.1, "max_corr": 0.0, "turnover": 0.2},
            role="reversal_gap",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={"reversal_gap": DummySkillAgent("reversal_gap", ("GOLD_GAP_RET", "NEG"))},
        planner=DummyPlanner(),
        reward_shaper=DummyRewardShaper(),
        critic=DummyCritic(),
    )

    step = manager.run_step(frame, target, pool, commit=True, validation_data=frame, validation_target=target)

    assert not step.accepted
    assert step.decision_reason == "critic_negative_walk_forward_proxy"


class RoutingCritic:
    def estimate(self, *args, **kwargs) -> DistributionalCriticEstimate:
        role = kwargs["role"]
        if role == "regime_filter":
            return DistributionalCriticEstimate(
                expected_gain=0.01,
                risk_adjusted_gain=0.18,
                quantiles=(-0.01, 0.04, 0.18),
                uncertainty=0.1,
                accepted=False,
                reason="fast_ic_screen",
                train_reward=0.02,
                validation_gain=0.01,
                trade_proxy_gain=0.0,
                walk_forward_proxy_gain=0.0,
                baseline_walk_forward_proxy=0.0,
                new_walk_forward_proxy=0.0,
                preview=None,
            )
        return DistributionalCriticEstimate(
            expected_gain=0.01,
            risk_adjusted_gain=0.03,
            quantiles=(0.0, 0.02, 0.03),
            uncertainty=0.02,
            accepted=True,
            reason="accepted",
            train_reward=0.0,
            validation_gain=0.0,
            trade_proxy_gain=0.0,
            walk_forward_proxy_gain=0.0,
            baseline_walk_forward_proxy=0.0,
            new_walk_forward_proxy=0.0,
            preview=None,
        )


def test_hierarchical_manager_reroutes_after_repeated_rejections() -> None:
    frame, target = make_frame()
    manager = HierarchicalManagerAgent(
        agents={
            "regime_filter": DummySkillAgent("regime_filter", ("GOLD_GAP_RET", "NEG")),
            "volatility_liquidity": DummySkillAgent("volatility_liquidity", ("GOLD_HL_SPREAD", "RANK")),
        },
        planner=DummyPlanner(("regime_filter", "volatility_liquidity")),
        reward_shaper=DummyRewardShaper(),
        critic=RoutingCritic(),
        rejection_cooldown_threshold=3,
        cooldown_steps=2,
    )
    pool = FactorPool(max_size=4)
    manager.accepted_counts["volatility_liquidity"] = 1

    steps = [manager.run_step(frame, target, pool, commit=False, validation_data=frame, validation_target=target) for _ in range(4)]

    assert [step.selected_agent for step in steps[:3]] == ["regime_filter", "regime_filter", "regime_filter"]
    assert steps[3].selected_agent == "volatility_liquidity"


class BootstrapCritic:
    def estimate(self, *args, **kwargs) -> DistributionalCriticEstimate:
        role = kwargs["role"]
        if role == "volatility_liquidity":
            return DistributionalCriticEstimate(
                expected_gain=0.01,
                risk_adjusted_gain=-0.03,
                quantiles=(-0.05, -0.03, 0.0),
                uncertainty=0.05,
                accepted=True,
                reason="accepted",
                train_reward=0.1,
                validation_gain=0.02,
                trade_proxy_gain=0.01,
                walk_forward_proxy_gain=-0.01,
                baseline_walk_forward_proxy=0.02,
                new_walk_forward_proxy=0.01,
                preview=None,
            )
        return DistributionalCriticEstimate(
            expected_gain=0.05,
            risk_adjusted_gain=0.02,
            quantiles=(0.0, 0.02, 0.04),
            uncertainty=0.04,
            accepted=True,
            reason="accepted",
            train_reward=0.1,
            validation_gain=0.02,
            trade_proxy_gain=0.01,
            walk_forward_proxy_gain=0.0,
            baseline_walk_forward_proxy=0.0,
            new_walk_forward_proxy=0.0,
            preview=None,
        )


class PairSelectionCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, data, target, kwargs
        if role == "short_horizon_flow" and len(pool.records) == 0:
            return DistributionalCriticEstimate(
                expected_gain=0.12,
                risk_adjusted_gain=0.18,
                quantiles=(0.0, 0.10, 0.18),
                uncertainty=0.04,
                accepted=True,
                reason="accepted",
                train_reward=0.12,
                validation_gain=0.06,
                trade_proxy_gain=0.03,
                walk_forward_proxy_gain=0.01,
                baseline_walk_forward_proxy=0.0,
                new_walk_forward_proxy=0.01,
                preview=None,
            )
        if role == "price_structure" and len(pool.records) == 0:
            return DistributionalCriticEstimate(
                expected_gain=0.08,
                risk_adjusted_gain=0.08,
                quantiles=(0.0, 0.05, 0.08),
                uncertainty=0.03,
                accepted=True,
                reason="accepted",
                train_reward=0.08,
                validation_gain=0.03,
                trade_proxy_gain=0.01,
                walk_forward_proxy_gain=0.0,
                baseline_walk_forward_proxy=0.0,
                new_walk_forward_proxy=0.0,
                preview=None,
            )
        if role == "short_horizon_flow":
            return DistributionalCriticEstimate(
                expected_gain=0.08,
                risk_adjusted_gain=0.06,
                quantiles=(0.0, 0.03, 0.06),
                uncertainty=0.02,
                accepted=True,
                reason="accepted",
                train_reward=0.06,
                validation_gain=0.03,
                trade_proxy_gain=0.01,
                walk_forward_proxy_gain=0.05,
                baseline_walk_forward_proxy=0.01,
                new_walk_forward_proxy=0.14,
                preview=None,
            )
        return DistributionalCriticEstimate(
            expected_gain=0.02,
            risk_adjusted_gain=0.02,
            quantiles=(0.0, 0.01, 0.02),
            uncertainty=0.01,
            accepted=True,
            reason="accepted",
            train_reward=0.02,
            validation_gain=0.01,
            trade_proxy_gain=0.0,
            walk_forward_proxy_gain=0.01,
            baseline_walk_forward_proxy=0.01,
            new_walk_forward_proxy=0.04,
            preview=None,
        )


class PriceOverrideCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, data, target, kwargs
        if role == "price_structure" and any(record.role == "short_horizon_flow" for record in pool.records):
            return DistributionalCriticEstimate(
                expected_gain=0.01,
                risk_adjusted_gain=-0.02,
                quantiles=(-0.02, -0.01, 0.0),
                uncertainty=0.02,
                accepted=False,
                reason="full_validation",
                train_reward=-0.02,
                validation_gain=-0.01,
                trade_proxy_gain=-0.01,
                walk_forward_proxy_gain=-0.01,
                baseline_walk_forward_proxy=0.02,
                new_walk_forward_proxy=0.01,
                preview=None,
            )
        if role == "price_structure":
            return DistributionalCriticEstimate(
                expected_gain=0.06,
                risk_adjusted_gain=0.05,
                quantiles=(0.0, 0.03, 0.05),
                uncertainty=0.02,
                accepted=True,
                reason="accepted",
                train_reward=0.05,
                validation_gain=0.02,
                trade_proxy_gain=0.01,
                walk_forward_proxy_gain=0.04,
                baseline_walk_forward_proxy=0.02,
                new_walk_forward_proxy=0.08,
                preview=None,
            )
        if role == "short_horizon_flow":
            return DistributionalCriticEstimate(
                expected_gain=0.03,
                risk_adjusted_gain=0.01,
                quantiles=(0.0, 0.01, 0.02),
                uncertainty=0.01,
                accepted=True,
                reason="accepted",
                train_reward=0.02,
                validation_gain=0.01,
                trade_proxy_gain=0.0,
                walk_forward_proxy_gain=0.0,
                baseline_walk_forward_proxy=0.02,
                new_walk_forward_proxy=0.02,
                preview=None,
            )
        return DistributionalCriticEstimate(
            expected_gain=0.04,
            risk_adjusted_gain=0.03,
            quantiles=(0.0, 0.02, 0.03),
            uncertainty=0.02,
            accepted=True,
            reason="accepted",
            train_reward=0.03,
            validation_gain=0.02,
            trade_proxy_gain=0.0,
            walk_forward_proxy_gain=0.0,
            baseline_walk_forward_proxy=0.0,
            new_walk_forward_proxy=0.0,
            preview=None,
        )


def test_hierarchical_manager_allows_bootstrap_commit_for_non_gap_skill() -> None:
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("GOLD_GAP_RET", "NEG"),
            canonical="GOLD_GAP_RET NEG",
            signal=frame["GOLD_GAP_RET"],
            metrics={"rank_ic": 0.1, "max_corr": 0.0, "turnover": 0.2},
            role="reversal_gap",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={
            "reversal_gap": DummySkillAgent("reversal_gap", ("GOLD_GAP_RET", "NEG")),
            "volatility_liquidity": DummySkillAgent("volatility_liquidity", ("GOLD_HL_SPREAD", "RANK")),
        },
        planner=DummyPlanner(("volatility_liquidity",)),
        reward_shaper=DummyRewardShaper(),
        critic=BootstrapCritic(),
    )

    step = manager.run_step(frame, target, pool, commit=True, validation_data=frame, validation_target=target)

    assert step.accepted
    assert step.selected_agent == "volatility_liquidity"


def test_hierarchical_manager_prefers_price_anchor_when_pair_proxy_is_better() -> None:
    frame, target = make_frame()
    manager = HierarchicalManagerAgent(
        agents={
            "short_horizon_flow": DummySkillAgent("short_horizon_flow", ("GOLD_GAP_RET", "NEG")),
            "price_structure": DummySkillAgent("price_structure", ("GOLD_GAP_RET", "GOLD_OC_RET", "SUB")),
        },
        planner=DummyPlanner(("short_horizon_flow", "price_structure")),
        reward_shaper=PairAwareRewardShaper(),
        critic=PairSelectionCritic(),
    )
    pool = FactorPool(max_size=4)

    step = manager.run_step(frame, target, pool, commit=True, validation_data=frame, validation_target=target)

    assert step.accepted
    assert step.selected_agent == "price_structure"


def test_hierarchical_manager_price_override_replaces_flow_baseline() -> None:
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("GOLD_GAP_RET", "NEG"),
            canonical="GOLD_GAP_RET NEG",
            signal=frame["GOLD_GAP_RET"],
            metrics={"rank_ic": 0.1, "max_corr": 0.0, "turnover": 0.2},
            role="short_horizon_flow",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={
            "short_horizon_flow": DummySkillAgent("short_horizon_flow", ("GOLD_GAP_RET", "NEG")),
            "price_structure": DummySkillAgent("price_structure", ("GOLD_GAP_RET", "GOLD_OC_RET", "SUB")),
        },
        planner=DummyPlanner(("price_structure",)),
        reward_shaper=PairAwareRewardShaper(),
        critic=PriceOverrideCritic(),
    )

    step = manager.run_step(frame, target, pool, commit=True, validation_data=frame, validation_target=target)

    assert step.accepted
    assert step.selected_agent == "price_structure"
    assert [record.role for record in pool.records] == ["price_structure"]


class FullValidationRewardShaper(DummyRewardShaper):
    def shape(self, formula, data, target, pool, commit=True, role=None) -> RewardOutcome:
        del data, target, pool
        record = FactorRecord(
            tokens=tuple(formula),
            canonical=" ".join(formula),
            signal=pd.Series([1.0, 2.0, 3.0]),
            metrics={"rank_ic": 0.10, "max_corr": 0.0, "turnover": 0.0},
            role=role,
        )
        if role == "regime_filter" and commit:
            decision = AdmissionDecision(False, "full_validation", record, 0.0, trade_proxy_gain=0.0)
            return RewardOutcome(reward=-0.1, clipped_reward=-0.1, decision=decision, components={})
        decision = AdmissionDecision(True, "accepted", record, 0.05, trade_proxy_gain=0.01)
        return RewardOutcome(reward=0.2, clipped_reward=0.2, decision=decision, components={})


class CrossSectionalFullValidationRewardShaper(DummyRewardShaper):
    def shape(self, formula, data, target, pool, commit=True, role=None) -> RewardOutcome:
        del target
        record = FactorRecord(
            tokens=tuple(formula),
            canonical=" ".join(formula),
            signal=pd.Series(np.linspace(0.0, 1.0, len(data)), index=data.index),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.05,
                "sharpe": 0.25,
                "annual_return": 0.01,
                "max_drawdown": -0.03,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        decision = AdmissionDecision(False, "full_validation", record, -0.005, trade_proxy_gain=0.01)
        return RewardOutcome(reward=-0.05, clipped_reward=-0.05, decision=decision, components={})


class CrossSectionalReplacementRewardShaper(DummyRewardShaper):
    def shape(self, formula, data, target, pool, commit=True, role=None) -> RewardOutcome:
        del target
        record = FactorRecord(
            tokens=tuple(formula),
            canonical=" ".join(formula),
            signal=pd.Series(np.linspace(0.0, 1.0, len(data)), index=data.index),
            metrics={
                "rank_ic": 0.004,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.8,
                "annual_return": 0.08,
                "max_drawdown": -0.03,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        replaced_canonical = pool.records[0].canonical if pool.records else None
        if commit and pool.records:
            pool.replace(0, record)
        elif commit:
            pool.add(record)
        decision = AdmissionDecision(
            True,
            "replaced_baseline",
            record,
            0.03,
            replaced_canonical=replaced_canonical,
            trade_proxy_gain=0.02,
        )
        return RewardOutcome(reward=0.03, clipped_reward=0.03, decision=decision, components={})


class CrossSectionalFastScreenReplacementRewardShaper(DummyRewardShaper):
    def shape(self, formula, data, target, pool, commit=True, role=None) -> RewardOutcome:
        del target
        record = FactorRecord(
            tokens=tuple(formula),
            canonical=" ".join(formula),
            signal=pd.Series(np.linspace(0.0, 1.0, len(data)), index=data.index),
            metrics={
                "rank_ic": 0.004,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.8,
                "annual_return": 0.08,
                "max_drawdown": -0.03,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        decision = AdmissionDecision(False, "fast_ic_screen", record, 0.0, trade_proxy_gain=-0.002)
        return RewardOutcome(reward=-0.02, clipped_reward=-0.02, decision=decision, components={})


class CrossSectionalValidationCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, data, target, pool, kwargs
        preview_record = FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="PROFITABILITY_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.4,
                "annual_return": 0.03,
                "max_drawdown": -0.02,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        return DistributionalCriticEstimate(
            expected_gain=0.02,
            risk_adjusted_gain=0.03,
            quantiles=(0.0, 0.03, 0.04),
            uncertainty=0.04,
            accepted=False,
            reason="full_validation",
            train_reward=-0.05,
            validation_gain=0.01,
            trade_proxy_gain=0.01,
            walk_forward_proxy_gain=0.02,
            baseline_walk_forward_proxy=0.0,
            new_walk_forward_proxy=0.02,
            preview=CandidatePoolPreview(
                accepted=True,
                reason="accepted",
                record=preview_record,
                marginal_gain=0.01,
                baseline_score=0.0,
                new_score=0.01,
                trade_proxy_gain=0.01,
                baseline_trade_proxy=0.0,
                new_trade_proxy=0.01,
            ),
        )


class CrossSectionalReplacementCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, data, target, pool, kwargs
        preview_record = FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK"),
            canonical="CASH_RATIO_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={
                "rank_ic": 0.004,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.8,
                "annual_return": 0.08,
                "max_drawdown": -0.03,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        return DistributionalCriticEstimate(
            expected_gain=0.03,
            risk_adjusted_gain=0.04,
            quantiles=(0.01, 0.04, 0.05),
            uncertainty=0.04,
            accepted=True,
            reason="replaced_baseline",
            train_reward=0.02,
            validation_gain=0.03,
            trade_proxy_gain=0.02,
            walk_forward_proxy_gain=0.03,
            baseline_walk_forward_proxy=0.01,
            new_walk_forward_proxy=0.04,
            preview=CandidatePoolPreview(
                accepted=True,
                reason="replaced_baseline",
                record=preview_record,
                marginal_gain=0.03,
                baseline_score=0.0,
                new_score=0.01,
                replaced_canonical="PROFITABILITY_Q RANK",
                trade_proxy_gain=0.02,
                baseline_trade_proxy=0.01,
                new_trade_proxy=0.03,
            ),
        )


class CrossSectionalValidationReplacementCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, data, target, kwargs
        preview_record = FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            canonical="CASH_RATIO_Q RANK SALES_TO_ASSETS_Q RANK ADD",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={
                "rank_ic": 0.021,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 1.0,
                "annual_return": 0.07,
                "max_drawdown": -0.04,
                "stability_score": -0.005,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        baseline = pool.records[0].canonical if pool.records else None
        return DistributionalCriticEstimate(
            expected_gain=0.02,
            risk_adjusted_gain=0.03,
            quantiles=(0.0, 0.03, 0.04),
            uncertainty=0.04,
            accepted=False,
            reason="fast_ic_screen",
            train_reward=-0.02,
            validation_gain=0.01,
            trade_proxy_gain=-0.005,
            walk_forward_proxy_gain=0.004,
            baseline_walk_forward_proxy=0.033,
            new_walk_forward_proxy=0.037,
            preview=CandidatePoolPreview(
                accepted=True,
                reason="replaced_baseline",
                record=preview_record,
                marginal_gain=0.01,
                baseline_score=0.0,
                new_score=0.01,
                replaced_canonical=baseline,
                trade_proxy_gain=-0.01,
                baseline_trade_proxy=0.03,
                new_trade_proxy=0.02,
            ),
        )


class CrossSectionalSecondUpgradeCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, data, target, kwargs
        preview_record = FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            canonical="CASH_RATIO_Q RANK SALES_TO_ASSETS_Q RANK ADD",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={
                "rank_ic": 0.006,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 1.1,
                "annual_return": 0.08,
                "max_drawdown": -0.03,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        return DistributionalCriticEstimate(
            expected_gain=0.03,
            risk_adjusted_gain=0.04,
            quantiles=(0.01, 0.04, 0.05),
            uncertainty=0.03,
            accepted=False,
            reason="fast_ic_screen",
            train_reward=-0.02,
            validation_gain=-0.01,
            trade_proxy_gain=-0.01,
            walk_forward_proxy_gain=0.012,
            baseline_walk_forward_proxy=0.020,
            new_walk_forward_proxy=0.032,
            preview=CandidatePoolPreview(
                accepted=False,
                reason="full_validation",
                record=preview_record,
                marginal_gain=-0.01,
                baseline_score=0.02,
                new_score=0.01,
                trade_proxy_gain=-0.01,
                baseline_trade_proxy=0.03,
                new_trade_proxy=0.02,
            ),
        )


class CrossSectionalNegativeTradeProxyCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, target, kwargs
        preview_record = FactorRecord(
            tokens=("RET_1", "NEG"),
            canonical="RET_1 NEG",
            signal=pd.Series(np.linspace(0.0, 1.0, len(data)), index=data.index),
            metrics={
                "rank_ic": 0.018,
                "rank_icir": 0.10,
                "max_corr": 0.0,
                "turnover": 0.8,
                "sharpe": 0.3,
                "annual_return": 0.02,
                "max_drawdown": -0.05,
                "cross_sectional": 1.0,
            },
            role=role,
        )
        return DistributionalCriticEstimate(
            expected_gain=0.02,
            risk_adjusted_gain=0.04,
            quantiles=(0.0, 0.04, 0.05),
            uncertainty=0.03,
            accepted=True,
            reason="accepted",
            train_reward=0.02,
            validation_gain=0.01,
            trade_proxy_gain=-0.01,
            walk_forward_proxy_gain=0.01,
            baseline_walk_forward_proxy=0.01,
            new_walk_forward_proxy=0.02,
            preview=CandidatePoolPreview(
                accepted=True,
                reason="accepted",
                record=preview_record,
                marginal_gain=0.005,
                baseline_score=0.01,
                new_score=0.015,
                trade_proxy_gain=-0.01,
                baseline_trade_proxy=0.02,
                new_trade_proxy=0.01,
            ),
        )


class CrossSectionalUnstableFlowCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, target, kwargs
        preview_record = FactorRecord(
            tokens=("RET_1", "NEG"),
            canonical="RET_1 NEG",
            signal=pd.Series(np.linspace(0.0, 1.0, len(data)), index=data.index),
            metrics={
                "rank_ic": 0.018,
                "rank_icir": 0.08,
                "max_corr": 0.0,
                "turnover": 0.78,
                "sharpe": 0.4,
                "annual_return": 0.03,
                "max_drawdown": -0.05,
                "cross_sectional": 1.0,
                "stability_score": -0.01,
                "rank_ic_window_positive_frac": 0.75,
                "ls_return_window_positive_frac": 0.50,
            },
            role=role,
        )
        return DistributionalCriticEstimate(
            expected_gain=0.02,
            risk_adjusted_gain=0.04,
            quantiles=(0.0, 0.04, 0.05),
            uncertainty=0.03,
            accepted=True,
            reason="accepted",
            train_reward=0.02,
            validation_gain=0.01,
            trade_proxy_gain=0.01,
            walk_forward_proxy_gain=0.02,
            baseline_walk_forward_proxy=0.01,
            new_walk_forward_proxy=0.03,
            preview=CandidatePoolPreview(
                accepted=True,
                reason="accepted",
                record=preview_record,
                marginal_gain=0.005,
                baseline_score=0.01,
                new_score=0.015,
                trade_proxy_gain=0.01,
                baseline_trade_proxy=0.02,
                new_trade_proxy=0.03,
            ),
        )


class CrossSectionalPositiveFlowCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, target, kwargs
        preview_record = FactorRecord(
            tokens=("RET_1", "NEG"),
            canonical="RET_1 NEG",
            signal=pd.Series(np.linspace(0.0, 1.0, len(data)), index=data.index),
            metrics={
                "rank_ic": 0.018,
                "rank_icir": 0.08,
                "max_corr": 0.0,
                "turnover": 0.20,
                "sharpe": 0.4,
                "annual_return": 0.03,
                "max_drawdown": -0.05,
                "cross_sectional": 1.0,
                "stability_score": 0.01,
                "rank_ic_window_positive_frac": 1.0,
                "ls_return_window_positive_frac": 1.0,
            },
            role=role,
        )
        return DistributionalCriticEstimate(
            expected_gain=0.02,
            risk_adjusted_gain=0.04,
            quantiles=(0.0, 0.04, 0.05),
            uncertainty=0.03,
            accepted=True,
            reason="accepted",
            train_reward=0.02,
            validation_gain=0.01,
            trade_proxy_gain=0.01,
            walk_forward_proxy_gain=0.02,
            baseline_walk_forward_proxy=0.01,
            new_walk_forward_proxy=0.03,
            preview=CandidatePoolPreview(
                accepted=True,
                reason="accepted",
                record=preview_record,
                marginal_gain=0.005,
                baseline_score=0.01,
                new_score=0.015,
                trade_proxy_gain=0.01,
                baseline_trade_proxy=0.02,
                new_trade_proxy=0.03,
            ),
        )


def test_hierarchical_manager_reroutes_after_full_validation_loop() -> None:
    frame, target = make_frame()
    manager = HierarchicalManagerAgent(
        agents={
            "regime_filter": DummySkillAgent("regime_filter", ("GOLD_GAP_RET", "NEG")),
            "volatility_liquidity": DummySkillAgent("volatility_liquidity", ("GOLD_HL_SPREAD", "RANK")),
        },
        planner=DummyPlanner(("regime_filter", "volatility_liquidity")),
        reward_shaper=FullValidationRewardShaper(),
        critic=BootstrapCritic(),
        rejection_cooldown_threshold=3,
        cooldown_steps=2,
    )
    pool = FactorPool(max_size=4)
    manager.accepted_counts["volatility_liquidity"] = 1

    steps = [manager.run_step(frame, target, pool, commit=True, validation_data=frame, validation_target=target) for _ in range(4)]

    assert [step.selected_agent for step in steps[:3]] == ["regime_filter", "regime_filter", "regime_filter"]
    assert steps[3].selected_agent == "volatility_liquidity"


def test_hierarchical_manager_allows_validation_backed_cross_sectional_bootstrap() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "TARGET_XS_RET_1": [0.01, 0.02, 0.015, 0.025],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    manager = HierarchicalManagerAgent(
        agents={"trend_structure": DummySkillAgent("trend_structure", ("PROFITABILITY_Q", "RANK"))},
        planner=DummyPlanner(("trend_structure",)),
        reward_shaper=CrossSectionalFullValidationRewardShaper(),
        critic=CrossSectionalValidationCritic(),
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert step.accepted
    assert step.decision_reason == "validation_backed_commit"
    assert step.selected_agent == "trend_structure"
    assert [record.canonical for record in pool.records] == ["PROFITABILITY_Q RANK"]


def test_hierarchical_manager_allows_validation_backed_cross_sectional_bootstrap_after_fast_ic_screen() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "CASH_RATIO_Q": [2.0, 1.0, 2.1, 1.1],
            "SALES_TO_ASSETS_Q": [0.5, 1.5, 0.6, 1.6],
            "TARGET_XS_RET_1": [0.01, 0.02, 0.015, 0.025],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    manager = HierarchicalManagerAgent(
        agents={
            "trend_structure": DummySkillAgent(
                "trend_structure",
                ("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            )
        },
        planner=DummyPlanner(("trend_structure",)),
        reward_shaper=CrossSectionalFastScreenReplacementRewardShaper(),
        critic=CrossSectionalValidationCritic(),
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert step.accepted
    assert step.decision_reason == "validation_backed_commit"


def test_hierarchical_manager_allows_validation_backed_cross_sectional_replacement() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "CASH_RATIO_Q": [2.0, 1.0, 2.1, 1.1],
            "TARGET_XS_RET_1": [0.01, 0.02, 0.015, 0.025],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="PROFITABILITY_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5], index=panel.index),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.4,
                "annual_return": 0.03,
                "max_drawdown": -0.02,
                "cross_sectional": 1.0,
            },
            role="trend_structure",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={"trend_structure": DummySkillAgent("trend_structure", ("CASH_RATIO_Q", "RANK"))},
        planner=DummyPlanner(("trend_structure",)),
        reward_shaper=CrossSectionalReplacementRewardShaper(),
        critic=CrossSectionalReplacementCritic(),
        bootstrap_anchor_skill="trend_structure",
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert step.accepted
    assert step.decision_reason == "replaced_baseline"
    assert [record.canonical for record in pool.records] == ["CASH_RATIO_Q RANK"]


def test_hierarchical_manager_allows_validation_backed_cross_sectional_replacement_after_fast_screen() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "CASH_RATIO_Q": [2.0, 1.0, 2.1, 1.1],
            "SALES_TO_ASSETS_Q": [0.5, 1.5, 0.6, 1.6],
            "TARGET_XS_RET_1": [0.01, 0.02, 0.015, 0.025],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK"),
            canonical="CASH_RATIO_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5], index=panel.index),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.4,
                "annual_return": 0.03,
                "max_drawdown": -0.02,
                "cross_sectional": 1.0,
            },
            role="trend_structure",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={
            "trend_structure": DummySkillAgent(
                "trend_structure",
                ("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            )
        },
        planner=DummyPlanner(("trend_structure",)),
        reward_shaper=CrossSectionalFastScreenReplacementRewardShaper(),
        critic=CrossSectionalValidationReplacementCritic(),
        bootstrap_anchor_skill="trend_structure",
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert step.accepted
    assert step.decision_reason == "validation_backed_replacement"
    assert [record.canonical for record in pool.records] == ["CASH_RATIO_Q RANK SALES_TO_ASSETS_Q RANK ADD"]


def test_hierarchical_manager_allows_validation_backed_cross_sectional_second_upgrade() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "CASH_RATIO_Q": [2.0, 1.0, 2.1, 1.1],
            "SALES_TO_ASSETS_Q": [0.5, 1.5, 0.6, 1.6],
            "TARGET_XS_RET_1": [0.01, 0.02, 0.015, 0.025],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK", "PROFITABILITY_Q", "RANK", "ADD"),
            canonical="CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5], index=panel.index),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.5,
                "annual_return": 0.03,
                "max_drawdown": -0.03,
                "cross_sectional": 1.0,
            },
            role="trend_structure",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={
            "trend_structure": DummySkillAgent(
                "trend_structure",
                ("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            )
        },
        planner=DummyPlanner(("trend_structure",)),
        reward_shaper=CrossSectionalFastScreenReplacementRewardShaper(),
        critic=CrossSectionalSecondUpgradeCritic(),
        bootstrap_anchor_skill="trend_structure",
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert step.accepted
    assert step.decision_reason == "validation_backed_upgrade"
    assert [record.canonical for record in pool.records] == ["CASH_RATIO_Q RANK SALES_TO_ASSETS_Q RANK ADD"]


def test_hierarchical_manager_blocks_flow_bootstrap_shortcut_when_residual_is_negative() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "RET_1": [0.1, -0.1, 0.1, -0.1],
            "CASH_RATIO_Q": [2.0, 1.0, 2.1, 1.1],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "TARGET_XS_RET_1": [0.01, 0.02, 0.015, 0.025],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK", "PROFITABILITY_Q", "RANK", "ADD"),
            canonical="CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5], index=panel.index),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.5,
                "annual_return": 0.03,
                "max_drawdown": -0.03,
                "cross_sectional": 1.0,
            },
            role="trend_structure",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={"short_horizon_flow": DummySkillAgent("short_horizon_flow", ("RET_1", "NEG"))},
        planner=DummyPlanner(("short_horizon_flow",)),
        reward_shaper=DummyRewardShaper(),
        critic=CrossSectionalNegativeTradeProxyCritic(),
        bootstrap_anchor_skill="trend_structure",
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert not step.accepted
    assert step.decision_reason == "critic_negative_trade_proxy"
    assert [record.canonical for record in pool.records] == ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"]


def test_hierarchical_manager_keeps_bootstrap_anchor_ahead_of_other_bootstrap_skills() -> None:
    manager = HierarchicalManagerAgent(
        agents={
            "short_horizon_flow": DummySkillAgent("short_horizon_flow", ("RET_1", "NEG")),
            "trend_structure": DummySkillAgent("trend_structure", ("PROFITABILITY_Q", "RANK")),
            "price_structure": DummySkillAgent("price_structure", ("RET_1", "OC_RET", "SUB")),
        },
        planner=DummyPlanner(("short_horizon_flow", "trend_structure", "price_structure")),
        reward_shaper=DummyRewardShaper(),
        critic=DummyCritic(),
        bootstrap_anchor_skill="trend_structure",
    )

    ordered = manager._candidate_skill_order(("short_horizon_flow", "trend_structure", "price_structure"))

    assert ordered[0] == "trend_structure"


def test_hierarchical_manager_route_b_empty_pool_commits_anchor_before_pairs() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "RET_1": [0.01, -0.02, 0.03, -0.01],
            "OC_RET": [0.02, 0.01, -0.01, -0.02],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "TARGET_XS_RET_1": [0.02, -0.01, 0.01, 0.03],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    manager = HierarchicalManagerAgent(
        agents={
            "short_horizon_flow": DummySkillAgent("short_horizon_flow", ("RET_1", "NEG")),
            "price_structure": DummySkillAgent("price_structure", ("RET_1", "OC_RET", "SUB")),
            "trend_structure": DummySkillAgent("trend_structure", ("PROFITABILITY_Q", "RANK")),
        },
        planner=DummyPlanner(("short_horizon_flow", "price_structure", "trend_structure")),
        reward_shaper=CrossSectionalFullValidationRewardShaper(),
        critic=CrossSectionalValidationCritic(),
        bootstrap_anchor_skill="trend_structure",
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert step.accepted
    assert step.selected_agent == "trend_structure"
    assert [record.canonical for record in pool.records] == ["PROFITABILITY_Q RANK"]


class PositiveCrossSectionalCritic:
    def estimate(self, formula, data, target, pool, role=None, **kwargs) -> DistributionalCriticEstimate:
        del formula, data, target, kwargs
        preview_record = FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical=f"{role} preview",
            signal=pd.Series([1.0, 2.0, 1.1, 2.1], index=pool.records[0].signal.index if pool.records else None),
            metrics={
                "rank_ic": 0.02,
                "sharpe": 0.4,
                "annual_return": 0.03,
                "turnover": 0.01,
                "max_drawdown": -0.05,
                "cross_sectional": 1.0,
                "stability_score": 0.01,
                "rank_ic_window_positive_frac": 1.0,
                "ls_return_window_positive_frac": 1.0,
            },
            role=role,
        )
        preview = CandidatePoolPreview(
            accepted=True,
            reason="accepted",
            record=preview_record,
            marginal_gain=0.02,
            baseline_score=0.01,
            new_score=0.03,
            trade_proxy_gain=0.02,
            baseline_trade_proxy=0.01,
            new_trade_proxy=0.03,
        )
        return DistributionalCriticEstimate(
            expected_gain=0.03,
            risk_adjusted_gain=0.04,
            quantiles=(0.0, 0.03, 0.04),
            uncertainty=0.01,
            accepted=True,
            reason="accepted",
            train_reward=0.02,
            validation_gain=0.02,
            trade_proxy_gain=0.02,
            walk_forward_proxy_gain=0.02,
            baseline_walk_forward_proxy=0.01,
            new_walk_forward_proxy=0.03,
            preview=preview,
        )


def test_hierarchical_manager_route_b_forces_second_slow_family_before_flow() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "RET_1": [0.01, -0.02, 0.03, -0.01],
            "CASH_RATIO_Q": [1.0, 2.0, 1.1, 2.1],
            "SALES_TO_ASSETS_Q": [0.8, 1.2, 0.9, 1.3],
            "BOOK_TO_MARKET_Q": [0.7, 1.1, 0.8, 1.0],
            "TARGET_XS_RET_1": [0.02, -0.01, 0.01, 0.03],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK"),
            canonical="CASH_RATIO_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.1, 2.1], index=panel.index),
            metrics={"rank_ic": 0.02, "max_corr": 0.0, "turnover": 0.01, "cross_sectional": 1.0},
            role="quality_solvency",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={
            "quality_solvency": DummySkillAgent("quality_solvency", ("CASH_RATIO_Q", "RANK")),
            "efficiency_growth": DummySkillAgent("efficiency_growth", ("SALES_TO_ASSETS_Q", "RANK")),
            "valuation_size": DummySkillAgent("valuation_size", ("BOOK_TO_MARKET_Q", "RANK")),
            "short_horizon_flow": DummySkillAgent("short_horizon_flow", ("RET_1", "NEG")),
        },
        planner=DummyPlanner(("short_horizon_flow", "efficiency_growth", "valuation_size")),
        reward_shaper=DummyRewardShaper(),
        critic=PositiveCrossSectionalCritic(),
        bootstrap_anchor_skill="quality_solvency",
    )
    manager.accepted_counts["quality_solvency"] = 1

    step = manager.run_step(panel, target, pool, commit=False, validation_data=panel, validation_target=target)

    assert step.selected_agent == "efficiency_growth"


def test_hierarchical_manager_blocks_cross_sectional_flow_with_negative_trade_proxy() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "RET_1": [0.01, -0.02, 0.03, -0.01],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "TARGET_XS_RET_1": [0.02, -0.01, 0.01, 0.03],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="PROFITABILITY_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.1, 2.1], index=panel.index),
            metrics={"rank_ic": 0.02, "max_corr": 0.0, "turnover": 0.01, "cross_sectional": 1.0},
            role="trend_structure",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={"short_horizon_flow": DummySkillAgent("short_horizon_flow", ("RET_1", "NEG"))},
        planner=DummyPlanner(("short_horizon_flow",)),
        reward_shaper=DummyRewardShaper(),
        critic=CrossSectionalNegativeTradeProxyCritic(),
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert not step.accepted
    assert step.decision_reason == "critic_negative_trade_proxy"


def test_hierarchical_manager_blocks_unstable_cross_sectional_flow_residual() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "RET_1": [0.01, -0.02, 0.03, -0.01],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "TARGET_XS_RET_1": [0.02, -0.01, 0.01, 0.03],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="PROFITABILITY_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.1, 2.1], index=panel.index),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.01,
                "cross_sectional": 1.0,
                "stability_score": 0.01,
                "rank_ic_window_positive_frac": 1.0,
                "ls_return_window_positive_frac": 0.75,
            },
            role="trend_structure",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={"short_horizon_flow": DummySkillAgent("short_horizon_flow", ("RET_1", "NEG"))},
        planner=DummyPlanner(("short_horizon_flow",)),
        reward_shaper=DummyRewardShaper(),
        critic=CrossSectionalUnstableFlowCritic(),
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert not step.accepted
    assert step.decision_reason == "critic_unstable_flow_residual"


def test_hierarchical_manager_keeps_trend_baseline_when_cross_sectional_flow_is_accepted() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "permno": [1, 2, 1, 2],
            "RET_1": [0.01, -0.02, 0.03, -0.01],
            "PROFITABILITY_Q": [1.0, 2.0, 1.1, 2.1],
            "TARGET_XS_RET_1": [0.02, -0.01, 0.01, 0.03],
        }
    )
    target = panel["TARGET_XS_RET_1"]
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="PROFITABILITY_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.1, 2.1], index=panel.index),
            metrics={
                "rank_ic": 0.02,
                "max_corr": 0.0,
                "turnover": 0.01,
                "cross_sectional": 1.0,
                "stability_score": 0.01,
                "rank_ic_window_positive_frac": 1.0,
                "ls_return_window_positive_frac": 1.0,
            },
            role="trend_structure",
        )
    )
    manager = HierarchicalManagerAgent(
        agents={"short_horizon_flow": DummySkillAgent("short_horizon_flow", ("RET_1", "NEG"))},
        planner=DummyPlanner(("short_horizon_flow",)),
        reward_shaper=PairAwareRewardShaper(),
        critic=CrossSectionalPositiveFlowCritic(),
    )

    step = manager.run_step(panel, target, pool, commit=True, validation_data=panel, validation_target=target)

    assert step.accepted
    assert {record.canonical for record in pool.records} == {"PROFITABILITY_Q RANK", "RET_1 NEG"}
