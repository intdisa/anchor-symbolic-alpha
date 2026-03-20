import numpy as np
import pandas as pd

from knowledge_guided_symbolic_alpha.agents import MacroAgent, ManagerAgent, MicroAgent, ReviewerAgent
from knowledge_guided_symbolic_alpha.agents.base import BaseRoleAgent
from knowledge_guided_symbolic_alpha.envs.common_knowledge import CommonKnowledgeState
from knowledge_guided_symbolic_alpha.evaluation import AdmissionDecision, AdmissionPolicy, FactorPool
from knowledge_guided_symbolic_alpha.evaluation.factor_pool import FactorRecord
from knowledge_guided_symbolic_alpha.models.controllers import GatingNet, LibraryPlanner, RegimeController
from knowledge_guided_symbolic_alpha.search import ProposalCandidate, SearchEvaluation
from knowledge_guided_symbolic_alpha.training import MultiAgentTrainer, PoolRewardShaper
from knowledge_guided_symbolic_alpha.agents.skill_family_agent import SkillFamilyAgent


def make_frame(high_vix: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    gold_close = pd.Series(100 + np.cumsum(0.4 + 0.02 * np.arange(len(index))), index=index)
    gold_volume = pd.Series(1000 + 40 * np.sin(np.arange(len(index)) / 2), index=index)
    cpi_step = 0.015 if high_vix else 0.0
    tnx_step = 0.006 if high_vix else 0.001
    cpi = pd.Series(100 + cpi_step * np.arange(len(index)), index=index)
    tnx = pd.Series(1.5 + tnx_step * np.arange(len(index)), index=index)
    vix_base = 30 if high_vix else 12
    vix_slope = 0.15 if high_vix else 0.01
    vix = pd.Series(vix_base + vix_slope * np.arange(len(index)), index=index)
    dxy = pd.Series(100 + (0.03 if high_vix else 0.0) * np.arange(len(index)), index=index)
    frame = pd.DataFrame(
        {
            "GOLD_CLOSE": gold_close,
            "GOLD_VOLUME": gold_volume,
            "CPI": cpi,
            "TNX": tnx,
            "VIX": vix,
            "DXY": dxy,
        }
    )
    target = frame["GOLD_CLOSE"].pct_change().shift(-1).fillna(0.0)
    return frame.iloc[:-1], target.iloc[:-1]


def test_macro_and_micro_agents_respect_feature_isolation() -> None:
    macro = MacroAgent()
    micro = MicroAgent()

    macro_candidate = next(candidate for candidate in macro.ranked_candidates() if candidate.valid)
    micro_candidate = next(candidate for candidate in micro.ranked_candidates() if candidate.valid)

    macro_features = {token for token in macro_candidate.body_tokens if token in macro.allowed_features | micro.allowed_features}
    micro_features = {token for token in micro_candidate.body_tokens if token in macro.allowed_features | micro.allowed_features}
    assert macro_features
    assert micro_features
    assert macro_features.issubset(macro.allowed_features)
    assert micro_features.issubset(micro.allowed_features)


def test_reviewer_can_veto_accepted_candidate() -> None:
    frame, target = make_frame()
    policy = AdmissionPolicy(min_abs_rank_ic=0.0, max_correlation=0.99)
    reviewer = ReviewerAgent(max_turnover=0.01)
    pool = FactorPool(max_size=4)

    decision = policy.screen("GOLD_CLOSE DELTA_1", frame, target, pool)
    assert decision.accepted
    review = reviewer.review(decision, pool)
    assert not review.approved
    assert review.reason == "review_high_turnover"


def test_reviewer_allows_cross_sectional_replaced_baseline_with_low_rank_ic() -> None:
    reviewer = ReviewerAgent()
    pool = FactorPool(max_size=4)
    decision = AdmissionDecision(
        accepted=True,
        reason="replaced_baseline",
        candidate=FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK"),
            canonical="RANK(CASH_RATIO_Q)",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={
                "rank_ic": 0.004,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 0.8,
                "annual_return": 0.08,
                "max_drawdown": -0.05,
                "cross_sectional": 1.0,
            },
            role="trend_structure",
        ),
        marginal_gain=0.03,
        trade_proxy_gain=0.02,
    )

    review = reviewer.review(decision, pool, role="trend_structure")

    assert review.approved
    assert review.reason == "review_accept"


def test_reviewer_allows_cross_sectional_replaced_baseline_with_small_negative_trade_proxy() -> None:
    reviewer = ReviewerAgent()
    pool = FactorPool(max_size=4)
    decision = AdmissionDecision(
        accepted=True,
        reason="replaced_baseline",
        candidate=FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            canonical="RANK(CASH_RATIO_Q)+RANK(SALES_TO_ASSETS_Q)",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={
                "rank_ic": 0.004,
                "max_corr": 0.0,
                "turnover": 0.01,
                "sharpe": 1.0,
                "annual_return": 0.07,
                "max_drawdown": -0.04,
                "cross_sectional": 1.0,
            },
            role="trend_structure",
        ),
        marginal_gain=0.01,
        trade_proxy_gain=-0.005,
    )

    review = reviewer.review(decision, pool, role="trend_structure")

    assert review.approved
    assert review.reason == "review_accept"


def test_cross_sectional_target_price_seed_override_can_force_novel_seed() -> None:
    agent = BaseRoleAgent(role="trend_structure", allowed_features=frozenset({"CASH_RATIO_Q", "PROFITABILITY_Q"}))
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK"),
            canonical="CASH_RATIO_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={"rank_ic": 0.02, "cross_sectional": 1.0},
            role="trend_structure",
        )
    )
    candidates = [
        ProposalCandidate(
            source="seed",
            body_tokens=("PROFITABILITY_Q", "RANK"),
            score=0.25,
            valid=True,
            terminal_error=None,
        )
    ]

    chosen = agent._select_novel_seed_override(
        candidates,
        pool,
        evaluator=lambda tokens: SearchEvaluation(score=0.1, accepted=False, reason="fast_ic_screen"),
    )

    assert chosen is not None
    assert chosen.body_tokens == ("PROFITABILITY_Q", "RANK")


def test_cross_sectional_target_price_seed_override_prefers_additive_seed_over_rank_only_seed() -> None:
    agent = BaseRoleAgent(
        role="trend_structure",
        allowed_features=frozenset({"CASH_RATIO_Q", "PROFITABILITY_Q", "SALES_TO_ASSETS_Q"}),
    )
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("CASH_RATIO_Q", "RANK"),
            canonical="CASH_RATIO_Q RANK",
            signal=pd.Series([1.0, 2.0, 1.5, 2.5]),
            metrics={"rank_ic": 0.02, "cross_sectional": 1.0},
            role="trend_structure",
        )
    )
    candidates = [
        ProposalCandidate(
            source="seed",
            body_tokens=("PROFITABILITY_Q", "RANK"),
            score=0.25,
            valid=True,
            terminal_error=None,
        ),
        ProposalCandidate(
            source="seed",
            body_tokens=("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            score=0.25,
            valid=True,
            terminal_error=None,
        ),
    ]

    chosen = agent._select_novel_seed_override(
        candidates,
        pool,
        evaluator=lambda tokens: SearchEvaluation(score=0.1, accepted=False, reason="fast_ic_screen"),
    )

    assert chosen is not None
    assert chosen.body_tokens == ("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD")


def test_cross_sectional_target_price_bootstrap_seed_override_prefers_additive_seed() -> None:
    agent = BaseRoleAgent(
        role="trend_structure",
        allowed_features=frozenset({"CASH_RATIO_Q", "PROFITABILITY_Q", "SALES_TO_ASSETS_Q"}),
    )
    candidates = [
        ProposalCandidate(
            source="seed",
            body_tokens=("CASH_RATIO_Q", "RANK"),
            score=0.25,
            valid=True,
            terminal_error=None,
        ),
        ProposalCandidate(
            source="seed",
            body_tokens=("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD"),
            score=0.25,
            valid=True,
            terminal_error=None,
        ),
    ]

    chosen = agent._select_bootstrap_seed_override(
        candidates,
        FactorPool(max_size=4),
        evaluator=lambda tokens: SearchEvaluation(score=0.1, accepted=False, reason="fast_ic_screen"),
    )

    assert chosen is not None
    assert chosen.body_tokens == ("CASH_RATIO_Q", "RANK", "SALES_TO_ASSETS_Q", "RANK", "ADD")


def test_manager_switches_bias_by_regime() -> None:
    calm_frame, calm_target = make_frame(high_vix=False)
    stress_frame, _ = make_frame(high_vix=True)
    del calm_target
    manager = ManagerAgent(
        macro_agent=MacroAgent(),
        micro_agent=MicroAgent(),
        reviewer_agent=ReviewerAgent(),
        gating_net=GatingNet(),
        regime_controller=RegimeController(),
        reward_shaper=PoolRewardShaper(),
    )
    pool = FactorPool(max_size=4)

    calm_agent, calm_regime, calm_macro, calm_micro = manager.select_agent(calm_frame, pool)
    stress_agent, stress_regime, stress_macro, stress_micro = manager.select_agent(stress_frame, pool)

    assert calm_regime == "BALANCED"
    assert calm_agent == "micro"
    assert calm_micro > calm_macro
    assert stress_regime == "HIGH_VOLATILITY"
    assert stress_agent == "macro"
    assert stress_macro > stress_micro


def test_multi_agent_trainer_runs_smoke() -> None:
    frame, target = make_frame(high_vix=True)
    split = 40
    manager = ManagerAgent(selection_mode="greedy", seed=5)
    trainer = MultiAgentTrainer(manager, pool_max_size=4)

    summary = trainer.train(
        frame.iloc[:split],
        target.iloc[:split],
        episodes=10,
        validation_data=frame.iloc[split:],
        validation_target=target.iloc[split:],
    )
    assert len(summary.history) == 10
    assert np.isfinite(summary.best_validation_pool_score)
    assert {episode.selected_agent for episode in summary.history}.issubset({"macro", "micro"})


def test_library_planner_keeps_flow_price_pair_in_shortlist() -> None:
    planner = LibraryPlanner(
        skill_names=(
            "short_horizon_flow",
            "price_structure",
            "trend_structure",
            "cross_asset_context",
        ),
        max_shortlist=3,
        base_skill_weights={
            "short_horizon_flow": 0.20,
            "price_structure": -0.10,
            "trend_structure": 0.15,
            "cross_asset_context": 0.12,
        },
    )
    state = CommonKnowledgeState(
        dataset_name="gold",
        regime="BALANCED",
        summary_vector=(0.0,),
        dataset_embedding=(0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.1),
        pool_embedding=(0.0,),
        pool_size=0,
        max_pool_size=4,
        occupied_skills=(),
        missing_skills=("short_horizon_flow", "price_structure"),
        redundancy=0.0,
        pool_trade_proxy=0.0,
        validation_pool_score=0.0,
    )

    decision = planner.plan(state)

    assert "short_horizon_flow" in decision.ordered_skills
    assert "price_structure" in decision.ordered_skills


def test_library_planner_boosts_trend_after_flow_is_occupied() -> None:
    planner = LibraryPlanner(
        skill_names=("short_horizon_flow", "quality_solvency", "efficiency_growth", "valuation_size"),
        max_shortlist=2,
        base_skill_weights={
            "short_horizon_flow": 0.10,
            "quality_solvency": 0.18,
            "efficiency_growth": 0.10,
            "valuation_size": 0.04,
        },
    )
    state = CommonKnowledgeState(
        dataset_name="route_b",
        regime="BALANCED",
        summary_vector=(0.0,),
        dataset_embedding=(0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.1),
        pool_embedding=(0.0,),
        pool_size=1,
        max_pool_size=4,
        occupied_skills=("short_horizon_flow",),
        missing_skills=("quality_solvency", "efficiency_growth", "valuation_size"),
        redundancy=0.0,
        pool_trade_proxy=0.02,
        validation_pool_score=0.01,
    )

    decision = planner.plan(state)

    assert decision.ordered_skills[0] == "quality_solvency"


def test_library_planner_prefers_efficiency_after_quality_anchor_for_route_b() -> None:
    planner = LibraryPlanner(
        skill_names=("quality_solvency", "efficiency_growth", "valuation_size", "short_horizon_flow"),
        max_shortlist=2,
        base_skill_weights={
            "quality_solvency": 0.18,
            "efficiency_growth": 0.10,
            "valuation_size": 0.04,
            "short_horizon_flow": 0.02,
        },
    )
    state = CommonKnowledgeState(
        dataset_name="route_b",
        regime="BALANCED",
        summary_vector=(0.0,),
        dataset_embedding=(0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.1),
        pool_embedding=(0.0,),
        pool_size=1,
        max_pool_size=4,
        occupied_skills=("quality_solvency",),
        missing_skills=("efficiency_growth", "valuation_size", "short_horizon_flow"),
        redundancy=0.0,
        pool_trade_proxy=0.02,
        validation_pool_score=0.01,
    )

    decision = planner.plan(state)

    assert decision.ordered_skills == ("efficiency_growth",)


def test_library_planner_prefers_trend_for_route_b_empty_pool() -> None:
    planner = LibraryPlanner(
        skill_names=("short_horizon_flow", "quality_solvency", "efficiency_growth", "valuation_size"),
        max_shortlist=2,
        base_skill_weights={
            "short_horizon_flow": 0.02,
            "quality_solvency": 0.18,
            "efficiency_growth": 0.10,
            "valuation_size": 0.04,
        },
    )
    state = CommonKnowledgeState(
        dataset_name="route_b",
        regime="BALANCED",
        summary_vector=(0.0,),
        dataset_embedding=(0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.1),
        pool_embedding=(0.0,),
        pool_size=0,
        max_pool_size=4,
        occupied_skills=(),
        missing_skills=("short_horizon_flow", "quality_solvency", "efficiency_growth", "valuation_size"),
        redundancy=0.0,
        pool_trade_proxy=0.0,
        validation_pool_score=0.0,
    )

    decision = planner.plan(state)

    assert decision.ordered_skills[0] == "quality_solvency"


def test_base_role_agent_prefers_novel_accepted_seed_over_duplicate() -> None:
    agent = SkillFamilyAgent(
        role="trend_structure",
        allowed_features=frozenset({"PROFITABILITY_Q", "CASH_RATIO_Q"}),
        seed_formulas=(("PROFITABILITY_Q", "RANK"), ("CASH_RATIO_Q", "RANK")),
    )
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="RANK(PROFITABILITY_Q)",
            signal=pd.Series([1.0, 2.0]),
            metrics={"rank_ic": 0.02},
            role="trend_structure",
        )
    )
    candidates = [
        ProposalCandidate(source="seed", body_tokens=("PROFITABILITY_Q", "RANK"), score=0.25, valid=True),
        ProposalCandidate(source="seed", body_tokens=("CASH_RATIO_Q", "RANK"), score=0.25, valid=True),
    ]

    def evaluator(tokens):
        if tokens == ("CASH_RATIO_Q", "RANK"):
            return SearchEvaluation(score=0.12, accepted=True, reason="accepted")
        return SearchEvaluation(score=-0.20, accepted=False, reason="duplicate_canonical")

    best = agent._select_novel_seed_override(candidates, pool, evaluator)

    assert best is not None
    assert best.body_tokens == ("CASH_RATIO_Q", "RANK")


def test_trend_structure_override_pool_can_replace_same_role_baseline() -> None:
    agent = SkillFamilyAgent(
        role="trend_structure",
        allowed_features=frozenset({"PROFITABILITY_Q", "CASH_RATIO_Q"}),
        seed_formulas=(("PROFITABILITY_Q", "RANK"), ("CASH_RATIO_Q", "RANK")),
    )
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="RANK(PROFITABILITY_Q)",
            signal=pd.Series([1.0, 2.0]),
            metrics={"rank_ic": 0.02},
            role="trend_structure",
        )
    )

    override_pool = agent._override_pool(pool)

    assert override_pool is not None
    assert override_pool.records == []


def test_base_role_agent_prefers_nonduplicate_ranked_candidate_when_top_is_duplicate() -> None:
    agent = SkillFamilyAgent(
        role="trend_structure",
        allowed_features=frozenset({"PROFITABILITY_Q", "CASH_RATIO_Q"}),
    )
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="RANK(PROFITABILITY_Q)",
            signal=pd.Series([1.0, 2.0]),
            metrics={"rank_ic": 0.02},
            role="trend_structure",
        )
    )
    ranked = [
        ProposalCandidate(source="beam", body_tokens=("PROFITABILITY_Q", "RANK"), score=1.0, valid=True),
        ProposalCandidate(source="seed", body_tokens=("CASH_RATIO_Q", "RANK"), score=0.8, valid=True),
    ]

    best = agent._prefer_novel_ranked_candidate(ranked, pool)

    assert best is not None
    assert best.body_tokens == ("CASH_RATIO_Q", "RANK")
