from dataclasses import dataclass

import pandas as pd

from knowledge_guided_symbolic_alpha.agents import AgentProposal, CompetitiveManagerAgent, ReviewerAgent
from knowledge_guided_symbolic_alpha.evaluation import AdmissionDecision, AdmissionPolicy, CandidatePoolPreview, FactorPool, FormulaEvaluator
from knowledge_guided_symbolic_alpha.evaluation.factor_pool import FactorRecord
from knowledge_guided_symbolic_alpha.evaluation.role_profiles import adapt_role_profile, resolve_role_profile
from knowledge_guided_symbolic_alpha.training.reward_shaping import RewardOutcome


def make_frame() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "GOLD_CLOSE": pd.Series(range(100, 120), index=index, dtype=float),
            "GOLD_VOLUME": pd.Series(range(1000, 1020), index=index, dtype=float),
            "CPI": pd.Series([100.0] * len(index), index=index),
            "TNX": pd.Series([2.0 + 0.01 * i for i in range(len(index))], index=index),
            "VIX": pd.Series([20.0 - 0.1 * i for i in range(len(index))], index=index),
            "DXY": pd.Series([100.0 + 0.05 * i for i in range(len(index))], index=index),
        }
    )
    target = frame["GOLD_CLOSE"].pct_change().shift(-1).fillna(0.0)
    return frame, target


def test_resolve_role_profile_returns_expected_thresholds() -> None:
    context = resolve_role_profile("context")
    assert context.commit_min_abs_rank_ic == 0.07
    assert context.commit_max_correlation == 0.75
    assert context.replacement_margin == 0.003
    assert context.commit_min_trade_proxy_gain == 0.002
    assert context.reviewer_max_turnover == 0.90
    assert context.resolved_preview_min_trade_proxy_gain == 0.001
    assert context.preview_min_validation_marginal_gain == 0.002

    target_price = resolve_role_profile("target_price")
    assert target_price.commit_min_abs_rank_ic == 0.05
    assert target_price.resolved_preview_min_abs_rank_ic == 0.025
    assert target_price.commit_min_trade_proxy_gain == 0.0
    assert target_price.reviewer_max_corr == 0.80

    target_flow = resolve_role_profile("target_flow")
    assert target_flow.commit_min_abs_rank_ic == 0.025
    assert target_flow.resolved_preview_min_abs_rank_ic == 0.010
    assert target_flow.commit_min_trade_proxy_gain == 5e-4
    assert target_flow.resolved_preview_min_trade_proxy_gain == 0.0
    assert target_flow.resolved_preview_max_correlation == 0.85
    assert target_flow.reviewer_max_turnover == 1.20
    assert target_flow.pool_complexity_penalty_scale == 0.004
    assert target_flow.reward_trade_proxy_scale == 2.5


def test_cross_sectional_role_profile_relaxes_rank_gate() -> None:
    profile = adapt_role_profile(resolve_role_profile("target_flow"), "target_flow", cross_sectional=True)

    assert profile.commit_min_abs_rank_ic == 0.01
    assert profile.reviewer_min_abs_rank_ic == 0.008
    assert profile.resolved_preview_min_abs_rank_ic == 0.008


def test_context_admission_is_stricter_than_target_price(monkeypatch) -> None:
    frame, target = make_frame()
    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.admission.evaluate_formula_metrics",
        lambda formula, data, target, evaluator=None: type(
            "Result",
            (),
            {
                "evaluated": AdmissionPolicy().evaluator.evaluate(formula, data),
                "metrics": {"rank_ic": 0.06, "rank_icir": 0.0, "turnover": 0.2, "max_drawdown": -0.1},
            },
        )(),
    )
    policy = AdmissionPolicy(min_abs_rank_ic=0.05, max_correlation=0.95)
    formula = ("GOLD_CLOSE", "RANK")
    target_decision = policy.screen(formula, frame, target, FactorPool(max_size=4), role="target_price")
    context_decision = policy.screen(formula, frame, target, FactorPool(max_size=4), role="context")

    assert target_decision.accepted
    assert context_decision.reason == "fast_ic_screen"


def test_target_flow_preview_is_relaxed_but_commit_is_not(monkeypatch) -> None:
    frame, target = make_frame()
    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.pool_scoring.evaluate_formula_metrics",
        lambda formula, data, target, evaluator=None: type(
            "Result",
            (),
            {
                "evaluated": FormulaEvaluator().evaluate(formula, data),
                "metrics": {
                    "rank_ic": 0.02,
                    "rank_icir": 0.0,
                    "turnover": 0.4,
                    "max_drawdown": -0.2,
                    "sharpe": 0.0,
                    "annual_return": 0.0,
                },
            },
        )(),
    )
    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.admission.evaluate_formula_metrics",
        lambda formula, data, target, evaluator=None: type(
            "Result",
            (),
            {
                "evaluated": FormulaEvaluator().evaluate(formula, data),
                "metrics": {
                    "rank_ic": 0.02,
                    "rank_icir": 0.0,
                    "turnover": 0.4,
                    "max_drawdown": -0.2,
                    "sharpe": 0.0,
                    "annual_return": 0.0,
                },
            },
        )(),
    )

    from knowledge_guided_symbolic_alpha.evaluation import preview_candidate_on_dataset

    preview = preview_candidate_on_dataset(
        ("GOLD_CLOSE", "RANK"),
        FactorPool(max_size=4),
        frame,
        target,
        role="target_flow",
    )
    decision = AdmissionPolicy().screen(
        ("GOLD_CLOSE", "RANK"),
        frame,
        target,
        FactorPool(max_size=4),
        role="target_flow",
    )

    assert preview.accepted
    assert decision.reason == "fast_ic_screen"
    assert preview.trade_proxy_gain < resolve_role_profile("target_flow").commit_min_trade_proxy_gain


def test_context_trade_proxy_gate_rejects_formula_that_target_price_accepts(monkeypatch) -> None:
    frame, target = make_frame()
    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.admission.evaluate_formula_metrics",
        lambda formula, data, target, evaluator=None: type(
            "Result",
            (),
            {
                "evaluated": FormulaEvaluator().evaluate(formula, data),
                "metrics": {
                    "rank_ic": 0.08,
                    "rank_icir": 0.0,
                    "turnover": 1.2,
                    "max_drawdown": -0.8,
                    "sharpe": -0.5,
                    "annual_return": -0.1,
                },
            },
        )(),
    )
    policy = AdmissionPolicy(min_abs_rank_ic=0.05, max_correlation=0.95)
    formula = ("GOLD_CLOSE", "RANK")

    target_decision = policy.screen(formula, frame, target, FactorPool(max_size=4), role="target_price")
    context_decision = policy.screen(formula, frame, target, FactorPool(max_size=4), role="context")

    assert target_decision.accepted
    assert context_decision.reason == "trade_proxy_check"
    assert context_decision.trade_proxy_gain <= resolve_role_profile("context").commit_min_trade_proxy_gain


def test_reviewer_uses_role_specific_thresholds() -> None:
    reviewer = ReviewerAgent()
    pool = FactorPool(max_size=4)
    record = FactorRecord(
        tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
        canonical="NEG(DELTA_1(GOLD_CLOSE))",
        signal=pd.Series([1.0, 2.0, 3.0]),
        metrics={"rank_ic": 0.06, "max_corr": 0.75, "turnover": 1.0},
        role="context",
    )
    decision = AdmissionDecision(True, "accepted", record, 0.01)

    assert reviewer.review(decision, pool, role="target_price").approved
    assert not reviewer.review(decision, pool, role="context").approved


def test_reviewer_allows_slightly_higher_target_flow_turnover_with_positive_trade_proxy() -> None:
    reviewer = ReviewerAgent()
    pool = FactorPool(max_size=4)
    record = FactorRecord(
        tokens=("GOLD_HL_SPREAD", "DELAY_1", "GOLD_GAP_RET", "CORR_5"),
        canonical="CORR_5(DELAY_1(GOLD_HL_SPREAD),GOLD_GAP_RET)",
        signal=pd.Series([1.0, 2.0, 1.5]),
        metrics={"rank_ic": 0.04, "max_corr": 0.2, "turnover": 1.08},
        role="target_flow",
    )
    decision = AdmissionDecision(True, "accepted", record, 0.01, trade_proxy_gain=0.01)

    assert reviewer.review(decision, pool, role="target_flow").approved


@dataclass
class DummyAgent:
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
        decision = AdmissionDecision(True, "accepted", record, 0.01)
        return RewardOutcome(reward=0.2, clipped_reward=0.2, decision=decision, components={})


def test_competitive_manager_prefers_target_price_on_tie() -> None:
    manager = CompetitiveManagerAgent(
        agents={
            "context": DummyAgent("context", ("VIX", "NEG")),
            "target_flow": DummyAgent("target_flow", ("GOLD_VOLUME", "DELTA_1")),
            "target_price": DummyAgent("target_price", ("GOLD_CLOSE", "DELTA_1", "NEG")),
        },
        reward_shaper=DummyRewardShaper(),
    )
    frame, target = make_frame()
    step = manager.run_step(frame, target, FactorPool(max_size=4), commit=False)
    assert step.selected_agent == "target_price"


class CoordinationRewardShaper:
    def shape(self, formula, data, target, pool, commit=True, role=None) -> RewardOutcome:
        del data, target, commit
        metrics_by_role = {
            "target_price": {"rank_ic": 0.08, "max_corr": 0.20, "turnover": 1.0},
            "target_flow": {"rank_ic": 0.08, "max_corr": 0.10, "turnover": 0.4},
            "context": {"rank_ic": 0.08, "max_corr": 0.60, "turnover": 0.2},
        }
        record = FactorRecord(
            tokens=tuple(formula),
            canonical=" ".join(formula),
            signal=pd.Series([1.0, 2.0, 3.0]),
            metrics=metrics_by_role.get(role, {"rank_ic": 0.08, "max_corr": 0.0, "turnover": 0.0}),
            role=role,
        )
        marginal_gain = 0.01 if role != "context" else 0.005
        decision = AdmissionDecision(True, "accepted", record, marginal_gain)
        return RewardOutcome(reward=0.2, clipped_reward=0.2, decision=decision, components={})


def test_competitive_manager_coordination_bonus_prefers_target_flow_when_price_already_in_pool() -> None:
    manager = CompetitiveManagerAgent(
        agents={
            "context": DummyAgent("context", ("VIX", "NEG")),
            "target_flow": DummyAgent("target_flow", ("GOLD_HL_SPREAD", "GOLD_GAP_RET", "CORR_5")),
            "target_price": DummyAgent("target_price", ("GOLD_CLOSE", "DELTA_1", "NEG")),
        },
        reward_shaper=CoordinationRewardShaper(),
    )
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
            canonical="NEG(DELTA_1(GOLD_CLOSE))",
            signal=pd.Series([1.0, 2.0, 3.0]),
            metrics={"rank_ic": 0.06, "max_corr": 0.10, "turnover": 0.8},
            role="target_price",
        )
    )

    step = manager.run_step(frame, target, pool, commit=False)

    assert step.selected_agent == "target_flow"


def test_competitive_manager_requires_positive_flow_standalone_trade_proxy(monkeypatch) -> None:
    manager = CompetitiveManagerAgent(
        agents={
            "target_flow": DummyAgent("target_flow", ("GOLD_HL_SPREAD", "GOLD_GAP_RET", "CORR_5")),
            "target_price": DummyAgent("target_price", ("GOLD_CLOSE", "DELTA_1", "NEG")),
        },
        reward_shaper=DummyRewardShaper(),
    )
    frame, target = make_frame()

    def fake_preview(tokens, pool, data, target, evaluator=None, role=None, **kwargs):
        del tokens, data, target, evaluator, kwargs
        if role == "target_price":
            return CandidatePoolPreview(
                accepted=True,
                reason="accepted",
                record=FactorRecord(
                    tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
                    canonical="price",
                    signal=pd.Series([1.0, 2.0, 3.0]),
                    metrics={"rank_ic": 0.08, "turnover": 0.8, "max_drawdown": -0.2},
                    role=role,
                ),
                marginal_gain=0.01,
                baseline_score=0.0,
                new_score=0.01,
                trade_proxy_gain=0.002,
                baseline_trade_proxy=0.0,
                new_trade_proxy=0.002,
            )
        trade_proxy_gain = -0.002 if len(pool.records) == 0 else 0.01
        return CandidatePoolPreview(
            accepted=True,
            reason="accepted",
            record=FactorRecord(
                tokens=("GOLD_HL_SPREAD", "GOLD_GAP_RET", "CORR_5"),
                canonical="flow",
                signal=pd.Series([1.0, 2.0, 3.0]),
                metrics={"rank_ic": 0.08, "turnover": 0.6, "max_drawdown": -0.2},
                role=role,
            ),
            marginal_gain=0.02,
            baseline_score=0.0,
            new_score=0.02,
            trade_proxy_gain=trade_proxy_gain,
            baseline_trade_proxy=0.0,
            new_trade_proxy=trade_proxy_gain,
        )

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.agents.competitive_manager_agent.preview_candidate_on_dataset",
        fake_preview,
    )

    def fake_rescore(pool, data, target):
        del data, target
        scored = FactorPool(max_size=pool.max_size)
        for record in pool.records:
            if record.role == "target_price":
                scored.add(
                    FactorRecord(
                        tokens=record.tokens,
                        canonical=record.canonical,
                        signal=record.signal,
                        metrics={
                            "rank_ic": 0.01,
                            "turnover": 0.9,
                            "max_drawdown": -0.4,
                            "sharpe": -0.2,
                            "annual_return": -0.03,
                            "max_corr": 0.1,
                        },
                        role=record.role,
                    )
                )
        return scored

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.agents.competitive_manager_agent.rescore_pool_on_dataset",
        fake_rescore,
    )
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
            canonical="price_baseline",
            signal=pd.Series([1.0, 2.0, 3.0]),
            metrics={"rank_ic": 0.06, "max_corr": 0.10, "turnover": 0.8},
            role="target_price",
        )
    )

    step = manager.run_step(frame, target, pool, commit=False, validation_data=frame, validation_target=target)

    assert step.selected_agent == "target_price"


def test_competitive_manager_can_override_price_baseline_with_better_flow(monkeypatch) -> None:
    manager = CompetitiveManagerAgent(
        agents={
            "target_flow": DummyAgent("target_flow", ("GOLD_GAP_RET", "NEG")),
            "target_price": DummyAgent("target_price", ("GOLD_CLOSE", "DELTA_1", "NEG")),
        },
        reward_shaper=DummyRewardShaper(),
    )
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
            canonical="price_baseline",
            signal=pd.Series([1.0, 2.0, 3.0]),
            metrics={"rank_ic": 0.06, "max_corr": 0.10, "turnover": 0.8},
            role="target_price",
        )
    )

    def fake_preview(tokens, pool, data, target, evaluator=None, role=None, **kwargs):
        del tokens, data, target, evaluator, kwargs
        if role == "target_price":
            return CandidatePoolPreview(
                accepted=True,
                reason="accepted",
                record=FactorRecord(
                    tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
                    canonical="price",
                    signal=pd.Series([1.0, 2.0, 3.0]),
                    metrics={"rank_ic": 0.07, "turnover": 0.9, "max_drawdown": -0.4},
                    role=role,
                ),
                marginal_gain=0.01,
                baseline_score=0.01,
                new_score=0.02,
                trade_proxy_gain=0.001,
                baseline_trade_proxy=0.01,
                new_trade_proxy=0.011,
            )
        has_price = any(record.role == "target_price" for record in pool.records)
        if has_price:
            return CandidatePoolPreview(
                accepted=False,
                reason="flow_residual_gate",
                record=FactorRecord(
                    tokens=("GOLD_GAP_RET", "NEG"),
                    canonical="flow",
                    signal=pd.Series([1.0, 2.0, 3.0]),
                    metrics={"rank_ic": 0.08, "turnover": 0.7, "max_drawdown": -0.2},
                    role=role,
                ),
                marginal_gain=-0.001,
                baseline_score=0.01,
                new_score=0.009,
                trade_proxy_gain=-0.001,
                baseline_trade_proxy=0.01,
                new_trade_proxy=0.009,
            )
        return CandidatePoolPreview(
            accepted=True,
            reason="accepted",
            record=FactorRecord(
                tokens=("GOLD_GAP_RET", "NEG"),
                canonical="flow",
                signal=pd.Series([1.0, 2.0, 3.0]),
                metrics={"rank_ic": 0.08, "turnover": 0.6, "max_drawdown": -0.1},
                role=role,
            ),
            marginal_gain=0.03,
            baseline_score=0.0,
            new_score=0.04,
            trade_proxy_gain=0.02,
            baseline_trade_proxy=0.0,
            new_trade_proxy=0.02,
        )

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.agents.competitive_manager_agent.preview_candidate_on_dataset",
        fake_preview,
    )

    def fake_rescore(pool, data, target):
        del data, target
        scored = FactorPool(max_size=pool.max_size)
        for record in pool.records:
            if record.role == "target_price":
                scored.add(
                    FactorRecord(
                        tokens=record.tokens,
                        canonical=record.canonical,
                        signal=record.signal,
                        metrics={
                            "rank_ic": 0.01,
                            "turnover": 0.9,
                            "max_drawdown": -0.4,
                            "sharpe": -0.2,
                            "annual_return": -0.03,
                            "max_corr": 0.1,
                        },
                        role=record.role,
                    )
                )
        return scored

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.agents.competitive_manager_agent.rescore_pool_on_dataset",
        fake_rescore,
    )

    step = manager.run_step(frame, target, pool, commit=False, validation_data=frame, validation_target=target)

    assert step.selected_agent == "target_flow"
    assert step.decision_reason == "flow_override"
