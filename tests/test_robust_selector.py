from __future__ import annotations

import pandas as pd

from knowledge_guided_symbolic_alpha.benchmarks import (
    generate_synthetic_temporal_shift_panel,
    naive_rank_ic_selection,
)
from knowledge_guided_symbolic_alpha.generation import FormulaCandidate
from knowledge_guided_symbolic_alpha.selection import RobustSelectorConfig, RobustTemporalSelector
from knowledge_guided_symbolic_alpha.selection.robust_selector import _CandidateEvaluation


def test_robust_selector_prefers_stable_formula_over_spurious_mean_winner() -> None:
    benchmark = generate_synthetic_temporal_shift_panel(seed=11)
    naive_formula = naive_rank_ic_selection(benchmark.candidate_formulas, benchmark.frame, benchmark.target)
    selector = RobustTemporalSelector()
    outcome = selector.select(benchmark.candidate_formulas, benchmark.frame, benchmark.target)

    assert naive_formula == benchmark.spurious_formula
    assert outcome.selected_formulas[0] == benchmark.true_formula
    assert any(record.formula == benchmark.spurious_formula and not record.selected for record in outcome.records)


def test_robust_selector_breaks_near_neighbor_ties_with_prior_alignment(monkeypatch) -> None:
    selector = RobustTemporalSelector(RobustSelectorConfig(top_k=1))
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4, freq="D")})
    target = pd.Series([0.0, 0.0, 0.0, 0.0], index=frame.index)
    candidates = [
        FormulaCandidate(formula="PROFITABILITY_A CASH_RATIO_Q RANK ADD", source="test"),
        FormulaCandidate(formula="CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD", source="test"),
        FormulaCandidate(formula="CASH_RATIO_Q RANK", source="test"),
    ]

    shared_signal = pd.Series([0.2, 0.1, -0.1, -0.2], index=frame.index)
    signals = {
        "PROFITABILITY_A CASH_RATIO_Q RANK ADD": shared_signal,
        "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD": shared_signal * 0.98,
        "CASH_RATIO_Q RANK": pd.Series([0.25, 0.05, -0.05, -0.25], index=frame.index),
    }
    evaluations = {
        "PROFITABILITY_A CASH_RATIO_Q RANK ADD": _CandidateEvaluation(
            formula="PROFITABILITY_A CASH_RATIO_Q RANK ADD",
            source="test",
            role=None,
            admissible=True,
            robust_score=0.0400,
            full_metrics={"rank_ic": 0.0150, "sharpe": 1.07},
            slice_rank_ic=[0.0126, -0.0013, 0.0261, 0.0228],
            slice_sharpe=[1.0, 1.0, 1.0, 1.0],
            slice_turnover=[0.008, 0.008, 0.008, 0.008],
            signal=signals["PROFITABILITY_A CASH_RATIO_Q RANK ADD"],
            diagnostics={
                "valid_slices": 4,
                "mean_rank_ic": 0.0151,
                "rank_ic_std": 0.0107,
                "min_rank_ic": -0.0013,
                "mean_turnover": 0.0078,
                "knowledge_alignment": 0.0,
                "temporal_granularity_score": 0.0,
            },
        ),
        "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD": _CandidateEvaluation(
            formula="CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD",
            source="test",
            role=None,
            admissible=True,
            robust_score=0.0364,
            full_metrics={"rank_ic": 0.0154, "sharpe": 1.03},
            slice_rank_ic=[0.0183, -0.0032, 0.0261, 0.0204],
            slice_sharpe=[1.0, 1.0, 1.0, 1.0],
            slice_turnover=[0.010, 0.010, 0.010, 0.010],
            signal=signals["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"],
            diagnostics={
                "valid_slices": 4,
                "mean_rank_ic": 0.0154,
                "rank_ic_std": 0.0111,
                "min_rank_ic": -0.0032,
                "mean_turnover": 0.0096,
                "knowledge_alignment": 1.0 / 3.0,
                "temporal_granularity_score": 1.0,
            },
        ),
        "CASH_RATIO_Q RANK": _CandidateEvaluation(
            formula="CASH_RATIO_Q RANK",
            source="test",
            role=None,
            admissible=True,
            robust_score=0.0330,
            full_metrics={"rank_ic": 0.0101, "sharpe": 0.98},
            slice_rank_ic=[0.0060, -0.0080, 0.0180, 0.0240],
            slice_sharpe=[1.0, 1.0, 1.0, 1.0],
            slice_turnover=[0.007, 0.007, 0.007, 0.007],
            signal=signals["CASH_RATIO_Q RANK"],
            diagnostics={
                "valid_slices": 4,
                "mean_rank_ic": 0.0100,
                "rank_ic_std": 0.0128,
                "min_rank_ic": -0.0080,
                "mean_turnover": 0.0067,
                "knowledge_alignment": 0.0,
                "temporal_granularity_score": 1.0,
            },
        ),
    }
    single_scores = {
        "PROFITABILITY_A CASH_RATIO_Q RANK ADD": 0.0376,
        "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD": 0.0371,
        "CASH_RATIO_Q RANK": 0.0360,
    }

    def fake_evaluate_candidate(
        self: RobustTemporalSelector,
        candidate: FormulaCandidate,
        frame_arg: pd.DataFrame,
        target_arg: pd.Series,
    ) -> _CandidateEvaluation:
        return evaluations[candidate.formula]

    def fake_subset_score(
        self: RobustTemporalSelector,
        selected: list[_CandidateEvaluation],
        frame_arg: pd.DataFrame,
        target_arg: pd.Series,
    ) -> float:
        if len(selected) == 1:
            return single_scores[selected[0].formula]
        return max(single_scores[item.formula] for item in selected) - 0.01

    monkeypatch.setattr(RobustTemporalSelector, "_evaluate_candidate", fake_evaluate_candidate)
    monkeypatch.setattr(RobustTemporalSelector, "_subset_score", fake_subset_score)

    outcome = selector.select(candidates, frame, target)

    assert outcome.selected_formulas == ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"]


def _fake_eval(
    formula: str,
    *,
    mean_rank_ic: float,
    min_slice_rank_ic: float,
    rank_ic_std: float,
    turnover: float,
    complexity: float,
) -> _CandidateEvaluation:
    signal = pd.Series([0.1, -0.1, 0.2, -0.2], dtype=float)
    return _CandidateEvaluation(
        formula=formula,
        source="test",
        role=None,
        admissible=True,
        robust_score=0.0,
        full_metrics={"rank_ic": mean_rank_ic},
        slice_rank_ic=[mean_rank_ic],
        slice_sharpe=[0.0],
        slice_turnover=[turnover],
        signal=signal,
        diagnostics={
            "mean_rank_ic": mean_rank_ic,
            "min_rank_ic": min_slice_rank_ic,
            "rank_ic_std": rank_ic_std,
            "mean_turnover": turnover,
            "complexity": complexity,
            "knowledge_alignment": 0.0,
            "temporal_granularity_score": 0.0,
        },
        temporal_objective_vector={
            "mean_rank_ic": mean_rank_ic,
            "min_slice_rank_ic": min_slice_rank_ic,
            "rank_ic_std": rank_ic_std,
            "turnover": turnover,
            "complexity": complexity,
        },
    )


def test_temporal_pareto_excludes_complexity_from_dominance() -> None:
    compact = _fake_eval(
        "CASH_RATIO_Q RANK",
        mean_rank_ic=0.02,
        min_slice_rank_ic=0.01,
        rank_ic_std=0.01,
        turnover=0.02,
        complexity=2.0,
    )
    richer = _fake_eval(
        "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD",
        mean_rank_ic=0.02,
        min_slice_rank_ic=0.01,
        rank_ic_std=0.01,
        turnover=0.02,
        complexity=5.0,
    )

    pareto_selector = RobustTemporalSelector(RobustSelectorConfig(selection_mode="pareto"))
    legacy_selector = RobustTemporalSelector(RobustSelectorConfig(selection_mode="pareto_discrete_legacy"))

    pareto_ranks = pareto_selector._temporal_rank_data([compact, richer])  # noqa: SLF001
    legacy_ranks = legacy_selector._temporal_rank_data([compact, richer])  # noqa: SLF001

    assert pareto_ranks.rank_map[compact.formula] == 1
    assert pareto_ranks.rank_map[richer.formula] == 1
    assert legacy_ranks.rank_map[compact.formula] == 1
    assert legacy_ranks.rank_map[richer.formula] > 1


def test_temporal_pareto_uses_crowding_distance_within_shared_front() -> None:
    left = _fake_eval(
        "LEFT",
        mean_rank_ic=0.030,
        min_slice_rank_ic=0.010,
        rank_ic_std=0.030,
        turnover=0.030,
        complexity=2.0,
    )
    center = _fake_eval(
        "CENTER",
        mean_rank_ic=0.025,
        min_slice_rank_ic=0.015,
        rank_ic_std=0.020,
        turnover=0.020,
        complexity=2.0,
    )
    right = _fake_eval(
        "RIGHT",
        mean_rank_ic=0.020,
        min_slice_rank_ic=0.020,
        rank_ic_std=0.010,
        turnover=0.010,
        complexity=2.0,
    )

    selector = RobustTemporalSelector(RobustSelectorConfig(selection_mode="pareto", enable_crowding_distance=True))
    ranking = selector._temporal_rank_data([left, center, right])  # noqa: SLF001

    assert ranking.rank_map[left.formula] == 1
    assert ranking.rank_map[center.formula] == 1
    assert ranking.rank_map[right.formula] == 1
    assert ranking.crowding_map[left.formula] == float("inf")
    assert ranking.crowding_map[right.formula] == float("inf")
    assert ranking.crowding_map[center.formula] < float("inf")
