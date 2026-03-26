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
            robust_score=0.0411,
            full_metrics={"rank_ic": 0.0142, "sharpe": 1.02},
            slice_rank_ic=[0.0100, -0.0011, 0.0238, 0.0243],
            slice_sharpe=[1.0, 1.0, 1.0, 1.0],
            slice_turnover=[0.007, 0.007, 0.007, 0.007],
            signal=signals["CASH_RATIO_Q RANK"],
            diagnostics={
                "valid_slices": 4,
                "mean_rank_ic": 0.0143,
                "rank_ic_std": 0.0106,
                "min_rank_ic": -0.0011,
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
