from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from knowledge_guided_symbolic_alpha.generation import FormulaCandidate
from knowledge_guided_symbolic_alpha.selection import CrossSeedConsensusSelector, CrossSeedSelectionRun
from knowledge_guided_symbolic_alpha.selection.robust_selector import RobustSelectorOutcome, RobustSelectorRecord


def test_cross_seed_consensus_selector_prefers_supported_champion() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "RET_1": [0.1, -0.2, 0.0, 0.1, 0.15, -0.1, 0.05, -0.02],
            "TARGET_RET_1": [0.03, -0.04, 0.0, 0.02, 0.03, -0.01, 0.02, -0.01],
        }
    )
    target = frame["TARGET_RET_1"]
    runs = [
        CrossSeedSelectionRun(
            seed=7,
            candidate_records=("A", "B"),
            selector_records=("A",),
            champion_records=("A",),
            selector_ranked_records=(
                SimpleNamespace(formula="A", robust_score=0.30, source="selector", role="anchor"),
                SimpleNamespace(formula="B", robust_score=0.31, source="selector", role="anchor"),
            ),
        ),
        CrossSeedSelectionRun(
            seed=17,
            candidate_records=("A", "B"),
            selector_records=("B",),
            champion_records=("A",),
            selector_ranked_records=(
                SimpleNamespace(formula="B", robust_score=0.31, source="selector", role="anchor"),
                SimpleNamespace(formula="A", robust_score=0.29, source="selector", role="anchor"),
            ),
        ),
        CrossSeedSelectionRun(
            seed=27,
            candidate_records=("A", "B"),
            selector_records=("A",),
            champion_records=("A",),
            selector_ranked_records=(
                SimpleNamespace(formula="A", robust_score=0.30, source="selector", role="anchor"),
                SimpleNamespace(formula="B", robust_score=0.30, source="selector", role="anchor"),
            ),
        ),
    ]
    base_candidates = [
        FormulaCandidate(formula="RET_1 NEG", source="test", role="anchor"),
        FormulaCandidate(formula="RET_1", source="test", role="anchor"),
    ]
    # Map support labels to real formulas for the rerank step.
    runs = [
        CrossSeedSelectionRun(
            seed=run.seed,
            candidate_records=tuple("RET_1 NEG" if item == "A" else "RET_1" for item in run.candidate_records),
            selector_records=tuple("RET_1 NEG" if item == "A" else "RET_1" for item in run.selector_records),
            champion_records=tuple("RET_1 NEG" if item == "A" else "RET_1" for item in run.champion_records),
            selector_ranked_records=tuple(
                SimpleNamespace(
                    formula="RET_1 NEG" if item.formula == "A" else "RET_1",
                    robust_score=item.robust_score,
                    source=item.source,
                    role=item.role,
                )
                for item in run.selector_ranked_records
            ),
        )
        for run in runs
    ]

    fake_records = [
        RobustSelectorRecord(
            formula="RET_1",
            source="test",
            role="anchor",
            selected=False,
            admissible=True,
            robust_score=0.0400,
            pairwise_wins=1,
            full_metrics={"rank_ic": 0.0400},
            slice_rank_ic=[0.04, 0.04],
            slice_sharpe=[0.0, 0.0],
            slice_turnover=[0.0, 0.0],
            diagnostics={},
        ),
        RobustSelectorRecord(
            formula="RET_1 NEG",
            source="test",
            role="anchor",
            selected=True,
            admissible=True,
            robust_score=0.0365,
            pairwise_wins=1,
            full_metrics={"rank_ic": 0.0365},
            slice_rank_ic=[0.03, 0.04],
            slice_sharpe=[0.0, 0.0],
            slice_turnover=[0.0, 0.0],
            diagnostics={},
        ),
    ]

    class FakeTemporalSelector:
        def select(self, candidates, frame_arg, target_arg):
            return RobustSelectorOutcome(
                selected_formulas=["RET_1 NEG"],
                fallback_used=False,
                records=fake_records,
                config={"selected_count": 1},
            )

    outcome = CrossSeedConsensusSelector(temporal_selector=FakeTemporalSelector()).select(
        runs,
        frame,
        target,
        base_candidates=base_candidates,
    )

    assert outcome.selected_formulas == ["RET_1 NEG"]
    assert outcome.ranked_records[0].champion_seed_support == 3


def test_cross_seed_consensus_selector_prefers_richer_near_neighbor_signal() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "RET_1": [0.01, -0.02, 0.03, -0.01, 0.02, -0.01],
            "TARGET_RET_1": [0.02, -0.01, 0.01, -0.02, 0.03, -0.01],
        }
    )
    target = frame["TARGET_RET_1"]
    rich = "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"
    simple = "CASH_RATIO_Q RANK"

    runs = [
        CrossSeedSelectionRun(
            seed=7,
            candidate_records=(rich, simple),
            selector_records=(simple,),
            champion_records=(rich,),
            selector_ranked_records=(
                SimpleNamespace(formula=simple, robust_score=0.041, source="selector", role="anchor", temporal_pareto_rank=1, temporal_tiebreak_rank=1),
                SimpleNamespace(formula=rich, robust_score=0.040, source="selector", role="anchor", temporal_pareto_rank=2, temporal_tiebreak_rank=2),
            ),
        ),
        CrossSeedSelectionRun(
            seed=17,
            candidate_records=(rich, simple),
            selector_records=(simple,),
            champion_records=(rich,),
            selector_ranked_records=(
                SimpleNamespace(formula=simple, robust_score=0.041, source="selector", role="anchor", temporal_pareto_rank=1, temporal_tiebreak_rank=1),
                SimpleNamespace(formula=rich, robust_score=0.040, source="selector", role="anchor", temporal_pareto_rank=2, temporal_tiebreak_rank=2),
            ),
        ),
        CrossSeedSelectionRun(
            seed=27,
            candidate_records=(rich, simple),
            selector_records=(simple,),
            champion_records=(rich,),
            selector_ranked_records=(
                SimpleNamespace(formula=simple, robust_score=0.041, source="selector", role="anchor", temporal_pareto_rank=1, temporal_tiebreak_rank=1),
                SimpleNamespace(formula=rich, robust_score=0.040, source="selector", role="anchor", temporal_pareto_rank=2, temporal_tiebreak_rank=2),
            ),
        ),
        CrossSeedSelectionRun(
            seed=37,
            candidate_records=(rich, simple),
            selector_records=(simple,),
            champion_records=tuple(),
            selector_ranked_records=(
                SimpleNamespace(formula=simple, robust_score=0.041, source="selector", role="anchor", temporal_pareto_rank=1, temporal_tiebreak_rank=1),
                SimpleNamespace(formula=rich, robust_score=0.040, source="selector", role="anchor", temporal_pareto_rank=2, temporal_tiebreak_rank=2),
            ),
        ),
        CrossSeedSelectionRun(
            seed=47,
            candidate_records=(rich, simple),
            selector_records=tuple(),
            champion_records=tuple(),
            selector_ranked_records=(
                SimpleNamespace(formula=rich, robust_score=0.040, source="selector", role="anchor", temporal_pareto_rank=2, temporal_tiebreak_rank=2),
                SimpleNamespace(formula=simple, robust_score=0.039, source="selector", role="anchor", temporal_pareto_rank=1, temporal_tiebreak_rank=1),
            ),
        ),
    ]

    fake_records = [
        RobustSelectorRecord(
            formula=rich,
            source="test",
            role="anchor",
            selected=True,
            admissible=True,
            robust_score=0.0400,
            pairwise_wins=1,
            full_metrics={"rank_ic": 0.0400},
            slice_rank_ic=[0.03, 0.05],
            slice_sharpe=[0.0, 0.0],
            slice_turnover=[0.0, 0.0],
            diagnostics={},
        ),
        RobustSelectorRecord(
            formula=simple,
            source="test",
            role="anchor",
            selected=False,
            admissible=True,
            robust_score=0.0410,
            pairwise_wins=1,
            full_metrics={"rank_ic": 0.0410},
            slice_rank_ic=[0.04, 0.04],
            slice_sharpe=[0.0, 0.0],
            slice_turnover=[0.0, 0.0],
            diagnostics={},
        ),
    ]

    class FakeTemporalSelector:
        def select(self, candidates, frame_arg, target_arg):
            return RobustSelectorOutcome(
                selected_formulas=[rich],
                fallback_used=False,
                records=fake_records,
                config={"selected_count": 1},
            )

    outcome = CrossSeedConsensusSelector(temporal_selector=FakeTemporalSelector()).select(
        runs,
        frame,
        target,
        base_candidates=[FormulaCandidate(formula=rich, source="test", role="anchor"), FormulaCandidate(formula=simple, source="test", role="anchor")],
    )

    assert outcome.selected_formulas == [rich]
    assert outcome.ranked_records[0].formula == rich
