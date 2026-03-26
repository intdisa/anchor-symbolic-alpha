from types import SimpleNamespace

from experiments.run_multiseed import (
    aggregate_variant_runs,
    build_canonical_variant_result,
    build_consensus_candidate_pool,
    consensus_seed_support_threshold,
    consensus_metric_diagnostics,
    canonical_pairwise_comparison,
    pairwise_comparison,
    support_adjusted_consensus_selection,
)


def test_aggregate_variant_runs_counts_formula_and_context_survival() -> None:
    runs = [
        {
            "seed": 7,
            "best_validation_pool_score": 0.01,
            "final_pool_size": 2,
            "accepted_episodes": 3,
            "formula_count": 2,
            "context_formula_survived": False,
            "walk_forward_metrics": {"sharpe": -0.1, "turnover": 0.7},
        },
        {
            "seed": 17,
            "best_validation_pool_score": 0.02,
            "final_pool_size": 1,
            "accepted_episodes": 2,
            "formula_count": 0,
            "context_formula_survived": True,
            "walk_forward_metrics": {},
        },
    ]

    aggregated = aggregate_variant_runs(runs)

    assert aggregated["best_validation_pool_score"]["count"] == 2
    assert aggregated["formula_count"]["max"] == 2.0
    assert aggregated["seed_count_with_formulas"] == 1
    assert aggregated["context_formula_survived_count"] == 1
    assert aggregated["walk_forward_metrics"]["sharpe"]["count"] == 1


def test_pairwise_comparison_uses_metric_direction() -> None:
    runs_by_variant = {
        "full": [
            {
                "seed": 7,
                "best_validation_pool_score": 0.02,
                "walk_forward_metrics": {"sharpe": 0.1, "turnover": 0.6},
            }
        ],
        "valuation_size_only": [
            {
                "seed": 7,
                "best_validation_pool_score": 0.01,
                "walk_forward_metrics": {"sharpe": 0.05, "turnover": 0.8},
            }
        ],
    }

    comparison = pairwise_comparison(runs_by_variant, "full", "valuation_size_only")

    assert comparison["win_counts"]["best_validation_pool_score"] == 1
    assert comparison["win_counts"]["sharpe"] == 1
    assert comparison["win_counts"]["turnover"] == 1


def test_consensus_seed_support_threshold_uses_majority_rule() -> None:
    assert consensus_seed_support_threshold(1) == 1
    assert consensus_seed_support_threshold(2) == 2
    assert consensus_seed_support_threshold(5) == 3


def test_build_consensus_candidate_pool_filters_one_off_formulas() -> None:
    runs = [
        {
            "seed": 7,
            "candidate_records": ["A", "B", "X"],
            "selector_records": ["A"],
            "champion_records": ["A"],
            "selector_ranked_records": [
                {"formula": "A", "robust_score": 0.30, "source": "champion_records", "role": "anchor"},
                {"formula": "B", "robust_score": 0.20, "source": "candidate_records", "role": "anchor"},
                {"formula": "X", "robust_score": 0.10, "source": "candidate_records", "role": "anchor"},
            ],
        },
        {
            "seed": 17,
            "candidate_records": ["A", "B"],
            "selector_records": ["B"],
            "champion_records": ["A"],
            "selector_ranked_records": [
                {"formula": "B", "robust_score": 0.25, "source": "candidate_records", "role": "anchor"},
                {"formula": "A", "robust_score": 0.24, "source": "champion_records", "role": "anchor"},
            ],
        },
        {
            "seed": 27,
            "candidate_records": ["A", "B"],
            "selector_records": ["A"],
            "champion_records": ["A"],
            "selector_ranked_records": [
                {"formula": "A", "robust_score": 0.31, "source": "champion_records", "role": "anchor"},
                {"formula": "B", "robust_score": 0.19, "source": "candidate_records", "role": "anchor"},
            ],
        },
    ]

    candidates, support = build_consensus_candidate_pool(runs)

    assert [candidate.formula for candidate in candidates] == ["A", "B"]
    assert support["min_seed_support"] == 2
    assert support["candidate_pool_size"] == 2
    assert support["formula_support"]["X"]["candidate_seed_support"] == 1


def test_build_consensus_candidate_pool_accepts_dataclass_like_ranked_records() -> None:
    runs = [
        {
            "seed": 7,
            "candidate_records": ["A"],
            "selector_records": ["A"],
            "champion_records": ["A"],
            "selector_ranked_records": [
                SimpleNamespace(formula="A", robust_score=0.3, source="selector", role="anchor")
            ],
        },
        {
            "seed": 17,
            "candidate_records": ["A"],
            "selector_records": ["A"],
            "champion_records": ["A"],
            "selector_ranked_records": [
                SimpleNamespace(formula="A", robust_score=0.2, source="selector", role="anchor")
            ],
        },
    ]

    candidates, support = build_consensus_candidate_pool(runs)

    assert [candidate.formula for candidate in candidates] == ["A"]
    assert support["formula_support"]["A"]["mean_selector_robust_score"] == 0.25


def test_support_adjusted_consensus_selection_prefers_consistent_champion() -> None:
    class SelectorRecord:
        def __init__(self, formula: str, robust_score: float) -> None:
            self.formula = formula
            self.robust_score = robust_score

    class SelectorOutcome:
        def __init__(self) -> None:
            self.records = [
                SelectorRecord("CASH_ONLY", 0.041),
                SelectorRecord("CASH_PLUS_QUALITY", 0.036),
            ]

    selected, ranked = support_adjusted_consensus_selection(
        SelectorOutcome(),
        {
            "seed_count": 5,
            "formula_support": {
                "CASH_ONLY": {
                    "candidate_seed_support": 5,
                    "selector_seed_support": 3,
                    "champion_seed_support": 0,
                    "mean_selector_rank": 1.5,
                },
                "CASH_PLUS_QUALITY": {
                    "candidate_seed_support": 5,
                    "selector_seed_support": 1,
                    "champion_seed_support": 5,
                    "mean_selector_rank": 2.0,
                },
            },
        },
    )

    assert selected == ["CASH_PLUS_QUALITY"]
    assert ranked[0]["formula"] == "CASH_PLUS_QUALITY"


def test_build_canonical_variant_result_prefers_consensus_metrics() -> None:
    aggregated = {
        "walk_forward_metrics": {
            "sharpe": {"mean": 0.10, "std": 0.20, "min": -0.1, "max": 0.4},
            "mean_test_rank_ic": {"mean": 0.01, "std": 0.02, "min": -0.02, "max": 0.03},
        }
    }
    consensus = {
        "seed_count": 5,
        "min_seed_support": 3,
        "candidate_pool_size": 2,
        "fallback_used": False,
        "selector_fallback_used": False,
        "selector_records": ["CASH_PLUS_QUALITY"],
        "evaluation_formula_source": "selector_records",
        "walk_forward_metrics": {"sharpe": 0.55, "mean_test_rank_ic": 0.012},
        "support_adjusted_ranked_records": [{"formula": "CASH_PLUS_QUALITY", "support_adjusted_score": 0.08}],
    }

    canonical = build_canonical_variant_result("full", aggregated, consensus)

    assert canonical["result_kind"] == "cross_seed_consensus"
    assert canonical["selector_records"] == ["CASH_PLUS_QUALITY"]
    assert canonical["walk_forward_metrics"]["sharpe"] == 0.55
    assert canonical["raw_seed_diagnostics"]["sharpe"]["std"] == 0.20


def test_consensus_metric_diagnostics_extracts_seed_dispersion() -> None:
    diagnostics = consensus_metric_diagnostics(
        {
            "walk_forward_metrics": {
                "sharpe": {"mean": 0.3, "std": 0.1, "min": 0.0, "max": 0.5},
                "turnover": {"mean": 0.02, "std": 0.01, "min": 0.01, "max": 0.04},
            }
        }
    )

    assert diagnostics["sharpe"]["mean"] == 0.3
    assert diagnostics["turnover"]["max"] == 0.04


def test_canonical_pairwise_comparison_uses_consensus_outputs() -> None:
    comparison = canonical_pairwise_comparison(
        {
            "full": {
                "selector_records": ["A"],
                "walk_forward_metrics": {"sharpe": 0.5, "turnover": 0.01},
            },
            "short_horizon_flow_only": {
                "selector_records": ["B"],
                "walk_forward_metrics": {"sharpe": 0.2, "turnover": 0.05},
            },
        },
        "full",
        "short_horizon_flow_only",
    )

    assert comparison["metric_deltas"]["sharpe"] == 0.3
    assert comparison["metric_deltas"]["turnover"] == -0.04
    assert comparison["left_selector_records"] == ["A"]
