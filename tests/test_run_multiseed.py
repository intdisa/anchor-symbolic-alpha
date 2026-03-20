from experiments.run_multiseed import aggregate_variant_runs, pairwise_comparison


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
        "target_price_only": [
            {
                "seed": 7,
                "best_validation_pool_score": 0.01,
                "walk_forward_metrics": {"sharpe": 0.05, "turnover": 0.8},
            }
        ],
    }

    comparison = pairwise_comparison(runs_by_variant, "full", "target_price_only")

    assert comparison["win_counts"]["best_validation_pool_score"] == 1
    assert comparison["win_counts"]["sharpe"] == 1
    assert comparison["win_counts"]["turnover"] == 1
