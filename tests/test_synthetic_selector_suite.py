from __future__ import annotations

from knowledge_guided_symbolic_alpha.benchmarks import generate_synthetic_selector_task, run_task_benchmark


def test_synthetic_selector_suite_recovers_true_formula_with_consensus() -> None:
    tasks = [
        generate_synthetic_selector_task("transient_spuriosity", seed=7, samples_per_env=32),
        generate_synthetic_selector_task("transient_spuriosity", seed=17, samples_per_env=32),
        generate_synthetic_selector_task("transient_spuriosity", seed=27, samples_per_env=32),
    ]
    result = run_task_benchmark(tasks)

    assert result.baselines["support_adjusted_cross_seed_consensus"].selected_formula == result.true_formula
    assert result.baselines["support_adjusted_cross_seed_consensus"].selection_accuracy >= result.baselines["naive_rank_ic"].selection_accuracy


def test_synthetic_near_neighbor_prefers_support_adjusted_consensus() -> None:
    tasks = [
        generate_synthetic_selector_task("near_neighbor_ambiguity", seed=7, samples_per_env=24),
        generate_synthetic_selector_task("near_neighbor_ambiguity", seed=17, samples_per_env=24),
        generate_synthetic_selector_task("near_neighbor_ambiguity", seed=27, samples_per_env=24),
        generate_synthetic_selector_task("near_neighbor_ambiguity", seed=37, samples_per_env=24),
        generate_synthetic_selector_task("near_neighbor_ambiguity", seed=47, samples_per_env=24),
    ]
    result = run_task_benchmark(tasks)

    assert result.baselines["support_adjusted_cross_seed_consensus"].selected_formula == result.true_formula
    assert (
        result.baselines["support_adjusted_cross_seed_consensus"].selection_accuracy
        > result.baselines["best_validation_mean_rank_ic"].selection_accuracy
    )
