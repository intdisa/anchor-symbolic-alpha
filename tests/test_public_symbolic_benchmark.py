from __future__ import annotations

from knowledge_guided_symbolic_alpha.benchmarks import generate_public_symbolic_task, run_task_benchmark


def test_public_symbolic_benchmark_prefers_consensus_method() -> None:
    tasks = [
        generate_public_symbolic_task("feynman_linear_add", seed=7, samples_per_env=32),
        generate_public_symbolic_task("feynman_linear_add", seed=17, samples_per_env=32),
    ]
    result = run_task_benchmark(tasks)

    assert result.baselines["support_adjusted_cross_seed_consensus"].selection_accuracy >= result.baselines["naive_rank_ic"].selection_accuracy
    assert result.baselines["support_adjusted_cross_seed_consensus"].selected_formula == "X0 X1 ADD"
    assert "pareto_front_selector" in result.baselines
    assert "lasso_formula_screening" in result.baselines


def test_public_symbolic_ratio_rewards_cross_seed_consensus() -> None:
    tasks = [
        generate_public_symbolic_task("feynman_ratio", seed=7, samples_per_env=24),
        generate_public_symbolic_task("feynman_ratio", seed=17, samples_per_env=24),
        generate_public_symbolic_task("feynman_ratio", seed=27, samples_per_env=24),
        generate_public_symbolic_task("feynman_ratio", seed=37, samples_per_env=24),
        generate_public_symbolic_task("feynman_ratio", seed=47, samples_per_env=24),
    ]
    result = run_task_benchmark(tasks)

    assert result.baselines["support_adjusted_cross_seed_consensus"].selected_formula == "X0 X1 DIV"
    assert (
        result.baselines["support_adjusted_cross_seed_consensus"].selection_accuracy
        >= result.baselines["best_validation_mean_rank_ic"].selection_accuracy
    )


def test_public_symbolic_product_seed_shift_favors_support_adjusted_consensus() -> None:
    tasks = [
        generate_public_symbolic_task("feynman_product_seed_shift", seed=7, samples_per_env=24),
        generate_public_symbolic_task("feynman_product_seed_shift", seed=17, samples_per_env=24),
        generate_public_symbolic_task("feynman_product_seed_shift", seed=27, samples_per_env=24),
        generate_public_symbolic_task("feynman_product_seed_shift", seed=37, samples_per_env=24),
        generate_public_symbolic_task("feynman_product_seed_shift", seed=47, samples_per_env=24),
    ]
    result = run_task_benchmark(tasks)

    assert result.baselines["support_adjusted_cross_seed_consensus"].selected_formula == "X0 X1 MUL"
    assert result.baselines["support_adjusted_cross_seed_consensus"].selection_accuracy == 1.0
    assert result.baselines["cross_seed_mean_score_consensus"].selection_accuracy == 0.0
    assert (
        result.baselines["support_adjusted_cross_seed_consensus"].selection_accuracy
        >= result.baselines["pareto_front_selector"].selection_accuracy
    )
