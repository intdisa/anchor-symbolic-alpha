from __future__ import annotations

from scripts.run_finance_hyperparameter_study import build_ablation_variants, single_factor_configs
from scripts.run_finance_rolling_meta_validation import BASE_CONFIG


def test_hyperparameter_ablation_variants_cover_expected_controls() -> None:
    variants = build_ablation_variants(dict(BASE_CONFIG))

    assert "no_temporal_pareto" in variants
    assert variants["no_temporal_pareto"]["temporal_selection_mode"] == "legacy_linear"
    assert variants["no_cross_seed_pareto"]["cross_seed_selection_mode"] == "legacy_linear"
    assert variants["pareto_discrete_legacy"]["temporal_selection_mode"] == "pareto_discrete_legacy"
    assert variants["no_near_neighbor_tie_break"]["enable_near_neighbor_tie_break"] is False
    assert variants["no_redundancy_gate"]["enable_redundancy_gate"] is False
    assert variants["legacy_linear_selector"]["temporal_selection_mode"] == "legacy_linear"
    assert variants["legacy_linear_selector"]["cross_seed_selection_mode"] == "legacy_linear"


def test_single_factor_configs_include_center_and_threshold_perturbations() -> None:
    labels = [label for label, _ in single_factor_configs(dict(BASE_CONFIG))]
    assert "center" in labels
    assert "min_mean_rank_ic_down" in labels
    assert "near_neighbor_signal_corr_up" in labels
    assert "min_seed_support_down" in labels
