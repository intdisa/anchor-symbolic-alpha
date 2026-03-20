import pandas as pd

from experiments.common import (
    build_manager,
    dataset_columns,
    dataset_input_frame,
    competitive_feature_partitions,
    feature_partitions,
    load_yaml,
    select_evaluation_formulas,
    resolve_route_b_skill_aliases,
    skill_family_biases,
    skill_family_feature_partitions,
    skill_family_operator_whitelists,
    skill_family_seed_formulas,
    three_way_feature_partitions,
)
from knowledge_guided_symbolic_alpha.agents import CompetitiveManagerAgent, HierarchicalManagerAgent, ManagerAgent
from knowledge_guided_symbolic_alpha.memory import ExperienceMemory


def test_feature_partitions_split_target_and_context_for_gold() -> None:
    context_features, target_features = feature_partitions("gold")
    assert "GOLD_CLOSE" in target_features
    assert "CRUDE_OIL_CLOSE" in context_features
    assert "VIX" in context_features
    assert not context_features & target_features


def test_three_way_feature_partitions_split_target_price_and_flow_for_gold() -> None:
    context_features, price_features, flow_features = three_way_feature_partitions("gold")
    assert "GOLD_CLOSE" in price_features
    assert "GOLD_GAP_RET" in price_features
    assert "GOLD_GAP_RET" in flow_features
    assert "GOLD_OC_RET" in flow_features
    assert "GOLD_REALIZED_VOL_5" in flow_features
    assert "GOLD_VOLUME_ZSCORE_20" in flow_features
    assert "CRUDE_OIL_CLOSE" in context_features
    assert "VIX" in context_features
    assert not context_features & price_features
    assert not context_features & flow_features


def test_competitive_feature_partitions_split_flow_into_vol_and_gap_roles() -> None:
    context_features, price_features, flow_vol_features, flow_gap_features = competitive_feature_partitions("gold")
    assert "GOLD_CLOSE" in price_features
    assert "GOLD_VOLUME" in flow_vol_features
    assert "GOLD_REALIZED_VOL_20" in flow_vol_features
    assert "GOLD_GAP_RET" in flow_gap_features
    assert "GOLD_OC_RET" in flow_gap_features
    assert "CRUDE_OIL_CLOSE" in context_features
    assert not context_features & flow_vol_features
    assert not context_features & flow_gap_features


def test_skill_family_feature_partitions_cover_new_hierarchy_roles() -> None:
    partitions = skill_family_feature_partitions("gold")

    assert "short_horizon_flow" in partitions
    assert "price_structure" in partitions
    assert "cross_asset_context" in partitions
    assert "GOLD_GAP_RET" in partitions["short_horizon_flow"]
    assert "GOLD_REALIZED_VOL_20" in partitions["short_horizon_flow"]
    assert "GOLD_VOLUME" not in partitions["short_horizon_flow"]
    assert "CRUDE_OIL_CLOSE" in partitions["cross_asset_context"]


def test_skill_family_feature_partitions_support_route_b() -> None:
    partitions = skill_family_feature_partitions("route_b")

    assert set(partitions) == {"quality_solvency", "efficiency_growth", "valuation_size", "short_horizon_flow"}
    assert "RET_1" in partitions["short_horizon_flow"]
    assert "CASH_RATIO_Q" in partitions["quality_solvency"]
    assert "SALES_TO_ASSETS_Q" in partitions["efficiency_growth"]
    assert "BOOK_TO_MARKET_Q" in partitions["valuation_size"]


def test_route_b_slow_skill_biases_and_seeds_follow_split_neighborhoods() -> None:
    biases = skill_family_biases("route_b")
    seeds = skill_family_seed_formulas("route_b")
    operators = skill_family_operator_whitelists()

    assert biases["quality_solvency"]["CASH_RATIO_Q"] > biases["quality_solvency"]["LEVERAGE_A"]
    assert biases["efficiency_growth"]["SALES_TO_ASSETS_Q"] > biases["efficiency_growth"]["PROFITABILITY_A"]
    assert biases["valuation_size"]["BOOK_TO_MARKET_Q"] > biases["valuation_size"]["SIZE_LOG_MCAP"]
    assert "ADD" in operators["quality_solvency"]
    assert "ADD" in operators["efficiency_growth"]
    assert seeds["quality_solvency"][0] == ("CASH_RATIO_Q", "RANK", "PROFITABILITY_Q", "RANK", "ADD")
    assert seeds["efficiency_growth"][0] == ("SALES_TO_ASSETS_Q", "RANK")
    assert seeds["valuation_size"][0] == ("BOOK_TO_MARKET_Q", "RANK")


def test_route_b_skill_aliases_expand_slow_family_group() -> None:
    assert resolve_route_b_skill_aliases(("trend_structure",)) == (
        "quality_solvency",
        "efficiency_growth",
        "valuation_size",
    )
    assert resolve_route_b_skill_aliases(("price_structure", "short_horizon_flow")) == (
        "valuation_size",
        "short_horizon_flow",
    )


def test_dataset_columns_support_route_b() -> None:
    assert dataset_columns("route_b") == ("TARGET_XS_RET_1", "TARGET_RET_1")


def test_dataset_input_frame_keeps_panel_ids_for_route_b() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "permno": [10001],
            "RET_1": [0.01],
            "TARGET_XS_RET_1": [0.02],
        }
    )
    view = dataset_input_frame(frame, ("RET_1",), "route_b")

    assert list(view.columns) == ["date", "permno", "RET_1"]


def test_build_manager_defaults_to_skill_hierarchy_for_experiments() -> None:
    manager = build_manager(load_yaml("configs/training.yaml"), dataset_name="gold")
    assert isinstance(manager, HierarchicalManagerAgent)
    assert manager.agent_names == (
        "short_horizon_flow",
        "price_structure",
        "trend_structure",
        "cross_asset_context",
        "regime_filter",
    )
    assert "CRUDE_OIL_CLOSE" in manager.agents["cross_asset_context"].allowed_features
    assert "GOLD_CLOSE" in manager.agents["price_structure"].allowed_features
    assert "GOLD_REALIZED_VOL_5" in manager.agents["short_horizon_flow"].allowed_features
    assert "GOLD_GAP_RET" in manager.agents["short_horizon_flow"].allowed_features
    assert manager.agents["short_horizon_flow"].token_score_biases["GOLD_GAP_RET"] > 0.0
    assert manager.agents["short_horizon_flow"].token_score_biases["GOLD_REALIZED_VOL_5"] > 0.0
    assert manager.agents["short_horizon_flow"].token_score_biases["CORR_5"] > 0.0
    assert manager.agents["regime_filter"].token_score_biases["VIX"] > 0.0
    assert ("GOLD_GAP_RET", "NEG") in manager.agents["short_horizon_flow"].seed_formulas
    assert ("GOLD_GAP_RET", "GOLD_REALIZED_VOL_5", "CORR_5") in manager.agents["short_horizon_flow"].seed_formulas
    assert ("VIX", "RANK", "NEG") in manager.agents["regime_filter"].seed_formulas


def test_build_manager_can_disable_memory_and_preserve_three_way_roles() -> None:
    manager = build_manager(
        load_yaml("configs/training.yaml"),
        dataset_name="gold",
        partition_mode="competitive_three_way",
        no_memory=True,
    )
    assert isinstance(manager, CompetitiveManagerAgent)
    assert isinstance(manager.experience_memory, ExperienceMemory)
    assert manager.experience_memory.success_scale == 0.0
    assert manager.experience_memory.failure_scale == 0.0


def test_build_manager_seed_override_changes_agent_generator_and_sampler_seeds() -> None:
    first = build_manager(
        load_yaml("configs/training.yaml"),
        dataset_name="gold",
        partition_mode="competitive_three_way",
        seed_override=101,
    )
    second = build_manager(
        load_yaml("configs/training.yaml"),
        dataset_name="gold",
        partition_mode="competitive_three_way",
        seed_override=202,
    )

    assert isinstance(first, CompetitiveManagerAgent)
    assert isinstance(second, CompetitiveManagerAgent)
    assert first.agents["context"].generator.seed == 198
    assert first.agents["target_price"].generator.seed == 295
    assert first.agents["target_flow_vol"].generator.seed == 392
    assert first.agents["target_flow_gap"].generator.seed == 489
    assert first.agents["context"].sampler.config.seed == 294
    assert first.agents["target_price"].sampler.config.seed == 487
    assert first.agents["target_flow_vol"].sampler.config.seed == 680
    assert first.agents["target_flow_gap"].sampler.config.seed == 873
    assert second.agents["context"].generator.seed != first.agents["context"].generator.seed
    assert second.agents["target_price"].sampler.config.seed != first.agents["target_price"].sampler.config.seed


def test_build_manager_keeps_target_context_mode_available() -> None:
    manager = build_manager(
        load_yaml("configs/training.yaml"),
        dataset_name="gold",
        partition_mode="target_context",
    )
    assert isinstance(manager, ManagerAgent)
    assert manager.agent_names == ("context", "target")
    assert "GOLD_CLOSE" in manager.micro_agent.allowed_features
    assert "CRUDE_OIL_CLOSE" in manager.macro_agent.allowed_features


def test_build_manager_can_build_skill_hierarchy_mode() -> None:
    manager = build_manager(
        load_yaml("configs/training.yaml"),
        dataset_name="gold",
        partition_mode="skill_hierarchy",
    )

    assert isinstance(manager, HierarchicalManagerAgent)
    assert manager.agent_names == (
        "short_horizon_flow",
        "price_structure",
        "trend_structure",
        "cross_asset_context",
        "regime_filter",
    )
    assert "GOLD_GAP_RET" in manager.agents["short_horizon_flow"].allowed_features
    assert "NEG" in manager.agents["short_horizon_flow"].allowed_operators
    assert manager.agents["short_horizon_flow"].max_length_cap == 5
    assert "CORR_5" in manager.agents["short_horizon_flow"].allowed_operators
    assert "TS_STD_5" not in manager.agents["short_horizon_flow"].allowed_operators
    assert "VIX" in manager.agents["regime_filter"].allowed_features


def test_build_manager_route_b_defaults_to_anchor_mode() -> None:
    manager = build_manager(
        load_yaml("configs/training_route_b_smoke.yaml"),
        dataset_name="route_b",
        partition_mode="skill_hierarchy",
    )

    assert isinstance(manager, HierarchicalManagerAgent)
    assert manager.agent_names == ("quality_solvency",)
    assert manager.bootstrap_anchor_skill == "quality_solvency"


def test_select_evaluation_formulas_prefers_best_validation_pool_when_available() -> None:
    formulas, source = select_evaluation_formulas(
        ("GOLD_CLOSE DELTA_1 NEG",),
        ("GOLD_GAP_RET NEG", "GOLD_OC_RET NEG"),
    )

    assert formulas == ["GOLD_CLOSE DELTA_1 NEG"]
    assert source == "champion_records"
