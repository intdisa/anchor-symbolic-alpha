from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from knowledge_guided_symbolic_alpha.agents import (
    CompetitiveManagerAgent,
    FeatureGroupAgent,
    HierarchicalManagerAgent,
    ManagerAgent,
    SkillFamilyAgent,
)
from knowledge_guided_symbolic_alpha.backtest import PortfolioConfig, SignalFusionConfig, WalkForwardConfig
from knowledge_guided_symbolic_alpha.dataio import build_gold_dataset, load_processed_route_b_splits
from knowledge_guided_symbolic_alpha.domain.feature_registry import FEATURE_REGISTRY
from knowledge_guided_symbolic_alpha.envs import CommonKnowledgeEncoder, StateEncoder
from knowledge_guided_symbolic_alpha.memory import ExperienceMemory
from knowledge_guided_symbolic_alpha.models.generator import RNNGenerator, TransformerGenerator
from knowledge_guided_symbolic_alpha.models.controllers import AgentWeights, GatingNet, LibraryPlanner
from knowledge_guided_symbolic_alpha.search import BeamSearchConfig, GrammarMCTSConfig, SamplingConfig
from knowledge_guided_symbolic_alpha.training import FormulaCurriculum


DEFAULT_DATA_CONFIG = Path("configs/data.yaml")
DEFAULT_TRAINING_CONFIG = Path("configs/training.yaml")
DEFAULT_BACKTEST_CONFIG = Path("configs/backtest.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/gold.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs")

TARGET_COLUMNS = {
    "gold": "TARGET_GOLD_FWD_RET_1",
    "crude_oil": "TARGET_CRUDE_OIL_FWD_RET_1",
    "sp500": "TARGET_SP500_FWD_RET_1",
    "route_b": "TARGET_XS_RET_1",
}

RETURN_COLUMNS = {
    "gold": "GOLD_RET_1",
    "crude_oil": "CRUDE_OIL_RET_1",
    "sp500": "SP500_RET_1",
    "route_b": "TARGET_RET_1",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_name(path: str | Path) -> str:
    payload = load_yaml(path)
    return str(payload["experiment"]["dataset"])


def load_dataset_bundle(data_config: str | Path = DEFAULT_DATA_CONFIG):
    payload = load_yaml(data_config)
    if "route_b_subset" in payload:
        return load_processed_route_b_splits(payload["route_b_subset"]["split_root"])
    return build_gold_dataset(config_path=data_config)


def dataset_columns(dataset_name: str) -> tuple[str, str]:
    if dataset_name not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported dataset {dataset_name!r}.")
    return TARGET_COLUMNS[dataset_name], RETURN_COLUMNS[dataset_name]


def dataset_prefix(dataset_name: str) -> str:
    mapping = {
        "gold": "GOLD",
        "crude_oil": "CRUDE_OIL",
        "sp500": "SP500",
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset {dataset_name!r}.")
    return mapping[dataset_name]


def feature_partitions(dataset_name: str) -> tuple[frozenset[str], frozenset[str]]:
    prefix = dataset_prefix(dataset_name) + "_"
    target_features = frozenset(
        name
        for name, spec in FEATURE_REGISTRY.items()
        if spec.is_micro and name.startswith(prefix)
    )
    context_features = frozenset(FEATURE_REGISTRY) - target_features
    return context_features, target_features


def three_way_feature_partitions(dataset_name: str) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    prefix = dataset_prefix(dataset_name)
    target_price_features = frozenset(
        {
            f"{prefix}_CLOSE",
            f"{prefix}_OPEN",
            f"{prefix}_HIGH",
            f"{prefix}_LOW",
            f"{prefix}_OC_RET",
            f"{prefix}_GAP_RET",
        }
    )
    target_flow_features = frozenset(
        {
            f"{prefix}_VOLUME",
            f"{prefix}_HL_SPREAD",
            f"{prefix}_REALIZED_VOL_5",
            f"{prefix}_REALIZED_VOL_20",
            f"{prefix}_VOLUME_ZSCORE_20",
            f"{prefix}_OC_RET",
            f"{prefix}_GAP_RET",
        }
    )
    context_features = frozenset(FEATURE_REGISTRY) - target_price_features - target_flow_features
    return context_features, target_price_features, target_flow_features


def competitive_feature_partitions(
    dataset_name: str,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    prefix = dataset_prefix(dataset_name)
    target_price_features = frozenset(
        {
            f"{prefix}_CLOSE",
            f"{prefix}_OPEN",
            f"{prefix}_HIGH",
            f"{prefix}_LOW",
            f"{prefix}_OC_RET",
            f"{prefix}_GAP_RET",
        }
    )
    target_flow_vol_features = frozenset(
        {
            f"{prefix}_VOLUME",
            f"{prefix}_HL_SPREAD",
            f"{prefix}_REALIZED_VOL_5",
            f"{prefix}_REALIZED_VOL_20",
            f"{prefix}_VOLUME_ZSCORE_20",
        }
    )
    target_flow_gap_features = frozenset(
        {
            f"{prefix}_GAP_RET",
            f"{prefix}_OC_RET",
            f"{prefix}_HL_SPREAD",
        }
    )
    context_features = (
        frozenset(FEATURE_REGISTRY)
        - target_price_features
        - target_flow_vol_features
        - target_flow_gap_features
    )
    return context_features, target_price_features, target_flow_vol_features, target_flow_gap_features


def skill_family_feature_partitions(dataset_name: str) -> dict[str, frozenset[str]]:
    if dataset_name == "route_b":
        return {
            "quality_solvency": frozenset(
                {
                    "CASH_RATIO_Q",
                    "PROFITABILITY_Q",
                    "PROFITABILITY_A",
                    "LEVERAGE_Q",
                    "LEVERAGE_A",
                    "SIZE_LOG_MCAP",
                }
            ),
            "efficiency_growth": frozenset(
                {
                    "SALES_TO_ASSETS_Q",
                    "ASSET_GROWTH_A",
                    "PROFITABILITY_A",
                    "SIZE_LOG_MCAP",
                }
            ),
            "valuation_size": frozenset(
                {
                    "BOOK_TO_MARKET_Q",
                    "BOOK_TO_MARKET_A",
                    "SIZE_LOG_MCAP",
                }
            ),
            "short_horizon_flow": frozenset(
                {
                    "RET_1",
                    "RET_5",
                    "VOLATILITY_20",
                    "TURNOVER_20",
                    "DOLLAR_VOLUME_20",
                    "AMIHUD_20",
                    "PRICE_TO_252_HIGH",
                }
            ),
        }
    prefix = dataset_prefix(dataset_name)
    macro_features = frozenset({"CPI", "TNX", "VIX", "DXY"})
    target_price_features = frozenset(
        {
            f"{prefix}_CLOSE",
            f"{prefix}_OPEN",
            f"{prefix}_HIGH",
            f"{prefix}_LOW",
            f"{prefix}_OC_RET",
            f"{prefix}_GAP_RET",
            f"{prefix}_HL_SPREAD",
        }
    )
    target_vol_features = frozenset(
        {
            f"{prefix}_GAP_RET",
            f"{prefix}_OC_RET",
            f"{prefix}_HL_SPREAD",
            f"{prefix}_REALIZED_VOL_5",
            f"{prefix}_REALIZED_VOL_20",
        }
    )
    context_features = frozenset(FEATURE_REGISTRY) - frozenset(
        feature for feature in FEATURE_REGISTRY if feature.startswith(f"{prefix}_")
    )
    return {
        "short_horizon_flow": frozenset(
            {
                f"{prefix}_GAP_RET",
                f"{prefix}_OC_RET",
                f"{prefix}_HL_SPREAD",
                f"{prefix}_REALIZED_VOL_5",
                f"{prefix}_REALIZED_VOL_20",
            }
        ),
        "price_structure": target_price_features,
        "trend_structure": frozenset(
            {
                f"{prefix}_CLOSE",
                f"{prefix}_OPEN",
                f"{prefix}_HIGH",
                f"{prefix}_LOW",
                f"{prefix}_REALIZED_VOL_20",
            }
        ),
        "cross_asset_context": context_features,
        "regime_filter": macro_features,
    }


def skill_family_operator_whitelists() -> dict[str, frozenset[str]]:
    return {
        "short_horizon_flow": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "CORR_5", "DIV"}),
        "quality_solvency": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "ADD"}),
        "efficiency_growth": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "ADD"}),
        "valuation_size": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "ADD"}),
        "price_structure": frozenset({"NEG", "DELAY_1", "DELTA_1", "SUB", "RANK", "ABS"}),
        "trend_structure": frozenset({"NEG", "DELAY_1", "DELTA_1", "TS_MEAN_5", "RANK", "SUB", "ADD"}),
        "cross_asset_context": frozenset({"NEG", "DELAY_1", "RANK", "CORR_5", "SUB"}),
        "regime_filter": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "CORR_5"}),
    }


def target_feature_biases(dataset_name: str) -> dict[str, float]:
    prefix = dataset_prefix(dataset_name)
    biases = {
        f"{prefix}_GAP_RET": 0.20,
        f"{prefix}_OC_RET": 0.12,
        f"{prefix}_HL_SPREAD": 0.10,
        f"{prefix}_REALIZED_VOL_5": 0.18,
        f"{prefix}_REALIZED_VOL_20": 0.12,
        f"{prefix}_VOLUME_ZSCORE_20": 0.14,
        "DELTA_1": 0.12,
        "NEG": 0.10,
        "RANK": 0.08,
        "TS_STD_5": 0.10,
        "TS_MEAN_5": 0.06,
    }
    return biases


def target_price_biases(dataset_name: str) -> dict[str, float]:
    prefix = dataset_prefix(dataset_name)
    return {
        f"{prefix}_GAP_RET": 0.24,
        f"{prefix}_OC_RET": 0.20,
        f"{prefix}_LOW": 0.08,
        f"{prefix}_HIGH": 0.06,
        f"{prefix}_CLOSE": 0.08,
        "DELTA_1": 0.14,
        "NEG": 0.12,
        "RANK": 0.08,
        "TS_MEAN_5": 0.06,
        "SUB": 0.04,
    }


def target_flow_biases(dataset_name: str) -> dict[str, float]:
    prefix = dataset_prefix(dataset_name)
    return {
        f"{prefix}_REALIZED_VOL_5": 0.28,
        f"{prefix}_REALIZED_VOL_20": 0.24,
        f"{prefix}_VOLUME_ZSCORE_20": 0.16,
        f"{prefix}_HL_SPREAD": 0.22,
        f"{prefix}_VOLUME": 0.10,
        f"{prefix}_GAP_RET": 0.10,
        f"{prefix}_OC_RET": 0.08,
        "TS_STD_5": 0.08,
        "TS_MEAN_5": 0.10,
        "DELTA_1": 0.14,
        "NEG": 0.06,
        "RANK": 0.12,
        "CORR_5": 0.05,
        "SUB": 0.08,
        "DIV": 0.12,
        "MUL": -0.04,
    }


def target_flow_vol_biases(dataset_name: str) -> dict[str, float]:
    prefix = dataset_prefix(dataset_name)
    return {
        f"{prefix}_REALIZED_VOL_5": 0.28,
        f"{prefix}_REALIZED_VOL_20": 0.24,
        f"{prefix}_VOLUME_ZSCORE_20": 0.22,
        f"{prefix}_HL_SPREAD": 0.18,
        f"{prefix}_VOLUME": 0.08,
        "TS_STD_5": 0.14,
        "TS_MEAN_5": 0.12,
        "DELTA_1": 0.10,
        "NEG": 0.08,
        "RANK": 0.10,
        "CORR_5": 0.08,
        "SUB": 0.06,
        "DIV": 0.08,
        "MUL": -0.06,
    }


def target_flow_gap_biases(dataset_name: str) -> dict[str, float]:
    prefix = dataset_prefix(dataset_name)
    return {
        f"{prefix}_GAP_RET": 0.26,
        f"{prefix}_OC_RET": 0.22,
        f"{prefix}_HL_SPREAD": 0.12,
        "DELTA_1": 0.14,
        "NEG": 0.12,
        "RANK": 0.10,
        "CORR_5": 0.08,
        "SUB": 0.10,
        "MUL": 0.02,
        "DIV": -0.04,
    }


def context_feature_biases(dataset_name: str) -> dict[str, float]:
    del dataset_name
    return {
        "VIX": 0.10,
        "TNX": 0.08,
        "DXY": 0.08,
        "CPI": 0.05,
        "DELAY_1": 0.10,
        "RANK": 0.08,
        "CORR_5": 0.06,
        "NEG": 0.04,
        "MUL": -0.10,
        "DIV": -0.12,
    }


def target_seed_formulas(dataset_name: str) -> tuple[tuple[str, ...], ...]:
    prefix = dataset_prefix(dataset_name)
    return (
        (f"{prefix}_CLOSE", "DELTA_1", "NEG"),
        (f"{prefix}_GAP_RET", "NEG"),
        (f"{prefix}_GAP_RET", "RANK", "NEG"),
        (f"{prefix}_REALIZED_VOL_5", "TS_MEAN_5"),
        (f"{prefix}_LOW", "TS_STD_5"),
        (f"{prefix}_CLOSE", "TS_STD_5"),
        (f"{prefix}_VOLUME_ZSCORE_20", "DELTA_1", "NEG"),
    )


def target_price_seed_formulas(dataset_name: str) -> tuple[tuple[str, ...], ...]:
    prefix = dataset_prefix(dataset_name)
    return (
        (f"{prefix}_CLOSE", "DELTA_1", "NEG"),
        (f"{prefix}_GAP_RET", "NEG"),
        (f"{prefix}_GAP_RET", "RANK", "NEG"),
        (f"{prefix}_OC_RET", "NEG"),
        (f"{prefix}_LOW", f"{prefix}_CLOSE", "SUB"),
    )


def target_flow_seed_formulas(dataset_name: str) -> tuple[tuple[str, ...], ...]:
    prefix = dataset_prefix(dataset_name)
    return (
        (f"{prefix}_GAP_RET", "RANK", f"{prefix}_HL_SPREAD", "CORR_5"),
        (f"{prefix}_OC_RET", "NEG"),
        (f"{prefix}_OC_RET", f"{prefix}_REALIZED_VOL_5", "CORR_5"),
        (f"{prefix}_REALIZED_VOL_20", "DELTA_1", f"{prefix}_HL_SPREAD", "DELTA_1", "DIV"),
        (f"{prefix}_HL_SPREAD", f"{prefix}_REALIZED_VOL_5", "DELTA_1", "DIV"),
        (f"{prefix}_HL_SPREAD", "RANK", f"{prefix}_VOLUME", "RANK", "DIV"),
        (f"{prefix}_HL_SPREAD", "RANK", f"{prefix}_VOLUME_ZSCORE_20", "RANK", "SUB"),
        (f"{prefix}_VOLUME_ZSCORE_20", "TS_STD_5", f"{prefix}_VOLUME", "DELTA_1", "CORR_5"),
        (f"{prefix}_REALIZED_VOL_5", "TS_MEAN_5", "NEG"),
        (f"{prefix}_VOLUME_ZSCORE_20", "DELTA_1", "NEG"),
    )


def target_flow_vol_seed_formulas(dataset_name: str) -> tuple[tuple[str, ...], ...]:
    prefix = dataset_prefix(dataset_name)
    return (
        (f"{prefix}_REALIZED_VOL_5", "TS_MEAN_5", "NEG"),
        (f"{prefix}_REALIZED_VOL_20", "DELTA_1", f"{prefix}_HL_SPREAD", "DELTA_1", "DIV"),
        (f"{prefix}_HL_SPREAD", f"{prefix}_REALIZED_VOL_5", "DELTA_1", "DIV"),
        (f"{prefix}_HL_SPREAD", "RANK", f"{prefix}_VOLUME_ZSCORE_20", "RANK", "SUB"),
        (f"{prefix}_VOLUME_ZSCORE_20", "TS_STD_5", f"{prefix}_VOLUME", "DELTA_1", "CORR_5"),
        (f"{prefix}_VOLUME_ZSCORE_20", "DELTA_1", "NEG"),
    )


def target_flow_gap_seed_formulas(dataset_name: str) -> tuple[tuple[str, ...], ...]:
    prefix = dataset_prefix(dataset_name)
    return (
        (f"{prefix}_GAP_RET", "NEG"),
        (f"{prefix}_OC_RET", "NEG"),
        (f"{prefix}_GAP_RET", "RANK", f"{prefix}_HL_SPREAD", "CORR_5"),
        (f"{prefix}_GAP_RET", f"{prefix}_OC_RET", "SUB", "NEG"),
        (f"{prefix}_OC_RET", "DELTA_1", "NEG"),
        (f"{prefix}_GAP_RET", f"{prefix}_HL_SPREAD", "SUB", "NEG"),
    )


def skill_family_biases(dataset_name: str) -> dict[str, dict[str, float]]:
    if dataset_name == "route_b":
        return {
            "quality_solvency": {
                "CASH_RATIO_Q": 0.30,
                "PROFITABILITY_Q": 0.24,
                "PROFITABILITY_A": 0.10,
                "LEVERAGE_Q": 0.18,
                "LEVERAGE_A": 0.06,
                "SIZE_LOG_MCAP": 0.04,
                "RANK": 0.18,
                "ADD": 0.18,
                "SUB": 0.10,
                "NEG": 0.10,
            },
            "efficiency_growth": {
                "SALES_TO_ASSETS_Q": 0.28,
                "ASSET_GROWTH_A": 0.18,
                "PROFITABILITY_A": 0.10,
                "SIZE_LOG_MCAP": 0.04,
                "RANK": 0.18,
                "ADD": 0.14,
                "SUB": 0.14,
                "NEG": 0.12,
            },
            "valuation_size": {
                "BOOK_TO_MARKET_Q": 0.24,
                "BOOK_TO_MARKET_A": 0.18,
                "SIZE_LOG_MCAP": 0.16,
                "RANK": 0.16,
                "SUB": 0.14,
                "ADD": 0.10,
                "NEG": 0.10,
            },
            "short_horizon_flow": {
                "RET_1": 0.24,
                "RET_5": 0.18,
                "VOLATILITY_20": 0.18,
                "TURNOVER_20": 0.16,
                "DOLLAR_VOLUME_20": 0.12,
                "AMIHUD_20": 0.16,
                "PRICE_TO_252_HIGH": 0.10,
                "NEG": 0.12,
                "RANK": 0.10,
                "DELTA_1": 0.08,
                "TS_STD_5": 0.12,
                "TS_MEAN_5": 0.08,
                "CORR_5": 0.10,
            },
        }
    prefix = dataset_prefix(dataset_name)
    return {
        "short_horizon_flow": {
            f"{prefix}_GAP_RET": 0.28,
            f"{prefix}_OC_RET": 0.22,
            f"{prefix}_HL_SPREAD": 0.14,
            f"{prefix}_REALIZED_VOL_5": 0.26,
            f"{prefix}_REALIZED_VOL_20": 0.10,
            "NEG": 0.14,
            "DELAY_1": 0.08,
            "DELTA_1": 0.10,
            "RANK": 0.08,
            "SUB": 0.08,
            "CORR_5": 0.18,
            "DIV": 0.10,
        },
        "price_structure": {
            f"{prefix}_OPEN": 0.08,
            f"{prefix}_CLOSE": 0.10,
            f"{prefix}_HIGH": 0.06,
            f"{prefix}_LOW": 0.06,
            f"{prefix}_OC_RET": 0.18,
            f"{prefix}_HL_SPREAD": 0.12,
            "SUB": 0.12,
            "NEG": 0.10,
            "DELAY_1": 0.08,
            "DELTA_1": 0.08,
            "RANK": 0.06,
        },
        "trend_structure": {
            f"{prefix}_CLOSE": 0.12,
            f"{prefix}_OPEN": 0.08,
            f"{prefix}_HIGH": 0.06,
            f"{prefix}_LOW": 0.06,
            f"{prefix}_REALIZED_VOL_20": 0.10,
            "DELAY_1": 0.12,
            "TS_MEAN_5": 0.10,
            "DELTA_1": 0.08,
            "NEG": 0.06,
            "RANK": 0.06,
        },
        "cross_asset_context": context_feature_biases(dataset_name),
        "regime_filter": {
            "VIX": 0.12,
            "DXY": 0.10,
            "TNX": 0.08,
            "CPI": 0.05,
            "DELAY_1": 0.10,
            "DELTA_1": 0.08,
            "NEG": 0.06,
            "RANK": 0.06,
            "CORR_5": 0.06,
        },
    }


def skill_family_seed_formulas(dataset_name: str) -> dict[str, tuple[tuple[str, ...], ...]]:
    if dataset_name == "route_b":
        return {
            "quality_solvency": (
                ("CASH_RATIO_Q", "RANK", "PROFITABILITY_Q", "RANK", "ADD"),
                ("CASH_RATIO_Q", "RANK"),
                ("PROFITABILITY_Q", "RANK"),
                ("PROFITABILITY_A", "RANK"),
                ("LEVERAGE_Q", "RANK", "NEG"),
                ("CASH_RATIO_Q", "RANK", "LEVERAGE_Q", "RANK", "SUB"),
                ("PROFITABILITY_Q", "RANK", "LEVERAGE_Q", "RANK", "SUB"),
            ),
            "efficiency_growth": (
                ("SALES_TO_ASSETS_Q", "RANK"),
                ("ASSET_GROWTH_A", "RANK", "NEG"),
                ("SALES_TO_ASSETS_Q", "RANK", "ASSET_GROWTH_A", "RANK", "SUB"),
                ("SALES_TO_ASSETS_Q", "RANK", "PROFITABILITY_A", "RANK", "ADD"),
                ("PROFITABILITY_A", "RANK", "ASSET_GROWTH_A", "RANK", "SUB"),
            ),
            "valuation_size": (
                ("BOOK_TO_MARKET_Q", "RANK"),
                ("BOOK_TO_MARKET_A", "RANK"),
                ("SIZE_LOG_MCAP", "RANK", "NEG"),
                ("BOOK_TO_MARKET_Q", "RANK", "SIZE_LOG_MCAP", "RANK", "SUB"),
                ("BOOK_TO_MARKET_A", "RANK", "SIZE_LOG_MCAP", "RANK", "SUB"),
            ),
            "short_horizon_flow": (
                ("RET_1", "NEG"),
                ("RET_5", "NEG"),
                ("TURNOVER_20", "RANK", "NEG"),
                ("AMIHUD_20", "RANK", "NEG"),
                ("RET_1", "VOLATILITY_20", "CORR_5"),
                ("PRICE_TO_252_HIGH", "NEG"),
            ),
        }
    prefix = dataset_prefix(dataset_name)
    return {
        "short_horizon_flow": (
            (f"{prefix}_GAP_RET", "NEG"),
            (f"{prefix}_OC_RET", "NEG"),
            (f"{prefix}_GAP_RET", f"{prefix}_REALIZED_VOL_5", "CORR_5"),
            (f"{prefix}_GAP_RET", f"{prefix}_REALIZED_VOL_20", "CORR_5"),
            (f"{prefix}_OC_RET", f"{prefix}_HL_SPREAD", "DIV"),
            (f"{prefix}_GAP_RET", f"{prefix}_OC_RET", "SUB", "NEG"),
        ),
        "price_structure": (
            (f"{prefix}_CLOSE", f"{prefix}_OPEN", "SUB"),
            (f"{prefix}_HIGH", f"{prefix}_CLOSE", "SUB"),
            (f"{prefix}_CLOSE", f"{prefix}_LOW", "SUB"),
            (f"{prefix}_OC_RET", "NEG"),
        ),
        "trend_structure": (
            (f"{prefix}_CLOSE", "DELAY_1", f"{prefix}_CLOSE", "SUB", "NEG"),
            (f"{prefix}_CLOSE", "TS_MEAN_5", "DELTA_1", "NEG"),
            (f"{prefix}_HIGH", f"{prefix}_LOW", "SUB"),
            (f"{prefix}_CLOSE", "RANK"),
        ),
        "cross_asset_context": (
            ("VIX", "DELTA_1", "NEG"),
            ("DXY", "DELAY_1", "NEG"),
            ("TNX", "DELAY_1", "DXY", "DELAY_1", "CORR_5"),
        ),
        "regime_filter": (
            ("VIX", "RANK", "NEG"),
            ("TNX", "DELTA_1", "NEG"),
            ("DXY", "DELAY_1", "NEG"),
            ("VIX", "DXY", "CORR_5"),
        ),
    }


def resolve_route_b_skill_aliases(skill_names: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if skill_names is None:
        return None
    alias_map = {
        "trend_structure": ("quality_solvency", "efficiency_growth", "valuation_size"),
        "price_structure": ("valuation_size",),
    }
    expanded: list[str] = []
    for name in skill_names:
        expanded.extend(alias_map.get(name, (name,)))
    return tuple(dict.fromkeys(expanded))


def build_curriculum(training_config: dict[str, Any]) -> FormulaCurriculum | None:
    enabled = bool(training_config.get("training", {}).get("curriculum", {}).get("enabled", True))
    return FormulaCurriculum() if enabled else None


def training_seed(training_config: dict[str, Any], seed_override: int | None = None) -> int:
    training = training_config.get("training", {})
    return int(seed_override if seed_override is not None else training.get("seed", 7))


def build_sequence_generator(training_config: dict[str, Any], seed: int):
    training = training_config.get("training", {})
    generator_name = str(training.get("generator", "transformer")).lower()
    pretraining = training.get("pretraining", {})
    if generator_name == "transformer":
        generator = TransformerGenerator(seed=seed)
    elif generator_name == "rnn":
        generator = RNNGenerator(seed=seed)
    else:
        raise ValueError(f"Unsupported generator {generator_name!r}.")
    checkpoint = pretraining.get("checkpoint")
    if bool(pretraining.get("load_checkpoint", False)) and checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            generator.load_checkpoint(checkpoint_path)
    return generator


def role_agent_kwargs(training_config: dict[str, Any], base_seed: int, role_index: int) -> dict[str, Any]:
    generator_seed = base_seed + 97 * (role_index + 1)
    sampling_seed = base_seed + 193 * (role_index + 1)
    return {
        "generator": build_sequence_generator(training_config, seed=generator_seed),
        "sampling_config": SamplingConfig(seed=sampling_seed),
    }


def boosted_flow_search_kwargs(base_seed: int, role_index: int) -> dict[str, Any]:
    return {
        "beam_config": BeamSearchConfig(
            beam_width=6,
            per_node_top_k=4,
            max_steps=18,
            length_penalty=0.02,
        ),
        "mcts_config": GrammarMCTSConfig(
            simulations=28,
            top_k_expansion=5,
            rollout_depth=6,
            exploration_constant=1.15,
        ),
        "sampling_config": SamplingConfig(
            seed=base_seed + 193 * (role_index + 1),
            temperature=0.8,
            top_k=4,
            max_steps=18,
        ),
    }


def build_manager(
    training_config: dict[str, Any],
    dataset_name: str = "gold",
    partition_mode: str = "skill_hierarchy",
    fixed_agent_name: str | None = None,
    no_memory: bool = False,
    seed_override: int | None = None,
    included_agent_names: tuple[str, ...] | None = None,
    seed_priority_enabled: bool = True,
    allow_validation_backed_bootstrap: bool = True,
    allow_validation_backed_replacement: bool = True,
    allow_validation_backed_upgrade: bool = True,
    enforce_flow_residual_gate: bool = True,
) -> ManagerAgent | CompetitiveManagerAgent | HierarchicalManagerAgent:
    base_seed = training_seed(training_config, seed_override=seed_override)
    shared_memory = ExperienceMemory(success_scale=0.0, failure_scale=0.0) if no_memory else ExperienceMemory()
    if partition_mode == "macro_micro":
        if fixed_agent_name is not None:
            raise ValueError("fixed_agent_name is only supported for competitive_three_way partition mode.")
        return ManagerAgent(
            selection_mode="greedy",
            seed=base_seed,
            experience_memory=shared_memory,
        )
    if partition_mode == "target_context":
        if fixed_agent_name is not None:
            raise ValueError("fixed_agent_name is only supported for competitive_three_way partition mode.")
        context_features, target_features = feature_partitions(dataset_name)
        context_agent = FeatureGroupAgent(
            role="context",
            allowed_features=context_features,
            token_score_biases=context_feature_biases(dataset_name),
            max_length_cap=12,
            experience_memory=shared_memory,
            **role_agent_kwargs(training_config, base_seed, role_index=0),
        )
        target_agent = FeatureGroupAgent(
            role="target",
            allowed_features=target_features,
            token_score_biases=target_feature_biases(dataset_name),
            max_length_cap=7,
            seed_formulas=target_seed_formulas(dataset_name),
            experience_memory=shared_memory,
            **role_agent_kwargs(training_config, base_seed, role_index=1),
        )
        gating_net = GatingNet(
            agent_names=("context", "target"),
            base_weights={
                "BALANCED": AgentWeights(macro=0.35, micro=0.65),
                "HIGH_VOLATILITY": AgentWeights(macro=0.55, micro=0.45),
                "RATE_HIKING": AgentWeights(macro=0.65, micro=0.35),
                "INFLATION_SHOCK": AgentWeights(macro=0.60, micro=0.40),
                "USD_STRENGTH": AgentWeights(macro=0.60, micro=0.40),
            },
        )
        state_encoder = StateEncoder(
            first_group_features=context_features,
            second_group_features=target_features,
            latest_columns=tuple(dict.fromkeys((f"{dataset_prefix(dataset_name)}_CLOSE", "VIX", "DXY", "TNX", "CPI"))),
        )
        return ManagerAgent(
            macro_agent=context_agent,
            micro_agent=target_agent,
            gating_net=gating_net,
            state_encoder=state_encoder,
            selection_mode="greedy",
            seed=base_seed,
            experience_memory=shared_memory,
            agent_names=("context", "target"),
        )
    if partition_mode == "skill_hierarchy":
        if dataset_name == "route_b":
            included_agent_names = resolve_route_b_skill_aliases(included_agent_names)
            if fixed_agent_name is not None:
                resolved_fixed = resolve_route_b_skill_aliases((fixed_agent_name,))
                fixed_agent_name = resolved_fixed[0] if resolved_fixed else fixed_agent_name
        if fixed_agent_name is not None and fixed_agent_name not in skill_family_feature_partitions(dataset_name):
            raise ValueError(f"Unknown skill family {fixed_agent_name!r}.")
        skill_features = skill_family_feature_partitions(dataset_name)
        operator_whitelists = skill_family_operator_whitelists()
        biases = skill_family_biases(dataset_name)
        seeds = skill_family_seed_formulas(dataset_name)
        if dataset_name == "route_b" and included_agent_names is None and fixed_agent_name is None:
            included_agent_names = (
                "quality_solvency",
            )
        agents = {
            name: SkillFamilyAgent(
                role=name,
                allowed_features=features,
                allowed_operators=operator_whitelists[name],
                token_score_biases=biases[name],
                max_length_cap=6 if name in {"regime_filter"} else (5 if name == "short_horizon_flow" else 7),
                seed_formulas=seeds[name],
                seed_priority_enabled=seed_priority_enabled,
                experience_memory=shared_memory,
                **(
                    {
                        **role_agent_kwargs(training_config, base_seed, role_index=index),
                        **(
                            boosted_flow_search_kwargs(base_seed, role_index=index)
                            if name == "short_horizon_flow"
                            else {}
                        ),
                    }
                ),
            )
            for index, (name, features) in enumerate(skill_features.items())
        }
        if included_agent_names is not None:
            agents = {name: agents[name] for name in included_agent_names}
        planner = LibraryPlanner(
            skill_names=tuple(agents.keys()),
            max_shortlist=min(3, len(agents)),
            base_skill_weights=(
                {
                    "quality_solvency": 0.18,
                    "efficiency_growth": 0.10,
                    "valuation_size": 0.04,
                    "short_horizon_flow": 0.02,
                }
                if dataset_name == "route_b"
                else {
                    "short_horizon_flow": 0.12,
                    "price_structure": 0.04,
                    "trend_structure": 0.04,
                    "cross_asset_context": -0.02,
                    "regime_filter": 0.00,
                }
            ),
        )
        if fixed_agent_name is not None:
            planner = LibraryPlanner(
                skill_names=(fixed_agent_name,),
                max_shortlist=1,
                base_skill_weights={fixed_agent_name: 1.0},
            )
            agents = {fixed_agent_name: agents[fixed_agent_name]}
        return HierarchicalManagerAgent(
            agents=agents,
            planner=planner,
            experience_memory=shared_memory,
            common_knowledge_encoder=CommonKnowledgeEncoder(),
            bootstrap_anchor_skill="quality_solvency" if dataset_name == "route_b" and "quality_solvency" in agents else None,
            allow_validation_backed_bootstrap=allow_validation_backed_bootstrap,
            allow_validation_backed_replacement=allow_validation_backed_replacement,
            allow_validation_backed_upgrade=allow_validation_backed_upgrade,
            enforce_flow_residual_gate=enforce_flow_residual_gate,
        )
    if partition_mode == "competitive_three_way":
        context_features, target_price_features, target_flow_vol_features, target_flow_gap_features = (
            competitive_feature_partitions(dataset_name)
        )
        agents = {
            "context": FeatureGroupAgent(
                role="context",
                allowed_features=context_features,
                token_score_biases=context_feature_biases(dataset_name),
                max_length_cap=12,
                experience_memory=shared_memory,
                **role_agent_kwargs(training_config, base_seed, role_index=0),
            ),
            "target_price": FeatureGroupAgent(
                role="target_price",
                allowed_features=target_price_features,
                token_score_biases=target_price_biases(dataset_name),
                max_length_cap=7,
                seed_formulas=target_price_seed_formulas(dataset_name),
                experience_memory=shared_memory,
                **role_agent_kwargs(training_config, base_seed, role_index=1),
            ),
            "target_flow_vol": FeatureGroupAgent(
                role="target_flow_vol",
                allowed_features=target_flow_vol_features,
                token_score_biases=target_flow_vol_biases(dataset_name),
                max_length_cap=7,
                seed_formulas=target_flow_vol_seed_formulas(dataset_name),
                experience_memory=shared_memory,
                **{
                    **role_agent_kwargs(training_config, base_seed, role_index=2),
                    **boosted_flow_search_kwargs(base_seed, role_index=2),
                },
            ),
            "target_flow_gap": FeatureGroupAgent(
                role="target_flow_gap",
                allowed_features=target_flow_gap_features,
                token_score_biases=target_flow_gap_biases(dataset_name),
                max_length_cap=6,
                seed_formulas=target_flow_gap_seed_formulas(dataset_name),
                experience_memory=shared_memory,
                **{
                    **role_agent_kwargs(training_config, base_seed, role_index=3),
                    **boosted_flow_search_kwargs(base_seed, role_index=3),
                },
            ),
        }
        if included_agent_names is not None:
            agents = {name: agents[name] for name in included_agent_names}
        return CompetitiveManagerAgent(
            agents=agents,
            experience_memory=shared_memory,
            fixed_agent_name=fixed_agent_name,
        )
    raise ValueError(f"Unsupported partition_mode {partition_mode!r}.")


def build_walk_forward_config(backtest_config: dict[str, Any]) -> WalkForwardConfig:
    payload = backtest_config.get("backtest", {})
    return WalkForwardConfig(
        train_size=int(payload.get("train_size", 756)),
        test_size=int(payload.get("test_size", 252)),
        step_size=int(payload.get("step_size", 126)),
        top_k=int(payload.get("top_k", 3)),
    )


def build_portfolio_config(backtest_config: dict[str, Any]) -> PortfolioConfig:
    payload = backtest_config.get("backtest", {})
    return PortfolioConfig(
        transaction_cost_bps=float(payload.get("transaction_cost_bps", 5.0)),
        signal_threshold=float(payload.get("signal_threshold", 0.0)),
        leverage=float(payload.get("leverage", 1.0)),
    )


def build_signal_fusion_config(backtest_config: dict[str, Any]) -> SignalFusionConfig:
    payload = backtest_config.get("backtest", {})
    return SignalFusionConfig(
        normalize=bool(payload.get("normalize", True)),
        clip_zscore=float(payload.get("clip_zscore", 5.0)),
    )


def ensure_output_dirs(root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Path]:
    root_path = Path(root)
    paths = {
        "root": root_path,
        "reports": root_path / "reports",
        "factors": root_path / "factors",
        "logs": root_path / "logs",
        "checkpoints": root_path / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def summary_counters(history: list[Any], field: str) -> dict[str, int]:
    return dict(Counter(str(getattr(item, field)) for item in history))


def group_formula_summaries_by_role(items: list[Any] | tuple[Any, ...]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in items:
        role = getattr(item, "role", None) or "unassigned"
        grouped.setdefault(str(role), []).append(str(getattr(item, "formula")))
    return grouped


def select_evaluation_formulas(
    champion_records: list[str] | tuple[str, ...],
    final_records: list[str] | tuple[str, ...],
) -> tuple[list[str], str]:
    champion = [str(item) for item in champion_records if str(item)]
    final = [str(item) for item in final_records if str(item)]
    if champion:
        return list(dict.fromkeys(champion)), "champion_records"
    if final:
        return list(dict.fromkeys(final)), "final_records"
    return [], "none"


def dataset_diagnostics(bundle) -> dict[str, Any]:
    if not hasattr(bundle, "frame"):
        split_rows = {
            "train": len(bundle.splits.train),
            "valid": len(bundle.splits.valid),
            "test": len(bundle.splits.test),
        }
        date_start = min(
            pd.Timestamp(bundle.splits.train["date"].min()),
            pd.Timestamp(bundle.splits.valid["date"].min()),
            pd.Timestamp(bundle.splits.test["date"].min()),
        )
        date_end = max(
            pd.Timestamp(bundle.splits.train["date"].max()),
            pd.Timestamp(bundle.splits.valid["date"].max()),
            pd.Timestamp(bundle.splits.test["date"].max()),
        )
        permno_count = int(
            pd.Index(bundle.splits.train["permno"])
            .union(bundle.splits.valid["permno"])
            .union(bundle.splits.test["permno"])
            .nunique()
        )
        return {
            "date_start": date_start,
            "date_end": date_end,
            "total_rows": int(sum(split_rows.values())),
            "feature_count": len(bundle.feature_columns),
            "feature_columns": list(bundle.feature_columns),
            "split_rows": split_rows,
            "permno_count": permno_count,
            "cross_sectional": True,
        }
    frame = bundle.frame
    macro_columns = [column for column in bundle.feature_columns if column in {"CPI", "TNX", "VIX", "DXY"}]
    micro_columns = [column for column in bundle.feature_columns if column not in macro_columns]
    gold_micro_columns = [column for column in micro_columns if column.startswith("GOLD_")]
    target_micro_columns = [
        column for column in micro_columns if column.startswith("CRUDE_OIL_") or column.startswith("SP500_")
    ]
    split_rows = {
        "train": len(bundle.splits.train),
        "valid": len(bundle.splits.valid),
        "test": len(bundle.splits.test),
    }
    macro_uniques = {
        column: {
            "train": int(bundle.splits.train[column].nunique(dropna=True)),
            "valid": int(bundle.splits.valid[column].nunique(dropna=True)),
            "test": int(bundle.splits.test[column].nunique(dropna=True)),
        }
        for column in macro_columns
    }
    return {
        "date_start": frame.index.min(),
        "date_end": frame.index.max(),
        "total_rows": len(frame),
        "feature_count": len(bundle.feature_columns),
        "micro_feature_count": len(micro_columns),
        "macro_feature_count": len(macro_columns),
        "feature_columns": list(bundle.feature_columns),
        "split_rows": split_rows,
        "macro_unique_values": macro_uniques,
        "transfer_gap": {
            "source_micro_assets": gold_micro_columns,
            "target_specific_micro_features_in_language": target_micro_columns,
        },
    }


def dataset_input_frame(split: pd.DataFrame, feature_columns: tuple[str, ...], dataset_name: str) -> pd.DataFrame:
    if dataset_name == "route_b":
        columns = ["date", "permno", *feature_columns]
        return split.loc[:, columns].copy()
    return split.loc[:, list(feature_columns)].copy()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(index): to_jsonable(item) for index, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=True)
        handle.write("\n")
