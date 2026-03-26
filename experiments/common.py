from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from knowledge_guided_symbolic_alpha.agents import HierarchicalManagerAgent, SkillFamilyAgent
from knowledge_guided_symbolic_alpha.backtest import PortfolioConfig, SignalFusionConfig, WalkForwardConfig
from knowledge_guided_symbolic_alpha.dataio import load_processed_us_equities_splits
from knowledge_guided_symbolic_alpha.envs import CommonKnowledgeEncoder
from knowledge_guided_symbolic_alpha.generation import build_anchor_generation_summary
from knowledge_guided_symbolic_alpha.memory import ExperienceMemory
from knowledge_guided_symbolic_alpha.models.controllers import LibraryPlanner
from knowledge_guided_symbolic_alpha.models.generator import RNNGenerator, TransformerGenerator
from knowledge_guided_symbolic_alpha.runtime import DEFAULT_RUNS_ROOT, ensure_run_output_dirs
from knowledge_guided_symbolic_alpha.search import BeamSearchConfig, GrammarMCTSConfig, SamplingConfig
from knowledge_guided_symbolic_alpha.selection import RobustTemporalSelector
from knowledge_guided_symbolic_alpha.training import FormulaCurriculum


DEFAULT_DATA_CONFIG = Path("configs/us_equities_smoke.yaml")
DEFAULT_TRAINING_CONFIG = Path("configs/training.yaml")
DEFAULT_BACKTEST_CONFIG = Path("configs/backtest.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
DEFAULT_OUTPUT_ROOT = DEFAULT_RUNS_ROOT

TARGET_COLUMNS = {"us_equities": "TARGET_XS_RET_1"}
RETURN_COLUMNS = {"us_equities": "TARGET_RET_1"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_name(path: str | Path) -> str:
    payload = load_yaml(path)
    return str(payload["experiment"]["dataset"])


def canonical_dataset_name(dataset_name: str) -> str:
    if dataset_name != "us_equities":
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Only 'us_equities' is supported.")
    return dataset_name


def load_dataset_bundle(data_config: str | Path = DEFAULT_DATA_CONFIG):
    payload = load_yaml(data_config)
    subset_config = payload.get("us_equities_subset")
    if subset_config is None:
        raise ValueError("Missing `us_equities_subset` in data config.")
    return load_processed_us_equities_splits(Path(subset_config["split_root"]))


def dataset_columns(dataset_name: str) -> tuple[str, str]:
    dataset_name = canonical_dataset_name(dataset_name)
    return TARGET_COLUMNS[dataset_name], RETURN_COLUMNS[dataset_name]


def skill_family_feature_partitions(dataset_name: str) -> dict[str, frozenset[str]]:
    canonical_dataset_name(dataset_name)
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


def skill_family_operator_whitelists() -> dict[str, frozenset[str]]:
    return {
        "quality_solvency": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "ADD"}),
        "efficiency_growth": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "ADD"}),
        "valuation_size": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "ADD"}),
        "short_horizon_flow": frozenset({"NEG", "DELAY_1", "DELTA_1", "RANK", "SUB", "CORR_5", "DIV"}),
    }


def skill_family_biases(dataset_name: str) -> dict[str, dict[str, float]]:
    canonical_dataset_name(dataset_name)
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


def skill_family_seed_formulas(dataset_name: str) -> dict[str, tuple[tuple[str, ...], ...]]:
    canonical_dataset_name(dataset_name)
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


def resolve_us_equities_skill_aliases(skill_names: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if skill_names is None:
        return None
    return tuple(dict.fromkeys(skill_names))


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
    dataset_name: str = "us_equities",
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
) -> HierarchicalManagerAgent:
    canonical_dataset_name(dataset_name)
    if partition_mode != "skill_hierarchy":
        raise ValueError("Mainline only supports `skill_hierarchy` partition mode.")

    base_seed = training_seed(training_config, seed_override=seed_override)
    shared_memory = ExperienceMemory(success_scale=0.0, failure_scale=0.0) if no_memory else ExperienceMemory()

    included_agent_names = resolve_us_equities_skill_aliases(included_agent_names)
    if fixed_agent_name is not None:
        resolved_fixed = resolve_us_equities_skill_aliases((fixed_agent_name,))
        fixed_agent_name = resolved_fixed[0] if resolved_fixed else fixed_agent_name

    skill_features = skill_family_feature_partitions(dataset_name)
    operator_whitelists = skill_family_operator_whitelists()
    biases = skill_family_biases(dataset_name)
    seeds = skill_family_seed_formulas(dataset_name)

    if included_agent_names is None and fixed_agent_name is None:
        included_agent_names = ("quality_solvency",)

    agents = {
        name: SkillFamilyAgent(
            role=name,
            allowed_features=features,
            allowed_operators=operator_whitelists[name],
            token_score_biases=biases[name],
            max_length_cap=5 if name == "short_horizon_flow" else 7,
            seed_formulas=seeds[name],
            seed_priority_enabled=seed_priority_enabled,
            experience_memory=shared_memory,
            **{
                **role_agent_kwargs(training_config, base_seed, role_index=index),
                **(boosted_flow_search_kwargs(base_seed, role_index=index) if name == "short_horizon_flow" else {}),
            },
        )
        for index, (name, features) in enumerate(skill_features.items())
    }

    if included_agent_names is not None:
        agents = {name: agents[name] for name in included_agent_names}

    planner = LibraryPlanner(
        skill_names=tuple(agents.keys()),
        max_shortlist=min(3, len(agents)),
        base_skill_weights={
            "quality_solvency": 0.18,
            "efficiency_growth": 0.10,
            "valuation_size": 0.04,
            "short_horizon_flow": 0.02,
        },
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
        bootstrap_anchor_skill="quality_solvency" if "quality_solvency" in agents else None,
        allow_validation_backed_bootstrap=allow_validation_backed_bootstrap,
        allow_validation_backed_replacement=allow_validation_backed_replacement,
        allow_validation_backed_upgrade=allow_validation_backed_upgrade,
        enforce_flow_residual_gate=enforce_flow_residual_gate,
    )


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


def ensure_output_dirs(root: str | Path = DEFAULT_OUTPUT_ROOT, run_name: str = "run") -> dict[str, Path]:
    return ensure_run_output_dirs(root, run_name)


def summary_counters(history: list[Any], field: str) -> dict[str, int]:
    return dict(Counter(str(getattr(item, field)) for item in history))


def group_formula_summaries_by_role(items: list[Any] | tuple[Any, ...]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for item in items:
        role = getattr(item, "role", None) or "unassigned"
        grouped.setdefault(str(role), []).append(str(getattr(item, "formula")))
    return grouped


def build_generation_summary(summary: Any):
    return build_anchor_generation_summary(summary)


def run_validation_selector(
    summary: Any,
    validation_frame: pd.DataFrame,
    validation_target: pd.Series,
):
    generation_summary = build_generation_summary(summary)
    selector = RobustTemporalSelector()
    selector_outcome = selector.select(generation_summary.candidates, validation_frame, validation_target)
    return generation_summary, selector_outcome


def select_evaluation_formulas(
    selector_records: list[str] | tuple[str, ...] | None,
    champion_records: list[str] | tuple[str, ...],
    final_records: list[str] | tuple[str, ...],
) -> tuple[list[str], str]:
    selected = [str(item) for item in (selector_records or ()) if str(item)]
    champion = [str(item) for item in champion_records if str(item)]
    final = [str(item) for item in final_records if str(item)]
    if selected:
        return list(dict.fromkeys(selected)), "selector_records"
    if champion:
        return list(dict.fromkeys(champion)), "champion_records"
    if final:
        return list(dict.fromkeys(final)), "final_records"
    return [], "none"


def dataset_diagnostics(bundle) -> dict[str, Any]:
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


def dataset_input_frame(split: pd.DataFrame, feature_columns: tuple[str, ...], dataset_name: str) -> pd.DataFrame:
    canonical_dataset_name(dataset_name)
    columns = ["date", "permno", *feature_columns]
    return split.loc[:, columns].copy()


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
