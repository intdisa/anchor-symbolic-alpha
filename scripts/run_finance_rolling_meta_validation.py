#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (
    build_portfolio_config,
    build_signal_fusion_config,
    build_walk_forward_config,
    dataset_columns,
    load_dataset_bundle,
    load_experiment_name,
    load_yaml,
)
from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
from knowledge_guided_symbolic_alpha.generation import FormulaCandidate
from knowledge_guided_symbolic_alpha.selection import (
    CrossSeedConsensusConfig,
    CrossSeedConsensusSelector,
    CrossSeedSelectionRun,
    FormulaEvaluationCache,
    RobustSelectorConfig,
    RobustScoreScaleStats,
    TemporalRobustSelector,
    estimate_robust_score_scales,
)


UNIVERSE_SOURCES = {
    "liquid500": {
        "data_config": Path("configs/us_equities_liquid500.yaml"),
        "canonical": Path("outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json"),
        "multiseed": Path("outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed.json"),
    },
    "liquid1000": {
        "data_config": Path("configs/us_equities_liquid1000.yaml"),
        "canonical": Path("outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json"),
        "multiseed": Path("outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed.json"),
    },
}
BACKTEST_CONFIG = Path("configs/backtest.yaml")
EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
BASE_CONFIG = {
    "temporal_selection_mode": "legacy_linear",
    "cross_seed_selection_mode": "legacy_linear",
    "min_valid_slices": 2,
    "min_mean_rank_ic": 0.0,
    "min_slice_rank_ic": -0.01,
    "near_neighbor_signal_corr": 0.80,
    "near_neighbor_token_overlap": 0.50,
    "near_neighbor_feature_family_overlap": 0.95,
    "min_seed_support": 3,
    "max_pairwise_signal_corr": 0.93,
    "enable_near_neighbor_tie_break": True,
    "enable_admissibility_gate": True,
    "enable_redundancy_gate": True,
}


@dataclass(frozen=True)
class RollingWindow:
    window_id: str
    calibration_frame: pd.DataFrame
    calibration_target: pd.Series
    meta_frame: pd.DataFrame
    meta_target: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling threshold meta-validation for finance signal selection.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--universes", type=str, default="liquid500,liquid1000")
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--coarse-top-k", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_universes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_candidate_pool(raw_runs: list[dict[str, Any]]) -> list[FormulaCandidate]:
    seen: set[str] = set()
    formulas: list[str] = []
    for run in raw_runs:
        for formula in run.get("candidate_records", []):
            if formula and formula not in seen:
                seen.add(formula)
                formulas.append(formula)
    return [FormulaCandidate(formula=formula, source="finance_rolling_meta", role="finance") for formula in formulas]


def load_universe_inputs(universe: str) -> tuple[list[dict[str, Any]], list[FormulaCandidate], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    multiseed = load_json(UNIVERSE_SOURCES[universe]["multiseed"])
    raw_runs = multiseed["runs_by_variant"]["full"]
    candidates = build_candidate_pool(raw_runs)
    bundle = load_dataset_bundle(UNIVERSE_SOURCES[universe]["data_config"])
    valid_frame = bundle.splits.valid.copy()
    valid_target = valid_frame["TARGET_XS_RET_1"].copy()
    test_frame = bundle.splits.test.copy()
    test_target = test_frame["TARGET_XS_RET_1"].copy()
    return raw_runs, candidates, valid_frame, valid_target, test_frame, test_target


def load_universe_train_split(universe: str) -> tuple[pd.DataFrame, pd.Series]:
    bundle = load_dataset_bundle(UNIVERSE_SOURCES[universe]["data_config"])
    train_frame = bundle.splits.train.copy()
    train_target = train_frame["TARGET_XS_RET_1"].copy()
    return train_frame, train_target


def load_universe_scale_stats(
    universe: str,
    candidates: list[FormulaCandidate],
    cache: FormulaEvaluationCache | None = None,
) -> RobustScoreScaleStats:
    train_frame, train_target = load_universe_train_split(universe)
    return estimate_robust_score_scales(
        candidates,
        train_frame,
        train_target,
        slice_count=BASE_CONFIG.get("slice_count", 4) if isinstance(BASE_CONFIG.get("slice_count"), int) else 4,
        evaluation_cache=cache,
        context_key=f"{universe}:train_scale",
    )


def build_rolling_windows(frame: pd.DataFrame, target: pd.Series, window_count: int) -> list[RollingWindow]:
    dates = pd.Index(pd.to_datetime(frame["date"]).sort_values().unique())
    chunks = [chunk for chunk in np.array_split(dates.to_numpy(), min(window_count, len(dates))) if len(chunk)]
    windows: list[RollingWindow] = []
    for window_index, chunk in enumerate(chunks, start=1):
        chunk_dates = pd.to_datetime(chunk)
        chunk_frame = frame[pd.to_datetime(frame["date"]).isin(chunk_dates)].copy()
        chunk_target = target.loc[chunk_frame.index]
        unique_dates = pd.Index(pd.to_datetime(chunk_frame["date"]).sort_values().unique())
        midpoint = len(unique_dates) // 2
        calibration_dates = set(unique_dates[:midpoint])
        meta_dates = set(unique_dates[midpoint:])
        calibration_frame = chunk_frame[pd.to_datetime(chunk_frame["date"]).isin(calibration_dates)].copy()
        meta_frame = chunk_frame[pd.to_datetime(chunk_frame["date"]).isin(meta_dates)].copy()
        if calibration_frame.empty or meta_frame.empty:
            continue
        windows.append(
            RollingWindow(
                window_id=f"window_{window_index}",
                calibration_frame=calibration_frame,
                calibration_target=chunk_target.loc[calibration_frame.index],
                meta_frame=meta_frame,
                meta_target=chunk_target.loc[meta_frame.index],
            )
        )
    return windows


def _safe_spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 3 or len(values_b) < 3:
        return 0.0
    frame = pd.DataFrame({"a": values_a, "b": values_b}).dropna()
    if len(frame) < 3:
        return 0.0
    corr = frame["a"].corr(frame["b"], method="spearman")
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)


def _rank_ic_retention(
    calibration_rank_ic: float,
    meta_rank_ic: float,
    calibration_candidate_rank_ics: list[float],
) -> tuple[float, float]:
    calibration_array = np.asarray(calibration_candidate_rank_ics, dtype=float)
    finite = calibration_array[np.isfinite(calibration_array)]
    scale = float(np.nanstd(finite)) if finite.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-6:
        mean_abs = float(np.nanmean(np.abs(finite))) if finite.size else 0.0
        scale = max(abs(calibration_rank_ic), mean_abs, 1e-3)
    retention = float(np.exp(-abs(meta_rank_ic - calibration_rank_ic) / scale))
    return float(np.clip(retention, 0.0, 1.0)), float(scale)


def selection_score(
    *,
    calibration_metrics: dict[str, float],
    meta_metrics: dict[str, float],
    calibration_candidate_rank_ics: list[float],
    meta_candidate_rank_ics: list[float],
    meta_candidate_turnovers: list[float] | None = None,
) -> dict[str, float]:
    calibration_rank_ic = float(calibration_metrics.get("rank_ic") or 0.0)
    meta_rank_ic = float(meta_metrics.get("rank_ic") or 0.0)
    turnover = float(meta_metrics.get("turnover") or 0.0)
    rank_consistency = _safe_spearman(calibration_candidate_rank_ics, meta_candidate_rank_ics)
    retention, retention_scale = _rank_ic_retention(
        calibration_rank_ic,
        meta_rank_ic,
        calibration_candidate_rank_ics,
    )
    turnover_array = np.asarray(meta_candidate_turnovers or [turnover], dtype=float)
    finite_turnover = turnover_array[np.isfinite(turnover_array)]
    turnover_scale = float(np.nanstd(finite_turnover)) if finite_turnover.size else 0.0
    if not np.isfinite(turnover_scale) or turnover_scale <= 1e-6:
        turnover_scale = max(float(np.nanmean(finite_turnover)) if finite_turnover.size else 0.0, 1e-3)
    standardized_turnover = float(np.tanh(turnover / turnover_scale))
    score = rank_consistency + retention - standardized_turnover
    return {
        "meta_score": float(score),
        "rank_consistency": float(rank_consistency),
        "rank_ic_retention": float(retention),
        "rank_ic_retention_scale": float(retention_scale),
        "turnover_scale": float(turnover_scale),
        "standardized_turnover": float(standardized_turnover),
    }


def config_key(cfg: dict[str, Any]) -> str:
    return (
        f"tm={cfg['temporal_selection_mode']}|cm={cfg['cross_seed_selection_mode']}|"
        f"vs={cfg['min_valid_slices']}|mr={cfg['min_mean_rank_ic']:.3f}|ms={cfg['min_slice_rank_ic']:.3f}|"
        f"nnc={cfg['near_neighbor_signal_corr']:.2f}|nnt={cfg['near_neighbor_token_overlap']:.2f}|"
        f"nnf={cfg['near_neighbor_feature_family_overlap']:.2f}|ss={cfg['min_seed_support']}|"
        f"corr={cfg['max_pairwise_signal_corr']:.2f}|adm={int(cfg['enable_admissibility_gate'])}|"
        f"nbr={int(cfg['enable_near_neighbor_tie_break'])}|red={int(cfg['enable_redundancy_gate'])}"
    )


def coarse_config_grid() -> list[dict[str, Any]]:
    options = {
        "min_valid_slices": [1, 2, 3],
        "min_mean_rank_ic": [-0.005, 0.0, 0.005],
        "min_slice_rank_ic": [-0.02, -0.01, 0.0],
        "near_neighbor_signal_corr": [0.70, 0.80, 0.90],
        "near_neighbor_token_overlap": [0.40, 0.50, 0.60],
        "near_neighbor_feature_family_overlap": [0.85, 0.95, 1.00],
        "min_seed_support": [2, 3, 4],
        "max_pairwise_signal_corr": [0.90, 0.93, 0.96],
    }
    configs: dict[str, dict[str, Any]] = {config_key(BASE_CONFIG): dict(BASE_CONFIG)}
    for key, values in options.items():
        for value in values:
            cfg = dict(BASE_CONFIG)
            cfg[key] = value
            configs[config_key(cfg)] = cfg
    paired_variants = [
        {"min_mean_rank_ic": 0.005, "min_slice_rank_ic": 0.0},
        {"min_mean_rank_ic": -0.005, "min_slice_rank_ic": -0.02},
        {"near_neighbor_signal_corr": 0.90, "near_neighbor_token_overlap": 0.60},
        {"near_neighbor_signal_corr": 0.70, "near_neighbor_token_overlap": 0.40},
        {"min_seed_support": 4, "max_pairwise_signal_corr": 0.90},
        {"min_seed_support": 2, "max_pairwise_signal_corr": 0.96},
        {"enable_admissibility_gate": False, "min_valid_slices": 1},
        {"enable_near_neighbor_tie_break": False, "near_neighbor_signal_corr": 0.80},
        {"enable_redundancy_gate": False, "max_pairwise_signal_corr": 1.0},
    ]
    for patch in paired_variants:
        cfg = dict(BASE_CONFIG)
        cfg.update(patch)
        configs[config_key(cfg)] = cfg
    return list(configs.values())


def fine_neighbors(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    step_values = {
        "min_valid_slices": [1, 2, 3],
        "min_mean_rank_ic": [-0.005, 0.0, 0.005, 0.010],
        "min_slice_rank_ic": [-0.02, -0.01, 0.0],
        "near_neighbor_signal_corr": [0.75, 0.80, 0.85, 0.90],
        "near_neighbor_token_overlap": [0.45, 0.50, 0.55, 0.60],
        "near_neighbor_feature_family_overlap": [0.90, 0.95, 1.00],
        "min_seed_support": [2, 3, 4],
        "max_pairwise_signal_corr": [0.90, 0.93, 0.96, 1.00],
    }
    expanded: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        expanded[config_key(cfg)] = dict(cfg)
        for key, values in step_values.items():
            current = cfg[key]
            if current not in values:
                continue
            index = values.index(current)
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(values):
                    candidate = dict(cfg)
                    candidate[key] = values[neighbor_index]
                    expanded[config_key(candidate)] = candidate
    return list(expanded.values())


def _selector_config(cfg: dict[str, Any], scale_stats: RobustScoreScaleStats | None = None) -> RobustSelectorConfig:
    return RobustSelectorConfig(
        selection_mode=str(cfg.get("temporal_selection_mode", "pareto")),
        min_valid_slices=int(cfg.get("min_valid_slices", 2)),
        min_mean_rank_ic=float(cfg.get("min_mean_rank_ic", 0.0)),
        min_slice_rank_ic=float(cfg.get("min_slice_rank_ic", -0.01)),
        near_neighbor_signal_corr=float(cfg.get("near_neighbor_signal_corr", 0.80)),
        near_neighbor_token_overlap=float(cfg.get("near_neighbor_token_overlap", 0.50)),
        near_neighbor_feature_family_overlap=float(cfg.get("near_neighbor_feature_family_overlap", 0.95)),
        max_pairwise_signal_corr=float(cfg.get("max_pairwise_signal_corr", 0.93)),
        enable_near_neighbor_tie_break=bool(cfg.get("enable_near_neighbor_tie_break", True)),
        enable_admissibility_gate=bool(cfg.get("enable_admissibility_gate", True)),
        enable_redundancy_gate=bool(cfg.get("enable_redundancy_gate", True)),
        sharpe_scale=float(cfg.get("sharpe_scale", scale_stats.sharpe_scale if scale_stats is not None else 2.0)),
        annual_return_scale=float(cfg.get("annual_return_scale", scale_stats.annual_return_scale if scale_stats is not None else 0.20)),
    )


def _consensus_config(cfg: dict[str, Any]) -> CrossSeedConsensusConfig:
    return CrossSeedConsensusConfig(
        selection_mode=str(cfg.get("cross_seed_selection_mode", "pareto")),
        min_seed_support=int(cfg.get("min_seed_support", 3)),
        rerank_mode="shared_frame",
    )


def derive_seed_runs(
    raw_runs: list[dict[str, Any]],
    frame: pd.DataFrame,
    target: pd.Series,
    temporal_selector: TemporalRobustSelector,
    *,
    evaluation_context: str,
) -> list[CrossSeedSelectionRun]:
    derived: list[CrossSeedSelectionRun] = []
    for raw in raw_runs:
        candidates = [
            FormulaCandidate(formula=formula, source="finance_rolling_meta", role="finance")
            for formula in raw.get("candidate_records", [])
            if formula
        ]
        outcome = temporal_selector.select(candidates, frame, target, evaluation_context=evaluation_context)
        champion_formula = ""
        champion_score = float("-inf")
        for record in outcome.records:
            score = float(record.diagnostics.get("mean_rank_ic") or float("-inf"))
            if score > champion_score:
                champion_score = score
                champion_formula = record.formula
        derived.append(
            CrossSeedSelectionRun(
                seed=int(raw["seed"]),
                candidate_records=tuple(candidate.formula for candidate in candidates),
                selector_records=tuple(outcome.selected_formulas),
                champion_records=(champion_formula,) if champion_formula else tuple(),
                selector_ranked_records=tuple(outcome.records),
            )
        )
    return derived


def cached_metrics(
    cache: FormulaEvaluationCache,
    formula: str,
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    context_key: str,
) -> dict[str, float]:
    metrics = cache.get(formula, frame, target, slice_count=4, context_key=context_key).full_metrics
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and np.isfinite(value)}


def default_formula(universe: str) -> str:
    payload = load_json(UNIVERSE_SOURCES[universe]["canonical"])
    return str(payload["canonical_by_variant"]["full"]["selector_records"][0])


def backtest_formula(universe: str, formula: str) -> dict[str, float]:
    bundle = load_dataset_bundle(UNIVERSE_SOURCES[universe]["data_config"])
    backtest_frame = pd.concat([bundle.splits.valid, bundle.splits.test], axis=0)
    dataset_name = load_experiment_name(EXPERIMENT_CONFIG)
    target_column, return_column = dataset_columns(dataset_name)
    backtest_config = load_yaml(BACKTEST_CONFIG)
    report = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    ).run(
        formulas=[formula],
        frame=backtest_frame,
        feature_columns=bundle.feature_columns,
        target_column=target_column,
        return_column=return_column,
        config=build_walk_forward_config(backtest_config),
    )
    return {key: float(value) for key, value in report.aggregate_metrics.items() if isinstance(value, (int, float))}


def evaluate_config_rows(
    universe: str,
    stage: str,
    configs: list[dict[str, Any]],
    raw_runs: list[dict[str, Any]],
    candidates: list[FormulaCandidate],
    windows: list[RollingWindow],
    cache: FormulaEvaluationCache,
    scale_stats: RobustScoreScaleStats,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate_formulas = [candidate.formula for candidate in candidates]
    for cfg in configs:
        selector = TemporalRobustSelector(_selector_config(cfg, scale_stats), evaluation_cache=cache)
        consensus = CrossSeedConsensusSelector(
            temporal_selector=selector,
            config=_consensus_config(cfg),
        )
        cfg_key = config_key(cfg)
        for window in windows:
            calib_context = f"{universe}:{window.window_id}:calibration"
            meta_context = f"{universe}:{window.window_id}:meta"
            seed_runs = derive_seed_runs(
                raw_runs,
                window.calibration_frame,
                window.calibration_target,
                selector,
                evaluation_context=calib_context,
            )
            outcome = consensus.select(
                seed_runs,
                window.calibration_frame,
                window.calibration_target,
                base_candidates=candidates,
                evaluation_context=calib_context,
            )
            formula = outcome.selected_formulas[0] if outcome.selected_formulas else ""
            calibration_metrics = (
                cached_metrics(
                    cache,
                    formula,
                    window.calibration_frame,
                    window.calibration_target,
                    context_key=calib_context,
                )
                if formula
                else {}
            )
            meta_metrics = (
                cached_metrics(
                    cache,
                    formula,
                    window.meta_frame,
                    window.meta_target,
                    context_key=meta_context,
                )
                if formula
                else {}
            )
            calibration_candidate_rank_ics: list[float] = []
            meta_candidate_rank_ics: list[float] = []
            meta_candidate_turnovers: list[float] = []
            for candidate_formula in candidate_formulas:
                candidate_calib_metrics = cached_metrics(
                    cache,
                    candidate_formula,
                    window.calibration_frame,
                    window.calibration_target,
                    context_key=calib_context,
                )
                candidate_meta_metrics = cached_metrics(
                    cache,
                    candidate_formula,
                    window.meta_frame,
                    window.meta_target,
                    context_key=meta_context,
                )
                calibration_candidate_rank_ics.append(float(candidate_calib_metrics.get("rank_ic") or 0.0))
                meta_candidate_rank_ics.append(float(candidate_meta_metrics.get("rank_ic") or 0.0))
                meta_candidate_turnovers.append(float(candidate_meta_metrics.get("turnover") or 0.0))
            objective = (
                selection_score(
                    calibration_metrics=calibration_metrics,
                    meta_metrics=meta_metrics,
                    calibration_candidate_rank_ics=calibration_candidate_rank_ics,
                    meta_candidate_rank_ics=meta_candidate_rank_ics,
                    meta_candidate_turnovers=meta_candidate_turnovers,
                )
                if formula
                else {
                    "meta_score": float("-inf"),
                    "rank_consistency": float("-inf"),
                    "rank_ic_retention": float("-inf"),
                    "turnover_scale": float("nan"),
                    "standardized_turnover": float("nan"),
                }
            )
            rows.append(
                {
                    "universe": universe,
                    "stage": stage,
                    "window_id": window.window_id,
                    "config_key": cfg_key,
                    "formula": formula,
                    "meta_score": round(float(objective["meta_score"]), 6) if formula else float("-inf"),
                    "meta_rank_ic": round(float(meta_metrics.get("rank_ic") or 0.0), 6) if meta_metrics else None,
                    "meta_sharpe": round(float(meta_metrics.get("sharpe") or 0.0), 6) if meta_metrics else None,
                    "meta_turnover": round(float(meta_metrics.get("turnover") or 0.0), 6) if meta_metrics else None,
                    "calibration_rank_ic": round(float(calibration_metrics.get("rank_ic") or 0.0), 6) if calibration_metrics else None,
                    "rank_consistency": round(float(objective["rank_consistency"]), 6) if formula else None,
                    "rank_ic_retention": round(float(objective["rank_ic_retention"]), 6) if formula else None,
                    "meta_turnover_scale": round(float(objective["turnover_scale"]), 6) if formula else None,
                    "meta_turnover_standardized": round(float(objective["standardized_turnover"]), 6) if formula else None,
                    **cfg,
                }
            )
    return rows


def summarize_config_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for (universe, cfg_key), group in rows.groupby(["universe", "config_key"], sort=False):
        formulas = group["formula"].dropna().tolist()
        formula_counts = pd.Series(formulas).value_counts(dropna=False) if formulas else pd.Series(dtype=int)
        top_formula = str(formula_counts.index[0]) if not formula_counts.empty else ""
        formula_stability = float(formula_counts.iloc[0] / len(formulas)) if not formula_counts.empty else 0.0
        first = group.iloc[0]
        summaries.append(
            {
                "universe": universe,
                "config_key": cfg_key,
                "mean_meta_score": round(float(group["meta_score"].mean()), 6),
                "mean_meta_rank_ic": round(float(group["meta_rank_ic"].mean()), 6),
                "mean_meta_sharpe": round(float(group["meta_sharpe"].mean()), 6),
                "mean_meta_turnover": round(float(group["meta_turnover"].mean()), 6),
                "mean_calibration_rank_ic": round(float(group["calibration_rank_ic"].mean()), 6),
                "mean_rank_consistency": round(float(group["rank_consistency"].mean()), 6),
                "mean_rank_ic_retention": round(float(group["rank_ic_retention"].mean()), 6),
                "mean_meta_turnover_scale": round(float(group["meta_turnover_scale"].mean()), 6),
                "selected_formula": top_formula,
                "formula_stability": round(formula_stability, 6),
                "window_count": int(len(group)),
                "temporal_selection_mode": str(first["temporal_selection_mode"]),
                "cross_seed_selection_mode": str(first["cross_seed_selection_mode"]),
                "min_valid_slices": int(first["min_valid_slices"]),
                "min_mean_rank_ic": float(first["min_mean_rank_ic"]),
                "min_slice_rank_ic": float(first["min_slice_rank_ic"]),
                "near_neighbor_signal_corr": float(first["near_neighbor_signal_corr"]),
                "near_neighbor_token_overlap": float(first["near_neighbor_token_overlap"]),
                "near_neighbor_feature_family_overlap": float(first["near_neighbor_feature_family_overlap"]),
                "min_seed_support": int(first["min_seed_support"]),
                "max_pairwise_signal_corr": float(first["max_pairwise_signal_corr"]),
                "enable_near_neighbor_tie_break": bool(first["enable_near_neighbor_tie_break"]),
                "enable_admissibility_gate": bool(first["enable_admissibility_gate"]),
                "enable_redundancy_gate": bool(first["enable_redundancy_gate"]),
            }
        )
    return pd.DataFrame(summaries).sort_values(
        ["universe", "mean_meta_score", "mean_rank_consistency", "formula_stability", "mean_meta_rank_ic"],
        ascending=[True, False, False, False, False],
    )


def build_window_best_frequency(rows: pd.DataFrame) -> pd.DataFrame:
    winners = rows.sort_values(
        ["universe", "window_id", "meta_score", "rank_consistency", "meta_rank_ic"],
        ascending=[True, True, False, False, False],
    ).groupby(["universe", "window_id"], as_index=False).head(1)
    summary = (
        winners.groupby(["universe", "config_key", "formula"], as_index=False)
        .size()
        .rename(columns={"size": "window_win_count"})
        .sort_values(["universe", "window_win_count", "formula"], ascending=[True, False, True])
    )
    return summary


def choose_best_configs(config_summary: pd.DataFrame, coarse_top_k: int) -> list[dict[str, Any]]:
    top = config_summary.head(coarse_top_k)
    keys = [
        "temporal_selection_mode",
        "cross_seed_selection_mode",
        "min_valid_slices",
        "min_mean_rank_ic",
        "min_slice_rank_ic",
        "near_neighbor_signal_corr",
        "near_neighbor_token_overlap",
        "near_neighbor_feature_family_overlap",
        "min_seed_support",
        "max_pairwise_signal_corr",
        "enable_near_neighbor_tie_break",
        "enable_admissibility_gate",
        "enable_redundancy_gate",
    ]
    return [{key: getattr(row, key) for key in keys} for row in top.itertuples(index=False)]


def run_universe(universe: str, window_count: int, coarse_top_k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_runs, candidates, valid_frame, valid_target, test_frame, test_target = load_universe_inputs(universe)
    windows = build_rolling_windows(valid_frame, valid_target, window_count=window_count)
    cache = FormulaEvaluationCache()
    scale_stats = load_universe_scale_stats(universe, candidates, cache)

    coarse_rows = pd.DataFrame(
        evaluate_config_rows(universe, "coarse", coarse_config_grid(), raw_runs, candidates, windows, cache, scale_stats)
    )
    coarse_summary = summarize_config_rows(coarse_rows)
    fine_configs = fine_neighbors(choose_best_configs(coarse_summary, coarse_top_k))
    fine_rows = pd.DataFrame(
        evaluate_config_rows(universe, "fine", fine_configs, raw_runs, candidates, windows, cache, scale_stats)
    )
    combined_rows = pd.concat([coarse_rows, fine_rows], axis=0, ignore_index=True)
    combined_rows = combined_rows.drop_duplicates(subset=["universe", "stage", "window_id", "config_key"], keep="last")
    combined_summary = summarize_config_rows(combined_rows)
    window_best = build_window_best_frequency(combined_rows)
    best = combined_summary.iloc[0]

    best_cfg = {
        "temporal_selection_mode": str(best["temporal_selection_mode"]),
        "cross_seed_selection_mode": str(best["cross_seed_selection_mode"]),
        "min_valid_slices": int(best["min_valid_slices"]),
        "min_mean_rank_ic": float(best["min_mean_rank_ic"]),
        "min_slice_rank_ic": float(best["min_slice_rank_ic"]),
        "near_neighbor_signal_corr": float(best["near_neighbor_signal_corr"]),
        "near_neighbor_token_overlap": float(best["near_neighbor_token_overlap"]),
        "near_neighbor_feature_family_overlap": float(best["near_neighbor_feature_family_overlap"]),
        "min_seed_support": int(best["min_seed_support"]),
        "max_pairwise_signal_corr": float(best["max_pairwise_signal_corr"]),
        "enable_near_neighbor_tie_break": bool(best["enable_near_neighbor_tie_break"]),
        "enable_admissibility_gate": bool(best["enable_admissibility_gate"]),
        "enable_redundancy_gate": bool(best["enable_redundancy_gate"]),
    }
    final_selector = TemporalRobustSelector(_selector_config(best_cfg, scale_stats), evaluation_cache=cache)
    final_consensus = CrossSeedConsensusSelector(
        temporal_selector=final_selector,
        config=_consensus_config(best_cfg),
    )
    valid_context = f"{universe}:valid_full"
    final_seed_runs = derive_seed_runs(raw_runs, valid_frame, valid_target, final_selector, evaluation_context=valid_context)
    final_outcome = final_consensus.select(
        final_seed_runs,
        valid_frame,
        valid_target,
        base_candidates=candidates,
        evaluation_context=valid_context,
    )
    selected_formula = final_outcome.selected_formulas[0] if final_outcome.selected_formulas else ""
    test_metrics = cached_metrics(cache, selected_formula, test_frame, test_target, context_key=f"{universe}:test_full") if selected_formula else {}
    walk_forward_metrics = backtest_formula(universe, selected_formula) if selected_formula else {}
    default = default_formula(universe)
    summary = {
        "universe": universe,
        "default_formula": default,
        "rolling_meta_selected_formula": selected_formula,
        "matches_default_formula": selected_formula == default,
        "selected_formula_stability": float(best["formula_stability"]),
        "window_best_config_frequency": window_best.to_dict(orient="records"),
        "best_config": {"config_key": str(best["config_key"]), **best_cfg},
        "mean_meta_metrics": {
            "meta_score": float(best["mean_meta_score"]),
            "rank_consistency": float(best["mean_rank_consistency"]),
            "rank_ic_retention": float(best["mean_rank_ic_retention"]),
            "calibration_rank_ic": float(best["mean_calibration_rank_ic"]),
            "meta_rank_ic": float(best["mean_meta_rank_ic"]),
            "meta_sharpe": float(best["mean_meta_sharpe"]),
            "meta_turnover": float(best["mean_meta_turnover"]),
        },
        "test_metrics": {key: round(float(value), 6) for key, value in test_metrics.items()},
        "walk_forward_metrics": {key: round(float(value), 6) for key, value in walk_forward_metrics.items()},
        "cache_stats": cache.stats(),
        "score_scale_stats": {
            "sharpe_scale": round(float(scale_stats.sharpe_scale), 6),
            "annual_return_scale": round(float(scale_stats.annual_return_scale), 6),
            "sharpe_std": round(float(scale_stats.sharpe_std), 6) if scale_stats.sharpe_std is not None else None,
            "annual_return_std": round(float(scale_stats.annual_return_std), 6) if scale_stats.annual_return_std is not None else None,
            "candidate_count": int(scale_stats.candidate_count),
        },
        "window_count": len(windows),
        "coarse_config_count": len(coarse_summary),
        "fine_config_count": int(fine_rows["config_key"].nunique()) if not fine_rows.empty else 0,
    }
    return combined_rows, combined_summary, window_best, summary


def build_markdown(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# Finance Rolling Meta-Validation Report",
        "",
        "| Universe | Default Signal | Rolling Meta Signal | Matches Default | Formula Stability | Mean Rank Consistency | Mean Retention | Walk-Forward Sharpe | Test Rank-IC | Cache Hits | Cache Misses |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        test_rank_ic = item["test_metrics"].get("rank_ic") if item["test_metrics"] else None
        wf_sharpe = item["walk_forward_metrics"].get("sharpe") if item["walk_forward_metrics"] else None
        meta_metrics = item.get("mean_meta_metrics", {})
        cache_stats = item.get("cache_stats", {})
        lines.append(
            f"| {item['universe']} | {item['default_formula']} | {item['rolling_meta_selected_formula']} | {item['matches_default_formula']} | "
            f"{item['selected_formula_stability']:.2f} | {meta_metrics.get('rank_consistency', 0.0):.4f} | "
            f"{meta_metrics.get('rank_ic_retention', 0.0):.4f} | {wf_sharpe:.4f} | {test_rank_ic:.4f} | "
            f"{cache_stats.get('hits', 0)} | {cache_stats.get('misses', 0)} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    detail_rows: list[pd.DataFrame] = []
    config_rows: list[pd.DataFrame] = []
    window_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for universe in parse_universes(args.universes):
        detail, config_summary, window_best, summary = run_universe(universe, args.window_count, args.coarse_top_k)
        detail_rows.append(detail)
        config_rows.append(config_summary)
        window_rows.append(window_best)
        summaries.append(summary)

    detail_frame = pd.concat(detail_rows, axis=0, ignore_index=True)
    config_frame = pd.concat(config_rows, axis=0, ignore_index=True)
    window_frame = pd.concat(window_rows, axis=0, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)

    detail_csv = args.output_root / "finance_rolling_meta_validation_detail.csv"
    config_csv = args.output_root / "finance_rolling_meta_validation_config_rankings.csv"
    window_csv = args.output_root / "finance_rolling_meta_validation_window_best.csv"
    summary_csv = args.output_root / "finance_rolling_meta_validation_summary.csv"
    json_path = args.output_root / "finance_rolling_meta_validation_report.json"
    md_path = args.output_root / "finance_rolling_meta_validation_report.md"

    detail_frame.to_csv(detail_csv, index=False)
    config_frame.to_csv(config_csv, index=False)
    window_frame.to_csv(window_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    payload = {
        "summaries": summaries,
        "detail_csv": str(detail_csv),
        "config_csv": str(config_csv),
        "window_best_csv": str(window_csv),
        "summary_csv": str(summary_csv),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(summaries) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "detail_csv": str(detail_csv),
                "config_csv": str(config_csv),
                "window_best_csv": str(window_csv),
                "summary_csv": str(summary_csv),
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
