#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.selection import (
    CrossSeedConsensusSelector,
    FormulaEvaluationCache,
    TemporalRobustSelector,
)
from scripts.run_finance_rolling_meta_validation import (
    BASE_CONFIG,
    _consensus_config,
    _selector_config,
    backtest_formula,
    cached_metrics,
    default_formula,
    derive_seed_runs,
    load_universe_scale_stats,
    load_universe_inputs,
)


SENSITIVITY_STEPS = {
    "min_valid_slices": [1, 2, 3],
    "min_mean_rank_ic": [-0.005, 0.0, 0.005],
    "min_slice_rank_ic": [-0.02, -0.01, 0.0],
    "near_neighbor_signal_corr": [0.70, 0.80, 0.90],
    "near_neighbor_token_overlap": [0.40, 0.50, 0.60],
    "near_neighbor_feature_family_overlap": [0.85, 0.95, 1.00],
    "min_seed_support": [2, 3, 4],
    "max_pairwise_signal_corr": [0.90, 0.93, 0.96],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finance selector ablation and sensitivity studies.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--universes", type=str, default="liquid500,liquid1000")
    return parser.parse_args()


def parse_universes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_best_configs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {universe: dict(BASE_CONFIG) for universe in ("liquid500", "liquid1000")}
    frame = pd.read_csv(path)
    mapping: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        best_cfg = row.get("best_config")
        if isinstance(best_cfg, str) and best_cfg:
            try:
                best_cfg = json.loads(best_cfg.replace("'", '"'))
            except Exception:
                best_cfg = None
        config = dict(BASE_CONFIG)
        if isinstance(best_cfg, dict):
            config.update(best_cfg)
        else:
            for key in BASE_CONFIG:
                prefixed = f"best_config_{key}"
                if prefixed in row and pd.notna(row[prefixed]):
                    config[key] = row[prefixed]
        mapping[str(row["universe"])] = config
    for universe in ("liquid500", "liquid1000"):
        mapping.setdefault(universe, dict(BASE_CONFIG))
    return mapping


def build_ablation_variants(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "full": dict(base),
        "no_temporal_pareto": {**base, "temporal_selection_mode": "legacy_linear"},
        "no_cross_seed_pareto": {**base, "cross_seed_selection_mode": "legacy_linear"},
        "pareto_discrete_legacy": {
            **base,
            "temporal_selection_mode": "pareto_discrete_legacy",
            "cross_seed_selection_mode": "pareto_discrete_legacy",
        },
        "no_near_neighbor_tie_break": {**base, "enable_near_neighbor_tie_break": False},
        "no_redundancy_gate": {**base, "enable_redundancy_gate": False, "max_pairwise_signal_corr": 1.0},
        "no_admissibility_gate": {
            **base,
            "enable_admissibility_gate": False,
            "min_valid_slices": 1,
            "min_mean_rank_ic": -1.0,
            "min_slice_rank_ic": -1.0,
        },
        "legacy_linear_selector": {
            **base,
            "temporal_selection_mode": "legacy_linear",
            "cross_seed_selection_mode": "legacy_linear",
        },
    }


def single_factor_configs(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = [("center", dict(base))]
    for key, values in SENSITIVITY_STEPS.items():
        current = base[key]
        if current not in values:
            continue
        index = values.index(current)
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(values):
                cfg = dict(base)
                cfg[key] = values[neighbor_index]
                suffix = "down" if neighbor_index < index else "up"
                rows.append((f"{key}_{suffix}", cfg))
    return rows


def joint_group_configs(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for min_mean in (-0.005, 0.0, 0.005):
        for min_slice in (-0.02, -0.01, 0.0):
            cfg = dict(base)
            cfg["min_mean_rank_ic"] = min_mean
            cfg["min_slice_rank_ic"] = min_slice
            rows.append((f"temporal_{min_mean:.3f}_{min_slice:.3f}", cfg))
    for support in (2, 3, 4):
        for corr in (0.90, 0.93, 0.96):
            cfg = dict(base)
            cfg["min_seed_support"] = support
            cfg["max_pairwise_signal_corr"] = corr
            rows.append((f"support_{support}_{corr:.2f}", cfg))
    return rows


def evaluate_selection(
    universe: str,
    label: str,
    study_type: str,
    cfg: dict[str, Any],
    *,
    raw_runs: list[dict[str, Any]],
    candidates: list[Any],
    valid_frame: pd.DataFrame,
    valid_target: pd.Series,
    test_frame: pd.DataFrame,
    test_target: pd.Series,
    cache: FormulaEvaluationCache,
    wf_cache: dict[str, dict[str, float]],
    scale_stats: Any,
) -> dict[str, Any]:
    temporal_selector = TemporalRobustSelector(_selector_config(cfg, scale_stats), evaluation_cache=cache)
    consensus_selector = CrossSeedConsensusSelector(
        temporal_selector=temporal_selector,
        config=_consensus_config(cfg),
    )
    valid_context = f"{universe}:hyper_valid"
    test_context = f"{universe}:hyper_test"
    seed_runs = derive_seed_runs(raw_runs, valid_frame, valid_target, temporal_selector, evaluation_context=valid_context)
    outcome = consensus_selector.select(
        seed_runs,
        valid_frame,
        valid_target,
        base_candidates=candidates,
        evaluation_context=valid_context,
    )
    selected_formula = outcome.selected_formulas[0] if outcome.selected_formulas else ""
    test_metrics = cached_metrics(cache, selected_formula, test_frame, test_target, context_key=test_context) if selected_formula else {}
    if selected_formula and selected_formula not in wf_cache:
        wf_cache[selected_formula] = backtest_formula(universe, selected_formula)
    walk_metrics = wf_cache.get(selected_formula, {})
    off_formulas = [run.selector_records[0] for run in seed_runs if run.selector_records and run.selector_records[0] != selected_formula]
    off_sharpes = []
    for formula in sorted(set(off_formulas)):
        if formula not in wf_cache:
            wf_cache[formula] = backtest_formula(universe, formula)
        off_sharpes.append(float(wf_cache[formula].get("sharpe") or 0.0))
    canonical = default_formula(universe)
    return {
        "universe": universe,
        "label": label,
        "variant": label,
        "study_type": study_type,
        "selected_formula": selected_formula,
        "matches_default_formula": selected_formula == canonical,
        "test_rank_ic": round(float(test_metrics.get("rank_ic") or 0.0), 6),
        "test_sharpe": round(float(test_metrics.get("sharpe") or 0.0), 6),
        "walk_forward_sharpe": round(float(walk_metrics.get("sharpe") or 0.0), 6),
        "walk_forward_max_drawdown": round(float(walk_metrics.get("max_drawdown") or 0.0), 6),
        "off_consensus_seed_fraction": round(float(len(off_formulas) / max(len(seed_runs), 1)), 6),
        "mean_off_consensus_wf_sharpe": round(float(np.mean(off_sharpes)), 6) if off_sharpes else None,
        "worst_off_consensus_wf_sharpe": round(float(np.min(off_sharpes)), 6) if off_sharpes else None,
        "temporal_selection_mode": str(cfg["temporal_selection_mode"]),
        "cross_seed_selection_mode": str(cfg["cross_seed_selection_mode"]),
        "min_valid_slices": int(cfg["min_valid_slices"]),
        "min_mean_rank_ic": float(cfg["min_mean_rank_ic"]),
        "min_slice_rank_ic": float(cfg["min_slice_rank_ic"]),
        "near_neighbor_signal_corr": float(cfg["near_neighbor_signal_corr"]),
        "near_neighbor_token_overlap": float(cfg["near_neighbor_token_overlap"]),
        "near_neighbor_feature_family_overlap": float(cfg["near_neighbor_feature_family_overlap"]),
        "min_seed_support": int(cfg["min_seed_support"]),
        "max_pairwise_signal_corr": float(cfg["max_pairwise_signal_corr"]),
        "enable_near_neighbor_tie_break": bool(cfg["enable_near_neighbor_tie_break"]),
        "enable_admissibility_gate": bool(cfg["enable_admissibility_gate"]),
        "enable_redundancy_gate": bool(cfg["enable_redundancy_gate"]),
    }


def build_markdown(title: str, rows: list[dict[str, Any]], row_key: str) -> str:
    lines = [f"# {title}", "", "| Universe | Variant | Signal | Matches S* | WF Sharpe | Test Rank-IC | Off-Consensus Seeds |", "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            f"| {row['universe']} | {row[row_key]} | {row['selected_formula']} | {row['matches_default_formula']} | "
            f"{row['walk_forward_sharpe']:.4f} | {row['test_rank_ic']:.4f} | {row['off_consensus_seed_fraction']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply operated on the grouping columns.*")
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    best_configs = load_best_configs(args.output_root / "finance_rolling_meta_validation_summary.csv")

    ablation_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for universe in parse_universes(args.universes):
        raw_runs, candidates, valid_frame, valid_target, test_frame, test_target = load_universe_inputs(universe)
        cache = FormulaEvaluationCache()
        wf_cache: dict[str, dict[str, float]] = {}
        scale_stats = load_universe_scale_stats(universe, candidates, cache)
        base = dict(best_configs[universe])

        for label, cfg in build_ablation_variants(base).items():
            ablation_rows.append(
                evaluate_selection(
                    universe,
                    label,
                    "ablation",
                    cfg,
                    raw_runs=raw_runs,
                    candidates=candidates,
                    valid_frame=valid_frame,
                    valid_target=valid_target,
                    test_frame=test_frame,
                    test_target=test_target,
                    cache=cache,
                    wf_cache=wf_cache,
                    scale_stats=scale_stats,
                )
            )

        for label, cfg in single_factor_configs(base) + joint_group_configs(base):
            sensitivity_rows.append(
                evaluate_selection(
                    universe,
                    label,
                    "sensitivity",
                    cfg,
                    raw_runs=raw_runs,
                    candidates=candidates,
                    valid_frame=valid_frame,
                    valid_target=valid_target,
                    test_frame=test_frame,
                    test_target=test_target,
                    cache=cache,
                    wf_cache=wf_cache,
                    scale_stats=scale_stats,
                )
            )

    ablation_frame = pd.DataFrame(ablation_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    ablation_csv = args.output_root / "finance_hyperparameter_ablation.csv"
    ablation_json = args.output_root / "finance_hyperparameter_ablation.json"
    ablation_md = args.output_root / "finance_hyperparameter_ablation.md"
    sensitivity_csv = args.output_root / "finance_sensitivity_surface.csv"
    sensitivity_json = args.output_root / "finance_sensitivity_surface.json"
    sensitivity_md = args.output_root / "finance_sensitivity_surface.md"

    ablation_frame.to_csv(ablation_csv, index=False)
    ablation_json.write_text(ablation_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    ablation_md.write_text(build_markdown("Finance Hyperparameter Ablation", ablation_rows, "label") + "\n", encoding="utf-8")
    sensitivity_frame.to_csv(sensitivity_csv, index=False)
    sensitivity_json.write_text(sensitivity_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    sensitivity_md.write_text(build_markdown("Finance Sensitivity Surface", sensitivity_rows, "label") + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ablation_csv": str(ablation_csv),
                "ablation_json": str(ablation_json),
                "ablation_markdown": str(ablation_md),
                "sensitivity_csv": str(sensitivity_csv),
                "sensitivity_json": str(sensitivity_json),
                "sensitivity_markdown": str(sensitivity_md),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
