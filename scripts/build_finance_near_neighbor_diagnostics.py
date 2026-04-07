#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.evaluation.finance_baselines import (
    feature_family_overlap,
    formula_feature_families,
    token_overlap,
)
from knowledge_guided_symbolic_alpha.evaluation.panel_dispatch import evaluate_formula_metrics
from knowledge_guided_symbolic_alpha.selection import (
    CrossSeedConsensusConfig,
    CrossSeedConsensusSelector,
    FormulaEvaluationCache,
    RobustSelectorConfig,
    TemporalRobustSelector,
)
from scripts.run_finance_rolling_meta_validation import (
    BASE_CONFIG,
    _consensus_config,
    _selector_config,
    derive_seed_runs,
    load_universe_inputs,
    load_universe_scale_stats,
)


BASELINE_FILES = (
    "finance_walkforward_baselines_pareto.csv",
    "finance_walkforward_baselines_extended.csv",
    "finance_walkforward_baselines.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build near-neighbor diagnostics for finance signal families.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _baseline_frame(output_root: Path) -> pd.DataFrame | None:
    for name in BASELINE_FILES:
        path = output_root / name
        if path.exists():
            return pd.read_csv(path)
    return None


def _selection_payload(universe: str, output_root: Path, top_k: int) -> tuple[str, list[dict[str, Any]], list[str], pd.DataFrame, pd.Series]:
    raw_runs, candidates, valid_frame, valid_target, _, _ = load_universe_inputs(universe)
    cache = FormulaEvaluationCache()
    scale_stats = load_universe_scale_stats(universe, candidates, cache)
    temporal_selector = TemporalRobustSelector(_selector_config(BASE_CONFIG, scale_stats), evaluation_cache=cache)
    consensus_selector = CrossSeedConsensusSelector(
        temporal_selector=temporal_selector,
        config=_consensus_config(BASE_CONFIG),
    )
    eval_context = f"{universe}:near_neighbor:pareto"
    seed_runs = derive_seed_runs(raw_runs, valid_frame, valid_target, temporal_selector, evaluation_context=eval_context)
    outcome = consensus_selector.select(
        seed_runs,
        valid_frame,
        valid_target,
        base_candidates=candidates,
        evaluation_context=eval_context,
    )
    canonical = outcome.selected_formulas[0] if outcome.selected_formulas else ""
    ranked_records = [asdict(record) for record in outcome.ranked_records[:top_k]]

    formulas: list[str] = [canonical]
    formulas.extend(str(item["formula"]) for item in ranked_records)

    baseline_frame = _baseline_frame(output_root)
    if baseline_frame is not None:
        formulas.extend(
            str(value)
            for value in baseline_frame.loc[baseline_frame["universe"] == universe, "formula"].dropna().tolist()
        )

    ordered: list[str] = []
    seen: set[str] = set()
    for formula in formulas:
        if formula and formula not in seen:
            seen.add(formula)
            ordered.append(formula)
    return canonical, ranked_records, ordered, valid_frame, valid_target


def signal_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 5:
        return 0.0
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return 0.0 if value is None or not np.isfinite(value) else float(value)


def evaluate_signals(formulas: list[str], frame: pd.DataFrame, target: pd.Series) -> dict[str, pd.Series]:
    signals: dict[str, pd.Series] = {}
    for formula in formulas:
        evaluated = evaluate_formula_metrics(formula, frame, target).evaluated
        signals[formula] = pd.Series(evaluated.signal, index=frame.index, dtype=float)
    return signals


def _formula_rows(universe: str, canonical: str, ranked: list[dict[str, Any]], formulas: list[str], signals: dict[str, pd.Series]) -> list[dict[str, Any]]:
    support_lookup = {str(item["formula"]): item for item in ranked}
    rows: list[dict[str, Any]] = []
    for formula in formulas:
        support = support_lookup.get(formula, {})
        rows.append(
            {
                "universe": universe,
                "formula": formula,
                "is_canonical": formula == canonical,
                "feature_families": ",".join(formula_feature_families(formula)),
                "complexity": len(formula.split()),
                "candidate_seed_support": int(support.get("candidate_seed_support") or 0),
                "selector_seed_support": int(support.get("selector_seed_support") or 0),
                "champion_seed_support": int(support.get("champion_seed_support") or 0),
                "consensus_pareto_rank": int(support.get("consensus_pareto_rank") or 999),
                "consensus_tiebreak_rank": int(support.get("consensus_tiebreak_rank") or 999),
                "mean_temporal_pareto_rank": float(support.get("mean_temporal_pareto_rank") or np.nan),
                "mean_temporal_tiebreak_rank": float(support.get("mean_temporal_tiebreak_rank") or np.nan),
                "mean_temporal_score": float(support.get("mean_temporal_score") or np.nan),
                "support_adjusted_score": float(support.get("support_adjusted_score") or np.nan),
                "signal_non_null_fraction": round(float(signals[formula].notna().mean()), 4),
            }
        )
    rows.sort(key=lambda item: (item["universe"], not item["is_canonical"], item["consensus_pareto_rank"], item["consensus_tiebreak_rank"], item["complexity"]))
    return rows


def _pair_rows(universe: str, canonical: str, formulas: list[str], formula_rows: list[dict[str, Any]], signals: dict[str, pd.Series]) -> list[dict[str, Any]]:
    metadata = {row["formula"]: row for row in formula_rows}
    rows: list[dict[str, Any]] = []
    for index, left_formula in enumerate(formulas):
        for right_formula in formulas[index + 1 :]:
            left_support = metadata[left_formula]
            right_support = metadata[right_formula]
            rows.append(
                {
                    "universe": universe,
                    "left_formula": left_formula,
                    "right_formula": right_formula,
                    "left_is_canonical": left_formula == canonical,
                    "right_is_canonical": right_formula == canonical,
                    "token_overlap": round(token_overlap(left_formula, right_formula), 4),
                    "feature_family_overlap": round(feature_family_overlap(left_formula, right_formula), 4),
                    "signal_spearman_corr": round(signal_corr(signals[left_formula], signals[right_formula]), 4),
                    "left_feature_families": left_support["feature_families"],
                    "right_feature_families": right_support["feature_families"],
                    "left_consensus_pareto_rank": left_support["consensus_pareto_rank"],
                    "right_consensus_pareto_rank": right_support["consensus_pareto_rank"],
                    "left_selector_support": left_support["selector_seed_support"],
                    "right_selector_support": right_support["selector_seed_support"],
                    "left_champion_support": left_support["champion_seed_support"],
                    "right_champion_support": right_support["champion_seed_support"],
                    "left_support_score": round(float(left_support.get("support_adjusted_score") or 0.0), 6),
                    "right_support_score": round(float(right_support.get("support_adjusted_score") or 0.0), 6),
                }
            )
    rows.sort(
        key=lambda item: (
            item["universe"],
            not (item["left_is_canonical"] or item["right_is_canonical"]),
            -item["signal_spearman_corr"],
            -item["token_overlap"],
        )
    )
    return rows


def build_payload(output_root: Path, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    formula_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for universe in ("liquid500", "liquid1000"):
        canonical, ranked, formulas, valid_frame, valid_target = _selection_payload(universe, output_root, top_k)
        signals = evaluate_signals(formulas, valid_frame, valid_target)
        universe_formula_rows = _formula_rows(universe, canonical, ranked, formulas, signals)
        formula_rows.extend(universe_formula_rows)
        pair_rows.extend(_pair_rows(universe, canonical, formulas, universe_formula_rows, signals))
    return formula_rows, pair_rows


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Finance Near-Neighbor Diagnostics",
        "",
        "| Universe | Left Signal | Right Signal | Token Overlap | Family Overlap | Signal Corr | Left Pareto Rank | Right Pareto Rank |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['universe']} | {row['left_formula']} | {row['right_formula']} | {row['token_overlap']:.4f} | "
            f"{row['feature_family_overlap']:.4f} | {row['signal_spearman_corr']:.4f} | {row['left_consensus_pareto_rank']} | {row['right_consensus_pareto_rank']} |"
        )
    return "\n".join(lines)


def build_cluster_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Finance Near-Neighbor Clusters",
        "",
        "| Universe | Signal | Canonical? | Pareto Rank | Champion Support | Selector Support | Candidate Support | Families |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['universe']} | {row['formula']} | {row['is_canonical']} | {row['consensus_pareto_rank']} | "
            f"{row['champion_seed_support']} | {row['selector_seed_support']} | {row['candidate_seed_support']} | {row['feature_families']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    formula_rows, pair_rows = build_payload(args.output_root, args.top_k)

    pair_frame = pd.DataFrame(pair_rows)
    formula_frame = pd.DataFrame(formula_rows)

    csv_path = args.output_root / "finance_near_neighbor_diagnostics.csv"
    json_path = args.output_root / "finance_near_neighbor_diagnostics.json"
    md_path = args.output_root / "finance_near_neighbor_diagnostics.md"
    clusters_csv = args.output_root / "finance_near_neighbor_clusters.csv"
    clusters_json = args.output_root / "finance_near_neighbor_clusters.json"
    clusters_md = args.output_root / "finance_near_neighbor_clusters.md"

    pair_frame.to_csv(csv_path, index=False)
    json_path.write_text(pair_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(pair_rows) + "\n", encoding="utf-8")

    formula_frame.to_csv(clusters_csv, index=False)
    clusters_json.write_text(formula_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    clusters_md.write_text(build_cluster_markdown(formula_rows) + "\n", encoding="utf-8")

    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path), "clusters_csv": str(clusters_csv)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
