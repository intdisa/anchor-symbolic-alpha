#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

SELECTION_MODES = (
    ("pareto_cross_seed_consensus", "pareto"),
    ("pareto_discrete_legacy", "pareto_discrete_legacy"),
    ("legacy_linear_selector", "legacy_linear"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build finance Pareto front diagnostics.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def _selected_record(records: list[Any], formula: str, prefix: str) -> dict[str, Any]:
    for record in records:
        if getattr(record, "formula", None) == formula:
            return {
                "selected_formula": formula,
                "front_size": getattr(record, f"{prefix}_front_size", None),
                "front_share": getattr(record, f"{prefix}_front_share", None),
                "crowding_distance": getattr(record, f"{prefix}_crowding_distance", None),
                "used_near_neighbor_tiebreak": bool(getattr(record, "used_near_neighbor_tiebreak", False)),
            }
    return {
        "selected_formula": formula,
        "front_size": None,
        "front_share": None,
        "crowding_distance": None,
        "used_near_neighbor_tiebreak": False,
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for universe in ("liquid500", "liquid1000"):
        raw_runs, candidates, valid_frame, valid_target, _, _ = load_universe_inputs(universe)
        cache = FormulaEvaluationCache()
        scale_stats = load_universe_scale_stats(universe, candidates, cache)
        for baseline_name, selection_mode in SELECTION_MODES:
            selector_cfg = dict(BASE_CONFIG)
            selector_cfg["temporal_selection_mode"] = selection_mode
            selector_cfg["cross_seed_selection_mode"] = selection_mode
            temporal_selector = TemporalRobustSelector(_selector_config(selector_cfg, scale_stats), evaluation_cache=cache)
            seed_runs = derive_seed_runs(
                raw_runs,
                valid_frame,
                valid_target,
                temporal_selector,
                evaluation_context=f"{universe}:front_diag:{selection_mode}",
            )
            temporal_rows: list[dict[str, Any]] = []
            for run in seed_runs:
                selected_formula = run.selector_records[0] if run.selector_records else ""
                ranked_records = list(run.selector_ranked_records)
                first_front_size = max((int(getattr(record, "temporal_front_size", 0) or 0) for record in ranked_records if int(getattr(record, "temporal_pareto_rank", 999) or 999) == 1), default=0)
                first_front_share = max((float(getattr(record, "temporal_front_share", 0.0) or 0.0) for record in ranked_records if int(getattr(record, "temporal_pareto_rank", 999) or 999) == 1), default=0.0)
                selected = _selected_record(ranked_records, selected_formula, "temporal")
                row = {
                    "universe": universe,
                    "baseline": baseline_name,
                    "layer": "temporal",
                    "seed": int(run.seed),
                    "first_front_size": first_front_size,
                    "first_front_share": round(float(first_front_share), 6),
                    **selected,
                }
                detail_rows.append(row)
                temporal_rows.append(row)

            consensus_selector = CrossSeedConsensusSelector(
                temporal_selector=temporal_selector,
                config=_consensus_config(selector_cfg),
            )
            consensus_outcome = consensus_selector.select(
                seed_runs,
                valid_frame,
                valid_target,
                base_candidates=candidates,
                evaluation_context=f"{universe}:front_diag:{selection_mode}",
            )
            selected_formula = consensus_outcome.selected_formulas[0] if consensus_outcome.selected_formulas else ""
            ranked_records = list(consensus_outcome.ranked_records)
            first_front_size = max((int(record.consensus_front_size or 0) for record in ranked_records if int(record.consensus_pareto_rank or 999) == 1), default=0)
            first_front_share = max((float(record.consensus_front_share or 0.0) for record in ranked_records if int(record.consensus_pareto_rank or 999) == 1), default=0.0)
            selected = _selected_record(ranked_records, selected_formula, "consensus")
            consensus_row = {
                "universe": universe,
                "baseline": baseline_name,
                "layer": "consensus",
                "seed": None,
                "first_front_size": first_front_size,
                "first_front_share": round(float(first_front_share), 6),
                **selected,
            }
            detail_rows.append(consensus_row)

            for layer_name, layer_rows in (("temporal", temporal_rows), ("consensus", [consensus_row])):
                frame = pd.DataFrame(layer_rows)
                summary_rows.append(
                    {
                        "universe": universe,
                        "baseline": baseline_name,
                        "layer": layer_name,
                        "mean_first_front_size": round(float(frame["first_front_size"].mean()), 6),
                        "mean_first_front_share": round(float(frame["first_front_share"].mean()), 6),
                        "mean_selected_crowding_distance": round(float(frame["crowding_distance"].dropna().mean()), 6) if frame["crowding_distance"].notna().any() else None,
                        "near_neighbor_tiebreak_rate": round(float(frame["used_near_neighbor_tiebreak"].mean()), 6),
                    }
                )
    return detail_rows, summary_rows


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Finance Pareto Front Diagnostics",
        "",
        "| Universe | Baseline | Layer | Mean Front Size | Mean Front Share | Mean Crowding | Tie-break Rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        crowding = "NA" if row["mean_selected_crowding_distance"] is None else f"{float(row['mean_selected_crowding_distance']):.4f}"
        lines.append(
            f"| {row['universe']} | {row['baseline']} | {row['layer']} | {float(row['mean_first_front_size']):.2f} | {float(row['mean_first_front_share']):.4f} | {crowding} | {float(row['near_neighbor_tiebreak_rate']):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    detail_rows, summary_rows = build_rows()
    detail_frame = pd.DataFrame(detail_rows)
    summary_frame = pd.DataFrame(summary_rows)
    csv_path = args.output_root / "finance_pareto_front_diagnostics.csv"
    json_path = args.output_root / "finance_pareto_front_diagnostics.json"
    md_path = args.output_root / "finance_pareto_front_diagnostics.md"
    detail_csv = args.output_root / "finance_pareto_front_detail.csv"
    detail_frame.to_csv(detail_csv, index=False)
    summary_frame.to_csv(csv_path, index=False)
    json_path.write_text(summary_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(summary_rows) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "detail_csv": str(detail_csv), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
