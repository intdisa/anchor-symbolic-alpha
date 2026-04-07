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

from experiments.common import dataset_columns, load_dataset_bundle, load_experiment_name
from knowledge_guided_symbolic_alpha.selection import (
    CrossSeedConsensusConfig,
    CrossSeedConsensusSelector,
    FormulaEvaluationCache,
    RobustSelectorConfig,
    TemporalRobustSelector,
)
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_metrics import cross_sectional_ic_summary, cross_sectional_long_short_returns
from knowledge_guided_symbolic_alpha.evaluation.risk_metrics import max_drawdown, sharpe_ratio
from scripts.run_finance_rolling_meta_validation import (
    BASE_CONFIG,
    EXPERIMENT_CONFIG,
    _consensus_config,
    _selector_config,
    derive_seed_runs,
    load_universe_inputs,
    load_universe_scale_stats,
)


REFERENCE_BASELINES = (
    "legacy_linear_selector",
    "support_adjusted_cross_seed_consensus",
    "pareto_discrete_legacy",
    "pareto_cross_seed_consensus",
)
BASELINE_FILES = (
    "finance_walkforward_baselines_pareto.csv",
    "finance_walkforward_baselines_extended.csv",
    "finance_walkforward_baselines.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a report showing how consensus filtering removes pseudo-alpha winners.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument(
        "--skip-live-selection",
        action="store_true",
        help="Build pseudo-alpha reports from existing baseline files only, without recomputing live seed-level selection payloads.",
    )
    return parser.parse_args()


def _baseline_path(output_root: Path) -> Path | None:
    for name in BASELINE_FILES:
        path = output_root / name
        if path.exists():
            return path
    return None


def _current_selection_payload(universe: str) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    raw_runs, candidates, valid_frame, valid_target, _, _ = load_universe_inputs(universe)
    cache = FormulaEvaluationCache()
    scale_stats = load_universe_scale_stats(universe, candidates, cache)
    temporal_selector = TemporalRobustSelector(_selector_config(BASE_CONFIG, scale_stats), evaluation_cache=cache)
    consensus_selector = CrossSeedConsensusSelector(
        temporal_selector=temporal_selector,
        config=_consensus_config(BASE_CONFIG),
    )
    eval_context = f"{universe}:selector_filtering:pareto"
    seed_runs = derive_seed_runs(raw_runs, valid_frame, valid_target, temporal_selector, evaluation_context=eval_context)
    outcome = consensus_selector.select(
        seed_runs,
        valid_frame,
        valid_target,
        base_candidates=candidates,
        evaluation_context=eval_context,
    )
    consensus_formula = outcome.selected_formulas[0] if outcome.selected_formulas else ""
    consensus_metrics = test_formula_metrics(universe, consensus_formula) if consensus_formula else {}

    formula_metrics: dict[str, dict[str, float]] = {}

    def metrics_for(formula: str) -> dict[str, float]:
        if formula not in formula_metrics:
            formula_metrics[formula] = test_formula_metrics(universe, formula) if formula else {}
        return formula_metrics[formula]

    seed_rows: list[dict[str, object]] = []
    off_consensus_rows: list[dict[str, object]] = []
    for run in seed_runs:
        selector_formula = run.selector_records[0] if run.selector_records else ""
        champion_formula = run.champion_records[0] if run.champion_records else ""
        selector_metrics = metrics_for(selector_formula)
        row = {
            "universe": universe,
            "seed": int(run.seed),
            "selector_formula": selector_formula,
            "champion_formula": champion_formula,
            "consensus_formula": consensus_formula,
            "selector_matches_consensus": selector_formula == consensus_formula,
            "champion_matches_consensus": champion_formula == consensus_formula,
            "walk_sharpe": round(float(selector_metrics.get("sharpe") or 0.0), 4),
            "mean_test_rank_ic": round(float(selector_metrics.get("mean_test_rank_ic") or 0.0), 4),
            "max_drawdown": round(float(selector_metrics.get("max_drawdown") or 0.0), 4),
            "consensus_sharpe": round(float(consensus_metrics.get("sharpe") or 0.0), 4),
            "consensus_rank_ic": round(float(consensus_metrics.get("mean_test_rank_ic") or 0.0), 4),
            "sharpe_gap_vs_consensus": round(float(selector_metrics.get("sharpe") or 0.0) - float(consensus_metrics.get("sharpe") or 0.0), 4),
            "rank_ic_gap_vs_consensus": round(float(selector_metrics.get("mean_test_rank_ic") or 0.0) - float(consensus_metrics.get("mean_test_rank_ic") or 0.0), 4),
        }
        seed_rows.append(row)
        if selector_formula != consensus_formula:
            off_consensus_rows.append(row)

    off_sorted = sorted(off_consensus_rows, key=lambda item: float(item["walk_sharpe"]))
    summary = {
        "universe": universe,
        "consensus_formula": consensus_formula,
        "consensus_sharpe": round(float(consensus_metrics.get("sharpe") or 0.0), 4),
        "consensus_rank_ic": round(float(consensus_metrics.get("mean_test_rank_ic") or 0.0), 4),
        "seed_count": len(seed_rows),
        "off_consensus_seed_count": len(off_consensus_rows),
        "off_consensus_seed_fraction": round(len(off_consensus_rows) / max(len(seed_rows), 1), 4),
        "mean_off_consensus_sharpe": round(sum(float(item["walk_sharpe"]) for item in off_consensus_rows) / max(len(off_consensus_rows), 1), 4) if off_consensus_rows else None,
        "mean_off_consensus_rank_ic": round(sum(float(item["mean_test_rank_ic"]) for item in off_consensus_rows) / max(len(off_consensus_rows), 1), 4) if off_consensus_rows else None,
        "worst_off_consensus_formula": off_sorted[0]["selector_formula"] if off_sorted else None,
        "worst_off_consensus_sharpe": off_sorted[0]["walk_sharpe"] if off_sorted else None,
    }
    return seed_rows, summary, {"summary": summary, "seed_rows": seed_rows}


def test_formula_metrics(universe: str, formula: str) -> dict[str, float]:
    bundle = load_dataset_bundle(Path(f"configs/us_equities_{universe}.yaml"))
    frame = bundle.splits.test.copy()
    dataset_name = load_experiment_name(EXPERIMENT_CONFIG)
    _, return_column = dataset_columns(dataset_name)
    evaluator = CrossSectionalFormulaEvaluator()
    evaluated = evaluator.evaluate(formula, frame)
    signal = evaluated.signal.astype(float)
    aligned = frame.loc[signal.index]
    returns, _ = cross_sectional_long_short_returns(
        signal,
        aligned[return_column],
        aligned["date"],
        aligned["permno"],
        quantile=0.2,
        weight_scheme="equal",
    )
    ic_metrics = cross_sectional_ic_summary(signal, aligned[return_column], aligned["date"])
    return {
        "sharpe": float(sharpe_ratio(returns)),
        "mean_test_rank_ic": float(ic_metrics["rank_ic"]),
        "max_drawdown": float(max_drawdown(returns)),
    }


def build_payload(output_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    payload: dict[str, Any] = {"universes": {}}
    for universe in ("liquid500", "liquid1000"):
        universe_seed_rows, summary, universe_payload = _current_selection_payload(universe)
        seed_rows.extend(universe_seed_rows)
        summary_rows.append(summary)
        payload["universes"][universe] = universe_payload
    return seed_rows, summary_rows, payload


def build_pseudoalpha_cases(
    seed_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    output_root: Path,
) -> list[dict[str, object]]:
    consensus_lookup = {row["universe"]: row for row in summary_rows}
    cases: list[dict[str, object]] = []
    for row in seed_rows:
        if bool(row["selector_matches_consensus"]):
            continue
        consensus = consensus_lookup[row["universe"]]
        cases.append(
            {
                "universe": row["universe"],
                "case_type": "seed_off_consensus",
                "source": f"seed_{row['seed']}",
                "candidate_formula": row["selector_formula"],
                "reference_formula": consensus["consensus_formula"],
                "candidate_walk_sharpe": row["walk_sharpe"],
                "reference_walk_sharpe": consensus["consensus_sharpe"],
                "candidate_rank_ic": row["mean_test_rank_ic"],
                "reference_rank_ic": consensus["consensus_rank_ic"],
                "sharpe_gap_vs_reference": row["sharpe_gap_vs_consensus"],
                "rank_ic_gap_vs_reference": row["rank_ic_gap_vs_consensus"],
            }
        )

    baseline_path = _baseline_path(output_root)
    if baseline_path is not None:
        baseline_frame = pd.read_csv(baseline_path)
        for universe, group in baseline_frame.groupby("universe"):
            consensus_row = next(
                (group[group["baseline"] == baseline].iloc[0] for baseline in REFERENCE_BASELINES if (group["baseline"] == baseline).any()),
                None,
            )
            if consensus_row is None:
                continue
            reference_metrics = test_formula_metrics(str(universe), str(consensus_row.formula))
            for row in group.itertuples(index=False):
                if bool(getattr(row, "matches_consensus_formula", False)):
                    continue
                candidate_metrics = test_formula_metrics(str(universe), str(row.formula))
                cases.append(
                    {
                        "universe": universe,
                        "case_type": "baseline_misfire",
                        "source": row.baseline,
                        "candidate_formula": row.formula,
                        "reference_formula": consensus_row.formula,
                        "candidate_walk_sharpe": round(float(candidate_metrics.get("sharpe") or 0.0), 4),
                        "reference_walk_sharpe": round(float(reference_metrics.get("sharpe") or 0.0), 4),
                        "candidate_rank_ic": round(float(candidate_metrics.get("mean_test_rank_ic") or 0.0), 4),
                        "reference_rank_ic": round(float(reference_metrics.get("mean_test_rank_ic") or 0.0), 4),
                        "sharpe_gap_vs_reference": round(float(candidate_metrics.get("sharpe") or 0.0) - float(reference_metrics.get("sharpe") or 0.0), 4),
                        "rank_ic_gap_vs_reference": round(float(candidate_metrics.get("mean_test_rank_ic") or 0.0) - float(reference_metrics.get("mean_test_rank_ic") or 0.0), 4),
                    }
                )
    cases.sort(key=lambda item: (item["universe"], float(item["sharpe_gap_vs_reference"]), float(item["rank_ic_gap_vs_reference"])))
    return cases


def build_markdown(summary_rows: list[dict[str, object]], seed_rows: list[dict[str, object]]) -> str:
    lines = ["# Signal Filtering Report", "", "## Universe Summary", ""]
    lines.append("| Universe | Canonical Signal | Canonical WF Sharpe | Off-Consensus Seeds | Mean Off-Consensus Sharpe | Worst Off-Consensus Signal |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in summary_rows:
        lines.append(
            f"| {row['universe']} | {row['consensus_formula']} | {row['consensus_sharpe']:.4f} | {row['off_consensus_seed_count']} | "
            + (f"{row['mean_off_consensus_sharpe']:.4f}" if row['mean_off_consensus_sharpe'] is not None else "NA")
            + f" | {row['worst_off_consensus_formula'] or 'NA'} |"
        )
    lines.extend(["", "## Seed-Level Details", ""])
    lines.append("| Universe | Seed | Selector Signal | Champion Signal | Selector=Canonical | WF Sharpe | Rank-IC | Gap vs Canonical |")
    lines.append("| --- | ---: | --- | --- | --- | ---: | ---: | ---: |")
    for row in sorted(seed_rows, key=lambda item: (item['universe'], item['seed'])):
        lines.append(
            f"| {row['universe']} | {row['seed']} | {row['selector_formula']} | {row['champion_formula']} | "
            f"{row['selector_matches_consensus']} | {row['walk_sharpe']:.4f} | {row['mean_test_rank_ic']:.4f} | {row['sharpe_gap_vs_consensus']:.4f} |"
        )
    return "\n".join(lines)


def build_pseudoalpha_markdown(cases: list[dict[str, object]]) -> str:
    lines = [
        "# Pseudo-Alpha Cases",
        "",
        "| Universe | Case Type | Source | Candidate Signal | Reference Signal | Candidate WF Sharpe | Reference WF Sharpe | Candidate Rank-IC | Reference Rank-IC | Sharpe Gap |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cases:
        lines.append(
            f"| {row['universe']} | {row['case_type']} | {row['source']} | {row['candidate_formula']} | {row['reference_formula']} | "
            f"{float(row['candidate_walk_sharpe']):.4f} | {float(row['reference_walk_sharpe']):.4f} | {float(row['candidate_rank_ic']):.4f} | {float(row['reference_rank_ic']):.4f} | {float(row['sharpe_gap_vs_reference']):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.skip_live_selection:
        seed_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        payload: dict[str, object] = {"universes": {}, "live_selection_skipped": True}
    else:
        seed_rows, summary_rows, payload = build_payload(args.output_root)
    pseudoalpha_cases = build_pseudoalpha_cases(seed_rows, summary_rows, args.output_root)
    pd.DataFrame(seed_rows).to_csv(args.output_root / "selector_filtering_seed_rows.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_root / "selector_filtering_summary.csv", index=False)
    pseudoalpha_frame = pd.DataFrame(pseudoalpha_cases)
    pseudoalpha_frame.to_csv(args.output_root / "selector_pseudoalpha_cases.csv", index=False)
    pseudoalpha_frame.to_csv(args.output_root / "finance_pseudoalpha_cases.csv", index=False)
    (args.output_root / "selector_filtering_report.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "selector_filtering_report.md").write_text(build_markdown(summary_rows, seed_rows) + "\n", encoding="utf-8")
    pseudo_md = build_pseudoalpha_markdown(pseudoalpha_cases) + "\n"
    (args.output_root / "selector_pseudoalpha_cases.md").write_text(pseudo_md, encoding="utf-8")
    (args.output_root / "finance_pseudoalpha_cases.md").write_text(pseudo_md, encoding="utf-8")
    (args.output_root / "finance_pseudoalpha_cases.json").write_text(
        pseudoalpha_frame.to_json(orient="records", force_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary_csv": str(args.output_root / "selector_filtering_summary.csv"),
                "seed_rows_csv": str(args.output_root / "selector_filtering_seed_rows.csv"),
                "pseudoalpha_cases_csv": str(args.output_root / "finance_pseudoalpha_cases.csv"),
                "report_json": str(args.output_root / "selector_filtering_report.json"),
                "report_markdown": str(args.output_root / "selector_filtering_report.md"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
