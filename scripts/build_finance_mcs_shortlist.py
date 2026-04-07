#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
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
from scripts.run_finance_walkforward_baselines import run_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a proxy Model Confidence Set report for finance signal shortlists.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--shortlist-size", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    return parser.parse_args()


def bootstrap_indices(length: int, block_length: int, reps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.empty((reps, length), dtype=int)
    for rep in range(reps):
        cursor = 0
        chosen: list[int] = []
        while cursor < length:
            start = int(rng.integers(0, max(length - block_length + 1, 1)))
            block = list(range(start, min(start + block_length, length)))
            chosen.extend(block)
            cursor += len(block)
        indices[rep] = np.asarray(chosen[:length], dtype=int)
    return indices


def proxy_mcs(loss_frame: pd.DataFrame, block_length: int, reps: int) -> dict[str, dict[str, Any]]:
    mean_losses = loss_frame.mean(axis=0)
    best_formula = str(mean_losses.idxmin())
    indices = bootstrap_indices(len(loss_frame), block_length, reps)
    payload: dict[str, dict[str, Any]] = {}
    for formula in loss_frame.columns:
        gap = loss_frame[formula].to_numpy(dtype=float) - loss_frame[best_formula].to_numpy(dtype=float)
        observed_gap = float(np.mean(gap))
        boot_gaps = np.mean(gap[indices], axis=1)
        ci_low = float(np.quantile(boot_gaps, 0.05))
        ci_high = float(np.quantile(boot_gaps, 0.95))
        payload[str(formula)] = {
            "observed_gap": observed_gap,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "in_mcs": bool(ci_low <= 0.0),
        }
    payload[best_formula]["is_sample_best"] = True
    return payload


def build_rows(shortlist_size: int, block_length: int, reps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for universe in ("liquid500", "liquid1000"):
        raw_runs, candidates, valid_frame, valid_target, _, _ = load_universe_inputs(universe)
        cache = FormulaEvaluationCache()
        scale_stats = load_universe_scale_stats(universe, candidates, cache)
        temporal_selector = TemporalRobustSelector(_selector_config(BASE_CONFIG, scale_stats), evaluation_cache=cache)
        consensus_selector = CrossSeedConsensusSelector(temporal_selector=temporal_selector, config=_consensus_config(BASE_CONFIG))
        context = f"{universe}:mcs"
        seed_runs = derive_seed_runs(raw_runs, valid_frame, valid_target, temporal_selector, evaluation_context=context)
        outcome = consensus_selector.select(seed_runs, valid_frame, valid_target, base_candidates=candidates, evaluation_context=context)
        shortlist = [record.formula for record in outcome.ranked_records[:shortlist_size]]
        canonical = outcome.selected_formulas[0] if outcome.selected_formulas else ""
        returns_map: dict[str, pd.Series] = {}
        for formula in shortlist:
            _, returns = run_walk_forward(universe, formula)
            returns_map[formula] = pd.Series(returns, copy=True)
        returns_frame = pd.concat(returns_map, axis=1).dropna(how="any")
        loss_frame = -returns_frame
        mcs_payload = proxy_mcs(loss_frame, block_length, reps)
        mcs_members = [formula for formula, stats in mcs_payload.items() if bool(stats.get("in_mcs"))]
        summary_rows.append(
            {
                "universe": universe,
                "shortlist_size": len(shortlist),
                "mcs_set_size": len(mcs_members),
                "canonical_signal": canonical,
                "canonical_in_mcs": canonical in mcs_members,
                "sample_best_signal": str(loss_frame.mean(axis=0).idxmin()),
                "sample_best_in_mcs": True,
            }
        )
        for formula in shortlist:
            stats = mcs_payload[formula]
            detail_rows.append(
                {
                    "universe": universe,
                    "formula": formula,
                    "is_canonical": formula == canonical,
                    "is_sample_best": bool(stats.get("is_sample_best", False)),
                    "in_mcs": bool(stats["in_mcs"]),
                    "observed_gap": round(float(stats["observed_gap"]), 8),
                    "ci_low": round(float(stats["ci_low"]), 8),
                    "ci_high": round(float(stats["ci_high"]), 8),
                }
            )
    return summary_rows, detail_rows


def build_markdown(summary_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Finance MCS Shortlist (Proxy)",
        "",
        "## Summary",
        "",
        "| Universe | Shortlist Size | MCS Set Size | Canonical Signal | Canonical in MCS | Sample Best |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['universe']} | {row['shortlist_size']} | {row['mcs_set_size']} | {row['canonical_signal']} | {row['canonical_in_mcs']} | {row['sample_best_signal']} |"
        )
    lines.extend([
        "",
        "## Detail",
        "",
        "| Universe | Formula | Canonical? | Sample Best? | In MCS | Gap | CI Low | CI High |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    for row in detail_rows:
        lines.append(
            f"| {row['universe']} | {row['formula']} | {row['is_canonical']} | {row['is_sample_best']} | {row['in_mcs']} | {row['observed_gap']:.6f} | {row['ci_low']:.6f} | {row['ci_high']:.6f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_rows, detail_rows = build_rows(args.shortlist_size, args.block_length, args.bootstrap_reps)
    summary_frame = pd.DataFrame(summary_rows)
    detail_frame = pd.DataFrame(detail_rows)
    csv_path = args.output_root / "finance_mcs_shortlist.csv"
    json_path = args.output_root / "finance_mcs_shortlist.json"
    md_path = args.output_root / "finance_mcs_shortlist.md"
    detail_csv = args.output_root / "finance_mcs_shortlist_detail.csv"
    summary_frame.to_csv(csv_path, index=False)
    detail_frame.to_csv(detail_csv, index=False)
    json_path.write_text(json.dumps({"summary": summary_rows, "detail": detail_rows}, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(summary_rows, detail_rows) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "detail_csv": str(detail_csv), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
