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


DEFAULT_SUMMARY = Path("outputs/runs/selector_suite_honest_r1/reports/all_selector_benchmark_summary.json")
IGNORED_BASELINES = {"support_adjusted_cross_seed_consensus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark Pareto plot data from selector benchmark outputs.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def token_overlap(left: str, right: str) -> float:
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    return float(len(left_set & right_set) / len(left_set | right_set))


def on_pareto_frontier(frame: pd.DataFrame) -> pd.Series:
    flags = []
    for row in frame.itertuples(index=False):
        dominated = False
        for other in frame.itertuples(index=False):
            if other.baseline == row.baseline:
                continue
            better_or_equal = other.average_accuracy >= row.average_accuracy and other.selected_formula_stability >= row.selected_formula_stability
            strictly_better = other.average_accuracy > row.average_accuracy or other.selected_formula_stability > row.selected_formula_stability
            if better_or_equal and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=frame.index)


def build_rows(payload: dict[str, Any]) -> pd.DataFrame:
    stress_lookup = {row["baseline"]: row for row in payload.get("stress_summary", [])}
    proxy_counts: dict[str, list[bool]] = {}
    for task in payload.get("task_results", []):
        true_formula = str(task["true_formula"])
        for baseline, result in task["baselines"].items():
            if baseline in IGNORED_BASELINES:
                continue
            selected_formula = str(result.get("selected_formula") or "")
            near_miss = selected_formula and selected_formula != true_formula and token_overlap(selected_formula, true_formula) >= 0.5
            proxy_counts.setdefault(baseline, []).append(bool(near_miss))
    rows = []
    for row in payload.get("leaderboard", []):
        if row["baseline"] in IGNORED_BASELINES:
            continue
        stress = stress_lookup.get(row["baseline"], {})
        near_miss_rate = None
        if proxy_counts.get(row["baseline"]):
            near_miss_rate = float(sum(proxy_counts[row["baseline"]]) / len(proxy_counts[row["baseline"]]))
        rows.append(
            {
                "baseline": row["baseline"],
                "average_accuracy": float(row.get("selection_accuracy") or 0.0),
                "selected_formula_stability": float(row.get("selected_formula_stability") or 0.0),
                "worst_case_accuracy": float(stress.get("worst_case_accuracy") or 0.0),
                "failure_boundary_accuracy": float(stress.get("failure_boundary_accuracy") or 0.0),
                "mean_test_rank_ic": float(row.get("mean_test_rank_ic") or 0.0),
                "mean_test_sharpe": float(row.get("mean_test_sharpe") or 0.0),
                "near_neighbor_misselection_rate": near_miss_rate,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["average_accuracy", "selected_formula_stability"], ascending=[False, False])
    frame["pareto_frontier"] = on_pareto_frontier(frame)
    return frame


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Selector Benchmark Pareto Summary",
        "",
        "| Baseline | Avg Accuracy | Stability | Worst Case | Failure Boundary | Pareto Frontier | Near-Neighbor MisSel |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        near = row["near_neighbor_misselection_rate"]
        near_text = "NA" if near is None else f"{float(near):.4f}"
        lines.append(
            f"| {row['baseline']} | {float(row['average_accuracy']):.4f} | {float(row['selected_formula_stability']):.4f} | "
            f"{float(row['worst_case_accuracy']):.4f} | {float(row['failure_boundary_accuracy']):.4f} | "
            f"{row['pareto_frontier']} | {near_text} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    frame = build_rows(payload)
    csv_path = args.output_root / "selector_benchmark_pareto.csv"
    json_path = args.output_root / "selector_benchmark_pareto.json"
    md_path = args.output_root / "selector_benchmark_pareto.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(frame.to_dict(orient="records")) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
