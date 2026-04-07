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

DEFAULT_SUMMARY = Path("outputs/runs/selector_suite_pareto_r1/reports/all_selector_benchmark_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark front diagnostics from selector benchmark outputs.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def _consensus_diag(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
    return {
        "first_front_size": diagnostics.get("first_front_size"),
        "first_front_share": diagnostics.get("first_front_share"),
        "selected_crowding_distance": diagnostics.get("selected_crowding_distance"),
        "used_near_neighbor_tiebreak": diagnostics.get("used_near_neighbor_tiebreak"),
    }


def build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    bucket: dict[str, list[dict[str, Any]]] = {}
    for task in payload.get("task_results", []):
        for baseline, result in task.get("baselines", {}).items():
            diag = _consensus_diag(result)
            bucket.setdefault(baseline, []).append(diag)
    rows = []
    for baseline, items in sorted(bucket.items()):
        frame = pd.DataFrame(items)
        if frame.empty:
            continue
        rows.append(
            {
                "baseline": baseline,
                "mean_first_front_size": round(float(frame["first_front_size"].dropna().mean()), 6) if frame["first_front_size"].notna().any() else None,
                "mean_first_front_share": round(float(frame["first_front_share"].dropna().mean()), 6) if frame["first_front_share"].notna().any() else None,
                "mean_selected_crowding_distance": round(float(frame["selected_crowding_distance"].dropna().mean()), 6) if frame["selected_crowding_distance"].notna().any() else None,
                "near_neighbor_tiebreak_rate": round(
                    float(frame["used_near_neighbor_tiebreak"].astype("boolean").fillna(False).astype(float).mean()),
                    6,
                ),
            }
        )
    return rows


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Selector Benchmark Front Diagnostics",
        "",
        "| Baseline | Mean First Front Size | Mean First Front Share | Mean Selected Crowding | Tie-break Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        crowding = "NA" if row["mean_selected_crowding_distance"] is None else f"{float(row['mean_selected_crowding_distance']):.4f}"
        front_size = "NA" if row["mean_first_front_size"] is None else f"{float(row['mean_first_front_size']):.4f}"
        front_share = "NA" if row["mean_first_front_share"] is None else f"{float(row['mean_first_front_share']):.4f}"
        lines.append(
            f"| {row['baseline']} | {front_size} | {front_share} | {crowding} | {float(row['near_neighbor_tiebreak_rate']):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    rows = build_rows(payload)
    frame = pd.DataFrame(rows)
    csv_path = args.output_root / "selector_benchmark_front_diagnostics.csv"
    json_path = args.output_root / "selector_benchmark_front_diagnostics.json"
    md_path = args.output_root / "selector_benchmark_front_diagnostics.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(rows) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
