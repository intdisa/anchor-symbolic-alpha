#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_guided_symbolic_alpha.runtime import ensure_preflight, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic temporal-shift selector benchmark.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-name", type=str, default="synthetic_selector_benchmark")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dates", type=int, default=72)
    parser.add_argument("--entities", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight = ensure_preflight("core")

    from experiments.common import ensure_output_dirs, write_json
    from knowledge_guided_symbolic_alpha.benchmarks import (
        generate_synthetic_temporal_shift_panel,
        naive_rank_ic_selection,
    )
    from knowledge_guided_symbolic_alpha.selection import RobustTemporalSelector

    benchmark = generate_synthetic_temporal_shift_panel(
        num_dates=args.dates,
        num_entities=args.entities,
        seed=args.seed,
    )
    output_dirs = ensure_output_dirs(args.output_root, args.run_name)
    manifest_path = write_run_manifest(
        output_dirs,
        script_name="scripts/run_synthetic_selector_benchmark.py",
        profile="core",
        preflight=preflight.to_dict(),
        config_paths={},
        dataset_name="synthetic_temporal_shift",
        subset=f"dates_{args.dates}_entities_{args.entities}",
        seed=args.seed,
    )

    selector = RobustTemporalSelector()
    selector_outcome = selector.select(benchmark.candidate_formulas, benchmark.frame, benchmark.target)
    naive_formula = naive_rank_ic_selection(benchmark.candidate_formulas, benchmark.frame, benchmark.target)
    payload = {
        "dataset": "synthetic_temporal_shift",
        "seed": args.seed,
        "date_count": args.dates,
        "entity_count": args.entities,
        "true_formula": benchmark.true_formula,
        "spurious_formula": benchmark.spurious_formula,
        "naive_formula": naive_formula,
        "selector_records": selector_outcome.selected_formulas,
        "selector_fallback_used": selector_outcome.fallback_used,
        "selector_ranked_records": selector_outcome.records,
        "manifest": str(manifest_path),
    }
    report_path = output_dirs["reports"] / "synthetic_selector_benchmark.json"
    write_json(report_path, payload)
    print(f"true_formula={benchmark.true_formula}")
    print(f"spurious_formula={benchmark.spurious_formula}")
    print(f"naive_formula={naive_formula}")
    print(f"selector_records={selector_outcome.selected_formulas}")
    print(f"manifest={manifest_path}")
    print(f"benchmark_report={report_path}")


if __name__ == "__main__":
    main()
