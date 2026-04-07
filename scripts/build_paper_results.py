#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_LIQUID500_CANONICAL = Path(
    "outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json"
)
DEFAULT_LIQUID1000_CANONICAL = Path(
    "outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json"
)
DEFAULT_LIQUID500_ABLATION = Path(
    "outputs/runs/liquid500_ablation_e5_r2__ablation/reports/us_equities_ablation.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing summaries from canonical results.")
    parser.add_argument("--liquid500-canonical", type=Path, default=DEFAULT_LIQUID500_CANONICAL)
    parser.add_argument("--liquid1000-canonical", type=Path, default=DEFAULT_LIQUID1000_CANONICAL)
    parser.add_argument("--liquid500-ablation", type=Path, default=DEFAULT_LIQUID500_ABLATION)
    parser.add_argument("--synthetic-benchmark-summary", type=Path, default=None)
    parser.add_argument("--public-benchmark-summary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stem", type=str, default="us_equities_paper_results")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return load_json(path)


def resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else Path.cwd() / path


def round4(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def stringify_formula(records: list[str]) -> str:
    return " | ".join(str(item) for item in records if str(item)) or "NONE"


def comparison_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    comparisons = payload.get("canonical_comparisons", [])
    return {(item["left_variant"], item["right_variant"]): item for item in comparisons}


def build_universe_summary(universe: str, source_path: Path) -> dict[str, Any]:
    payload = load_json(source_path)
    canonical = payload["canonical_by_variant"]
    full = canonical["full"]
    quality = canonical.get("quality_solvency_only", {})
    flow = canonical.get("short_horizon_flow_only", {})
    comparisons = comparison_map(payload)
    full_vs_flow = comparisons.get(("full", "short_horizon_flow_only"), {})
    full_metrics = full.get("walk_forward_metrics", {})
    raw = full.get("raw_seed_diagnostics", {})
    seed_support = full.get("seed_support", {})
    full_report = resolve_path(payload.get("full_report"))
    return {
        "universe": universe,
        "source": str(source_path),
        "subset": payload.get("subset"),
        "seeds": list(payload.get("seeds", [])),
        "full_report": str(full_report) if full_report else None,
        "full": {
            "formula": stringify_formula(full.get("selector_records", [])),
            "sharpe": round4(full_metrics.get("sharpe")),
            "annual_return": round4(full_metrics.get("annual_return")),
            "max_drawdown": round4(full_metrics.get("max_drawdown")),
            "turnover": round4(full_metrics.get("turnover")),
            "mean_test_rank_ic": round4(full_metrics.get("mean_test_rank_ic")),
            "seed_count": int(seed_support.get("seed_count", 0)),
            "candidate_pool_size": int(seed_support.get("candidate_pool_size", 0)),
            "raw_seed_sharpe_mean": round4(raw.get("sharpe", {}).get("mean")),
            "raw_seed_sharpe_std": round4(raw.get("sharpe", {}).get("std")),
        },
        "quality_solvency_only": {
            "formula": stringify_formula(quality.get("selector_records", [])),
            "sharpe": round4(quality.get("walk_forward_metrics", {}).get("sharpe")),
        },
        "short_horizon_flow_only": {
            "formula": stringify_formula(flow.get("selector_records", [])),
            "sharpe": round4(flow.get("walk_forward_metrics", {}).get("sharpe")),
        },
        "controls": {
            "full_equals_quality_formula": full.get("selector_records", []) == quality.get("selector_records", []),
            "full_minus_flow_sharpe": round4(full_vs_flow.get("metric_deltas", {}).get("sharpe")),
            "full_minus_flow_turnover": round4(full_vs_flow.get("metric_deltas", {}).get("turnover")),
        },
        "selector_case_study": {
            "formula": stringify_formula(full.get("selector_records", [])),
            "support_adjusted_ranked_records": full.get("support_adjusted_ranked_records", []),
        },
    }


def build_finance_payload(liquid500_path: Path, liquid1000_path: Path) -> dict[str, Any]:
    universes = [
        build_universe_summary("liquid500", liquid500_path),
        build_universe_summary("liquid1000", liquid1000_path),
    ]
    shared_formula = len({item["full"]["formula"] for item in universes}) == 1
    return {
        "universes": universes,
        "cross_universe_summary": {
            "shared_full_formula": shared_formula,
            "full_formula": universes[0]["full"]["formula"] if shared_formula else None,
            "liquid500_sharpe": universes[0]["full"]["sharpe"],
            "liquid1000_sharpe": universes[1]["full"]["sharpe"],
        },
        "claims": [
            "Cross-seed consensus is the canonical result object.",
            "The anchor generator repeatedly recovers the cash-quality formula family.",
            "Full and quality_solvency_only remain identical on the current mainline runs.",
            "Short-horizon flow remains a control branch rather than a mainline winner.",
        ],
    }


def build_benchmark_section(label: str, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    leaderboard = payload.get("leaderboard", [])
    task_results = payload.get("task_results", [])
    return {
        "label": label,
        "source": payload.get("manifest") or label,
        "leaderboard": leaderboard,
        "task_results": task_results,
    }


def build_ablation_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    rows: list[dict[str, Any]] = []
    for partition_mode, records in payload.get("results_by_partition_mode", {}).items():
        for record in records:
            metrics = record.get("walk_forward_metrics", {})
            rows.append(
                {
                    "partition_mode": partition_mode,
                    "variant": record.get("variant"),
                    "formula": stringify_formula(record.get("selector_records", [])),
                    "sharpe": round4(metrics.get("sharpe")),
                    "mean_test_rank_ic": round4(metrics.get("mean_test_rank_ic")),
                    "turnover": round4(metrics.get("turnover")),
                    "accepted_episodes": int(record.get("accepted_episodes", 0)),
                    "final_pool_size": int(record.get("final_pool_size", 0)),
                }
            )
    return rows


def build_seed_dispersion_rows(universes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for universe in universes:
        full_report_path = resolve_path(universe.get("full_report"))
        if full_report_path is None or not full_report_path.exists():
            continue
        payload = load_json(full_report_path)
        runs = payload.get("runs_by_variant", {}).get("full", [])
        for run in runs:
            metrics = run.get("walk_forward_metrics", {})
            rows.append(
                {
                    "universe": universe["universe"],
                    "seed": int(run.get("seed", 0)),
                    "selected_formula": stringify_formula(run.get("selector_records", [])),
                    "sharpe": round4(metrics.get("sharpe")),
                    "mean_test_rank_ic": round4(metrics.get("mean_test_rank_ic")),
                    "turnover": round4(metrics.get("turnover")),
                }
            )
    return rows


def build_selector_case_studies(finance_payload: dict[str, Any], benchmark_sections: list[dict[str, Any]]) -> dict[str, Any]:
    case_studies: dict[str, Any] = {"finance": {}, "benchmarks": {}}
    for universe in finance_payload["universes"]:
        case_studies["finance"][universe["universe"]] = universe["selector_case_study"]
    for section in benchmark_sections:
        if section is None:
            continue
        first_task = None
        for task in section.get("task_results", []):
            baselines = task.get("baselines", {})
            baseline = (
                baselines.get("legacy_linear_selector")
                or baselines.get("support_adjusted_cross_seed_consensus")
                or baselines.get("pareto_discrete_legacy")
                or baselines.get("pareto_cross_seed_consensus")
            )
            if baseline:
                first_task = {
                    "task_id": task.get("task_id"),
                    "scenario": task.get("scenario"),
                    "true_formula": task.get("true_formula"),
                    "selected_formula": baseline.get("selected_formula"),
                    "support_adjusted_ranked_records": baseline.get("diagnostics", {}).get("support_adjusted_ranked_records", []),
                }
                break
        if first_task is not None:
            case_studies["benchmarks"][section["label"]] = first_task
    return case_studies


def benchmark_section_by_label(sections: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    for section in sections:
        if section.get("label") == label:
            return section
    return None


def benchmark_leader(section: dict[str, Any] | None, baseline: str) -> dict[str, Any] | None:
    if section is None:
        return None
    for row in section.get("leaderboard", []):
        if row.get("baseline") == baseline:
            return row
    return None


def build_claims_payload(payload: dict[str, Any]) -> dict[str, Any]:
    finance = payload["finance"]
    benchmark_sections = payload["benchmark_sections"]
    synthetic = benchmark_section_by_label(benchmark_sections, "synthetic")
    public = benchmark_section_by_label(benchmark_sections, "public_symbolic")
    liquid500 = finance["universes"][0]
    liquid1000 = finance["universes"][1]
    synthetic_support = (
        benchmark_leader(synthetic, "legacy_linear_selector")
        or benchmark_leader(synthetic, "support_adjusted_cross_seed_consensus")
        or benchmark_leader(synthetic, "pareto_discrete_legacy")
        or benchmark_leader(synthetic, "pareto_cross_seed_consensus")
    )
    synthetic_naive = benchmark_leader(synthetic, "naive_rank_ic")
    public_support = (
        benchmark_leader(public, "legacy_linear_selector")
        or benchmark_leader(public, "support_adjusted_cross_seed_consensus")
        or benchmark_leader(public, "pareto_discrete_legacy")
        or benchmark_leader(public, "pareto_cross_seed_consensus")
    )
    public_naive = benchmark_leader(public, "naive_rank_ic")
    public_mean = benchmark_leader(public, "cross_seed_mean_score_consensus")
    public_single = benchmark_leader(public, "single_seed_temporal_selector")

    claims: list[dict[str, Any]] = [
        {
            "id": "finance_consensus",
            "claim": "Cross-seed support-adjusted consensus recovers the same cash-quality formula on both liquid500 and liquid1000.",
            "evidence": {
                "formula": finance["cross_universe_summary"]["full_formula"],
                "liquid500_sharpe": liquid500["full"]["sharpe"],
                "liquid1000_sharpe": liquid1000["full"]["sharpe"],
                "shared_full_formula": finance["cross_universe_summary"]["shared_full_formula"],
            },
        },
        {
            "id": "control_structure",
            "claim": "Extra skill families are not part of the current main contribution because full and quality_solvency_only remain identical, while short_horizon_flow stays weaker.",
            "evidence": {
                "liquid500_full_equals_quality": liquid500["controls"]["full_equals_quality_formula"],
                "liquid1000_full_equals_quality": liquid1000["controls"]["full_equals_quality_formula"],
                "liquid500_full_minus_flow_sharpe": liquid500["controls"]["full_minus_flow_sharpe"],
                "liquid1000_full_minus_flow_sharpe": liquid1000["controls"]["full_minus_flow_sharpe"],
            },
        },
    ]
    if synthetic_support is not None and synthetic_naive is not None:
        claims.append(
            {
                "id": "synthetic_selector_gain",
                "claim": "On synthetic temporal-shift benchmarks, the Pareto cross-seed selector improves the accuracy-stability trade-off relative to naive validation ranking.",
                "evidence": {
                    "support_selection_accuracy": synthetic_support.get("selection_accuracy"),
                    "naive_selection_accuracy": synthetic_naive.get("selection_accuracy"),
                    "support_oracle_regret": synthetic_support.get("oracle_regret_rank_ic"),
                    "naive_oracle_regret": synthetic_naive.get("oracle_regret_rank_ic"),
                },
            }
        )
    if public_support is not None:
        claims.append(
            {
                "id": "public_selector_gain",
                "claim": "On public symbolic benchmarks, the Pareto cross-seed selector remains on the accuracy-stability frontier against single-seed, naive, and mean-score baselines.",
                "evidence": {
                    "support_selection_accuracy": public_support.get("selection_accuracy"),
                    "naive_selection_accuracy": public_naive.get("selection_accuracy") if public_naive else None,
                    "single_seed_selection_accuracy": public_single.get("selection_accuracy") if public_single else None,
                    "mean_score_selection_accuracy": public_mean.get("selection_accuracy") if public_mean else None,
                },
            }
        )

    limitations = [
        "The public benchmark edge is now clear on the suite headline, but it still depends materially on the seed-shift product task.",
        "The theory layer is lightweight and currently supports the method narrative rather than serving as a full theorem section.",
        "The real-world application evidence remains concentrated on two U.S. equities universes.",
    ]
    method_components = [
        {
            "name": "Anchor Generator",
            "role": "Produces a candidate pool of symbolic formulas; it is not the paper's main novelty object.",
        },
        {
            "name": "TemporalRobustSelector",
            "role": "Ranks candidate formulas with temporal Pareto dominance, crowding distance, admissibility gates, and near-neighbor tie-breaks.",
        },
        {
            "name": "CrossSeedConsensusSelector",
            "role": "Aggregates seed-level support and temporal diagnostics through cross-seed Pareto dominance and emits the canonical formula.",
        },
    ]
    return {
        "paper_object": "cross_seed_pareto_signal_selection",
        "title_candidates": [
            "Cross-Seed Pareto Signal Selection under Temporal Shift",
            "Selecting Stable Symbolic Signals from Rashomon Candidate Pools",
            "Pareto-Ranked Symbolic Signal Selection with Temporal and Seed Variability",
        ],
        "main_claims": claims,
        "method_components": method_components,
        "limitations": limitations,
    }


def build_outline_markdown(payload: dict[str, Any], claims_payload: dict[str, Any]) -> str:
    finance = payload["finance"]
    liquid500 = finance["universes"][0]
    liquid1000 = finance["universes"][1]
    lines = [
        "# Draft Outline",
        "",
        "## Working Title",
        "",
        f"- {claims_payload['title_candidates'][0]}",
        "",
        "## Core Thesis",
        "",
        "- One anchor generator produces candidate symbolic formulas.",
        "- A temporal selector ranks formulas within a seed under shift using Pareto dominance and crowding distance.",
        "- A cross-seed Pareto selector emits the canonical formula.",
        "",
        "## Main Claims",
        "",
    ]
    for item in claims_payload["main_claims"]:
        lines.extend([f"- {item['claim']}", f"  - evidence id: `{item['id']}`"])
    lines.extend(
        [
            "",
            "## Suggested Section Skeleton",
            "",
            "1. Introduction",
            "   - motivate symbolic selection under temporal and seed variability",
            "   - state that multi-agent synergy is not the final contribution",
            "2. Method",
            "   - define candidate pool, temporal Pareto dominance, and cross-seed Pareto consensus",
            "   - present TemporalRobustSelector and CrossSeedConsensusSelector",
            "3. Benchmarks",
            "   - synthetic temporal-shift suite",
            "   - public symbolic suite with seed-shift control tasks",
            "   - U.S. equities application",
            "4. Main Results",
            f"   - liquid500 canonical formula `{liquid500['full']['formula']}` with Sharpe `{_fmt(liquid500['full']['sharpe'])}`",
            f"   - liquid1000 canonical formula `{liquid1000['full']['formula']}` with Sharpe `{_fmt(liquid1000['full']['sharpe'])}`",
            "   - benchmark headline tables emphasizing the accuracy-stability frontier",
            "5. Controls and Ablations",
            "   - full vs quality_solvency_only equivalence",
            "   - short_horizon_flow as a control branch",
            "   - no-support / no-temporal / no-tie-break ablations",
            "6. Discussion",
            "   - where the method still depends on benchmark design",
            "   - limitations of current public-benchmark coverage",
            "",
            "## Writing Notes",
            "",
        ]
    )
    for item in claims_payload["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    finance_payload = build_finance_payload(args.liquid500_canonical, args.liquid1000_canonical)
    synthetic_payload = build_benchmark_section("synthetic", load_optional_json(args.synthetic_benchmark_summary))
    public_payload = build_benchmark_section("public_symbolic", load_optional_json(args.public_benchmark_summary))
    ablation_payload = load_optional_json(args.liquid500_ablation)
    benchmark_sections = [section for section in [synthetic_payload, public_payload] if section is not None]
    claims_payload = build_claims_payload(
        {
            "finance": finance_payload,
            "benchmark_sections": benchmark_sections,
        }
    )
    return {
        "result_kind": "paper_mainline_summary",
        "finance": finance_payload,
        "benchmark_sections": benchmark_sections,
        "ablation_rows": build_ablation_rows(ablation_payload),
        "seed_dispersion_rows": build_seed_dispersion_rows(finance_payload["universes"]),
        "selector_case_studies": build_selector_case_studies(finance_payload, benchmark_sections),
        "claims": claims_payload,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(payload: dict[str, Any]) -> str:
    finance = payload["finance"]
    main_rows = []
    control_rows = []
    for item in finance["universes"]:
        full = item["full"]
        main_rows.append(
            [
                item["universe"],
                full["formula"],
                _fmt(full["sharpe"]),
                _fmt(full["annual_return"]),
                _fmt(full["mean_test_rank_ic"]),
                _fmt(full["turnover"]),
                _fmt(full["raw_seed_sharpe_mean"]),
                _fmt(full["raw_seed_sharpe_std"]),
            ]
        )
        control_rows.append(
            [
                item["universe"],
                str(item["controls"]["full_equals_quality_formula"]),
                item["short_horizon_flow_only"]["formula"],
                _fmt(item["short_horizon_flow_only"]["sharpe"]),
                _fmt(item["controls"]["full_minus_flow_sharpe"]),
                _fmt(item["controls"]["full_minus_flow_turnover"]),
            ]
        )
    lines = [
        "# Paper Results",
        "",
        "## Finance Main Table",
        "",
        markdown_table(
            ["Universe", "Formula", "Sharpe", "AnnRet", "RankIC", "Turnover", "RawMeanSharpe", "RawStdSharpe"],
            main_rows,
        ),
        "",
        "## Finance Control Table",
        "",
        markdown_table(
            ["Universe", "FullEqQuality", "FlowFormula", "FlowSharpe", "FullMinusFlowSharpe", "FullMinusFlowTurnover"],
            control_rows,
        ),
    ]
    if payload["benchmark_sections"]:
        lines.extend(["", "## Benchmark Leaderboards", ""])
        for section in payload["benchmark_sections"]:
            rows = [
                [
                    row["baseline"],
                    _fmt(row.get("selection_accuracy")),
                    _fmt(row.get("misselection_rate")),
                    _fmt(row.get("oracle_regret_rank_ic")),
                    _fmt(row.get("mean_test_rank_ic")),
                    _fmt(row.get("mean_test_sharpe")),
                    _fmt(row.get("selected_formula_stability")),
                ]
                for row in section.get("leaderboard", [])
            ]
            lines.extend(
                [
                    f"### {section['label']}",
                    "",
                    markdown_table(
                        ["Baseline", "SelAcc", "MisSel", "OracleRegret", "RankIC", "Sharpe", "Stability"],
                        rows,
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finance_main_rows(finance_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in finance_payload["universes"]:
        full = item["full"]
        rows.append(
            {
                "universe": item["universe"],
                "formula": full["formula"],
                "sharpe": full["sharpe"],
                "annual_return": full["annual_return"],
                "mean_test_rank_ic": full["mean_test_rank_ic"],
                "turnover": full["turnover"],
                "raw_seed_sharpe_mean": full["raw_seed_sharpe_mean"],
                "raw_seed_sharpe_std": full["raw_seed_sharpe_std"],
                "seed_count": full["seed_count"],
                "candidate_pool_size": full["candidate_pool_size"],
            }
        )
    return rows


def benchmark_rows(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in sections:
        for row in section.get("leaderboard", []):
            rows.append({"suite": section["label"], **row})
    return rows


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{args.stem}.json"
    md_path = output_root / f"{args.stem}.md"
    main_csv = output_root / f"{args.stem}_main_table.csv"
    benchmark_csv = output_root / f"{args.stem}_benchmark_table.csv"
    ablation_csv = output_root / f"{args.stem}_ablation_table.csv"
    seed_dispersion_csv = output_root / f"{args.stem}_seed_dispersion.csv"
    case_studies_json = output_root / f"{args.stem}_selector_case_studies.json"
    claims_json = output_root / f"{args.stem}_claims.json"
    outline_md = output_root / f"{args.stem}_draft_outline.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload) + "\n", encoding="utf-8")
    write_csv(main_csv, finance_main_rows(payload["finance"]))
    write_csv(benchmark_csv, benchmark_rows(payload["benchmark_sections"]))
    write_csv(ablation_csv, payload["ablation_rows"])
    write_csv(seed_dispersion_csv, payload["seed_dispersion_rows"])
    case_studies_json.write_text(json.dumps(payload["selector_case_studies"], ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    claims_json.write_text(json.dumps(payload["claims"], ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    outline_md.write_text(build_outline_markdown(payload, payload["claims"]) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "markdown_report": str(md_path),
                "main_csv": str(main_csv),
                "benchmark_csv": str(benchmark_csv),
                "ablation_csv": str(ablation_csv),
                "seed_dispersion_csv": str(seed_dispersion_csv),
                "selector_case_studies_json": str(case_studies_json),
                "claims_json": str(claims_json),
                "draft_outline_md": str(outline_md),
                "shared_full_formula": payload["finance"]["cross_universe_summary"]["shared_full_formula"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
