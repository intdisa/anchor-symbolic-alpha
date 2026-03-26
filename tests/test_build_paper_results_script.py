from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _canonical_payload(sharpe: float, flow_sharpe: float, source_name: str, full_report: str) -> dict:
    return {
        "dataset": "us_equities",
        "subset": source_name,
        "partition_mode": "skill_hierarchy",
        "episodes": 5,
        "seeds": [7, 17, 27, 37, 47],
        "variants": ["full", "quality_solvency_only", "short_horizon_flow_only"],
        "canonical_result_kind": "cross_seed_consensus",
        "canonical_by_variant": {
            "full": {
                "variant": "full",
                "result_kind": "cross_seed_consensus",
                "selector_records": ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"],
                "evaluation_formula_source": "selector_records",
                "walk_forward_metrics": {
                    "sharpe": sharpe,
                    "annual_return": 0.05,
                    "max_drawdown": -0.25,
                    "turnover": 0.01,
                    "mean_test_rank_ic": 0.012,
                },
                "seed_support": {
                    "seed_count": 5,
                    "min_seed_support": 3,
                    "candidate_pool_size": 6,
                    "fallback_used": False,
                    "selector_fallback_used": False,
                },
                "raw_seed_diagnostics": {
                    "sharpe": {"mean": sharpe - 0.1, "std": 0.2, "min": 0.0, "max": sharpe},
                },
                "support_adjusted_ranked_records": [
                    {"formula": "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD", "support_adjusted_score": 0.08}
                ],
            },
            "quality_solvency_only": {
                "variant": "quality_solvency_only",
                "result_kind": "cross_seed_consensus",
                "selector_records": ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"],
                "walk_forward_metrics": {"sharpe": sharpe},
            },
            "short_horizon_flow_only": {
                "variant": "short_horizon_flow_only",
                "result_kind": "cross_seed_consensus",
                "selector_records": ["RET_1 NEG"],
                "walk_forward_metrics": {"sharpe": flow_sharpe},
            },
        },
        "canonical_comparisons": [
            {
                "left_variant": "full",
                "right_variant": "quality_solvency_only",
                "result_kind": "cross_seed_consensus",
                "left_selector_records": ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"],
                "right_selector_records": ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"],
                "metric_deltas": {"sharpe": 0.0, "turnover": 0.0},
            },
            {
                "left_variant": "full",
                "right_variant": "short_horizon_flow_only",
                "result_kind": "cross_seed_consensus",
                "left_selector_records": ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"],
                "right_selector_records": ["RET_1 NEG"],
                "metric_deltas": {"sharpe": sharpe - flow_sharpe, "turnover": -0.75},
            },
        ],
        "full_report": full_report,
        "manifest": "outputs/runs/example/run_manifest.json",
    }


def _raw_multiseed_payload() -> dict:
    return {
        "runs_by_variant": {
            "full": [
                {"seed": 7, "selector_records": ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"], "walk_forward_metrics": {"sharpe": 0.56, "mean_test_rank_ic": 0.012, "turnover": 0.01}},
                {"seed": 17, "selector_records": ["CASH_RATIO_Q RANK"], "walk_forward_metrics": {"sharpe": 0.11, "mean_test_rank_ic": 0.003, "turnover": 0.02}},
            ]
        }
    }


def _benchmark_summary() -> dict:
    return {
        "benchmark_name": "synthetic_selector_benchmark_suite",
        "leaderboard": [
            {
                "baseline": "support_adjusted_cross_seed_consensus",
                "selection_accuracy": 1.0,
                "misselection_rate": 0.0,
                "oracle_regret_rank_ic": 0.0,
                "mean_test_rank_ic": 0.21,
                "mean_test_sharpe": 1.12,
                "selected_formula_stability": 1.0,
            }
        ],
        "task_results": [
            {
                "task_id": "transient_spuriosity",
                "scenario": "transient_spuriosity",
                "true_formula": "X0 X1 ADD",
                "baselines": {
                    "support_adjusted_cross_seed_consensus": {
                        "selected_formula": "X0 X1 ADD",
                        "diagnostics": {
                            "support_adjusted_ranked_records": [
                                {"formula": "X0 X1 ADD", "support_adjusted_score": 0.11}
                            ]
                        },
                    }
                },
            }
        ],
    }


def _ablation_payload() -> dict:
    return {
        "results_by_partition_mode": {
            "skill_hierarchy": [
                {
                    "variant": "full",
                    "selector_records": ["CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"],
                    "accepted_episodes": 1,
                    "final_pool_size": 1,
                    "walk_forward_metrics": {"sharpe": 0.56, "mean_test_rank_ic": 0.012, "turnover": 0.01},
                }
            ]
        }
    }


def test_build_paper_results_script(tmp_path: Path) -> None:
    raw_report = tmp_path / "raw_multiseed.json"
    raw_report.write_text(json.dumps(_raw_multiseed_payload()), encoding="utf-8")
    liquid500 = tmp_path / "liquid500.json"
    liquid1000 = tmp_path / "liquid1000.json"
    liquid500.write_text(json.dumps(_canonical_payload(0.56, 0.13, "liquid500", str(raw_report))), encoding="utf-8")
    liquid1000.write_text(json.dumps(_canonical_payload(0.73, 0.22, "liquid1000", str(raw_report))), encoding="utf-8")
    benchmark_summary = tmp_path / "synthetic_benchmark.json"
    benchmark_summary.write_text(json.dumps(_benchmark_summary()), encoding="utf-8")
    ablation = tmp_path / "ablation.json"
    ablation.write_text(json.dumps(_ablation_payload()), encoding="utf-8")

    output_root = tmp_path / "reports"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_paper_results.py",
            "--liquid500-canonical",
            str(liquid500),
            "--liquid1000-canonical",
            str(liquid1000),
            "--liquid500-ablation",
            str(ablation),
            "--synthetic-benchmark-summary",
            str(benchmark_summary),
            "--output-root",
            str(output_root),
            "--stem",
            "paper_results_test",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    json_report = output_root / "paper_results_test.json"
    markdown_report = output_root / "paper_results_test.md"
    main_csv = output_root / "paper_results_test_main_table.csv"
    benchmark_csv = output_root / "paper_results_test_benchmark_table.csv"
    ablation_csv = output_root / "paper_results_test_ablation_table.csv"
    seed_dispersion_csv = output_root / "paper_results_test_seed_dispersion.csv"
    case_studies = output_root / "paper_results_test_selector_case_studies.json"
    claims_json = output_root / "paper_results_test_claims.json"
    draft_outline = output_root / "paper_results_test_draft_outline.md"

    assert payload["shared_full_formula"] is True
    assert json_report.exists()
    assert markdown_report.exists()
    assert main_csv.exists()
    assert benchmark_csv.exists()
    assert ablation_csv.exists()
    assert seed_dispersion_csv.exists()
    assert case_studies.exists()
    assert claims_json.exists()
    assert draft_outline.exists()
    report_payload = json.loads(json_report.read_text(encoding="utf-8"))
    claims_payload = json.loads(claims_json.read_text(encoding="utf-8"))
    assert report_payload["finance"]["universes"][0]["full"]["formula"] == "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"
    assert report_payload["benchmark_sections"][0]["leaderboard"][0]["baseline"] == "support_adjusted_cross_seed_consensus"
    assert claims_payload["paper_object"] == "cross_seed_robust_symbolic_selection"
    assert claims_payload["main_claims"][0]["id"] == "finance_consensus"
    assert "Benchmark Leaderboards" in markdown_report.read_text(encoding="utf-8")
    assert "Suggested Section Skeleton" in draft_outline.read_text(encoding="utf-8")
