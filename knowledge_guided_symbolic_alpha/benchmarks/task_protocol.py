from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.panel_dispatch import evaluate_formula_metrics
from ..generation import FormulaCandidate
from ..selection import (
    CrossSeedConsensusConfig,
    CrossSeedConsensusSelector,
    CrossSeedSelectionRun,
    TemporalRobustSelector,
)
from ..selection.cross_seed_selector import mean_score_consensus_formula
from ..selection.robust_selector import TemporalSelectorRecord


@dataclass(frozen=True)
class SelectorBenchmarkTask:
    benchmark_name: str
    task_id: str
    scenario: str
    seed: int
    frame: pd.DataFrame
    target: pd.Series
    candidate_formulas: list[FormulaCandidate]
    true_formula: str
    selector_envs: tuple[str, ...] = ("env_a", "env_b", "env_c")
    test_envs: tuple[str, ...] = ("env_d",)
    environment_column: str = "environment"


@dataclass(frozen=True)
class SeedBaselineResult:
    seed: int
    selected_formula: str
    test_metrics: dict[str, float | None]
    oracle_regret_rank_ic: float | None
    selection_correct: bool


@dataclass(frozen=True)
class BenchmarkBaselineAggregate:
    baseline: str
    selected_formula: str | None
    selection_accuracy: float
    misselection_rate: float
    oracle_regret_rank_ic: float | None
    selected_formula_stability: float
    mean_test_rank_ic: float | None
    mean_test_sharpe: float | None
    mean_test_turnover: float | None
    seed_count: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkTaskResult:
    benchmark_name: str
    task_id: str
    scenario: str
    seed_count: int
    true_formula: str
    baselines: dict[str, BenchmarkBaselineAggregate]
    seed_level_results: dict[str, list[SeedBaselineResult]]


def selector_split(task: SelectorBenchmarkTask) -> tuple[pd.DataFrame, pd.Series]:
    mask = task.frame[task.environment_column].isin(task.selector_envs)
    frame = task.frame.loc[mask].copy()
    target = task.target.loc[frame.index]
    return frame, target


def test_split(task: SelectorBenchmarkTask) -> tuple[pd.DataFrame, pd.Series]:
    mask = task.frame[task.environment_column].isin(task.test_envs)
    frame = task.frame.loc[mask].copy()
    target = task.target.loc[frame.index]
    return frame, target


def evaluate_formula_on_task(formula: str, task: SelectorBenchmarkTask, *, split: str) -> dict[str, float | None]:
    if not formula:
        return {}
    frame, target = selector_split(task) if split == "selector" else test_split(task)
    metrics = evaluate_formula_metrics(formula, frame, target).metrics
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and np.isfinite(value)}


def select_best_formula_by_metric(
    candidates: list[FormulaCandidate],
    frame: pd.DataFrame,
    target: pd.Series,
    metric: str,
) -> str:
    best_formula = ""
    best_value = float("-inf")
    for candidate in candidates:
        metrics = evaluate_formula_metrics(candidate.formula, frame, target).metrics
        value = float(metrics.get(metric, float("-inf")))
        if np.isfinite(value) and value > best_value:
            best_formula = candidate.formula
            best_value = value
    return best_formula


def mean_slice_rank_ic(formula: str, frame: pd.DataFrame, target: pd.Series, slice_count: int = 4) -> float:
    values: list[float] = []
    for slice_frame in temporal_slices(frame, slice_count=slice_count):
        slice_target = target.loc[slice_frame.index]
        try:
            metrics = evaluate_formula_metrics(formula, slice_frame, slice_target).metrics
        except Exception:
            continue
        value = metrics.get("rank_ic")
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("-inf")
    return float(np.mean(values))


def select_best_formula_by_mean_slice_rank_ic(
    candidates: list[FormulaCandidate],
    frame: pd.DataFrame,
    target: pd.Series,
    slice_count: int = 4,
) -> str:
    best_formula = ""
    best_value = float("-inf")
    for candidate in candidates:
        value = mean_slice_rank_ic(candidate.formula, frame, target, slice_count=slice_count)
        if np.isfinite(value) and value > best_value:
            best_formula = candidate.formula
            best_value = value
    return best_formula


def build_seed_run(
    task: SelectorBenchmarkTask,
    temporal_selector: TemporalRobustSelector,
) -> tuple[CrossSeedSelectionRun, list[TemporalSelectorRecord], str]:
    frame, target = selector_split(task)
    outcome = temporal_selector.select(task.candidate_formulas, frame, target)
    champion_formula = select_best_formula_by_mean_slice_rank_ic(task.candidate_formulas, frame, target)
    champion = (champion_formula,) if champion_formula else tuple()
    run = CrossSeedSelectionRun(
        seed=task.seed,
        candidate_records=tuple(candidate.formula for candidate in task.candidate_formulas),
        selector_records=tuple(outcome.selected_formulas),
        champion_records=tuple(champion),
        selector_ranked_records=tuple(outcome.records),
    )
    selected_formula = outcome.selected_formulas[0] if outcome.selected_formulas else ""
    return run, list(outcome.records), selected_formula


def aggregate_seed_baseline(
    baseline: str,
    selected_formulas: list[str],
    per_seed_metrics: list[dict[str, float | None]],
    true_formula: str,
    oracle_metrics: list[dict[str, float | None]],
    *,
    support_fraction: float | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> BenchmarkBaselineAggregate:
    seed_count = len(selected_formulas)
    correct = [formula == true_formula for formula in selected_formulas]
    selection_accuracy = float(np.mean(correct)) if correct else float("nan")
    misselection_rate = float(1.0 - selection_accuracy) if correct else float("nan")
    regret_values = []
    test_rank_ic_values = []
    test_sharpe_values = []
    test_turnover_values = []
    for selected, oracle in zip(per_seed_metrics, oracle_metrics):
        selected_rank_ic = selected.get("rank_ic")
        oracle_rank_ic = oracle.get("rank_ic")
        if isinstance(selected_rank_ic, (int, float)) and isinstance(oracle_rank_ic, (int, float)):
            regret_values.append(float(oracle_rank_ic) - float(selected_rank_ic))
        if isinstance(selected.get("rank_ic"), (int, float)):
            test_rank_ic_values.append(float(selected["rank_ic"]))
        if isinstance(selected.get("sharpe"), (int, float)):
            test_sharpe_values.append(float(selected["sharpe"]))
        if isinstance(selected.get("turnover"), (int, float)):
            test_turnover_values.append(float(selected["turnover"]))
    if support_fraction is None:
        stability = 0.0
        if selected_formulas:
            counts = pd.Series(selected_formulas).value_counts(dropna=False)
            stability = float(counts.iloc[0] / len(selected_formulas))
    else:
        stability = float(support_fraction)
    selected_formula = None
    if selected_formulas:
        counts = pd.Series(selected_formulas).value_counts(dropna=False)
        selected_formula = str(counts.index[0])
    return BenchmarkBaselineAggregate(
        baseline=baseline,
        selected_formula=selected_formula,
        selection_accuracy=selection_accuracy,
        misselection_rate=misselection_rate,
        oracle_regret_rank_ic=float(np.mean(regret_values)) if regret_values else None,
        selected_formula_stability=stability,
        mean_test_rank_ic=float(np.mean(test_rank_ic_values)) if test_rank_ic_values else None,
        mean_test_sharpe=float(np.mean(test_sharpe_values)) if test_sharpe_values else None,
        mean_test_turnover=float(np.mean(test_turnover_values)) if test_turnover_values else None,
        seed_count=seed_count,
        diagnostics=diagnostics or {},
    )


def run_task_benchmark(
    tasks: list[SelectorBenchmarkTask],
    *,
    temporal_selector: TemporalRobustSelector | None = None,
    consensus_selector: CrossSeedConsensusSelector | None = None,
) -> BenchmarkTaskResult:
    if not tasks:
        raise ValueError("At least one task instance is required.")
    temporal_selector = temporal_selector or TemporalRobustSelector()
    consensus_selector = consensus_selector or CrossSeedConsensusSelector(
        temporal_selector=temporal_selector,
        config=CrossSeedConsensusConfig(rerank_mode="support_only"),
    )
    benchmark_name = tasks[0].benchmark_name
    task_id = tasks[0].task_id
    scenario = tasks[0].scenario
    true_formula = tasks[0].true_formula

    seed_level_results: dict[str, list[SeedBaselineResult]] = {
        "naive_rank_ic": [],
        "best_validation_sharpe": [],
        "best_validation_mean_rank_ic": [],
        "single_seed_temporal_selector": [],
    }
    seed_runs: list[CrossSeedSelectionRun] = []
    base_candidates = tasks[0].candidate_formulas

    for task in tasks:
        sel_frame, sel_target = selector_split(task)
        tst_frame, tst_target = test_split(task)
        oracle_metrics = evaluate_formula_metrics(true_formula, tst_frame, tst_target).metrics
        baseline_choices = {
            "naive_rank_ic": select_best_formula_by_metric(task.candidate_formulas, sel_frame, sel_target, "rank_ic"),
            "best_validation_sharpe": select_best_formula_by_metric(task.candidate_formulas, sel_frame, sel_target, "sharpe"),
            "best_validation_mean_rank_ic": select_best_formula_by_mean_slice_rank_ic(task.candidate_formulas, sel_frame, sel_target),
        }
        seed_run, temporal_records, temporal_formula = build_seed_run(task, temporal_selector)
        seed_runs.append(seed_run)
        baseline_choices["single_seed_temporal_selector"] = temporal_formula

        for baseline, formula in baseline_choices.items():
            metrics = evaluate_formula_metrics(formula, tst_frame, tst_target).metrics if formula else {}
            rank_ic = metrics.get("rank_ic")
            oracle_rank_ic = oracle_metrics.get("rank_ic")
            regret = None
            if isinstance(rank_ic, (int, float)) and isinstance(oracle_rank_ic, (int, float)):
                regret = float(oracle_rank_ic) - float(rank_ic)
            seed_level_results[baseline].append(
                SeedBaselineResult(
                    seed=task.seed,
                    selected_formula=formula,
                    test_metrics={key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and np.isfinite(value)},
                    oracle_regret_rank_ic=regret,
                    selection_correct=(formula == true_formula),
                )
            )

    baselines: dict[str, BenchmarkBaselineAggregate] = {}
    for baseline, results in seed_level_results.items():
        baselines[baseline] = aggregate_seed_baseline(
            baseline,
            [item.selected_formula for item in results],
            [item.test_metrics for item in results],
            true_formula,
            [evaluate_formula_on_task(true_formula, task, split="test") for task in tasks],
        )

    selector_stack_frame, selector_stack_target = stacked_split(tasks, split="selector")
    consensus_outcome = consensus_selector.select(seed_runs, selector_stack_frame, selector_stack_target, base_candidates=base_candidates)
    support_formula = consensus_outcome.selected_formulas[0] if consensus_outcome.selected_formulas else ""
    mean_score_formula = mean_score_consensus_formula(seed_runs)

    consensus_metrics = [evaluate_formula_on_task(support_formula, task, split="test") for task in tasks] if support_formula else [{} for _ in tasks]
    mean_score_metrics = [evaluate_formula_on_task(mean_score_formula, task, split="test") for task in tasks] if mean_score_formula else [{} for _ in tasks]
    oracle_metrics = [evaluate_formula_on_task(true_formula, task, split="test") for task in tasks]

    support_payload = consensus_outcome.formula_support.get(support_formula, {}) if support_formula else {}
    mean_score_support = summarize_formula_support_for_formula(seed_runs, mean_score_formula)
    baselines["cross_seed_mean_score_consensus"] = aggregate_seed_baseline(
        "cross_seed_mean_score_consensus",
        [mean_score_formula] * len(tasks),
        mean_score_metrics,
        true_formula,
        oracle_metrics,
        support_fraction=float(mean_score_support / max(len(tasks), 1)),
        diagnostics={"selected_formula": mean_score_formula},
    )
    baselines["support_adjusted_cross_seed_consensus"] = aggregate_seed_baseline(
        "support_adjusted_cross_seed_consensus",
        [support_formula] * len(tasks),
        consensus_metrics,
        true_formula,
        oracle_metrics,
        support_fraction=float(support_payload.get("candidate_seed_support", 0) / max(len(tasks), 1)),
        diagnostics={
            "selected_formula": support_formula,
            "candidate_pool_size": len(consensus_outcome.candidate_pool),
            "support_adjusted_ranked_records": [
                {
                    "formula": record.formula,
                    "support_adjusted_score": record.support_adjusted_score,
                    "candidate_seed_support": record.candidate_seed_support,
                    "selector_seed_support": record.selector_seed_support,
                    "champion_seed_support": record.champion_seed_support,
                }
                for record in consensus_outcome.ranked_records
            ],
        },
    )
    return BenchmarkTaskResult(
        benchmark_name=benchmark_name,
        task_id=task_id,
        scenario=scenario,
        seed_count=len(tasks),
        true_formula=true_formula,
        baselines=baselines,
        seed_level_results=seed_level_results,
    )


def suite_leaderboard(task_results: list[BenchmarkTaskResult]) -> list[dict[str, Any]]:
    baselines = sorted({name for result in task_results for name in result.baselines})
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        accuracy = []
        misselection = []
        regret = []
        rank_ic = []
        sharpe = []
        turnover = []
        stability = []
        for result in task_results:
            payload = result.baselines[baseline]
            accuracy.append(payload.selection_accuracy)
            misselection.append(payload.misselection_rate)
            stability.append(payload.selected_formula_stability)
            if payload.oracle_regret_rank_ic is not None:
                regret.append(payload.oracle_regret_rank_ic)
            if payload.mean_test_rank_ic is not None:
                rank_ic.append(payload.mean_test_rank_ic)
            if payload.mean_test_sharpe is not None:
                sharpe.append(payload.mean_test_sharpe)
            if payload.mean_test_turnover is not None:
                turnover.append(payload.mean_test_turnover)
        rows.append(
            {
                "baseline": baseline,
                "task_count": len(task_results),
                "selection_accuracy": float(np.mean(accuracy)) if accuracy else None,
                "misselection_rate": float(np.mean(misselection)) if misselection else None,
                "oracle_regret_rank_ic": float(np.mean(regret)) if regret else None,
                "selected_formula_stability": float(np.mean(stability)) if stability else None,
                "mean_test_rank_ic": float(np.mean(rank_ic)) if rank_ic else None,
                "mean_test_sharpe": float(np.mean(sharpe)) if sharpe else None,
                "mean_test_turnover": float(np.mean(turnover)) if turnover else None,
            }
        )
    rows.sort(key=lambda item: (item["selection_accuracy"], item["mean_test_rank_ic"], item["mean_test_sharpe"]), reverse=True)
    return rows


def stacked_split(tasks: list[SelectorBenchmarkTask], *, split: str) -> tuple[pd.DataFrame, pd.Series]:
    frames = []
    targets = []
    for index, task in enumerate(tasks):
        frame, target = selector_split(task) if split == "selector" else test_split(task)
        local = frame.copy()
        local["seed"] = task.seed
        if "date" not in local.columns:
            local["date"] = pd.date_range("2000-01-01", periods=len(local), freq="D")
        else:
            local["date"] = pd.to_datetime(local["date"]) + pd.to_timedelta(index * 1000, unit="D")
        frames.append(local)
        targets.append(target.reset_index(drop=True))
    stacked_frame = pd.concat(frames, axis=0, ignore_index=True)
    stacked_target = pd.concat(targets, axis=0, ignore_index=True)
    stacked_target.index = stacked_frame.index
    return stacked_frame, stacked_target


def summarize_formula_support_for_formula(runs: list[CrossSeedSelectionRun], formula: str) -> int:
    if not formula:
        return 0
    support = 0
    for run in runs:
        if formula in run.candidate_records:
            support += 1
    return support


def temporal_slices(frame: pd.DataFrame, slice_count: int = 4) -> list[pd.DataFrame]:
    if "date" not in frame.columns:
        return [frame]
    dates = pd.Index(pd.to_datetime(frame["date"]).sort_values().unique())
    if dates.empty:
        return [frame]
    chunks = [chunk for chunk in np.array_split(dates.to_numpy(), min(slice_count, len(dates))) if len(chunk)]
    slices = []
    for chunk in chunks:
        mask = pd.to_datetime(frame["date"]).isin(pd.to_datetime(chunk))
        slices.append(frame.loc[mask].copy())
    return slices or [frame]


def write_suite_payload(output_root: Path, stem: str, payload: dict[str, Any], leaderboard: list[dict[str, Any]]) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{stem}.json"
    md_path = output_root / f"{stem}.md"
    csv_path = output_root / f"{stem}_leaderboard.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_suite_markdown(payload, leaderboard) + "\n", encoding="utf-8")
    pd.DataFrame(leaderboard).to_csv(csv_path, index=False)
    return {
        "json_report": str(json_path),
        "markdown_report": str(md_path),
        "csv_report": str(csv_path),
    }


def build_suite_markdown(payload: dict[str, Any], leaderboard: list[dict[str, Any]]) -> str:
    lines = [f"# {payload['benchmark_name']} Results", "", "## Leaderboard", ""]
    if leaderboard:
        headers = ["Baseline", "SelAcc", "MisSel", "OracleRegret", "RankIC", "Sharpe", "Stability"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in leaderboard:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["baseline"]),
                        _fmt(row["selection_accuracy"]),
                        _fmt(row["misselection_rate"]),
                        _fmt(row["oracle_regret_rank_ic"]),
                        _fmt(row["mean_test_rank_ic"]),
                        _fmt(row["mean_test_sharpe"]),
                        _fmt(row["selected_formula_stability"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Tasks", ""])
    for task in payload.get("task_results", []):
        lines.append(f"- `{task['task_id']}` / `{task['scenario']}` -> true `{task['true_formula']}`")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"
