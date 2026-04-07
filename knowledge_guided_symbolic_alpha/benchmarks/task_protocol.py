from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from ..evaluation.panel_dispatch import evaluate_formula_metrics
from ..generation import FormulaCandidate
from ..selection import (
    CrossSeedConsensusConfig,
    CrossSeedConsensusOutcome,
    CrossSeedConsensusSelector,
    CrossSeedSelectionRun,
    RobustSelectorConfig,
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


def formula_complexity(formula: str) -> int:
    return max(1, len(str(formula).split()))


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


def select_formula_by_pareto_front(
    candidates: list[FormulaCandidate],
    frame: pd.DataFrame,
    target: pd.Series,
    slice_count: int = 4,
) -> str:
    frontier: list[dict[str, float | str]] = []
    scored: list[dict[str, float | str]] = []
    for candidate in candidates:
        score = mean_slice_rank_ic(candidate.formula, frame, target, slice_count=slice_count)
        if not np.isfinite(score):
            continue
        scored.append(
            {
                "formula": candidate.formula,
                "score": float(score),
                "complexity": float(formula_complexity(candidate.formula)),
            }
        )
    if not scored:
        return ""
    for item in scored:
        dominated = False
        for other in scored:
            if other is item:
                continue
            better_or_equal = other["score"] >= item["score"] and other["complexity"] <= item["complexity"]
            strictly_better = other["score"] > item["score"] or other["complexity"] < item["complexity"]
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(item)
    frontier.sort(key=lambda item: (item["complexity"], item["score"]))
    if len(frontier) == 1:
        return str(frontier[0]["formula"])

    complexities = np.array([float(item["complexity"]) for item in frontier], dtype=float)
    scores = np.array([float(item["score"]) for item in frontier], dtype=float)
    comp_scale = max(complexities.max() - complexities.min(), 1.0)
    score_scale = max(scores.max() - scores.min(), 1e-12)
    xs = (complexities - complexities.min()) / comp_scale
    ys = (scores - scores.min()) / score_scale
    start = np.array([xs[0], ys[0]], dtype=float)
    end = np.array([xs[-1], ys[-1]], dtype=float)
    line = end - start
    line_norm = float(np.linalg.norm(line))
    if line_norm <= 1e-12:
        return str(max(frontier, key=lambda item: (item["score"], -item["complexity"]))["formula"])
    best_formula = str(frontier[-1]["formula"])
    best_distance = float("-inf")
    for item, x, y in zip(frontier, xs, ys):
        point = np.array([x, y], dtype=float)
        relative = point - start
        cross_2d = line[0] * relative[1] - line[1] * relative[0]
        distance = float(abs(cross_2d) / line_norm)
        tie_break = float(item["score"]) - 1e-3 * float(item["complexity"])
        if distance > best_distance or (np.isclose(distance, best_distance) and tie_break > 0):
            best_distance = distance
            best_formula = str(item["formula"])
    return best_formula


def select_formula_by_lasso_screening(
    candidates: list[FormulaCandidate],
    frame: pd.DataFrame,
    target: pd.Series,
) -> str:
    if not candidates:
        return ""
    signals: list[pd.Series] = []
    formulas: list[str] = []
    for candidate in candidates:
        try:
            evaluated = evaluate_formula_metrics(candidate.formula, frame, target).evaluated
        except Exception:
            continue
        signal = pd.Series(evaluated.signal, index=frame.index, dtype=float).replace([np.inf, -np.inf], np.nan)
        if signal.isna().all():
            continue
        signals.append(signal.fillna(signal.median()))
        formulas.append(candidate.formula)
    if not signals:
        return ""
    if len(signals) == 1:
        return formulas[0]
    design = pd.concat(signals, axis=1)
    design.columns = formulas
    values = design.to_numpy(dtype=float)
    values = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=0, keepdims=True)
    std[std < 1e-12] = 1.0
    values = values / std
    response = pd.Series(target, index=frame.index, dtype=float).fillna(float(pd.Series(target).median())).to_numpy(dtype=float)
    try:
        fit = LassoCV(cv=3, random_state=0, max_iter=10000).fit(values, response)
        coefs = np.abs(fit.coef_)
        if np.any(coefs > 1e-10):
            return formulas[int(np.argmax(coefs))]
    except Exception:
        pass
    correlations = np.abs(np.corrcoef(values, response, rowvar=False)[-1, :-1])
    correlations = np.nan_to_num(correlations, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return formulas[int(np.argmax(correlations))]


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
    temporal_selector = temporal_selector or TemporalRobustSelector(
        RobustSelectorConfig(selection_mode="pareto")
    )
    discrete_temporal_selector = TemporalRobustSelector(
        RobustSelectorConfig(selection_mode="pareto_discrete_legacy")
    )
    legacy_temporal_selector = TemporalRobustSelector(
        RobustSelectorConfig(selection_mode="legacy_linear")
    )
    consensus_selector = consensus_selector or CrossSeedConsensusSelector(
        temporal_selector=temporal_selector,
        config=CrossSeedConsensusConfig(selection_mode="pareto", rerank_mode="shared_frame"),
    )
    discrete_consensus_selector = CrossSeedConsensusSelector(
        temporal_selector=discrete_temporal_selector,
        config=CrossSeedConsensusConfig(selection_mode="pareto_discrete_legacy", rerank_mode="shared_frame"),
    )
    legacy_consensus_selector = CrossSeedConsensusSelector(
        temporal_selector=legacy_temporal_selector,
        config=CrossSeedConsensusConfig(selection_mode="legacy_linear", rerank_mode="shared_frame"),
    )
    benchmark_name = tasks[0].benchmark_name
    task_id = tasks[0].task_id
    scenario = tasks[0].scenario
    true_formula = tasks[0].true_formula

    seed_level_results: dict[str, list[SeedBaselineResult]] = {
        "naive_rank_ic": [],
        "best_validation_sharpe": [],
        "best_validation_mean_rank_ic": [],
        "pareto_front_selector": [],
        "lasso_formula_screening": [],
        "single_seed_temporal_selector": [],
        "pareto_discrete_temporal_selector": [],
        "legacy_linear_temporal_selector": [],
    }
    pareto_seed_runs: list[CrossSeedSelectionRun] = []
    discrete_seed_runs: list[CrossSeedSelectionRun] = []
    legacy_seed_runs: list[CrossSeedSelectionRun] = []
    base_candidates = tasks[0].candidate_formulas

    for task in tasks:
        sel_frame, sel_target = selector_split(task)
        tst_frame, tst_target = test_split(task)
        oracle_metrics = evaluate_formula_metrics(true_formula, tst_frame, tst_target).metrics
        baseline_choices = {
            "naive_rank_ic": select_best_formula_by_metric(task.candidate_formulas, sel_frame, sel_target, "rank_ic"),
            "best_validation_sharpe": select_best_formula_by_metric(task.candidate_formulas, sel_frame, sel_target, "sharpe"),
            "best_validation_mean_rank_ic": select_best_formula_by_mean_slice_rank_ic(task.candidate_formulas, sel_frame, sel_target),
            "pareto_front_selector": select_formula_by_pareto_front(task.candidate_formulas, sel_frame, sel_target),
            "lasso_formula_screening": select_formula_by_lasso_screening(task.candidate_formulas, sel_frame, sel_target),
        }
        pareto_seed_run, _, pareto_temporal_formula = build_seed_run(task, temporal_selector)
        discrete_seed_run, _, discrete_temporal_formula = build_seed_run(task, discrete_temporal_selector)
        legacy_seed_run, _, legacy_temporal_formula = build_seed_run(task, legacy_temporal_selector)
        pareto_seed_runs.append(pareto_seed_run)
        discrete_seed_runs.append(discrete_seed_run)
        legacy_seed_runs.append(legacy_seed_run)
        baseline_choices["single_seed_temporal_selector"] = pareto_temporal_formula
        baseline_choices["pareto_discrete_temporal_selector"] = discrete_temporal_formula
        baseline_choices["legacy_linear_temporal_selector"] = legacy_temporal_formula

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
    oracle_metrics = [evaluate_formula_on_task(true_formula, task, split="test") for task in tasks]
    for baseline, results in seed_level_results.items():
        baselines[baseline] = aggregate_seed_baseline(
            baseline,
            [item.selected_formula for item in results],
            [item.test_metrics for item in results],
            true_formula,
            oracle_metrics,
        )

    selector_stack_frame, selector_stack_target = stacked_split(tasks, split="selector")
    pareto_outcome = consensus_selector.select(
        pareto_seed_runs,
        selector_stack_frame,
        selector_stack_target,
        base_candidates=base_candidates,
    )
    legacy_outcome = legacy_consensus_selector.select(
        legacy_seed_runs,
        selector_stack_frame,
        selector_stack_target,
        base_candidates=base_candidates,
    )
    discrete_outcome = discrete_consensus_selector.select(
        discrete_seed_runs,
        selector_stack_frame,
        selector_stack_target,
        base_candidates=base_candidates,
    )
    mean_score_formula = mean_score_consensus_formula(pareto_seed_runs)

    def _consensus_payload(name: str, formula: str, outcome: CrossSeedConsensusOutcome) -> BenchmarkBaselineAggregate:
        metrics = [evaluate_formula_on_task(formula, task, split="test") for task in tasks] if formula else [{} for _ in tasks]
        support_payload = outcome.formula_support.get(formula, {}) if formula else {}
        return aggregate_seed_baseline(
            name,
            [formula] * len(tasks),
            metrics,
            true_formula,
            oracle_metrics,
            support_fraction=float(support_payload.get("candidate_seed_support", 0) / max(len(tasks), 1)),
            diagnostics={
                "selected_formula": formula,
                "candidate_pool_size": len(outcome.candidate_pool),
                "first_front_size": max(
                    (int(record.consensus_front_size or 0) for record in outcome.ranked_records if int(record.consensus_pareto_rank or 999) == 1),
                    default=0,
                ),
                "first_front_share": max(
                    (float(record.consensus_front_share or 0.0) for record in outcome.ranked_records if int(record.consensus_pareto_rank or 999) == 1),
                    default=0.0,
                ),
                "selected_crowding_distance": next(
                    (
                        float(record.consensus_crowding_distance)
                        for record in outcome.ranked_records
                        if record.formula == formula and record.consensus_crowding_distance is not None
                    ),
                    None,
                ),
                "used_near_neighbor_tiebreak": any(bool(record.used_near_neighbor_tiebreak) for record in outcome.ranked_records),
                "ranked_records": [
                    {
                        "formula": record.formula,
                        "support_adjusted_score": record.support_adjusted_score,
                        "mean_temporal_score": record.mean_temporal_score,
                        "candidate_seed_support": record.candidate_seed_support,
                        "selector_seed_support": record.selector_seed_support,
                        "champion_seed_support": record.champion_seed_support,
                        "mean_selector_rank": record.mean_selector_rank,
                        "consensus_pareto_rank": record.consensus_pareto_rank,
                        "consensus_tiebreak_rank": record.consensus_tiebreak_rank,
                        "consensus_front_size": record.consensus_front_size,
                        "consensus_front_share": record.consensus_front_share,
                        "consensus_crowding_distance": record.consensus_crowding_distance,
                        "used_near_neighbor_tiebreak": record.used_near_neighbor_tiebreak,
                    }
                    for record in outcome.ranked_records
                ],
            },
        )

    baselines["cross_seed_mean_score_consensus"] = aggregate_seed_baseline(
        "cross_seed_mean_score_consensus",
        [mean_score_formula] * len(tasks),
        [evaluate_formula_on_task(mean_score_formula, task, split="test") for task in tasks] if mean_score_formula else [{} for _ in tasks],
        true_formula,
        oracle_metrics,
        support_fraction=float(summarize_formula_support_for_formula(pareto_seed_runs, mean_score_formula) / max(len(tasks), 1)),
        diagnostics={"selected_formula": mean_score_formula},
    )
    pareto_formula = pareto_outcome.selected_formulas[0] if pareto_outcome.selected_formulas else ""
    discrete_formula = discrete_outcome.selected_formulas[0] if discrete_outcome.selected_formulas else ""
    legacy_formula = legacy_outcome.selected_formulas[0] if legacy_outcome.selected_formulas else ""
    baselines["pareto_cross_seed_consensus"] = _consensus_payload("pareto_cross_seed_consensus", pareto_formula, pareto_outcome)
    baselines["pareto_discrete_legacy"] = _consensus_payload("pareto_discrete_legacy", discrete_formula, discrete_outcome)
    baselines["legacy_linear_selector"] = _consensus_payload("legacy_linear_selector", legacy_formula, legacy_outcome)
    baselines["support_adjusted_cross_seed_consensus"] = baselines["legacy_linear_selector"]

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


def suite_stress_summary(task_results: list[BenchmarkTaskResult]) -> list[dict[str, Any]]:
    baselines = sorted({name for result in task_results for name in result.baselines})
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        per_task = []
        failure_boundary_scores = []
        for result in task_results:
            payload = result.baselines[baseline]
            task_row = {
                "task_id": result.task_id,
                "scenario": result.scenario,
                "selection_accuracy": float(payload.selection_accuracy),
                "misselection_rate": float(payload.misselection_rate),
            }
            per_task.append(task_row)
            if "adversarial" in result.scenario or "stress" in result.scenario or "outlier" in result.scenario:
                failure_boundary_scores.append(float(payload.selection_accuracy))
        per_task.sort(key=lambda item: (item["selection_accuracy"], -item["misselection_rate"]))
        worst = per_task[0] if per_task else {"task_id": None, "scenario": None, "selection_accuracy": None}
        rows.append(
            {
                "baseline": baseline,
                "task_count": len(per_task),
                "average_selection_accuracy": float(np.mean([item["selection_accuracy"] for item in per_task])) if per_task else None,
                "worst_case_accuracy": worst["selection_accuracy"],
                "worst_case_task_id": worst["task_id"],
                "worst_case_scenario": worst["scenario"],
                "failure_boundary_accuracy": float(np.mean(failure_boundary_scores)) if failure_boundary_scores else None,
            }
        )
    rows.sort(key=lambda item: (item["average_selection_accuracy"], item["worst_case_accuracy"]), reverse=True)
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
