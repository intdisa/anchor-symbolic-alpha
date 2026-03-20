from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from .evaluator import EvaluationError, FormulaEvaluator
from .factor_pool import FactorPool, FactorRecord
from .orthogonality import max_abs_correlation
from .panel_dispatch import evaluate_formula_metrics, is_cross_sectional_frame, resolve_formula_evaluator
from .role_profiles import adapt_role_profile, normalize_role, resolve_role_profile


@dataclass(frozen=True)
class CandidatePoolPreview:
    accepted: bool
    reason: str
    record: FactorRecord | None
    marginal_gain: float
    baseline_score: float
    new_score: float
    replaced_canonical: str | None = None
    trade_proxy_gain: float = 0.0
    baseline_trade_proxy: float = 0.0
    new_trade_proxy: float = 0.0


def evaluate_formula_record(
    formula: str | list[str] | tuple[str, ...],
    data: pd.DataFrame,
    target: pd.Series,
    evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
    role: str | None = None,
) -> FactorRecord:
    result = evaluate_formula_metrics(formula, data, target, evaluator=evaluator)
    evaluated = result.evaluated
    metrics = result.metrics
    return FactorRecord(
        tokens=evaluated.parsed.tokens,
        canonical=evaluated.parsed.canonical,
        signal=evaluated.signal,
        metrics=metrics,
        role=role,
    )


def rescore_pool_on_dataset(
    pool: FactorPool,
    data: pd.DataFrame,
    target: pd.Series,
    evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
) -> FactorPool:
    evaluator = resolve_formula_evaluator(data, evaluator=evaluator)
    scored_pool = FactorPool(max_size=pool.max_size)
    for record in pool.records:
        try:
            rescored = evaluate_formula_record(record.tokens, data, target, evaluator=evaluator, role=record.role)
        except EvaluationError:
            continue
        scored_pool.add(rescored)
    return scored_pool


def score_pool_on_dataset(
    pool: FactorPool,
    data: pd.DataFrame,
    target: pd.Series,
    evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
) -> float:
    scored_pool = rescore_pool_on_dataset(pool, data, target, evaluator=evaluator)
    score = scored_pool.pool_score()
    if not np.isfinite(score):
        return float("-inf")
    return score


def preview_candidate_on_dataset(
    formula: str | list[str] | tuple[str, ...],
    pool: FactorPool,
    data: pd.DataFrame,
    target: pd.Series,
    evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
    role: str | None = None,
    min_abs_rank_ic: float | None = None,
    max_correlation: float | None = None,
    replacement_margin: float | None = None,
    min_validation_marginal_gain: float | None = None,
    min_trade_proxy_gain: float | None = None,
) -> CandidatePoolPreview:
    evaluator = resolve_formula_evaluator(data, evaluator=evaluator)
    profile = resolve_role_profile(role)
    profile = adapt_role_profile(profile, role, cross_sectional=is_cross_sectional_frame(data))
    normalized_role = normalize_role(role)
    resolved_min_abs_rank_ic = (
        profile.resolved_preview_min_abs_rank_ic if min_abs_rank_ic is None else min_abs_rank_ic
    )
    resolved_max_correlation = (
        profile.resolved_preview_max_correlation if max_correlation is None else max_correlation
    )
    resolved_replacement_margin = profile.replacement_margin if replacement_margin is None else replacement_margin
    resolved_validation_marginal_gain = (
        profile.preview_min_validation_marginal_gain if min_validation_marginal_gain is None else min_validation_marginal_gain
    )
    resolved_min_trade_proxy_gain = (
        profile.resolved_preview_min_trade_proxy_gain if min_trade_proxy_gain is None else min_trade_proxy_gain
    )
    baseline_pool = rescore_pool_on_dataset(pool, data, target, evaluator=evaluator)
    baseline_score = baseline_pool.pool_score()
    baseline_trade_proxy = baseline_pool.trade_proxy_score()
    try:
        record = evaluate_formula_record(formula, data, target, evaluator=evaluator, role=role)
    except EvaluationError as exc:
        return CandidatePoolPreview(
            accepted=False,
            reason=str(exc),
            record=None,
            marginal_gain=0.0,
            baseline_score=baseline_score,
            new_score=baseline_score,
            baseline_trade_proxy=baseline_trade_proxy,
            new_trade_proxy=baseline_trade_proxy,
        )

    if record.canonical in baseline_pool.canonicals():
        return CandidatePoolPreview(
            accepted=False,
            reason="duplicate_canonical",
            record=record,
            marginal_gain=0.0,
            baseline_score=baseline_score,
            new_score=baseline_score,
            baseline_trade_proxy=baseline_trade_proxy,
            new_trade_proxy=baseline_trade_proxy,
        )

    weakest_index = baseline_pool.weakest_index() if len(baseline_pool) >= baseline_pool.max_size else None
    same_role_replace_index = None
    if is_cross_sectional_frame(data) and normalized_role == "target_price":
        same_role_indices = [
            index
            for index, existing in enumerate(baseline_pool.records)
            if normalize_role(existing.role) == normalized_role
        ]
        if same_role_indices:
            same_role_replace_index = min(
                same_role_indices,
                key=lambda index: float(baseline_pool.records[index].metrics.get("rank_ic", 0.0)),
            )

    replacement_preview = None
    if same_role_replace_index is not None:
        replacement_preview = _preview_candidate_path(
            record,
            baseline_pool,
            baseline_score,
            baseline_trade_proxy,
            normalized_role=normalized_role,
            cross_sectional=is_cross_sectional_frame(data),
            min_abs_rank_ic=resolved_min_abs_rank_ic,
            max_correlation=resolved_max_correlation,
            replacement_margin=resolved_replacement_margin,
            min_validation_marginal_gain=resolved_validation_marginal_gain,
            min_trade_proxy_gain=resolved_min_trade_proxy_gain,
            replace_index=same_role_replace_index,
        )
        if replacement_preview.accepted:
            return replacement_preview

    append_preview = _preview_candidate_path(
        record,
        baseline_pool,
        baseline_score,
        baseline_trade_proxy,
        normalized_role=normalized_role,
        cross_sectional=is_cross_sectional_frame(data),
        min_abs_rank_ic=resolved_min_abs_rank_ic,
        max_correlation=resolved_max_correlation,
        replacement_margin=resolved_replacement_margin,
        min_validation_marginal_gain=resolved_validation_marginal_gain,
        min_trade_proxy_gain=resolved_min_trade_proxy_gain,
        replace_index=weakest_index if len(baseline_pool) >= baseline_pool.max_size else None,
    )
    if replacement_preview is not None and append_preview.reason == "fast_ic_screen":
        return replacement_preview
    return append_preview


def _preview_candidate_path(
    record: FactorRecord,
    baseline_pool: FactorPool,
    baseline_score: float,
    baseline_trade_proxy: float,
    *,
    normalized_role: str | None,
    cross_sectional: bool,
    min_abs_rank_ic: float,
    max_correlation: float,
    replacement_margin: float,
    min_validation_marginal_gain: float,
    min_trade_proxy_gain: float,
    replace_index: int | None,
) -> CandidatePoolPreview:
    new_trade_proxy = baseline_pool.trade_proxy_with(record, replace_index=replace_index)
    trade_proxy_gain = new_trade_proxy - baseline_trade_proxy
    same_family_replacement = (
        cross_sectional
        and normalized_role == "target_price"
        and replace_index is not None
        and normalize_role(baseline_pool.records[replace_index].role) == normalized_role
    )
    baseline_replacement_gain = (
        baseline_pool.baseline_replacement_gain(record, replace_index) if same_family_replacement else float("-inf")
    )
    baseline_replacement_ok = same_family_replacement and baseline_replacement_gain > 0.0
    rank_ic = float(record.metrics.get("rank_ic", 0.0))
    meets_rank_gate = np.isfinite(rank_ic) and abs(rank_ic) >= min_abs_rank_ic
    meets_trade_proxy_gate = trade_proxy_gain > min_trade_proxy_gain
    if not meets_rank_gate:
        if not ((normalized_role == "target_flow" and meets_trade_proxy_gate) or baseline_replacement_ok):
            return CandidatePoolPreview(
                accepted=False,
                reason="fast_ic_screen",
                record=record,
                marginal_gain=0.0,
                baseline_score=baseline_score,
                new_score=baseline_score,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )

    corr_pool = baseline_pool if replace_index is None else _pool_without_index(baseline_pool, replace_index)
    current_signals = corr_pool.signals_frame()
    max_corr = max_abs_correlation(record.signal, current_signals, method="spearman")
    record.metrics["max_corr"] = max_corr
    if max_corr >= max_correlation:
        return CandidatePoolPreview(
            accepted=False,
            reason="correlation_check",
            record=record,
            marginal_gain=0.0,
            baseline_score=baseline_score,
            new_score=baseline_score,
            trade_proxy_gain=trade_proxy_gain,
            baseline_trade_proxy=baseline_trade_proxy,
            new_trade_proxy=new_trade_proxy,
        )

    requires_trade_proxy_gate = normalized_role == "context" and min_trade_proxy_gain > 0.0
    new_score = baseline_pool.score_with(record, replace_index=replace_index)
    marginal_gain = new_score - baseline_score
    effective_gain = baseline_replacement_gain if baseline_replacement_ok else marginal_gain
    if replace_index is None and len(baseline_pool) < baseline_pool.max_size:
        if marginal_gain <= min_validation_marginal_gain:
            return CandidatePoolPreview(
                accepted=False,
                reason="full_validation",
                record=record,
                marginal_gain=marginal_gain,
                baseline_score=baseline_score,
                new_score=new_score,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )
        if requires_trade_proxy_gate and trade_proxy_gain <= min_trade_proxy_gain:
            return CandidatePoolPreview(
                accepted=False,
                reason="trade_proxy_check",
                record=record,
                marginal_gain=marginal_gain,
                baseline_score=baseline_score,
                new_score=new_score,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )
        return CandidatePoolPreview(
            accepted=True,
            reason="accepted",
            record=record,
            marginal_gain=marginal_gain,
            baseline_score=baseline_score,
            new_score=new_score,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )

    if same_family_replacement and baseline_replacement_ok:
        replaced = baseline_pool.records[replace_index].canonical
        return CandidatePoolPreview(
            accepted=True,
            reason="replaced_baseline",
            record=record,
            marginal_gain=effective_gain,
            baseline_score=baseline_score,
            new_score=new_score,
            replaced_canonical=replaced,
            trade_proxy_gain=trade_proxy_gain,
            baseline_trade_proxy=baseline_trade_proxy,
            new_trade_proxy=new_trade_proxy,
        )

    if effective_gain <= replacement_margin:
        return CandidatePoolPreview(
            accepted=False,
            reason="replacement_check",
            record=record,
            marginal_gain=effective_gain,
            baseline_score=baseline_score,
            new_score=new_score,
            trade_proxy_gain=trade_proxy_gain,
            baseline_trade_proxy=baseline_trade_proxy,
            new_trade_proxy=new_trade_proxy,
        )
    if requires_trade_proxy_gate and trade_proxy_gain <= min_trade_proxy_gain:
        return CandidatePoolPreview(
            accepted=False,
            reason="trade_proxy_check",
            record=record,
            marginal_gain=marginal_gain,
            baseline_score=baseline_score,
            new_score=new_score,
            trade_proxy_gain=trade_proxy_gain,
            baseline_trade_proxy=baseline_trade_proxy,
            new_trade_proxy=new_trade_proxy,
        )
    replaced = baseline_pool.records[replace_index].canonical if replace_index is not None else None
    return CandidatePoolPreview(
        accepted=True,
        reason="replaced" if replace_index is not None else "accepted",
        record=record,
        marginal_gain=effective_gain,
        baseline_score=baseline_score,
        new_score=new_score,
        replaced_canonical=replaced,
        trade_proxy_gain=trade_proxy_gain,
        baseline_trade_proxy=baseline_trade_proxy,
        new_trade_proxy=new_trade_proxy,
    )


def _pool_without_index(pool: FactorPool, remove_index: int) -> FactorPool:
    clone = pool.copy()
    clone.records = [
        record
        for index, record in enumerate(clone.records)
        if index != remove_index
    ]
    return clone
