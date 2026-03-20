from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from .evaluator import EvaluationError, FormulaEvaluator
from .factor_pool import FactorPool, FactorRecord
from .orthogonality import max_abs_correlation
from .panel_dispatch import evaluate_formula_metrics, is_cross_sectional_frame, resolve_formula_evaluator
from .role_profiles import RoleProfile, adapt_role_profile, normalize_role, resolve_role_profile


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str
    candidate: FactorRecord | None
    marginal_gain: float
    replaced_canonical: str | None = None
    trade_proxy_gain: float = 0.0
    baseline_trade_proxy: float = 0.0
    new_trade_proxy: float = 0.0


class AdmissionPolicy:
    def __init__(
        self,
        evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
        min_abs_rank_ic: float = 0.05,
        max_correlation: float = 0.9,
        replacement_margin: float = 1e-4,
    ) -> None:
        self.evaluator = evaluator or FormulaEvaluator()
        self.min_abs_rank_ic = min_abs_rank_ic
        self.max_correlation = max_correlation
        self.replacement_margin = replacement_margin

    def screen(
        self,
        formula: str | list[str] | tuple[str, ...],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        role: str | None = None,
    ) -> AdmissionDecision:
        resolved_evaluator = resolve_formula_evaluator(data, evaluator=self.evaluator)
        profile = resolve_role_profile(
            role,
            default_commit_min_abs_rank_ic=self.min_abs_rank_ic,
            default_commit_max_correlation=self.max_correlation,
            default_replacement_margin=self.replacement_margin,
        )
        profile = adapt_role_profile(profile, role, cross_sectional=is_cross_sectional_frame(data))
        normalized_role = normalize_role(role)
        try:
            result = evaluate_formula_metrics(formula, data, target, evaluator=resolved_evaluator)
        except EvaluationError as exc:
            return AdmissionDecision(False, str(exc), None, 0.0)

        evaluated = result.evaluated
        metrics = result.metrics
        record = FactorRecord(
            tokens=evaluated.parsed.tokens,
            canonical=evaluated.parsed.canonical,
            signal=evaluated.signal,
            metrics=metrics,
            role=role,
        )

        if record.canonical in pool.canonicals():
            return AdmissionDecision(False, "duplicate_canonical", record, 0.0)

        weakest_index = pool.weakest_index() if len(pool) >= pool.max_size else None

        same_role_replace_index = None
        if is_cross_sectional_frame(data) and normalized_role == "target_price":
            same_role_indices = [
                index
                for index, existing in enumerate(pool.records)
                if normalize_role(existing.role) == normalized_role
            ]
            if same_role_indices:
                same_role_replace_index = min(
                    same_role_indices,
                    key=lambda index: float(pool.records[index].metrics.get("rank_ic", 0.0)),
                )

        replacement_decision = None
        if same_role_replace_index is not None:
            replacement_decision = self._screen_candidate_path(
                record,
                pool,
                profile,
                normalized_role=normalized_role,
                cross_sectional=is_cross_sectional_frame(data),
                replace_index=same_role_replace_index,
            )
            if replacement_decision.accepted:
                pool.replace(same_role_replace_index, record)
                return replacement_decision

        append_decision = self._screen_candidate_path(
            record,
            pool,
            profile,
            normalized_role=normalized_role,
            cross_sectional=is_cross_sectional_frame(data),
            replace_index=weakest_index if len(pool) >= pool.max_size else None,
        )
        if append_decision.accepted:
            if append_decision.replaced_canonical is not None and weakest_index is not None and len(pool) >= pool.max_size:
                pool.replace(weakest_index, record)
            else:
                pool.add(record)
        if replacement_decision is not None and append_decision.reason == "fast_ic_screen":
            return replacement_decision
        return append_decision

    def _screen_candidate_path(
        self,
        record: FactorRecord,
        pool: FactorPool,
        profile: RoleProfile,
        *,
        normalized_role: str | None,
        cross_sectional: bool,
        replace_index: int | None,
    ) -> AdmissionDecision:
        baseline_trade_proxy = pool.trade_proxy_score()
        new_trade_proxy = pool.trade_proxy_with(record, replace_index=replace_index)
        trade_proxy_gain = new_trade_proxy - baseline_trade_proxy
        same_family_replacement = (
            cross_sectional
            and normalized_role == "target_price"
            and replace_index is not None
            and normalize_role(pool.records[replace_index].role) == normalized_role
        )
        baseline_replacement_gain = (
            pool.baseline_replacement_gain(record, replace_index) if same_family_replacement else float("-inf")
        )
        baseline_replacement_ok = same_family_replacement and baseline_replacement_gain > 0.0

        rank_ic = float(record.metrics.get("rank_ic", 0.0))
        meets_rank_gate = np.isfinite(rank_ic) and abs(rank_ic) >= profile.commit_min_abs_rank_ic
        meets_trade_proxy_gate = trade_proxy_gain > profile.commit_min_trade_proxy_gain
        if not meets_rank_gate:
            if normalized_role == "target_flow" and meets_trade_proxy_gate:
                pass
            elif baseline_replacement_ok:
                pass
            else:
                return AdmissionDecision(
                    False,
                    "fast_ic_screen",
                    record,
                    0.0,
                    trade_proxy_gain=trade_proxy_gain,
                    baseline_trade_proxy=baseline_trade_proxy,
                    new_trade_proxy=new_trade_proxy,
                )

        corr_pool = pool if replace_index is None else self._pool_without_index(pool, replace_index)
        current_signals = corr_pool.signals_frame()
        max_corr = max_abs_correlation(record.signal, current_signals, method="spearman")
        record.metrics["max_corr"] = max_corr
        if max_corr >= profile.commit_max_correlation:
            return AdmissionDecision(
                False,
                "correlation_check",
                record,
                0.0,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )

        baseline = pool.pool_score()
        new_score = pool.score_with(record, replace_index=replace_index)
        marginal_gain = new_score - baseline
        effective_gain = baseline_replacement_gain if baseline_replacement_ok else marginal_gain
        requires_trade_proxy_gate = normalized_role == "context" and profile.commit_min_trade_proxy_gain > 0.0
        if replace_index is None and len(pool) < pool.max_size:
            if marginal_gain <= 0.0:
                return AdmissionDecision(
                    False,
                    "full_validation",
                    record,
                    marginal_gain,
                    trade_proxy_gain=trade_proxy_gain,
                    baseline_trade_proxy=baseline_trade_proxy,
                    new_trade_proxy=new_trade_proxy,
                )
            if requires_trade_proxy_gate and trade_proxy_gain <= profile.commit_min_trade_proxy_gain:
                return AdmissionDecision(
                    False,
                    "trade_proxy_check",
                    record,
                    marginal_gain,
                    trade_proxy_gain=trade_proxy_gain,
                    baseline_trade_proxy=baseline_trade_proxy,
                    new_trade_proxy=new_trade_proxy,
                )
            return AdmissionDecision(
                True,
                "accepted",
                record,
                marginal_gain,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )

        if same_family_replacement and baseline_replacement_ok:
            replaced = pool.records[replace_index].canonical
            return AdmissionDecision(
                True,
                "replaced_baseline",
                record,
                effective_gain,
                replaced_canonical=replaced,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )

        if effective_gain <= profile.replacement_margin:
            return AdmissionDecision(
                False,
                "replacement_check",
                record,
                effective_gain,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )
        if requires_trade_proxy_gate and trade_proxy_gain <= profile.commit_min_trade_proxy_gain:
            return AdmissionDecision(
                False,
                "trade_proxy_check",
                record,
                marginal_gain,
                trade_proxy_gain=trade_proxy_gain,
                baseline_trade_proxy=baseline_trade_proxy,
                new_trade_proxy=new_trade_proxy,
            )
        replaced = pool.records[replace_index].canonical if replace_index is not None else None
        return AdmissionDecision(
            True,
            "replaced" if replace_index is not None else "accepted",
            record,
            effective_gain,
            replaced_canonical=replaced,
            trade_proxy_gain=trade_proxy_gain,
            baseline_trade_proxy=baseline_trade_proxy,
            new_trade_proxy=new_trade_proxy,
        )

    @staticmethod
    def _pool_without_index(pool: FactorPool, remove_index: int) -> FactorPool:
        clone = pool.copy()
        clone.records = [
            record
            for index, record in enumerate(clone.records)
            if index != remove_index
        ]
        return clone
