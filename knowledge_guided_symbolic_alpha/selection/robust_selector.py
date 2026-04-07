from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..domain.feature_registry import FEATURE_REGISTRY
from ..domain.priors import PRIOR_RULES
from ..evaluation.panel_dispatch import evaluate_formula_metrics, score_signal_metrics
from ..generation import FormulaCandidate


@dataclass(frozen=True)
class RobustSelectorConfig:
    selection_mode: str = "pareto"
    top_k: int = 3
    slice_count: int = 4
    pairwise_margin: float = 1e-4
    min_valid_slices: int = 2
    min_mean_rank_ic: float = 0.0
    min_slice_rank_ic: float = -0.01
    enable_admissibility_gate: bool = True
    enable_near_neighbor_tie_break: bool = True
    enable_redundancy_gate: bool = True
    rank_ic_std_penalty: float = 0.35
    min_rank_ic_bonus: float = 0.55
    positive_slice_bonus: float = 0.02
    sharpe_weight: float = 0.03
    annual_return_weight: float = 0.02
    turnover_weight: float = 0.015
    complexity_weight: float = 0.0015
    stability_weight: float = 0.10
    min_subset_improvement: float = 0.002
    max_pairwise_signal_corr: float = 0.93
    subset_corr_penalty: float = 0.030
    subset_overlap_penalty: float = 0.015
    subset_single_score_weight: float = 0.20
    near_neighbor_score_gap: float = 0.005
    near_neighbor_signal_corr: float = 0.80
    near_neighbor_token_overlap: float = 0.50
    near_neighbor_feature_family_overlap: float = 0.95
    enable_crowding_distance: bool = True
    max_complexity: int | None = None
    sharpe_scale: float | None = None
    annual_return_scale: float | None = None


@dataclass(frozen=True)
class RobustSelectorRecord:
    formula: str
    source: str
    role: str | None
    selected: bool
    admissible: bool
    robust_score: float
    pairwise_wins: int
    full_metrics: dict[str, float | None]
    slice_rank_ic: list[float | None]
    slice_sharpe: list[float | None]
    slice_turnover: list[float | None]
    diagnostics: dict[str, float | int | None]
    temporal_objective_vector: dict[str, float | None] | None = None
    temporal_pareto_rank: int | None = None
    temporal_tiebreak_rank: int | None = None
    temporal_front_size: int | None = None
    temporal_front_share: float | None = None
    temporal_crowding_distance: float | None = None
    used_near_neighbor_tiebreak: bool = False


@dataclass(frozen=True)
class RobustSelectorOutcome:
    selected_formulas: list[str]
    fallback_used: bool
    records: list[RobustSelectorRecord]
    config: dict[str, float | int | str | None]


@dataclass(frozen=True)
class CachedFormulaEvaluation:
    full_metrics: dict[str, float | None]
    slice_rank_ic: list[float | None]
    slice_sharpe: list[float | None]
    slice_turnover: list[float | None]
    signal: pd.Series


@dataclass(frozen=True)
class RobustScoreScaleStats:
    sharpe_scale: float
    annual_return_scale: float
    sharpe_std: float | None = None
    annual_return_std: float | None = None
    candidate_count: int = 0


class FormulaEvaluationCache:
    def __init__(
        self,
        evaluator: Callable[..., object] | None = None,
    ) -> None:
        self._evaluator = evaluator or evaluate_formula_metrics
        self._cache: dict[tuple[str, str, int], CachedFormulaEvaluation] = {}
        self.hits = 0
        self.misses = 0

    def get(
        self,
        formula: str,
        frame: pd.DataFrame,
        target: pd.Series,
        *,
        slice_count: int,
        context_key: str | None = None,
    ) -> CachedFormulaEvaluation:
        resolved_context = context_key or _default_evaluation_context(frame)
        cache_key = (resolved_context, formula, int(slice_count))
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        full_evaluation = self._evaluator(formula, frame, target)
        full_metrics = {
            key: _finite_or_none(value)
            for key, value in full_evaluation.metrics.items()
            if isinstance(value, (int, float))
        }
        signal = pd.Series(full_evaluation.evaluated.signal, index=frame.index, dtype="float32")
        slice_rank_ic: list[float | None] = []
        slice_sharpe: list[float | None] = []
        slice_turnover: list[float | None] = []
        for slice_frame in _temporal_slices(frame, slice_count):
            slice_target = target.loc[slice_frame.index]
            try:
                slice_metrics = self._evaluator(formula, slice_frame, slice_target).metrics
            except Exception:
                slice_rank_ic.append(None)
                slice_sharpe.append(None)
                slice_turnover.append(None)
                continue
            slice_rank_ic.append(_finite_or_none(slice_metrics.get("rank_ic")))
            slice_sharpe.append(_finite_or_none(slice_metrics.get("sharpe")))
            slice_turnover.append(_finite_or_none(slice_metrics.get("turnover")))

        cached = CachedFormulaEvaluation(
            full_metrics=full_metrics,
            slice_rank_ic=slice_rank_ic,
            slice_sharpe=slice_sharpe,
            slice_turnover=slice_turnover,
            signal=signal,
        )
        self._cache[cache_key] = cached
        return cached

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._cache), "hits": self.hits, "misses": self.misses}


def estimate_robust_score_scales(
    candidates: list[FormulaCandidate],
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    slice_count: int = 4,
    evaluation_cache: FormulaEvaluationCache | None = None,
    context_key: str | None = None,
) -> RobustScoreScaleStats:
    mean_sharpes: list[float] = []
    annual_returns: list[float] = []
    for candidate in candidates:
        try:
            if evaluation_cache is not None:
                cached = evaluation_cache.get(
                    candidate.formula,
                    frame,
                    target,
                    slice_count=slice_count,
                    context_key=context_key,
                )
                full_metrics = cached.full_metrics
                slice_sharpe = list(cached.slice_sharpe)
            else:
                evaluation = evaluate_formula_metrics(candidate.formula, frame, target)
                full_metrics = evaluation.metrics
                slice_sharpe = []
                for slice_frame in _temporal_slices(frame, slice_count):
                    slice_target = target.loc[slice_frame.index]
                    try:
                        slice_metrics = evaluate_formula_metrics(candidate.formula, slice_frame, slice_target).metrics
                    except Exception:
                        continue
                    slice_value = _finite_or_none(slice_metrics.get("sharpe"))
                    if slice_value is not None:
                        slice_sharpe.append(slice_value)
        except Exception:
            continue
        valid_sharpe = np.asarray([value for value in slice_sharpe if value is not None], dtype=float)
        full_sharpe = _finite_or_none(full_metrics.get("sharpe"))
        annual_return = _finite_or_none(full_metrics.get("annual_return"))
        mean_sharpe = float(valid_sharpe.mean()) if valid_sharpe.size else (full_sharpe if full_sharpe is not None else None)
        if mean_sharpe is not None and np.isfinite(mean_sharpe):
            mean_sharpes.append(float(mean_sharpe))
        if annual_return is not None and np.isfinite(annual_return):
            annual_returns.append(float(annual_return))
    sharpe_scale, sharpe_std = _empirical_scale(mean_sharpes)
    annual_return_scale, annual_return_std = _empirical_scale(annual_returns)
    return RobustScoreScaleStats(
        sharpe_scale=sharpe_scale,
        annual_return_scale=annual_return_scale,
        sharpe_std=sharpe_std,
        annual_return_std=annual_return_std,
        candidate_count=len(candidates),
    )


class TemporalRobustSelector:
    def __init__(
        self,
        config: RobustSelectorConfig | None = None,
        evaluation_cache: FormulaEvaluationCache | None = None,
    ) -> None:
        self.config = config or RobustSelectorConfig()
        self.evaluation_cache = evaluation_cache
        self._active_evaluation_context: str | None = None

    def select(
        self,
        candidates: list[FormulaCandidate],
        frame: pd.DataFrame,
        target: pd.Series,
        *,
        evaluation_context: str | None = None,
    ) -> RobustSelectorOutcome:
        previous_context = self._active_evaluation_context
        self._active_evaluation_context = evaluation_context
        try:
            records = [self._evaluate_candidate(candidate, frame, target) for candidate in candidates]
            fallback_used = len([record for record in records if record.admissible]) == 0
            pool = [record for record in records if record.admissible] or records
            if self.config.selection_mode == "legacy_linear":
                pairwise_wins = self._legacy_pairwise_wins(records)
                selected_evaluations, subset_score = self._legacy_select_subset(pool, pairwise_wins, frame, target)
                rank_map, tiebreak_map = self._legacy_rank_maps(records, pairwise_wins)
                ranking_data = _ParetoRankingData(
                    rank_map=rank_map,
                    tiebreak_map=tiebreak_map,
                    front_size_map={record.formula: 1 for record in records},
                    front_share_map={record.formula: 1.0 / max(len(records), 1) for record in records},
                    crowding_map={record.formula: float("nan") for record in records},
                )
                used_near_neighbor_tiebreak = False
            else:
                pairwise_wins = self._temporal_pairwise_wins(records)
                ranking_data = self._temporal_rank_data(records)
                selected_evaluations, used_near_neighbor_tiebreak = self._select_pareto(pool, ranking_data)
                subset_score = None
            selected_formulas = [item.formula for item in selected_evaluations]
            selected_lookup = set(selected_formulas)
            finalized = [
                RobustSelectorRecord(
                    formula=record.formula,
                    source=record.source,
                    role=record.role,
                    selected=record.formula in selected_lookup,
                    admissible=record.admissible,
                    robust_score=record.robust_score,
                    pairwise_wins=pairwise_wins.get(record.formula, 0),
                    full_metrics={
                        key: value for key, value in record.full_metrics.items() if not isinstance(value, str)
                    },
                    slice_rank_ic=record.slice_rank_ic,
                    slice_sharpe=record.slice_sharpe,
                    slice_turnover=record.slice_turnover,
                    diagnostics={
                        **record.diagnostics,
                        "max_corr_to_selected": self._max_corr_to_selected(record, selected_evaluations),
                        "avg_token_overlap_to_selected": self._avg_token_overlap_to_selected(record, selected_evaluations),
                        "subset_score": subset_score if record.formula in selected_lookup else None,
                    },
                    temporal_objective_vector=record.temporal_objective_vector,
                    temporal_pareto_rank=ranking_data.rank_map.get(record.formula),
                    temporal_tiebreak_rank=ranking_data.tiebreak_map.get(record.formula),
                    temporal_front_size=ranking_data.front_size_map.get(record.formula),
                    temporal_front_share=ranking_data.front_share_map.get(record.formula),
                    temporal_crowding_distance=ranking_data.crowding_map.get(record.formula),
                    used_near_neighbor_tiebreak=bool(
                        used_near_neighbor_tiebreak and record.formula in selected_lookup
                    ),
                )
                for record in records
            ]
            finalized.sort(
                key=lambda item: (
                    item.formula not in selected_lookup,
                    item.temporal_pareto_rank if item.temporal_pareto_rank is not None else 10**9,
                    _crowding_order_value(item.temporal_crowding_distance),
                    item.temporal_tiebreak_rank if item.temporal_tiebreak_rank is not None else 10**9,
                    -item.pairwise_wins,
                    -item.robust_score,
                    item.formula,
                )
            )
            return RobustSelectorOutcome(
                selected_formulas=selected_formulas,
                fallback_used=fallback_used,
                records=finalized,
                config={
                    "selection_mode": self.config.selection_mode,
                    "top_k": self.config.top_k,
                    "selected_count": len(selected_formulas),
                    "slice_count": self.config.slice_count,
                    "min_valid_slices": self.config.min_valid_slices,
                    "min_mean_rank_ic": self.config.min_mean_rank_ic,
                    "min_slice_rank_ic": self.config.min_slice_rank_ic,
                    "subset_score": subset_score,
                },
            )
        finally:
            self._active_evaluation_context = previous_context

    def _evaluate_candidate(
        self,
        candidate: FormulaCandidate,
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> _CandidateEvaluation:
        try:
            if self.evaluation_cache is not None:
                cached = self.evaluation_cache.get(
                    candidate.formula,
                    frame,
                    target,
                    slice_count=self.config.slice_count,
                    context_key=self._active_evaluation_context,
                )
                full_metrics = cached.full_metrics
                slice_rank_ic = list(cached.slice_rank_ic)
                slice_sharpe = list(cached.slice_sharpe)
                slice_turnover = list(cached.slice_turnover)
                signal = cached.signal
            else:
                full_evaluation = evaluate_formula_metrics(candidate.formula, frame, target)
                full_metrics = full_evaluation.metrics
                signal = full_evaluation.evaluated.signal
                slice_rank_ic = []
                slice_sharpe = []
                slice_turnover = []
        except Exception as exc:  # pragma: no cover - guard against evaluator edge cases
            return _CandidateEvaluation(
                formula=candidate.formula,
                source=candidate.source,
                role=candidate.role,
                admissible=False,
                robust_score=float("-inf"),
                full_metrics={"error": str(exc)},
                slice_rank_ic=[],
                slice_sharpe=[],
                slice_turnover=[],
                signal=pd.Series(dtype=float),
                diagnostics={"valid_slices": 0, "mean_rank_ic": None, "min_rank_ic": None, "complexity": len(candidate.formula.split())},
            )

        if self.evaluation_cache is None:
            for slice_frame in _temporal_slices(frame, self.config.slice_count):
                slice_target = target.loc[slice_frame.index]
                try:
                    slice_metrics = evaluate_formula_metrics(candidate.formula, slice_frame, slice_target).metrics
                except Exception:
                    slice_rank_ic.append(None)
                    slice_sharpe.append(None)
                    slice_turnover.append(None)
                    continue
                slice_rank_ic.append(_finite_or_none(slice_metrics.get("rank_ic")))
                slice_sharpe.append(_finite_or_none(slice_metrics.get("sharpe")))
                slice_turnover.append(_finite_or_none(slice_metrics.get("turnover")))

        valid_rank_ic = np.asarray([value for value in slice_rank_ic if value is not None], dtype=float)
        valid_sharpe = np.asarray([value for value in slice_sharpe if value is not None], dtype=float)
        valid_turnover = np.asarray([value for value in slice_turnover if value is not None], dtype=float)
        full_sharpe = _finite_or_none(full_metrics.get("sharpe")) or 0.0
        full_return = _finite_or_none(full_metrics.get("annual_return")) or 0.0
        full_turnover = max(0.0, _finite_or_none(full_metrics.get("turnover")) or 0.0)
        full_stability = _finite_or_none(full_metrics.get("stability_score")) or 0.0
        valid_slices = int(valid_rank_ic.size)
        mean_rank_ic = float(valid_rank_ic.mean()) if valid_rank_ic.size else float("-inf")
        rank_ic_std = float(valid_rank_ic.std(ddof=0)) if valid_rank_ic.size else float("inf")
        min_rank_ic = float(valid_rank_ic.min()) if valid_rank_ic.size else float("-inf")
        positive_slice_frac = float(np.mean(valid_rank_ic > 0.0)) if valid_rank_ic.size else 0.0
        mean_sharpe = float(valid_sharpe.mean()) if valid_sharpe.size else full_sharpe
        mean_turnover = float(valid_turnover.mean()) if valid_turnover.size else full_turnover
        complexity = float(max(1, len(candidate.formula.split())))
        admissible = True
        if self.config.enable_admissibility_gate:
            admissible = (
                valid_slices >= self.config.min_valid_slices
                and mean_rank_ic >= self.config.min_mean_rank_ic
                and min_rank_ic >= self.config.min_slice_rank_ic
            )
        if self.config.max_complexity is not None:
            admissible = admissible and complexity <= float(self.config.max_complexity)
        sharpe_scale = _resolve_score_scale(self.config.sharpe_scale, fallback=2.0)
        annual_return_scale = _resolve_score_scale(self.config.annual_return_scale, fallback=0.20)
        robust_score = (
            mean_rank_ic
            + self.config.min_rank_ic_bonus * min_rank_ic
            - self.config.rank_ic_std_penalty * rank_ic_std
            + self.config.positive_slice_bonus * positive_slice_frac
            + self.config.sharpe_weight * np.tanh(mean_sharpe / sharpe_scale)
            + self.config.annual_return_weight * np.tanh(full_return / annual_return_scale)
            + self.config.stability_weight * max(min(full_stability, 0.05), -0.05)
            - self.config.turnover_weight * mean_turnover
            - self.config.complexity_weight * complexity
        )
        objective_vector = {
            "mean_rank_ic": mean_rank_ic if np.isfinite(mean_rank_ic) else None,
            "min_slice_rank_ic": min_rank_ic if np.isfinite(min_rank_ic) else None,
            "rank_ic_std": rank_ic_std if np.isfinite(rank_ic_std) else None,
            "turnover": mean_turnover if np.isfinite(mean_turnover) else None,
            "complexity": complexity if np.isfinite(complexity) else None,
        }
        return _CandidateEvaluation(
            formula=candidate.formula,
            source=candidate.source,
            role=candidate.role,
            admissible=admissible,
            robust_score=float(robust_score),
            full_metrics={
                key: _finite_or_none(value)
                for key, value in full_metrics.items()
                if isinstance(value, (int, float))
            },
            slice_rank_ic=slice_rank_ic,
            slice_sharpe=slice_sharpe,
            slice_turnover=slice_turnover,
            signal=pd.Series(signal, index=frame.index, dtype=float),
            diagnostics={
                "valid_slices": valid_slices,
                "mean_rank_ic": mean_rank_ic if np.isfinite(mean_rank_ic) else None,
                "rank_ic_std": rank_ic_std if np.isfinite(rank_ic_std) else None,
                "min_rank_ic": min_rank_ic if np.isfinite(min_rank_ic) else None,
                "positive_slice_frac": positive_slice_frac,
                "mean_sharpe": mean_sharpe if np.isfinite(mean_sharpe) else None,
                "mean_turnover": mean_turnover if np.isfinite(mean_turnover) else None,
                "knowledge_alignment": _knowledge_alignment(candidate.formula),
                "temporal_granularity_score": _temporal_granularity_score(candidate.formula),
                "complexity": complexity,
                "sharpe_scale": sharpe_scale,
                "annual_return_scale": annual_return_scale,
            },
            temporal_objective_vector=objective_vector,
        )

    def _temporal_pairwise_wins(self, records: list[_CandidateEvaluation]) -> dict[str, int]:
        wins = {record.formula: 0 for record in records}
        for left_index, left in enumerate(records):
            for right_index, right in enumerate(records):
                if left_index == right_index:
                    continue
                if self._dominates_temporal(left, right):
                    wins[left.formula] += 1
        return wins

    def _temporal_rank_data(self, records: list[_CandidateEvaluation]) -> _ParetoRankingData:
        dominates = self._dominates_temporal
        objective_directions = self._temporal_objective_directions()
        fronts = _pareto_fronts(records, dominates)
        rank_map: dict[str, int] = {}
        tiebreak_map: dict[str, int] = {}
        front_size_map: dict[str, int] = {}
        front_share_map: dict[str, float] = {}
        crowding_map: dict[str, float] = {}
        next_rank = 1
        total = max(len(records), 1)
        for front_index, front in enumerate(fronts, start=1):
            if self._uses_crowding_distance():
                crowding = _crowding_distances(
                    front,
                    objective_directions=objective_directions,
                    vector_getter=lambda item: item.temporal_objective_vector,
                )
                ordered = sorted(front, key=lambda item: (_crowding_order_value(crowding.get(item.formula)), *self._temporal_tiebreak_sort_key(item)))
            else:
                crowding = {item.formula: float("nan") for item in front}
                ordered = sorted(front, key=self._temporal_tiebreak_sort_key)
            for item in ordered:
                rank_map[item.formula] = front_index
                tiebreak_map[item.formula] = next_rank
                front_size_map[item.formula] = len(front)
                front_share_map[item.formula] = float(len(front) / total)
                crowding_map[item.formula] = float(crowding.get(item.formula, float("nan")))
                next_rank += 1
        return _ParetoRankingData(
            rank_map=rank_map,
            tiebreak_map=tiebreak_map,
            front_size_map=front_size_map,
            front_share_map=front_share_map,
            crowding_map=crowding_map,
        )

    def _select_pareto(
        self,
        pool: list[_CandidateEvaluation],
        ranking_data: _ParetoRankingData,
    ) -> tuple[list[_CandidateEvaluation], bool]:
        if not pool:
            return [], False
        ordered = sorted(
            pool,
            key=lambda item: (
                ranking_data.rank_map.get(item.formula, 10**9),
                _crowding_order_value(ranking_data.crowding_map.get(item.formula)),
                ranking_data.tiebreak_map.get(item.formula, 10**9),
                item.formula,
            ),
        )
        best = ordered[0]
        used_near_neighbor_tiebreak = False
        if not self.config.enable_near_neighbor_tie_break:
            return [best], used_near_neighbor_tiebreak
        best_rank = ranking_data.rank_map.get(best.formula)
        best_crowding = ranking_data.crowding_map.get(best.formula)
        cluster = [
            item
            for item in ordered
            if ranking_data.rank_map.get(item.formula) == best_rank
            and _same_crowding_bucket(ranking_data.crowding_map.get(item.formula), best_crowding)
            and self._is_near_neighbor_pareto(best, item)
        ]
        if not cluster:
            return [best], used_near_neighbor_tiebreak
        preferred = max(cluster, key=self._pareto_near_neighbor_priority)
        used_near_neighbor_tiebreak = preferred.formula != best.formula
        return [preferred], used_near_neighbor_tiebreak

    def _is_near_neighbor_pareto(self, anchor: _CandidateEvaluation, candidate: _CandidateEvaluation) -> bool:
        if anchor.formula == candidate.formula:
            return True
        if abs(anchor.robust_score - candidate.robust_score) > self.config.near_neighbor_score_gap:
            return False
        if _feature_family_overlap(anchor.formula, candidate.formula) < self.config.near_neighbor_feature_family_overlap:
            return False
        if _token_overlap(anchor.formula, candidate.formula) < self.config.near_neighbor_token_overlap:
            return False
        return _spearman_abs_corr(anchor.signal, candidate.signal) >= self.config.near_neighbor_signal_corr

    def _pareto_near_neighbor_priority(self, record: _CandidateEvaluation) -> tuple[float, ...]:
        return (
            float(record.diagnostics.get("knowledge_alignment") or 0.0),
            float(record.diagnostics.get("temporal_granularity_score") or 0.0),
            float(record.diagnostics.get("min_rank_ic") or float("-inf")),
            float(record.diagnostics.get("mean_rank_ic") or float("-inf")),
            -float(record.diagnostics.get("rank_ic_std") or float("inf")),
            -float(record.diagnostics.get("mean_turnover") or float("inf")),
            -float(record.diagnostics.get("complexity") or float("inf")),
            float(record.robust_score),
        )

    def _temporal_tiebreak_sort_key(self, record: _CandidateEvaluation) -> tuple[float, ...]:
        return (
            -_objective_sort_value(record.diagnostics.get("min_rank_ic"), maximize=True),
            _objective_sort_value(record.diagnostics.get("rank_ic_std"), maximize=False),
            -_objective_sort_value(record.diagnostics.get("mean_rank_ic"), maximize=True),
            _objective_sort_value(record.diagnostics.get("mean_turnover"), maximize=False),
            _objective_sort_value(record.diagnostics.get("complexity"), maximize=False),
            record.formula,
        )

    def _uses_crowding_distance(self) -> bool:
        return self.config.selection_mode == "pareto" and self.config.enable_crowding_distance

    def _temporal_objective_directions(self) -> tuple[tuple[str, bool], ...]:
        if self.config.selection_mode == "pareto_discrete_legacy":
            return (
                ("mean_rank_ic", True),
                ("min_slice_rank_ic", True),
                ("rank_ic_std", False),
                ("turnover", False),
                ("complexity", False),
            )
        return (
            ("mean_rank_ic", True),
            ("min_slice_rank_ic", True),
            ("rank_ic_std", False),
            ("turnover", False),
        )

    def _dominates_temporal(self, left: _CandidateEvaluation, right: _CandidateEvaluation) -> bool:
        return _dominates_objectives(
            left.temporal_objective_vector,
            right.temporal_objective_vector,
            objective_directions=self._temporal_objective_directions(),
        )

    def _legacy_pairwise_wins(self, records: list[_CandidateEvaluation]) -> dict[str, int]:
        wins = {record.formula: 0 for record in records}
        for left_index, left in enumerate(records):
            for right_index, right in enumerate(records):
                if left_index == right_index:
                    continue
                if self._legacy_dominates(left, right):
                    wins[left.formula] += 1
        return wins

    def _legacy_dominates(self, left: _CandidateEvaluation, right: _CandidateEvaluation) -> bool:
        if left.robust_score <= right.robust_score + self.config.pairwise_margin:
            return False
        left_min_rank_ic = left.diagnostics.get("min_rank_ic")
        right_min_rank_ic = right.diagnostics.get("min_rank_ic")
        left_turnover = left.diagnostics.get("mean_turnover")
        right_turnover = right.diagnostics.get("mean_turnover")
        if (
            left_min_rank_ic is not None
            and right_min_rank_ic is not None
            and float(left_min_rank_ic) + self.config.pairwise_margin < float(right_min_rank_ic)
        ):
            return False
        if (
            left_turnover is not None
            and right_turnover is not None
            and float(left_turnover) > float(right_turnover) + 0.10
        ):
            return False
        return True

    def _legacy_rank_maps(self, records: list[_CandidateEvaluation], pairwise_wins: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
        ordered = sorted(
            records,
            key=lambda item: (
                pairwise_wins.get(item.formula, 0),
                item.robust_score,
                float(item.full_metrics.get("rank_ic") or float("-inf")),
                float(item.full_metrics.get("sharpe") or float("-inf")),
            ),
            reverse=True,
        )
        rank_map: dict[str, int] = {}
        tiebreak_map: dict[str, int] = {}
        for index, item in enumerate(ordered, start=1):
            rank_map[item.formula] = index
            tiebreak_map[item.formula] = index
        return rank_map, tiebreak_map

    def _legacy_select_subset(
        self,
        pool: list[_CandidateEvaluation],
        pairwise_wins: dict[str, int],
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[list[_CandidateEvaluation], float | None]:
        ordered = sorted(
            pool,
            key=lambda item: (
                pairwise_wins.get(item.formula, 0),
                item.robust_score,
                float(item.full_metrics.get("rank_ic") or float("-inf")),
                float(item.full_metrics.get("sharpe") or float("-inf")),
            ),
            reverse=True,
        )
        if not ordered:
            return [], None
        single_scores = {item.formula: self._subset_score([item], frame, target) for item in ordered}
        best_single = self._legacy_choose_best_single(ordered, single_scores)
        selected = [best_single]
        current_score = single_scores.get(best_single.formula, float("-inf"))
        while len(selected) < self.config.top_k:
            best_candidate = None
            best_gain = float("-inf")
            best_score = current_score
            for candidate in ordered:
                if candidate.formula in {item.formula for item in selected}:
                    continue
                max_corr = self._max_signal_corr(candidate, selected)
                if self.config.enable_redundancy_gate and max_corr >= self.config.max_pairwise_signal_corr:
                    continue
                trial = [*selected, candidate]
                trial_score = self._subset_score(trial, frame, target)
                gain = trial_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_score = trial_score
                    best_candidate = candidate
            if best_candidate is None or best_gain <= self.config.min_subset_improvement:
                break
            selected.append(best_candidate)
            current_score = best_score
        return selected, current_score

    def _legacy_choose_best_single(
        self,
        ordered: list[_CandidateEvaluation],
        single_scores: dict[str, float],
    ) -> _CandidateEvaluation:
        best = max(
            ordered,
            key=lambda item: (
                single_scores.get(item.formula, float("-inf")),
                item.robust_score,
            ),
        )
        while True:
            cluster = [item for item in ordered if self._is_near_neighbor_legacy(best, item, single_scores)]
            if not cluster:
                return best
            preferred = max(cluster, key=lambda item: self._legacy_near_neighbor_priority(item, single_scores))
            if preferred.formula == best.formula:
                return best
            best = preferred

    def _is_near_neighbor_legacy(
        self,
        anchor: _CandidateEvaluation,
        candidate: _CandidateEvaluation,
        single_scores: dict[str, float],
    ) -> bool:
        if anchor.formula == candidate.formula:
            return True
        score_gap = abs(
            single_scores.get(anchor.formula, float("-inf"))
            - single_scores.get(candidate.formula, float("-inf"))
        )
        if not np.isfinite(score_gap) or score_gap > self.config.near_neighbor_score_gap:
            return False
        if _feature_family_overlap(anchor.formula, candidate.formula) < self.config.near_neighbor_feature_family_overlap:
            return False
        if _token_overlap(anchor.formula, candidate.formula) < self.config.near_neighbor_token_overlap:
            return False
        return _spearman_abs_corr(anchor.signal, candidate.signal) >= self.config.near_neighbor_signal_corr

    def _legacy_near_neighbor_priority(
        self,
        record: _CandidateEvaluation,
        single_scores: dict[str, float],
    ) -> tuple[float, ...]:
        return (
            float(record.diagnostics.get("knowledge_alignment") or 0.0),
            float(record.full_metrics.get("rank_ic") or float("-inf")),
            float(record.diagnostics.get("mean_rank_ic") or float("-inf")),
            float(record.diagnostics.get("temporal_granularity_score") or 0.0),
            -float(record.diagnostics.get("rank_ic_std") or float("inf")),
            float(record.diagnostics.get("min_rank_ic") or float("-inf")),
            -float(record.diagnostics.get("mean_turnover") or float("inf")),
            single_scores.get(record.formula, float("-inf")),
            record.robust_score,
        )

    def _subset_score(
        self,
        selected: list[_CandidateEvaluation],
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        fused_signal = _fuse_candidate_signals(selected, frame)
        if fused_signal.dropna().empty:
            return float("-inf")
        full_metrics = score_signal_metrics(fused_signal, frame.loc[fused_signal.index], target.loc[fused_signal.index])
        slice_rank_ic: list[float] = []
        for slice_frame in _temporal_slices(frame.loc[fused_signal.index], self.config.slice_count):
            slice_signal = fused_signal.loc[slice_frame.index]
            slice_target = target.loc[slice_frame.index]
            try:
                slice_metrics = score_signal_metrics(slice_signal, slice_frame, slice_target)
            except Exception:
                continue
            rank_ic = _finite_or_none(slice_metrics.get("rank_ic"))
            if rank_ic is not None:
                slice_rank_ic.append(rank_ic)
        valid_rank_ic = np.asarray(slice_rank_ic, dtype=float)
        mean_rank_ic = float(valid_rank_ic.mean()) if valid_rank_ic.size else float(full_metrics.get("rank_ic", float("-inf")))
        min_rank_ic = float(valid_rank_ic.min()) if valid_rank_ic.size else mean_rank_ic
        rank_ic_std = float(valid_rank_ic.std(ddof=0)) if valid_rank_ic.size else 0.0
        sharpe = _finite_or_none(full_metrics.get("sharpe")) or 0.0
        annual_return = _finite_or_none(full_metrics.get("annual_return")) or 0.0
        turnover = max(0.0, _finite_or_none(full_metrics.get("turnover")) or 0.0)
        stability = _finite_or_none(full_metrics.get("stability_score")) or 0.0
        avg_corr = self._avg_pairwise_signal_corr(selected)
        avg_overlap = self._avg_token_overlap(selected)
        mean_single_score = float(np.mean([item.robust_score for item in selected]))
        sharpe_scale = _resolve_score_scale(self.config.sharpe_scale, fallback=2.0)
        annual_return_scale = _resolve_score_scale(self.config.annual_return_scale, fallback=0.20)
        return float(
            mean_rank_ic
            + self.config.min_rank_ic_bonus * min_rank_ic
            - self.config.rank_ic_std_penalty * rank_ic_std
            + self.config.sharpe_weight * np.tanh(sharpe / sharpe_scale)
            + self.config.annual_return_weight * np.tanh(annual_return / annual_return_scale)
            + self.config.stability_weight * max(min(stability, 0.05), -0.05)
            - self.config.turnover_weight * turnover
            - self.config.subset_corr_penalty * avg_corr
            - self.config.subset_overlap_penalty * avg_overlap
            + self.config.subset_single_score_weight * mean_single_score
        )

    def _max_signal_corr(self, candidate: _CandidateEvaluation, selected: list[_CandidateEvaluation]) -> float:
        if not selected or candidate.signal.dropna().empty:
            return 0.0
        values = []
        for item in selected:
            corr = _spearman_abs_corr(candidate.signal, item.signal)
            if np.isfinite(corr):
                values.append(corr)
        return float(max(values)) if values else 0.0

    def _avg_pairwise_signal_corr(self, selected: list[_CandidateEvaluation]) -> float:
        if len(selected) <= 1:
            return 0.0
        values: list[float] = []
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                corr = _spearman_abs_corr(left.signal, right.signal)
                if np.isfinite(corr):
                    values.append(corr)
        return float(np.mean(values)) if values else 0.0

    def _avg_token_overlap(self, selected: list[_CandidateEvaluation]) -> float:
        if len(selected) <= 1:
            return 0.0
        overlaps: list[float] = []
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                overlaps.append(_token_overlap(left.formula, right.formula))
        return float(np.mean(overlaps)) if overlaps else 0.0

    def _max_corr_to_selected(
        self,
        record: _CandidateEvaluation,
        selected: list[_CandidateEvaluation],
    ) -> float | None:
        others = [item for item in selected if item.formula != record.formula]
        if not others:
            return None
        value = self._max_signal_corr(record, others)
        return value if np.isfinite(value) else None

    def _avg_token_overlap_to_selected(
        self,
        record: _CandidateEvaluation,
        selected: list[_CandidateEvaluation],
    ) -> float | None:
        others = [item for item in selected if item.formula != record.formula]
        if not others:
            return None
        overlaps = [_token_overlap(record.formula, item.formula) for item in others]
        return float(np.mean(overlaps)) if overlaps else None


@dataclass(frozen=True)
class _CandidateEvaluation:
    formula: str
    source: str
    role: str | None
    admissible: bool
    robust_score: float
    full_metrics: dict[str, float | None | str]
    slice_rank_ic: list[float | None]
    slice_sharpe: list[float | None]
    slice_turnover: list[float | None]
    signal: pd.Series
    diagnostics: dict[str, float | int | None]
    temporal_objective_vector: dict[str, float | None] | None = None


@dataclass(frozen=True)
class _ParetoRankingData:
    rank_map: dict[str, int]
    tiebreak_map: dict[str, int]
    front_size_map: dict[str, int]
    front_share_map: dict[str, float]
    crowding_map: dict[str, float]


def _finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _temporal_slices(frame: pd.DataFrame, slice_count: int) -> list[pd.DataFrame]:
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


def _default_evaluation_context(frame: pd.DataFrame) -> str:
    if "date" not in frame.columns or frame.empty:
        return f"rows={len(frame)}"
    dates = pd.to_datetime(frame["date"])
    return f"rows={len(frame)}|start={dates.min()}|end={dates.max()}"


def _empirical_scale(values: list[float]) -> tuple[float, float | None]:
    if not values:
        return 1.0, None
    arr = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return 1.0, None
    std = float(arr.std(ddof=0))
    if np.isfinite(std) and std > 0.0:
        return std, std
    abs_mean = float(np.mean(np.abs(arr)))
    if np.isfinite(abs_mean) and abs_mean > 0.0:
        return abs_mean, std if np.isfinite(std) else None
    max_abs = float(np.max(np.abs(arr)))
    if np.isfinite(max_abs) and max_abs > 0.0:
        return max_abs, std if np.isfinite(std) else None
    return 1.0, std if np.isfinite(std) else None


def _resolve_score_scale(value: float | None, *, fallback: float) -> float:
    if value is None:
        return float(fallback)
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return float(fallback)
    return value


def _spearman_abs_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    if aligned.iloc[:, 0].nunique(dropna=True) <= 1 or aligned.iloc[:, 1].nunique(dropna=True) <= 1:
        return 0.0
    left_rank = aligned.iloc[:, 0].rank(method="average")
    right_rank = aligned.iloc[:, 1].rank(method="average")
    corr = left_rank.corr(right_rank, method="pearson")
    if corr is None or not np.isfinite(corr):
        return float("nan")
    return float(abs(corr))


def _token_overlap(left_formula: str, right_formula: str) -> float:
    left = set(left_formula.split())
    right = set(right_formula.split())
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _formula_features(formula: str) -> frozenset[str]:
    return frozenset(token for token in formula.split() if token in FEATURE_REGISTRY)


def _feature_family(feature: str) -> str:
    if feature.endswith("_Q") or feature.endswith("_A"):
        return feature[:-2]
    return feature


def _feature_family_overlap(left_formula: str, right_formula: str) -> float:
    left = {_feature_family(feature) for feature in _formula_features(left_formula)}
    right = {_feature_family(feature) for feature in _formula_features(right_formula)}
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _knowledge_alignment(formula: str) -> float:
    if not PRIOR_RULES:
        return 0.0
    features = _formula_features(formula)
    matched = sum(1 for rule in PRIOR_RULES if set(rule.features).issubset(features))
    return float(matched / len(PRIOR_RULES))


def _temporal_granularity_score(formula: str) -> float:
    timed_features = [feature for feature in _formula_features(formula) if feature.endswith("_Q") or feature.endswith("_A")]
    if not timed_features:
        return 0.0
    quarterly = sum(1 for feature in timed_features if feature.endswith("_Q"))
    annual = sum(1 for feature in timed_features if feature.endswith("_A"))
    return float((quarterly - annual) / len(timed_features))


def _fuse_candidate_signals(selected: list[_CandidateEvaluation], frame: pd.DataFrame) -> pd.Series:
    if not selected:
        return pd.Series(dtype=float)
    signal_frame = pd.concat({item.formula: item.signal for item in selected}, axis=1).dropna(how="all")
    if signal_frame.empty:
        return pd.Series(dtype=float)
    weight_vector = np.asarray([max(item.robust_score, 0.0) for item in selected], dtype=float)
    if not np.isfinite(weight_vector).all() or np.allclose(weight_vector, 0.0):
        weight_vector = np.ones(len(selected), dtype=float)
    weight_vector = weight_vector / np.sum(np.abs(weight_vector))
    if "date" in frame.columns:
        dates = pd.to_datetime(frame.loc[signal_frame.index, "date"])
        processed = signal_frame.copy()
        for column in processed.columns:
            series = processed[column].replace([np.inf, -np.inf], np.nan)
            mean = series.groupby(dates).transform("mean")
            std = series.groupby(dates).transform(lambda values: values.std(ddof=0))
            processed[column] = ((series - mean) / std.replace(0.0, np.nan)).clip(-5.0, 5.0).fillna(0.0)
        fused = processed.to_numpy() @ weight_vector
        return pd.Series(fused, index=processed.index, name="fused_signal")
    processed = signal_frame.replace([np.inf, -np.inf], np.nan)
    for column in processed.columns:
        series = processed[column]
        mean = series.mean()
        std = series.std(ddof=0)
        processed[column] = 0.0 if (not np.isfinite(std) or std == 0.0) else ((series - mean) / std).clip(-5.0, 5.0)
    fused = processed.fillna(0.0).to_numpy() @ weight_vector
    return pd.Series(fused, index=processed.index, name="fused_signal")


def _objective_sort_value(value: float | int | None, *, maximize: bool) -> float:
    if value is None or not np.isfinite(float(value)):
        return float("-inf") if maximize else float("inf")
    return float(value)


def _same_crowding_bucket(left: float | None, right: float | None, *, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return False
    left_value = float(left)
    right_value = float(right)
    if np.isinf(left_value) and np.isinf(right_value):
        return True
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return False
    return abs(left_value - right_value) <= tolerance


def _crowding_order_value(value: float | None) -> float:
    if value is None:
        return float("inf")
    numeric = float(value)
    if np.isnan(numeric):
        return float("inf")
    if np.isinf(numeric):
        return float("-inf")
    return -numeric


def _dominates_objectives(
    left: dict[str, float | None] | None,
    right: dict[str, float | None] | None,
    *,
    objective_directions: tuple[tuple[str, bool], ...],
) -> bool:
    if not left or not right:
        return False
    better_or_equal = True
    strictly_better = False
    for key, maximize in objective_directions:
        left_value = _objective_sort_value(left.get(key), maximize=maximize)
        right_value = _objective_sort_value(right.get(key), maximize=maximize)
        if maximize:
            if left_value < right_value:
                better_or_equal = False
                break
            if left_value > right_value:
                strictly_better = True
        else:
            if left_value > right_value:
                better_or_equal = False
                break
            if left_value < right_value:
                strictly_better = True
    return better_or_equal and strictly_better


def _pareto_fronts(
    records: list[_CandidateEvaluation],
    dominates: Callable[[_CandidateEvaluation, _CandidateEvaluation], bool],
) -> list[list[_CandidateEvaluation]]:
    remaining = list(records)
    fronts: list[list[_CandidateEvaluation]] = []
    while remaining:
        front: list[_CandidateEvaluation] = []
        for candidate in remaining:
            dominated = any(
                other.formula != candidate.formula and dominates(other, candidate)
                for other in remaining
            )
            if not dominated:
                front.append(candidate)
        if not front:
            fronts.append(list(remaining))
            break
        fronts.append(front)
        front_formulas = {item.formula for item in front}
        remaining = [item for item in remaining if item.formula not in front_formulas]
    return fronts


def _crowding_distances(
    records: list[_CandidateEvaluation],
    *,
    objective_directions: tuple[tuple[str, bool], ...],
    vector_getter: Callable[[_CandidateEvaluation], dict[str, float | None] | None],
) -> dict[str, float]:
    if not records:
        return {}
    if len(records) <= 2:
        return {record.formula: float("inf") for record in records}
    distances = {record.formula: 0.0 for record in records}
    for objective, maximize in objective_directions:
        ranked: list[tuple[_CandidateEvaluation, float]] = []
        for record in records:
            vector = vector_getter(record) or {}
            value = vector.get(objective)
            if value is None or not np.isfinite(float(value)):
                continue
            scored = float(value) if maximize else -float(value)
            ranked.append((record, scored))
        if len(ranked) <= 2:
            for record, _ in ranked:
                distances[record.formula] = float("inf")
            continue
        ranked.sort(key=lambda item: item[1])
        low = ranked[0][1]
        high = ranked[-1][1]
        distances[ranked[0][0].formula] = float("inf")
        distances[ranked[-1][0].formula] = float("inf")
        if np.isclose(high, low):
            continue
        scale = high - low
        for index in range(1, len(ranked) - 1):
            formula = ranked[index][0].formula
            if np.isinf(distances[formula]):
                continue
            distances[formula] += (ranked[index + 1][1] - ranked[index - 1][1]) / scale
    return distances


TemporalSelectorConfig = RobustSelectorConfig
TemporalSelectorRecord = RobustSelectorRecord
TemporalSelectorOutcome = RobustSelectorOutcome
RobustTemporalSelector = TemporalRobustSelector
