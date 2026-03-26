from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..domain.feature_registry import FEATURE_REGISTRY
from ..domain.priors import PRIOR_RULES
from ..evaluation.panel_dispatch import evaluate_formula_metrics, score_signal_metrics
from ..generation import FormulaCandidate


@dataclass(frozen=True)
class RobustSelectorConfig:
    top_k: int = 3
    slice_count: int = 4
    pairwise_margin: float = 1e-4
    min_valid_slices: int = 2
    min_mean_rank_ic: float = 0.0
    min_slice_rank_ic: float = -0.01
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


@dataclass(frozen=True)
class RobustSelectorOutcome:
    selected_formulas: list[str]
    fallback_used: bool
    records: list[RobustSelectorRecord]
    config: dict[str, float | int]


class TemporalRobustSelector:
    def __init__(self, config: RobustSelectorConfig | None = None) -> None:
        self.config = config or RobustSelectorConfig()

    def select(
        self,
        candidates: list[FormulaCandidate],
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> RobustSelectorOutcome:
        records = [self._evaluate_candidate(candidate, frame, target) for candidate in candidates]
        pairwise_wins = self._pairwise_wins(records)
        fallback_used = len([record for record in records if record.admissible]) == 0
        pool = [record for record in records if record.admissible] or records
        selected_evaluations, subset_score = self._select_subset(pool, pairwise_wins, frame, target)
        selected_formulas = [item.formula for item in selected_evaluations]
        selected_lookup = set(selected_formulas)
        enriched = [
            RobustSelectorRecord(
                formula=record.formula,
                source=record.source,
                role=record.role,
                selected=record.formula in selected_lookup,
                admissible=record.admissible,
                robust_score=record.robust_score,
                pairwise_wins=pairwise_wins.get(record.formula, 0),
                full_metrics=record.full_metrics,
                slice_rank_ic=record.slice_rank_ic,
                slice_sharpe=record.slice_sharpe,
                slice_turnover=record.slice_turnover,
                diagnostics={
                    **record.diagnostics,
                    "max_corr_to_selected": self._max_corr_to_selected(record, selected_evaluations),
                    "avg_token_overlap_to_selected": self._avg_token_overlap_to_selected(record, selected_evaluations),
                    "subset_score": subset_score if record.formula in selected_lookup else None,
                },
            )
            for record in records
        ]
        finalized = [
            RobustSelectorRecord(
                formula=record.formula,
                source=record.source,
                role=record.role,
                selected=record.formula in selected_lookup,
                admissible=record.admissible,
                robust_score=record.robust_score,
                pairwise_wins=record.pairwise_wins,
                full_metrics=record.full_metrics,
                slice_rank_ic=record.slice_rank_ic,
                slice_sharpe=record.slice_sharpe,
                slice_turnover=record.slice_turnover,
                diagnostics=record.diagnostics,
            )
            for record in sorted(
                enriched,
                key=lambda item: (
                    item.formula not in selected_lookup,
                    -item.pairwise_wins,
                    -item.robust_score,
                ),
            )
        ]
        return RobustSelectorOutcome(
            selected_formulas=selected_formulas,
            fallback_used=fallback_used,
            records=finalized,
            config={
                "top_k": self.config.top_k,
                "selected_count": len(selected_formulas),
                "slice_count": self.config.slice_count,
                "pairwise_margin": self.config.pairwise_margin,
                "min_valid_slices": self.config.min_valid_slices,
                "subset_score": subset_score,
            },
        )

    def _evaluate_candidate(
        self,
        candidate: FormulaCandidate,
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> _CandidateEvaluation:
        try:
            full_evaluation = evaluate_formula_metrics(candidate.formula, frame, target)
            full_metrics = full_evaluation.metrics
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
                diagnostics={"valid_slices": 0, "mean_rank_ic": None, "min_rank_ic": None},
            )

        slice_rank_ic: list[float | None] = []
        slice_sharpe: list[float | None] = []
        slice_turnover: list[float | None] = []
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
        admissible = (
            valid_slices >= self.config.min_valid_slices
            and mean_rank_ic >= self.config.min_mean_rank_ic
            and min_rank_ic >= self.config.min_slice_rank_ic
        )
        robust_score = (
            mean_rank_ic
            + self.config.min_rank_ic_bonus * min_rank_ic
            - self.config.rank_ic_std_penalty * rank_ic_std
            + self.config.positive_slice_bonus * positive_slice_frac
            + self.config.sharpe_weight * np.tanh(mean_sharpe / 2.0)
            + self.config.annual_return_weight * np.tanh(full_return / 0.20)
            + self.config.stability_weight * max(min(full_stability, 0.05), -0.05)
            - self.config.turnover_weight * mean_turnover
            - self.config.complexity_weight * complexity
        )
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
            signal=full_evaluation.evaluated.signal,
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
            },
        )

    def _pairwise_wins(self, records: list["_CandidateEvaluation"]) -> dict[str, int]:
        wins = {record.formula: 0 for record in records}
        for left_index, left in enumerate(records):
            for right_index, right in enumerate(records):
                if left_index == right_index:
                    continue
                if self._dominates(left, right):
                    wins[left.formula] += 1
        return wins

    def _dominates(self, left: "_CandidateEvaluation", right: "_CandidateEvaluation") -> bool:
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

    def _select_subset(
        self,
        pool: list["_CandidateEvaluation"],
        pairwise_wins: dict[str, int],
        frame: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[list["_CandidateEvaluation"], float]:
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
            return [], float("-inf")
        single_scores = {
            item.formula: self._subset_score([item], frame, target)
            for item in ordered
        }
        best_single = self._choose_best_single(ordered, single_scores)
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
                if max_corr >= self.config.max_pairwise_signal_corr:
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

    def _choose_best_single(
        self,
        ordered: list["_CandidateEvaluation"],
        single_scores: dict[str, float],
    ) -> "_CandidateEvaluation":
        best = max(
            ordered,
            key=lambda item: (
                single_scores.get(item.formula, float("-inf")),
                item.robust_score,
            ),
        )
        while True:
            cluster = [item for item in ordered if self._is_near_neighbor(best, item, single_scores)]
            if not cluster:
                return best
            preferred = max(cluster, key=lambda item: self._near_neighbor_priority(item, single_scores))
            if preferred.formula == best.formula:
                return best
            best = preferred

    def _is_near_neighbor(
        self,
        anchor: "_CandidateEvaluation",
        candidate: "_CandidateEvaluation",
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

    def _near_neighbor_priority(
        self,
        record: "_CandidateEvaluation",
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
        selected: list["_CandidateEvaluation"],
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
        return float(
            mean_rank_ic
            + self.config.min_rank_ic_bonus * min_rank_ic
            - self.config.rank_ic_std_penalty * rank_ic_std
            + self.config.sharpe_weight * np.tanh(sharpe / 2.0)
            + self.config.annual_return_weight * np.tanh(annual_return / 0.20)
            + self.config.stability_weight * max(min(stability, 0.05), -0.05)
            - self.config.turnover_weight * turnover
            - self.config.subset_corr_penalty * avg_corr
            - self.config.subset_overlap_penalty * avg_overlap
            + self.config.subset_single_score_weight * mean_single_score
        )

    def _max_signal_corr(self, candidate: "_CandidateEvaluation", selected: list["_CandidateEvaluation"]) -> float:
        if not selected or candidate.signal.dropna().empty:
            return 0.0
        values = []
        for item in selected:
            corr = _spearman_abs_corr(candidate.signal, item.signal)
            if np.isfinite(corr):
                values.append(corr)
        return float(max(values)) if values else 0.0

    def _avg_pairwise_signal_corr(self, selected: list["_CandidateEvaluation"]) -> float:
        if len(selected) <= 1:
            return 0.0
        values: list[float] = []
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                corr = _spearman_abs_corr(left.signal, right.signal)
                if np.isfinite(corr):
                    values.append(corr)
        return float(np.mean(values)) if values else 0.0

    def _avg_token_overlap(self, selected: list["_CandidateEvaluation"]) -> float:
        if len(selected) <= 1:
            return 0.0
        overlaps: list[float] = []
        for index, left in enumerate(selected):
            for right in selected[index + 1 :]:
                overlaps.append(_token_overlap(left.formula, right.formula))
        return float(np.mean(overlaps)) if overlaps else 0.0

    def _max_corr_to_selected(
        self,
        record: "_CandidateEvaluation",
        selected: list["_CandidateEvaluation"],
    ) -> float | None:
        others = [item for item in selected if item.formula != record.formula]
        if not others:
            return None
        value = self._max_signal_corr(record, others)
        return value if np.isfinite(value) else None

    def _avg_token_overlap_to_selected(
        self,
        record: "_CandidateEvaluation",
        selected: list["_CandidateEvaluation"],
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


TemporalSelectorConfig = RobustSelectorConfig
TemporalSelectorRecord = RobustSelectorRecord
TemporalSelectorOutcome = RobustSelectorOutcome
RobustTemporalSelector = TemporalRobustSelector
