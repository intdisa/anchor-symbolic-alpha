from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..evaluation.finance_baselines import feature_family_overlap, formula_feature_families, token_overlap
from ..generation import FormulaCandidate
from .robust_selector import RobustSelectorOutcome, RobustSelectorRecord, TemporalRobustSelector


@dataclass(frozen=True)
class CrossSeedConsensusConfig:
    selection_mode: str = "pareto"
    min_seed_support: int | None = None
    rerank_mode: str = "shared_frame"
    enable_cross_seed_pareto: bool = True
    enable_candidate_support_gate: bool = True
    enable_crowding_distance: bool = True
    champion_support_weight: float = 0.03
    selector_support_weight: float = 0.01
    candidate_support_weight: float = 0.005
    selector_rank_penalty: float = 0.002


@dataclass(frozen=True)
class CrossSeedConsensusRecord:
    formula: str
    support_adjusted_score: float | None = None
    mean_temporal_score: float | None = None
    candidate_seed_support: int = 0
    selector_seed_support: int = 0
    champion_seed_support: int = 0
    mean_selector_rank: float | None = None
    support_objective_vector: dict[str, float | int | None] | None = None
    consensus_pareto_rank: int | None = None
    consensus_tiebreak_rank: int | None = None
    consensus_front_size: int | None = None
    consensus_front_share: float | None = None
    consensus_crowding_distance: float | None = None
    used_near_neighbor_tiebreak: bool = False


@dataclass(frozen=True)
class CrossSeedConsensusOutcome:
    selected_formulas: list[str]
    candidate_pool: list[FormulaCandidate]
    ranked_records: list[CrossSeedConsensusRecord]
    formula_support: dict[str, dict[str, Any]]
    temporal_records: list[RobustSelectorRecord]
    fallback_used: bool
    selector_fallback_used: bool
    seed_count: int
    min_seed_support: int


@dataclass(frozen=True)
class CrossSeedSelectionRun:
    seed: int
    candidate_records: tuple[str, ...]
    selector_records: tuple[str, ...]
    champion_records: tuple[str, ...]
    selector_ranked_records: tuple[Any, ...]


@dataclass(frozen=True)
class _ConsensusRankingData:
    rank_map: dict[str, int]
    tiebreak_map: dict[str, int]
    front_size_map: dict[str, int]
    front_share_map: dict[str, float]
    crowding_map: dict[str, float]


class CrossSeedConsensusSelector:
    def __init__(
        self,
        temporal_selector: TemporalRobustSelector | None = None,
        config: CrossSeedConsensusConfig | None = None,
    ) -> None:
        self.temporal_selector = temporal_selector or TemporalRobustSelector()
        self.config = config or CrossSeedConsensusConfig()

    def select(
        self,
        runs: list[dict[str, Any] | CrossSeedSelectionRun],
        frame: pd.DataFrame,
        target: pd.Series,
        base_candidates: list[FormulaCandidate] | None = None,
        *,
        evaluation_context: str | None = None,
    ) -> CrossSeedConsensusOutcome:
        normalized_runs = [normalize_seed_run(run) for run in runs]
        formula_support = summarize_formula_support(normalized_runs)
        seed_count = len({int(run.seed) for run in normalized_runs})
        threshold = self.config.min_seed_support or consensus_seed_support_threshold(seed_count)
        candidate_pool, fallback_used = build_consensus_candidate_pool(
            normalized_runs,
            formula_support=formula_support,
            min_seed_support=threshold,
            base_candidates=base_candidates,
        )
        if candidate_pool:
            if self.config.rerank_mode == "support_only":
                temporal_outcome = RobustSelectorOutcome(
                    selected_formulas=[],
                    fallback_used=False,
                    records=[
                        RobustSelectorRecord(
                            formula=candidate.formula,
                            source=candidate.source,
                            role=candidate.role,
                            selected=False,
                            admissible=True,
                            robust_score=float(formula_support.get(candidate.formula, {}).get("mean_selector_robust_score") or 0.0),
                            pairwise_wins=0,
                            full_metrics={},
                            slice_rank_ic=[],
                            slice_sharpe=[],
                            slice_turnover=[],
                            diagnostics={},
                            temporal_objective_vector=None,
                            temporal_pareto_rank=_int_or_none(formula_support.get(candidate.formula, {}).get("mean_temporal_pareto_rank")),
                            temporal_tiebreak_rank=_int_or_none(formula_support.get(candidate.formula, {}).get("mean_temporal_tiebreak_rank")),
                        )
                        for candidate in candidate_pool
                    ],
                    config={"rerank_mode": "support_only"},
                )
            else:
                if evaluation_context is None:
                    temporal_outcome = self.temporal_selector.select(candidate_pool, frame, target)
                else:
                    try:
                        temporal_outcome = self.temporal_selector.select(
                            candidate_pool,
                            frame,
                            target,
                            evaluation_context=evaluation_context,
                        )
                    except TypeError:
                        temporal_outcome = self.temporal_selector.select(candidate_pool, frame, target)
            if self.config.selection_mode == "legacy_linear":
                selected_formulas, ranked_records = support_adjusted_consensus_selection(
                    temporal_outcome.records,
                    formula_support,
                    seed_count=seed_count,
                    config=self.config,
                )
            else:
                selected_formulas, ranked_records = pareto_consensus_selection(
                    temporal_outcome.records,
                    formula_support,
                    seed_count=seed_count,
                    config=self.config,
                )
            selector_fallback_used = bool(temporal_outcome.fallback_used)
            temporal_records = list(temporal_outcome.records)
        else:
            selected_formulas = []
            ranked_records = []
            selector_fallback_used = False
            temporal_records = []
        return CrossSeedConsensusOutcome(
            selected_formulas=selected_formulas,
            candidate_pool=candidate_pool,
            ranked_records=ranked_records,
            formula_support=formula_support,
            temporal_records=temporal_records,
            fallback_used=fallback_used,
            selector_fallback_used=selector_fallback_used,
            seed_count=seed_count,
            min_seed_support=threshold,
        )


def consensus_seed_support_threshold(seed_count: int) -> int:
    if seed_count <= 1:
        return 1
    return max(2, int(math.ceil(seed_count / 2.0)))


def normalize_seed_run(run: dict[str, Any] | CrossSeedSelectionRun) -> CrossSeedSelectionRun:
    if isinstance(run, CrossSeedSelectionRun):
        return run
    return CrossSeedSelectionRun(
        seed=int(run["seed"]),
        candidate_records=tuple(str(item) for item in run.get("candidate_records", []) if str(item)),
        selector_records=tuple(str(item) for item in run.get("selector_records", []) if str(item)),
        champion_records=tuple(str(item) for item in run.get("champion_records", []) if str(item)),
        selector_ranked_records=tuple(run.get("selector_ranked_records", [])),
    )


def summarize_formula_support(runs: list[CrossSeedSelectionRun]) -> dict[str, dict[str, Any]]:
    candidate_support: dict[str, set[int]] = defaultdict(set)
    selector_support: dict[str, set[int]] = defaultdict(set)
    champion_support: dict[str, set[int]] = defaultdict(set)
    mean_ranks: dict[str, list[float]] = defaultdict(list)
    robust_scores: dict[str, list[float]] = defaultdict(list)
    temporal_pareto_ranks: dict[str, list[float]] = defaultdict(list)
    temporal_tiebreak_ranks: dict[str, list[float]] = defaultdict(list)
    seed_mean_rank_ic: dict[str, list[float]] = defaultdict(list)
    seed_min_rank_ic: dict[str, list[float]] = defaultdict(list)
    seed_rank_ic_std: dict[str, list[float]] = defaultdict(list)
    seed_turnover: dict[str, list[float]] = defaultdict(list)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for run in runs:
        seed = int(run.seed)
        for formula in set(run.candidate_records):
            candidate_support[formula].add(seed)
        for formula in set(run.selector_records):
            selector_support[formula].add(seed)
        for formula in set(run.champion_records):
            champion_support[formula].add(seed)
        for rank, record in enumerate(run.selector_ranked_records, start=1):
            formula = _ranked_record_field(record, "formula")
            if not formula:
                continue
            mean_ranks[formula].append(float(rank))
            robust_score = _ranked_record_field(record, "robust_score")
            if isinstance(robust_score, (int, float)):
                robust_scores[formula].append(float(robust_score))
            temporal_pareto_rank = _ranked_record_field(record, "temporal_pareto_rank")
            if isinstance(temporal_pareto_rank, (int, float)):
                temporal_pareto_ranks[formula].append(float(temporal_pareto_rank))
            temporal_tiebreak_rank = _ranked_record_field(record, "temporal_tiebreak_rank")
            if isinstance(temporal_tiebreak_rank, (int, float)):
                temporal_tiebreak_ranks[formula].append(float(temporal_tiebreak_rank))
            diagnostics = _ranked_record_field(record, "diagnostics")
            if isinstance(diagnostics, dict):
                mean_rank_ic = diagnostics.get("mean_rank_ic")
                min_rank_ic = diagnostics.get("min_rank_ic")
                rank_ic_std = diagnostics.get("rank_ic_std")
                mean_turnover = diagnostics.get("mean_turnover")
                if isinstance(mean_rank_ic, (int, float)):
                    seed_mean_rank_ic[formula].append(float(mean_rank_ic))
                if isinstance(min_rank_ic, (int, float)):
                    seed_min_rank_ic[formula].append(float(min_rank_ic))
                if isinstance(rank_ic_std, (int, float)):
                    seed_rank_ic_std[formula].append(float(rank_ic_std))
                if isinstance(mean_turnover, (int, float)):
                    seed_turnover[formula].append(float(mean_turnover))
            source = _ranked_record_field(record, "source")
            role = _ranked_record_field(record, "role")
            if source:
                source_counts[formula][str(source)] += 1
            if role:
                role_counts[formula][str(role)] += 1

    formulas = sorted(set(candidate_support) | set(selector_support) | set(champion_support))
    payload: dict[str, dict[str, Any]] = {}
    for formula in formulas:
        sources = source_counts.get(formula, Counter())
        roles = role_counts.get(formula, Counter())
        payload[formula] = {
            "candidate_seed_support": len(candidate_support.get(formula, set())),
            "selector_seed_support": len(selector_support.get(formula, set())),
            "champion_seed_support": len(champion_support.get(formula, set())),
            "mean_selector_rank": float(np.mean(mean_ranks[formula])) if mean_ranks.get(formula) else None,
            "mean_selector_robust_score": float(np.mean(robust_scores[formula])) if robust_scores.get(formula) else None,
            "mean_temporal_pareto_rank": float(np.mean(temporal_pareto_ranks[formula])) if temporal_pareto_ranks.get(formula) else None,
            "mean_temporal_tiebreak_rank": float(np.mean(temporal_tiebreak_ranks[formula])) if temporal_tiebreak_ranks.get(formula) else None,
            "mean_seed_mean_rank_ic": float(np.mean(seed_mean_rank_ic[formula])) if seed_mean_rank_ic.get(formula) else None,
            "mean_seed_min_rank_ic": float(np.mean(seed_min_rank_ic[formula])) if seed_min_rank_ic.get(formula) else None,
            "mean_seed_rank_ic_std": float(np.mean(seed_rank_ic_std[formula])) if seed_rank_ic_std.get(formula) else None,
            "mean_seed_turnover": float(np.mean(seed_turnover[formula])) if seed_turnover.get(formula) else None,
            "primary_source": sources.most_common(1)[0][0] if sources else "cross_seed_consensus",
            "primary_role": roles.most_common(1)[0][0] if roles else None,
        }
    return payload


def build_consensus_candidate_pool(
    runs: list[CrossSeedSelectionRun],
    *,
    formula_support: dict[str, dict[str, Any]] | None = None,
    min_seed_support: int | None = None,
    base_candidates: list[FormulaCandidate] | None = None,
) -> tuple[list[FormulaCandidate], bool]:
    support = formula_support or summarize_formula_support(runs)
    seed_count = len({int(run.seed) for run in runs})
    threshold = min_seed_support or consensus_seed_support_threshold(seed_count)
    base_lookup = {candidate.formula: candidate for candidate in base_candidates or []}

    ranked_formulas = sorted(
        support,
        key=lambda formula: (
            support[formula].get("candidate_seed_support") or 0,
            support[formula].get("champion_seed_support") or 0,
            support[formula].get("selector_seed_support") or 0,
            -(support[formula].get("mean_temporal_pareto_rank") or float("inf")),
            -(support[formula].get("mean_temporal_tiebreak_rank") or float("inf")),
            support[formula].get("mean_selector_robust_score") or float("-inf"),
        ),
        reverse=True,
    )
    selected_formulas = [
        formula
        for formula in ranked_formulas
        if int(support[formula].get("candidate_seed_support", 0)) >= threshold
    ]
    fallback_used = False
    if not selected_formulas and ranked_formulas:
        fallback_used = True
        selected_formulas = [ranked_formulas[0]]

    candidates: list[FormulaCandidate] = []
    for formula in selected_formulas:
        if formula in base_lookup:
            base = base_lookup[formula]
            candidates.append(FormulaCandidate(formula=base.formula, source=base.source, role=base.role))
        else:
            candidates.append(
                FormulaCandidate(
                    formula=formula,
                    source=str(support[formula].get("primary_source") or "cross_seed_consensus"),
                    role=support[formula].get("primary_role"),
                )
            )
    return candidates, fallback_used


def mean_score_consensus_formula(
    runs: list[CrossSeedSelectionRun],
    *,
    min_seed_support: int | None = None,
) -> str:
    support = summarize_formula_support(runs)
    threshold = min_seed_support or consensus_seed_support_threshold(len({run.seed for run in runs}))
    eligible = [
        formula
        for formula, payload in support.items()
        if int(payload.get("candidate_seed_support", 0)) >= threshold
    ]
    if not eligible:
        eligible = list(support)
    if not eligible:
        return ""
    return max(
        eligible,
        key=lambda formula: (
            support[formula].get("mean_selector_robust_score") or float("-inf"),
            support[formula].get("selector_seed_support") or 0,
            support[formula].get("champion_seed_support") or 0,
            -(support[formula].get("mean_selector_rank") or float("inf")),
        ),
    )


def support_adjusted_consensus_selection(
    temporal_records: list[RobustSelectorRecord],
    formula_support: dict[str, dict[str, Any]],
    *,
    seed_count: int,
    config: CrossSeedConsensusConfig,
) -> tuple[list[str], list[CrossSeedConsensusRecord]]:
    ranked_records: list[CrossSeedConsensusRecord] = []
    safe_seed_count = max(1, int(seed_count))
    for record in temporal_records:
        support = formula_support.get(record.formula, {})
        candidate_frac = float(support.get("candidate_seed_support", 0)) / safe_seed_count
        selector_frac = float(support.get("selector_seed_support", 0)) / safe_seed_count
        champion_frac = float(support.get("champion_seed_support", 0)) / safe_seed_count
        mean_rank = support.get("mean_selector_rank")
        rank_penalty = 0.0 if mean_rank is None else config.selector_rank_penalty * max(float(mean_rank) - 1.0, 0.0)
        support_adjusted_score = (
            float(record.robust_score)
            + config.champion_support_weight * champion_frac
            + config.selector_support_weight * selector_frac
            + config.candidate_support_weight * candidate_frac
            - rank_penalty
        )
        ranked_records.append(
            CrossSeedConsensusRecord(
                formula=record.formula,
                support_adjusted_score=float(support_adjusted_score),
                mean_temporal_score=support.get("mean_selector_robust_score"),
                candidate_seed_support=int(support.get("candidate_seed_support", 0)),
                selector_seed_support=int(support.get("selector_seed_support", 0)),
                champion_seed_support=int(support.get("champion_seed_support", 0)),
                mean_selector_rank=mean_rank if isinstance(mean_rank, (int, float)) else None,
            )
        )
    ranked_records.sort(
        key=lambda item: (
            item.support_adjusted_score if item.support_adjusted_score is not None else float("-inf"),
            item.champion_seed_support,
            item.selector_seed_support,
            item.candidate_seed_support,
            -(item.mean_selector_rank or float("inf")),
        ),
        reverse=True,
    )
    selected = [ranked_records[0].formula] if ranked_records else []
    return selected, ranked_records


def pareto_consensus_selection(
    temporal_records: list[RobustSelectorRecord],
    formula_support: dict[str, dict[str, Any]],
    *,
    seed_count: int,
    config: CrossSeedConsensusConfig,
) -> tuple[list[str], list[CrossSeedConsensusRecord]]:
    safe_seed_count = max(1, int(seed_count))
    eligible: list[CrossSeedConsensusRecord] = []
    all_records: list[CrossSeedConsensusRecord] = []
    for record in temporal_records:
        support = formula_support.get(record.formula, {})
        candidate_support = int(support.get("candidate_seed_support", 0))
        selector_support = int(support.get("selector_seed_support", 0))
        champion_support = int(support.get("champion_seed_support", 0))
        mean_temporal_score = support.get("mean_selector_robust_score")
        mean_selector_rank = support.get("mean_selector_rank")
        candidate_frac = float(candidate_support) / safe_seed_count
        selector_frac = float(selector_support) / safe_seed_count
        champion_frac = float(champion_support) / safe_seed_count
        complexity = float(max(1, len(record.formula.split())))
        if config.selection_mode == "pareto_discrete_legacy":
            objective_vector = {
                "champion_seed_support": champion_support,
                "selector_seed_support": selector_support,
                "candidate_seed_support": candidate_support,
                "mean_temporal_pareto_rank": support.get("mean_temporal_pareto_rank"),
                "mean_temporal_tiebreak_rank": support.get("mean_temporal_tiebreak_rank"),
                "complexity": complexity,
            }
        else:
            objective_vector = {
                "champion_seed_support_frac": champion_frac,
                "selector_seed_support_frac": selector_frac,
                "candidate_seed_support_frac": candidate_frac,
                "mean_seed_mean_rank_ic": support.get("mean_seed_mean_rank_ic"),
                "mean_seed_min_rank_ic": support.get("mean_seed_min_rank_ic"),
                "mean_seed_rank_ic_std": support.get("mean_seed_rank_ic_std"),
                "mean_seed_turnover": support.get("mean_seed_turnover"),
            }
        rank_penalty = 0.0 if mean_selector_rank is None else config.selector_rank_penalty * max(float(mean_selector_rank) - 1.0, 0.0)
        diagnostic_score = (
            float(record.robust_score)
            + config.champion_support_weight * champion_frac
            + config.selector_support_weight * selector_frac
            + config.candidate_support_weight * candidate_frac
            - rank_penalty
        )
        consensus_record = CrossSeedConsensusRecord(
            formula=record.formula,
            support_adjusted_score=float(diagnostic_score),
            mean_temporal_score=float(mean_temporal_score) if isinstance(mean_temporal_score, (int, float)) else None,
            candidate_seed_support=candidate_support,
            selector_seed_support=selector_support,
            champion_seed_support=champion_support,
            mean_selector_rank=float(mean_selector_rank) if isinstance(mean_selector_rank, (int, float)) else None,
            support_objective_vector=objective_vector,
        )
        all_records.append(consensus_record)
        if (not config.enable_candidate_support_gate) or candidate_support >= (config.min_seed_support or consensus_seed_support_threshold(seed_count)):
            eligible.append(consensus_record)
    if not eligible:
        eligible = list(all_records)
    if not eligible:
        return [], []

    ranking_data = _consensus_rank_data(
        eligible,
        selection_mode=config.selection_mode,
        use_crowding=config.enable_crowding_distance,
    )
    ranked_records = [
        CrossSeedConsensusRecord(
            formula=record.formula,
            support_adjusted_score=record.support_adjusted_score,
            mean_temporal_score=record.mean_temporal_score,
            candidate_seed_support=record.candidate_seed_support,
            selector_seed_support=record.selector_seed_support,
            champion_seed_support=record.champion_seed_support,
            mean_selector_rank=record.mean_selector_rank,
            support_objective_vector=record.support_objective_vector,
            consensus_pareto_rank=ranking_data.rank_map.get(record.formula),
            consensus_tiebreak_rank=ranking_data.tiebreak_map.get(record.formula),
            consensus_front_size=ranking_data.front_size_map.get(record.formula),
            consensus_front_share=ranking_data.front_share_map.get(record.formula),
            consensus_crowding_distance=ranking_data.crowding_map.get(record.formula),
        )
        for record in eligible
    ]
    ranked_records.sort(
        key=lambda item: (
            item.consensus_pareto_rank if item.consensus_pareto_rank is not None else 10**9,
            _crowding_order_value(item.consensus_crowding_distance),
            item.consensus_tiebreak_rank if item.consensus_tiebreak_rank is not None else 10**9,
            item.formula,
        )
    )
    if not ranked_records:
        return [], []
    best = ranked_records[0]
    best_rank = best.consensus_pareto_rank
    best_crowding = best.consensus_crowding_distance
    cluster = [
        item
        for item in ranked_records
        if item.consensus_pareto_rank == best_rank
        and _same_crowding_bucket(item.consensus_crowding_distance, best_crowding)
        and _consensus_near_neighbor(best.formula, item.formula)
    ]
    selected_record = max(cluster, key=_consensus_near_neighbor_priority) if cluster else best
    used_near_neighbor_tiebreak = bool(cluster and selected_record.formula != best.formula)
    ranked_records = [
        CrossSeedConsensusRecord(
            formula=record.formula,
            support_adjusted_score=record.support_adjusted_score,
            mean_temporal_score=record.mean_temporal_score,
            candidate_seed_support=record.candidate_seed_support,
            selector_seed_support=record.selector_seed_support,
            champion_seed_support=record.champion_seed_support,
            mean_selector_rank=record.mean_selector_rank,
            support_objective_vector=record.support_objective_vector,
            consensus_pareto_rank=record.consensus_pareto_rank,
            consensus_tiebreak_rank=record.consensus_tiebreak_rank,
            consensus_front_size=record.consensus_front_size,
            consensus_front_share=record.consensus_front_share,
            consensus_crowding_distance=record.consensus_crowding_distance,
            used_near_neighbor_tiebreak=used_near_neighbor_tiebreak and record.formula == selected_record.formula,
        )
        for record in ranked_records
    ]
    selected = [selected_record.formula]
    return selected, ranked_records


def _consensus_rank_data(
    records: list[CrossSeedConsensusRecord],
    *,
    selection_mode: str,
    use_crowding: bool,
) -> _ConsensusRankingData:
    dominates = _dominates_consensus if selection_mode != "pareto_discrete_legacy" else _dominates_consensus_discrete_legacy
    objective_directions = _consensus_objective_directions(selection_mode)
    fronts = _pareto_fronts(records, dominates)
    rank_map: dict[str, int] = {}
    tiebreak_map: dict[str, int] = {}
    front_size_map: dict[str, int] = {}
    front_share_map: dict[str, float] = {}
    crowding_map: dict[str, float] = {}
    next_rank = 1
    total = max(len(records), 1)
    for front_index, front in enumerate(fronts, start=1):
        if use_crowding and selection_mode != "pareto_discrete_legacy":
            crowding = _crowding_distances(
                front,
                objective_directions=objective_directions,
                vector_getter=lambda item: item.support_objective_vector,
            )
            ordered = sorted(front, key=lambda item: (_crowding_order_value(crowding.get(item.formula)), *_consensus_tiebreak_key(item, selection_mode)))
        else:
            crowding = {item.formula: float("nan") for item in front}
            ordered = sorted(front, key=lambda item: _consensus_tiebreak_key(item, selection_mode))
        for record in ordered:
            rank_map[record.formula] = front_index
            tiebreak_map[record.formula] = next_rank
            front_size_map[record.formula] = len(front)
            front_share_map[record.formula] = float(len(front) / total)
            crowding_map[record.formula] = float(crowding.get(record.formula, float("nan")))
            next_rank += 1
    return _ConsensusRankingData(
        rank_map=rank_map,
        tiebreak_map=tiebreak_map,
        front_size_map=front_size_map,
        front_share_map=front_share_map,
        crowding_map=crowding_map,
    )


def _consensus_tiebreak_key(record: CrossSeedConsensusRecord, selection_mode: str) -> tuple[float, ...]:
    support = record.support_objective_vector or {}
    if selection_mode != "pareto_discrete_legacy":
        return (
            -_objective_value(support.get("mean_seed_min_rank_ic"), maximize=True),
            _objective_value(support.get("mean_seed_rank_ic_std"), maximize=False),
            -_objective_value(support.get("mean_seed_mean_rank_ic"), maximize=True),
            _objective_value(support.get("mean_seed_turnover"), maximize=False),
            -float(record.champion_seed_support),
            -float(record.selector_seed_support),
            -float(record.candidate_seed_support),
            _objective_value(record.mean_selector_rank, maximize=False),
            _objective_value(float(max(1, len(record.formula.split()))), maximize=False),
            record.formula,
        )
    return (
        -float(record.champion_seed_support),
        -float(record.selector_seed_support),
        -float(record.candidate_seed_support),
        _objective_value(support.get("mean_temporal_pareto_rank"), maximize=False),
        _objective_value(support.get("mean_temporal_tiebreak_rank"), maximize=False),
        _objective_value(support.get("complexity"), maximize=False),
        -(record.mean_temporal_score if record.mean_temporal_score is not None else float("-inf")),
        record.formula,
    )


def _dominates_consensus(left: CrossSeedConsensusRecord, right: CrossSeedConsensusRecord) -> bool:
    return _dominates_objectives(
        left.support_objective_vector,
        right.support_objective_vector,
        objective_directions=(
            ("champion_seed_support_frac", True),
            ("selector_seed_support_frac", True),
            ("candidate_seed_support_frac", True),
            ("mean_seed_mean_rank_ic", True),
            ("mean_seed_min_rank_ic", True),
            ("mean_seed_rank_ic_std", False),
            ("mean_seed_turnover", False),
        ),
    )


def _dominates_consensus_discrete_legacy(left: CrossSeedConsensusRecord, right: CrossSeedConsensusRecord) -> bool:
    return _dominates_objectives(
        left.support_objective_vector,
        right.support_objective_vector,
        objective_directions=(
            ("champion_seed_support", True),
            ("selector_seed_support", True),
            ("candidate_seed_support", True),
            ("mean_temporal_pareto_rank", False),
            ("mean_temporal_tiebreak_rank", False),
            ("complexity", False),
        ),
    )


def _dominates_objectives(
    left: dict[str, float | int | None] | None,
    right: dict[str, float | int | None] | None,
    *,
    objective_directions: tuple[tuple[str, bool], ...],
) -> bool:
    if not left or not right:
        return False
    better_or_equal = True
    strictly_better = False
    for key, maximize in objective_directions:
        left_value = _objective_value(left.get(key), maximize=maximize)
        right_value = _objective_value(right.get(key), maximize=maximize)
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


def _objective_value(value: float | int | None, *, maximize: bool) -> float:
    if value is None:
        return float("-inf") if maximize else float("inf")
    value = float(value)
    if not np.isfinite(value):
        return float("-inf") if maximize else float("inf")
    return value


def _consensus_objective_directions(selection_mode: str) -> tuple[tuple[str, bool], ...]:
    if selection_mode == "pareto_discrete_legacy":
        return (
            ("champion_seed_support", True),
            ("selector_seed_support", True),
            ("candidate_seed_support", True),
            ("mean_temporal_pareto_rank", False),
            ("mean_temporal_tiebreak_rank", False),
            ("complexity", False),
        )
    return (
        ("champion_seed_support_frac", True),
        ("selector_seed_support_frac", True),
        ("candidate_seed_support_frac", True),
        ("mean_seed_mean_rank_ic", True),
        ("mean_seed_min_rank_ic", True),
        ("mean_seed_rank_ic_std", False),
        ("mean_seed_turnover", False),
    )


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


def _pareto_fronts(
    records: list[CrossSeedConsensusRecord],
    dominates: Callable[[CrossSeedConsensusRecord, CrossSeedConsensusRecord], bool],
) -> list[list[CrossSeedConsensusRecord]]:
    remaining = list(records)
    fronts: list[list[CrossSeedConsensusRecord]] = []
    while remaining:
        front: list[CrossSeedConsensusRecord] = []
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
    records: list[CrossSeedConsensusRecord],
    *,
    objective_directions: tuple[tuple[str, bool], ...],
    vector_getter: Callable[[CrossSeedConsensusRecord], dict[str, float | int | None] | None],
) -> dict[str, float]:
    if not records:
        return {}
    if len(records) <= 2:
        return {record.formula: float("inf") for record in records}
    distances = {record.formula: 0.0 for record in records}
    for objective, maximize in objective_directions:
        ranked: list[tuple[CrossSeedConsensusRecord, float]] = []
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


def _consensus_near_neighbor(anchor_formula: str, candidate_formula: str) -> bool:
    if anchor_formula == candidate_formula:
        return True
    left_tokens = anchor_formula.split()
    right_tokens = candidate_formula.split()
    if left_tokens and right_tokens and all(token.startswith("X") or token in {"ADD", "SUB", "MUL", "DIV", "NEG"} for token in left_tokens + right_tokens):
        return float(token_overlap(anchor_formula, candidate_formula)) >= 0.75
    overlap = float(token_overlap(anchor_formula, candidate_formula))
    family_overlap = float(feature_family_overlap(anchor_formula, candidate_formula))
    left_families = set(formula_feature_families(anchor_formula))
    right_families = set(formula_feature_families(candidate_formula))
    nested_family = bool(left_families and right_families and (left_families <= right_families or right_families <= left_families))
    return overlap >= 0.50 or (overlap >= 0.40 and family_overlap >= 0.50) or (nested_family and family_overlap >= 0.50)


def _consensus_near_neighbor_priority(record: CrossSeedConsensusRecord) -> tuple[float, ...]:
    support = record.support_objective_vector or {}
    family_count = float(len(formula_feature_families(record.formula)))
    if "mean_seed_min_rank_ic" in support:
        return (
            float(record.champion_seed_support),
            family_count,
            float(record.selector_seed_support),
            float(record.candidate_seed_support),
            _objective_value(support.get("mean_seed_min_rank_ic"), maximize=True),
            _objective_value(support.get("mean_seed_mean_rank_ic"), maximize=True),
            -_objective_value(support.get("mean_seed_rank_ic_std"), maximize=False),
            -_objective_value(support.get("mean_seed_turnover"), maximize=False),
            record.mean_temporal_score if record.mean_temporal_score is not None else float("-inf"),
        )
    return (
        float(record.champion_seed_support),
        family_count,
        float(record.selector_seed_support),
        float(record.candidate_seed_support),
        -_objective_value(support.get("mean_temporal_pareto_rank"), maximize=False),
        -_objective_value(support.get("mean_temporal_tiebreak_rank"), maximize=False),
        record.mean_temporal_score if record.mean_temporal_score is not None else float("-inf"),
        -_objective_value(support.get("complexity"), maximize=False),
    )


def _ranked_record_field(record: Any, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(round(value))
    return None
