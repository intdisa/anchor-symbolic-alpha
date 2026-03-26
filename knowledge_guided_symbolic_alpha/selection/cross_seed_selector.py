from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from ..generation import FormulaCandidate
from .robust_selector import RobustSelectorOutcome, RobustSelectorRecord, TemporalRobustSelector


@dataclass(frozen=True)
class CrossSeedConsensusConfig:
    min_seed_support: int | None = None
    champion_support_weight: float = 0.03
    selector_support_weight: float = 0.01
    candidate_support_weight: float = 0.005
    selector_rank_penalty: float = 0.002
    rerank_mode: str = "shared_frame"


@dataclass(frozen=True)
class CrossSeedConsensusRecord:
    formula: str
    support_adjusted_score: float
    mean_temporal_score: float | None
    candidate_seed_support: int
    selector_seed_support: int
    champion_seed_support: int
    mean_selector_rank: float | None


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
                        )
                        for candidate in candidate_pool
                    ],
                    config={"rerank_mode": "support_only"},
                )
            else:
                temporal_outcome = self.temporal_selector.select(candidate_pool, frame, target)
            selected_formulas, ranked_records = support_adjusted_consensus_selection(
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
            support[formula]["candidate_seed_support"],
            support[formula]["selector_seed_support"],
            support[formula]["champion_seed_support"],
            -(support[formula]["mean_selector_rank"] or float("inf")),
            support[formula]["mean_selector_robust_score"] or float("-inf"),
        ),
        reverse=True,
    )
    selected_formulas = [
        formula
        for formula in ranked_formulas
        if int(support[formula]["candidate_seed_support"]) >= threshold
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
            item.support_adjusted_score,
            item.champion_seed_support,
            item.selector_seed_support,
            item.candidate_seed_support,
            -(item.mean_selector_rank or float("inf")),
        ),
        reverse=True,
    )
    selected = [ranked_records[0].formula] if ranked_records else []
    return selected, ranked_records


def _ranked_record_field(record: Any, field: str) -> Any:
    if isinstance(record, dict):
        return record.get(field)
    return getattr(record, field, None)
