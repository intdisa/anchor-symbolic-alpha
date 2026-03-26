from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import pandas as pd

from ..evaluation import FactorPool, score_pool_on_dataset
from ..evaluation.factor_pool import FactorRecord
from ..generation import FormulaCandidate
from ..evaluation.panel_dispatch import is_cross_sectional_frame
from ..evaluation.pool_scoring import rescore_pool_on_dataset
from ..search import FormulaSampler
from .curriculum import FormulaCurriculum
from .reward_shaping import PoolRewardShaper


@dataclass(frozen=True)
class TrainingEpisode:
    formula: str
    tokens: tuple[str, ...]
    reward: float
    accepted: bool
    reason: str
    pool_size: int
    pool_score: float
    validation_pool_score: float
    validation_selection_score: float
    terminal_valid: bool


@dataclass(frozen=True)
class TrainingSummary:
    history: list[TrainingEpisode]
    best_validation_pool_score: float
    best_validation_selection_score: float
    champion_records: tuple[str, ...]
    final_records: tuple[str, ...]
    candidate_records: tuple[str, ...]
    final_pool_size: int
    candidate_record_summaries: tuple[FormulaCandidate, ...] = tuple()


@dataclass(frozen=True)
class MultiAgentTrainingEpisode:
    formula: str
    selected_agent: str
    regime: str
    reward: float
    accepted: bool
    decision_reason: str
    review_reason: str
    pool_size: int
    validation_pool_score: float
    validation_selection_score: float


@dataclass(frozen=True)
class PoolRecordSnapshot:
    formula: str
    role: str | None


@dataclass(frozen=True)
class MultiAgentTrainingSummary:
    history: list[MultiAgentTrainingEpisode]
    best_validation_pool_score: float
    best_validation_selection_score: float
    champion_records: tuple[str, ...]
    final_records: tuple[str, ...]
    candidate_records: tuple[str, ...]
    final_pool_size: int
    candidate_record_summaries: tuple[FormulaCandidate, ...] = tuple()
    champion_record_summaries: tuple[PoolRecordSnapshot, ...] = tuple()
    final_record_summaries: tuple[PoolRecordSnapshot, ...] = tuple()


class ManagerLike(Protocol):
    def apply_curriculum(self, stage) -> None: ...

    def warm_start(
        self,
        pool: FactorPool,
        data: pd.DataFrame,
        target: pd.Series,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> None: ...

    def run_step(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        commit: bool = True,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ): ...

    def generated_candidate_records(self) -> tuple[FormulaCandidate, ...]: ...


def _snapshot_pool(records: list[FactorRecord]) -> tuple[PoolRecordSnapshot, ...]:
    return tuple(
        PoolRecordSnapshot(
            formula=" ".join(record.tokens),
            role=record.role,
        )
        for record in records
    )


class SingleAgentTrainer:
    def __init__(
        self,
        sampler: FormulaSampler,
        reward_shaper: PoolRewardShaper,
        pool_max_size: int = 16,
        curriculum: FormulaCurriculum | None = None,
    ) -> None:
        self.sampler = sampler
        self.reward_shaper = reward_shaper
        self.pool_max_size = pool_max_size
        self.curriculum = curriculum

    def train(
        self,
        train_data: pd.DataFrame,
        train_target: pd.Series,
        episodes: int,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> TrainingSummary:
        pool = FactorPool(max_size=self.pool_max_size)
        history: list[TrainingEpisode] = []
        seen_candidates: list[str] = []
        candidate_summaries: list[FormulaCandidate] = []
        best_validation_pool_score = float("-inf")
        best_validation_selection_score = float("-inf")
        champion_records: tuple[str, ...] = tuple()

        for episode_index in range(episodes):
            if self.curriculum is not None:
                self._apply_sampler_curriculum(self.curriculum.stage_for_episode(episode_index, episodes))
            sample = self.sampler.sample()
            formula_text = " ".join(sample.body_tokens)
            if formula_text and formula_text not in seen_candidates:
                seen_candidates.append(formula_text)
                candidate_summaries.append(
                    FormulaCandidate(
                        formula=formula_text,
                        source="sampled_formula",
                        role=None,
                    )
                )
            outcome = self.reward_shaper.shape(
                sample.body_tokens,
                train_data,
                train_target,
                pool,
                commit=True,
            )
            if hasattr(self.sampler.generator, "observe"):
                self.sampler.generator.observe(
                    sample.body_tokens,
                    outcome.clipped_reward,
                    outcome.decision.accepted,
                )
            pool_score = pool.pool_score()
            validation_score = pool_score
            validation_selection_score = validation_score
            if validation_data is not None and validation_target is not None:
                validation_score, validation_selection_score = _validation_scores(
                    pool,
                    validation_data,
                    validation_target,
                )
            if validation_selection_score > best_validation_selection_score:
                best_validation_selection_score = validation_selection_score
                best_validation_pool_score = validation_score
                champion_records = tuple(" ".join(record.tokens) for record in pool.records)
            history.append(
                TrainingEpisode(
                    formula=formula_text,
                    tokens=sample.body_tokens,
                    reward=outcome.clipped_reward,
                    accepted=outcome.decision.accepted,
                    reason=outcome.decision.reason,
                    pool_size=len(pool),
                    pool_score=pool_score,
                    validation_pool_score=validation_score,
                    validation_selection_score=validation_selection_score,
                    terminal_valid=sample.valid,
                )
            )
        return TrainingSummary(
            history=history,
            best_validation_pool_score=best_validation_pool_score,
            best_validation_selection_score=best_validation_selection_score,
            champion_records=champion_records,
            final_records=tuple(" ".join(record.tokens) for record in pool.records),
            candidate_records=tuple(seen_candidates),
            candidate_record_summaries=tuple(candidate_summaries),
            final_pool_size=len(pool),
        )

    def _apply_sampler_curriculum(self, stage) -> None:
        self.sampler.grammar.max_length = stage.max_length
        self.sampler.grammar.min_length = stage.min_length
        self.sampler.config = replace(
            self.sampler.config,
            temperature=stage.temperature,
            top_k=stage.top_k,
            max_steps=stage.max_length + 2,
        )


class MultiAgentTrainer:
    def __init__(
        self,
        manager: ManagerLike,
        pool_max_size: int = 16,
        curriculum: FormulaCurriculum | None = None,
    ) -> None:
        self.manager = manager
        self.pool_max_size = pool_max_size
        self.curriculum = curriculum

    def train(
        self,
        train_data: pd.DataFrame,
        train_target: pd.Series,
        episodes: int,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> MultiAgentTrainingSummary:
        pool = FactorPool(max_size=self.pool_max_size)
        history: list[MultiAgentTrainingEpisode] = []
        seen_candidates: list[str] = []
        best_validation_pool_score = float("-inf")
        best_validation_selection_score = float("-inf")
        champion_records: tuple[str, ...] = tuple()
        champion_record_summaries: tuple[PoolRecordSnapshot, ...] = tuple()
        if hasattr(self.manager, "warm_start"):
            self.manager.warm_start(
                pool,
                train_data,
                train_target,
                validation_data=validation_data,
                validation_target=validation_target,
            )
        for episode_index in range(episodes):
            if self.curriculum is not None:
                self.manager.apply_curriculum(self.curriculum.stage_for_episode(episode_index, episodes))
            step = self.manager.run_step(
                train_data,
                train_target,
                pool,
                commit=True,
                validation_data=validation_data,
                validation_target=validation_target,
            )
            formula_text = " ".join(step.formula_tokens)
            if formula_text and formula_text not in seen_candidates:
                seen_candidates.append(formula_text)
            validation_score = pool.pool_score()
            validation_selection_score = validation_score
            if validation_data is not None and validation_target is not None:
                validation_score, validation_selection_score = _validation_scores(
                    pool,
                    validation_data,
                    validation_target,
                )
            if validation_selection_score > best_validation_selection_score:
                best_validation_selection_score = validation_selection_score
                best_validation_pool_score = validation_score
                champion_records = tuple(" ".join(record.tokens) for record in pool.records)
                champion_record_summaries = _snapshot_pool(pool.records)
            history.append(
                MultiAgentTrainingEpisode(
                    formula=formula_text,
                    selected_agent=step.selected_agent,
                    regime=step.regime,
                    reward=step.reward,
                    accepted=step.accepted,
                    decision_reason=step.decision_reason,
                    review_reason=step.review_reason,
                    pool_size=step.pool_size,
                    validation_pool_score=validation_score,
                    validation_selection_score=validation_selection_score,
                )
            )
        candidate_record_summaries = (
            tuple(self.manager.generated_candidate_records())
            if hasattr(self.manager, "generated_candidate_records")
            else tuple(
                FormulaCandidate(formula=formula, source="episode_formula", role=None)
                for formula in seen_candidates
            )
        )
        if candidate_record_summaries:
            seen_candidates = [item.formula for item in candidate_record_summaries]
        return MultiAgentTrainingSummary(
            history=history,
            best_validation_pool_score=best_validation_pool_score,
            best_validation_selection_score=best_validation_selection_score,
            champion_records=champion_records,
            final_records=tuple(" ".join(record.tokens) for record in pool.records),
            candidate_records=tuple(seen_candidates),
            candidate_record_summaries=tuple(candidate_record_summaries),
            final_pool_size=len(pool),
            champion_record_summaries=champion_record_summaries,
            final_record_summaries=_snapshot_pool(pool.records),
        )


def _validation_scores(
    pool: FactorPool,
    validation_data: pd.DataFrame,
    validation_target: pd.Series,
) -> tuple[float, float]:
    validation_pool_score = score_pool_on_dataset(pool, validation_data, validation_target)
    if not is_cross_sectional_frame(validation_data):
        return validation_pool_score, validation_pool_score
    rescored_pool = rescore_pool_on_dataset(pool, validation_data, validation_target)
    return validation_pool_score, rescored_pool.trade_proxy_score()
