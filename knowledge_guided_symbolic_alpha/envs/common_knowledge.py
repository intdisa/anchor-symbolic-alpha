from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation import FactorPool, score_pool_on_dataset
from ..models.embeddings import DatasetEmbedder, PoolEmbedder


@dataclass(frozen=True)
class CommonKnowledgeState:
    dataset_name: str
    regime: str
    summary_vector: tuple[float, ...]
    dataset_embedding: tuple[float, ...]
    pool_embedding: tuple[float, ...]
    pool_size: int
    max_pool_size: int
    occupied_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    redundancy: float
    pool_trade_proxy: float
    validation_pool_score: float


class CommonKnowledgeEncoder:
    def __init__(
        self,
        dataset_embedder: DatasetEmbedder | None = None,
        pool_embedder: PoolEmbedder | None = None,
    ) -> None:
        self.dataset_embedder = dataset_embedder or DatasetEmbedder()
        self.pool_embedder = pool_embedder or PoolEmbedder()

    def encode(
        self,
        dataset_name: str,
        regime: str,
        data: pd.DataFrame,
        pool: FactorPool,
        skill_names: tuple[str, ...],
        target: pd.Series | None = None,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> CommonKnowledgeState:
        allowed_features = frozenset(str(column) for column in data.columns)
        dataset_embedding = self.dataset_embedder.embed(
            data,
            allowed_features,
            target=target,
            regime=regime,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        pool_embedding = self.pool_embedder.embed(pool, allowed_features, role="context")
        occupied_skills = tuple(sorted({record.role for record in pool.records if record.role is not None}))
        missing_skills = tuple(skill for skill in skill_names if skill not in occupied_skills)
        redundancy = float(
            np.mean(
                [
                    abs(float(record.metrics.get("max_corr", 0.0)))
                    for record in pool.records
                    if np.isfinite(float(record.metrics.get("max_corr", 0.0)))
                ]
            )
        ) if pool.records else 0.0
        validation_pool_score = (
            score_pool_on_dataset(pool, validation_data, validation_target)
            if validation_data is not None and validation_target is not None
            else pool.pool_score()
        )
        summary_vector = (
            dataset_embedding.vector
            + pool_embedding.vector
            + (
                float(len(pool.records) / max(pool.max_size, 1)),
                redundancy,
                float(pool.trade_proxy_score()),
                float(validation_pool_score),
            )
        )
        return CommonKnowledgeState(
            dataset_name=dataset_name,
            regime=regime,
            summary_vector=summary_vector,
            dataset_embedding=dataset_embedding.vector,
            pool_embedding=pool_embedding.vector,
            pool_size=len(pool.records),
            max_pool_size=pool.max_size,
            occupied_skills=occupied_skills,
            missing_skills=missing_skills,
            redundancy=redundancy,
            pool_trade_proxy=float(pool.trade_proxy_score()),
            validation_pool_score=float(validation_pool_score),
        )
