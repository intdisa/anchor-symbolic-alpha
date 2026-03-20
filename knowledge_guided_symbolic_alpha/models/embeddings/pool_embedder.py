from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...evaluation import FactorPool
from ...evaluation.role_profiles import normalize_role


@dataclass(frozen=True)
class PoolEmbedding:
    vector: tuple[float, ...]
    token_biases: dict[str, float]


class PoolEmbedder:
    VECTOR_DIM = 8

    def embed(self, pool: FactorPool, allowed_features: frozenset[str], role: str) -> PoolEmbedding:
        normalized_role = normalize_role(role)
        feature_counts: dict[str, int] = {}
        role_counts = {"target_price": 0, "target_flow": 0, "context": 0}
        rank_ics: list[float] = []
        turnovers: list[float] = []
        max_corrs: list[float] = []

        for record in pool.records:
            record_role = normalize_role(record.role)
            if record_role in role_counts:
                role_counts[record_role] += 1
            rank_ics.append(abs(float(record.metrics.get("rank_ic", 0.0))))
            turnovers.append(float(record.metrics.get("turnover", 0.0)))
            max_corrs.append(float(record.metrics.get("max_corr", 0.0)))
            for token in record.tokens:
                feature_counts[token] = feature_counts.get(token, 0) + 1

        token_biases: dict[str, float] = {}
        if pool.records:
            max_count = max(feature_counts.values(), default=1)
            for feature in allowed_features:
                usage = feature_counts.get(feature, 0)
                if usage == 0:
                    token_biases[feature] = token_biases.get(feature, 0.0) + 0.06
                else:
                    token_biases[feature] = token_biases.get(feature, 0.0) - 0.08 * (usage / max_count)

        if normalized_role == "target_flow" and role_counts["target_price"] > 0 and role_counts["target_flow"] == 0:
            token_biases["NEG"] = token_biases.get("NEG", 0.0) + 0.06
            token_biases["RANK"] = token_biases.get("RANK", 0.0) + 0.05
        if normalized_role == "target_price" and role_counts["target_flow"] > 0 and role_counts["target_price"] == 0:
            token_biases["DELAY_1"] = token_biases.get("DELAY_1", 0.0) + 0.05
        if normalized_role == "context" and (role_counts["target_price"] > 0 or role_counts["target_flow"] > 0):
            token_biases["MUL"] = token_biases.get("MUL", 0.0) - 0.08
            token_biases["DIV"] = token_biases.get("DIV", 0.0) - 0.08

        size_ratio = float(len(pool.records) / max(pool.max_size, 1))
        avg_rank_ic = float(np.mean(rank_ics)) if rank_ics else 0.0
        avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
        avg_corr = float(np.mean(max_corrs)) if max_corrs else 0.0
        unique_roles = len({normalize_role(record.role) for record in pool.records if normalize_role(record.role) is not None})
        vector = (
            size_ratio,
            float(role_counts["target_price"] / max(len(pool.records), 1)),
            float(role_counts["target_flow"] / max(len(pool.records), 1)),
            float(role_counts["context"] / max(len(pool.records), 1)),
            avg_rank_ic,
            avg_turnover / 2.0,
            avg_corr,
            float(unique_roles / 3.0),
        )
        return PoolEmbedding(vector=vector, token_biases=token_biases)
