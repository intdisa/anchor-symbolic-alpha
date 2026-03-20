from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...domain.operator_registry import OPERATOR_REGISTRY
from ...language.tokens import FEATURE_TOKENS


@dataclass(frozen=True)
class FormulaEmbedding:
    vector: tuple[float, ...]


class FormulaEmbedder:
    VECTOR_DIM = 6

    def embed(self, tokens: tuple[str, ...]) -> FormulaEmbedding:
        if not tokens:
            return FormulaEmbedding(vector=(0.0,) * self.VECTOR_DIM)
        feature_count = sum(token in FEATURE_TOKENS for token in tokens)
        operator_count = sum(token in OPERATOR_REGISTRY for token in tokens)
        binary_count = sum(OPERATOR_REGISTRY[token].arity == 2 for token in tokens if token in OPERATOR_REGISTRY)
        unary_count = sum(OPERATOR_REGISTRY[token].arity == 1 for token in tokens if token in OPERATOR_REGISTRY)
        delay_count = sum(token == "DELAY_1" for token in tokens)
        rolling_count = sum(token.startswith("TS_") or token.startswith("CORR_") for token in tokens)
        length = max(len(tokens), 1)
        vector = (
            float(length / 16.0),
            float(feature_count / length),
            float(operator_count / length),
            float(binary_count / length),
            float(unary_count / length),
            float((delay_count + rolling_count) / length),
        )
        vector = tuple(float(np.clip(value, -1.0, 1.0)) for value in vector)
        return FormulaEmbedding(vector=vector)
