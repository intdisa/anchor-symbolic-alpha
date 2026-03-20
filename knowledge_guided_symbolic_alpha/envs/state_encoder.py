from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..domain.feature_registry import FEATURE_REGISTRY
from ..evaluation import FactorPool


@dataclass(frozen=True)
class EncodedState:
    regime: str
    pool_size: int
    macro_pool_fraction: float
    micro_pool_fraction: float
    latest_observation: dict[str, float]


class StateEncoder:
    def __init__(
        self,
        first_group_features: frozenset[str] | None = None,
        second_group_features: frozenset[str] | None = None,
        latest_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.first_group_features = first_group_features
        self.second_group_features = second_group_features
        self.latest_columns = latest_columns or ("GOLD_CLOSE", "GOLD_VOLUME", "CPI", "TNX", "VIX", "DXY")

    def encode(
        self,
        frame: pd.DataFrame,
        pool: FactorPool,
        regime: str,
    ) -> EncodedState:
        latest_observation = {
            column: float(frame[column].iloc[-1])
            for column in self.latest_columns
            if column in frame.columns and not frame.empty
        }
        first_group_features = self.first_group_features
        second_group_features = self.second_group_features
        if first_group_features is None or second_group_features is None:
            first_group_features = frozenset(
                token for token, spec in FEATURE_REGISTRY.items() if spec.is_macro
            )
            second_group_features = frozenset(
                token for token, spec in FEATURE_REGISTRY.items() if spec.is_micro
            )
        macro_count = 0
        micro_count = 0
        for record in pool.records:
            features = [token for token in record.tokens if token in FEATURE_REGISTRY]
            if not features:
                continue
            macro_hits = sum(1 for token in features if token in first_group_features)
            micro_hits = sum(1 for token in features if token in second_group_features)
            if macro_hits >= micro_hits:
                macro_count += 1
            if micro_hits >= macro_hits:
                micro_count += 1
        total = max(len(pool.records), 1)
        return EncodedState(
            regime=regime,
            pool_size=len(pool.records),
            macro_pool_fraction=macro_count / total,
            micro_pool_fraction=micro_count / total,
            latest_observation=latest_observation,
        )
