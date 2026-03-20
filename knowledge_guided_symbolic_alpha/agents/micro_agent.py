from __future__ import annotations

from ..domain.feature_registry import FEATURE_REGISTRY
from .base import BaseRoleAgent


class MicroAgent(BaseRoleAgent):
    def __init__(self, **kwargs) -> None:
        allowed_features = frozenset(
            name for name, spec in FEATURE_REGISTRY.items() if spec.is_micro
        )
        super().__init__(role="micro", allowed_features=allowed_features, **kwargs)
