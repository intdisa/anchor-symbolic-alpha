from __future__ import annotations

from ..domain.feature_registry import FEATURE_REGISTRY
from .base import BaseRoleAgent


class MacroAgent(BaseRoleAgent):
    def __init__(self, **kwargs) -> None:
        allowed_features = frozenset(
            name for name, spec in FEATURE_REGISTRY.items() if spec.is_macro
        )
        super().__init__(role="macro", allowed_features=allowed_features, **kwargs)
