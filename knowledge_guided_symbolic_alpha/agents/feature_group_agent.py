from __future__ import annotations

from .base import BaseRoleAgent


class FeatureGroupAgent(BaseRoleAgent):
    def __init__(self, role: str, allowed_features: frozenset[str], **kwargs) -> None:
        super().__init__(role=role, allowed_features=allowed_features, **kwargs)
