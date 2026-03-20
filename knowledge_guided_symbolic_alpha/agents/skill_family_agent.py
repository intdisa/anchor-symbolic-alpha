from __future__ import annotations

from .base import BaseRoleAgent


class SkillFamilyAgent(BaseRoleAgent):
    def __init__(
        self,
        role: str,
        allowed_features: frozenset[str],
        allowed_operators: frozenset[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            role=role,
            allowed_features=allowed_features,
            allowed_operators=allowed_operators,
            **kwargs,
        )
