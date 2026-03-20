from __future__ import annotations

from dataclasses import dataclass

from ..domain.operator_registry import OPERATOR_REGISTRY
from ..language.tokens import FEATURE_TOKENS


@dataclass(frozen=True)
class RoleActionMask:
    role: str
    allowed_features: frozenset[str]
    allowed_operators: frozenset[str] | None = None

    def filter_tokens(self, valid_tokens: tuple[str, ...]) -> tuple[str, ...]:
        filtered = tuple(
            token
            for token in valid_tokens
            if (
                (token not in FEATURE_TOKENS or token in self.allowed_features)
                and (
                    token not in OPERATOR_REGISTRY
                    or self.allowed_operators is None
                    or token in self.allowed_operators
                )
            )
        )
        if filtered:
            return filtered
        if "<EOS>" in valid_tokens:
            return ("<EOS>",)
        return valid_tokens
