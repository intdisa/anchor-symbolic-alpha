from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...language.grammar import GrammarState


@dataclass(frozen=True)
class GeneratorConditioningContext:
    summary_vector: tuple[float, ...]
    token_biases: dict[str, float]
    signature: tuple[str, ...] = ()


class TokenScorer(Protocol):
    def score_tokens(
        self,
        state: GrammarState,
        valid_tokens: tuple[str, ...],
    ) -> dict[str, float]:
        """Return a logit-like score for each valid token."""

    def observe(
        self,
        tokens: tuple[str, ...],
        reward: float,
        accepted: bool,
    ) -> None:
        """Update the scorer with the episode outcome."""

    def set_conditioning_context(
        self,
        context: GeneratorConditioningContext | None,
    ) -> None:
        """Update the dataset/pool conditioning for subsequent scoring."""
