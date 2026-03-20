from __future__ import annotations

from .failure_regions import FailureRegions
from .retrieval import RetrievedMemory
from .success_patterns import SuccessPatterns


class ExperienceMemory:
    def __init__(
        self,
        success_scale: float = 0.20,
        failure_scale: float = 0.30,
    ) -> None:
        self.success_patterns = SuccessPatterns()
        self.failure_regions = FailureRegions()
        self.success_scale = success_scale
        self.failure_scale = failure_scale

    def record(
        self,
        regime: str,
        role: str,
        tokens: tuple[str, ...],
        reward: float,
        accepted: bool,
        reason: str,
    ) -> None:
        if not tokens:
            return
        if accepted:
            self.success_patterns.record(regime, role, tokens, reward)
        else:
            self.failure_regions.record(regime, role, tokens, reason, reward)

    def retrieve(
        self,
        regime: str,
        role: str,
        valid_tokens: tuple[str, ...],
    ) -> RetrievedMemory:
        success_scores = self.success_patterns.token_scores(regime, role, valid_tokens)
        failure_scores = self.failure_regions.token_scores(regime, role, valid_tokens)
        token_biases = {
            token: self.success_scale * success_scores[token] - self.failure_scale * failure_scores[token]
            for token in valid_tokens
        }
        return RetrievedMemory(regime=regime, role=role, token_biases=token_biases)
