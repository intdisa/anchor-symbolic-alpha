from __future__ import annotations

from collections import Counter, defaultdict


class SuccessPatterns:
    def __init__(self) -> None:
        self.token_counts = defaultdict(Counter)
        self.formula_counts = Counter()

    def record(self, regime: str, role: str, tokens: tuple[str, ...], reward: float) -> None:
        del reward
        key = (regime, role)
        self.token_counts[key].update(tokens)
        self.formula_counts[key] += 1

    def token_scores(
        self,
        regime: str,
        role: str,
        valid_tokens: tuple[str, ...],
    ) -> dict[str, float]:
        key = (regime, role)
        total = self.formula_counts[key]
        if total == 0:
            return {token: 0.0 for token in valid_tokens}
        counts = self.token_counts[key]
        return {token: counts[token] / total for token in valid_tokens}
