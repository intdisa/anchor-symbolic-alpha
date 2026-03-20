from __future__ import annotations

from collections import Counter, defaultdict


class FailureRegions:
    def __init__(self) -> None:
        self.token_counts = defaultdict(Counter)
        self.reason_counts = defaultdict(Counter)
        self.formula_counts = Counter()

    def record(
        self,
        regime: str,
        role: str,
        tokens: tuple[str, ...],
        reason: str,
        reward: float,
    ) -> None:
        key = (regime, role)
        weight = self._severity(reason, reward)
        self.token_counts[key].update({token: weight for token in tokens})
        self.reason_counts[key][reason] += 1
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

    def _severity(self, reason: str, reward: float) -> float:
        if "Formula must reduce" in reason or "NaN" in reason:
            return 2.0
        if reason in {"fast_ic_screen", "full_validation", "replacement_check"}:
            return 1.0 + abs(min(0.0, reward))
        return 1.0
