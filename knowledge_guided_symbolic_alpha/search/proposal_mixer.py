from __future__ import annotations

from dataclasses import dataclass

from ..language import ParseError, RPNParser
from .grammar_mcts import SearchEvaluation


@dataclass(frozen=True)
class ProposalCandidate:
    source: str
    body_tokens: tuple[str, ...]
    score: float
    valid: bool
    terminal_error: str | None = None
    accepted: bool = False
    reason: str | None = None


class ProposalMixer:
    def __init__(
        self,
        parser: RPNParser | None = None,
        source_weights: dict[str, float] | None = None,
        evaluation_weight: float = 0.80,
        acceptance_bonus: float = 0.15,
    ) -> None:
        self.parser = parser or RPNParser()
        self.source_weights = source_weights or {"mcts": 0.10, "beam": 0.05, "seed": 0.08, "sample": 0.0}
        self.evaluation_weight = evaluation_weight
        self.acceptance_bonus = acceptance_bonus

    def rerank(
        self,
        candidates: list[ProposalCandidate],
        evaluator=None,
    ) -> list[ProposalCandidate]:
        unique = self._dedupe(candidates)
        ranked: list[ProposalCandidate] = []
        for candidate in unique:
            source_bonus = self.source_weights.get(candidate.source, 0.0)
            if evaluator is not None and candidate.valid:
                evaluation = evaluator(candidate.body_tokens)
                mixed_score = (
                    (1.0 - self.evaluation_weight) * candidate.score
                    + self.evaluation_weight * evaluation.score
                    + source_bonus
                    + (self.acceptance_bonus if evaluation.accepted else 0.0)
                )
                ranked.append(
                    ProposalCandidate(
                        source=candidate.source,
                        body_tokens=candidate.body_tokens,
                        score=float(mixed_score),
                        valid=candidate.valid,
                        terminal_error=candidate.terminal_error,
                        accepted=evaluation.accepted,
                        reason=evaluation.reason,
                    )
                )
                continue
            fallback_score = candidate.score + source_bonus - (1.0 if not candidate.valid else 0.0)
            ranked.append(
                ProposalCandidate(
                    source=candidate.source,
                    body_tokens=candidate.body_tokens,
                    score=float(fallback_score),
                    valid=candidate.valid,
                    terminal_error=candidate.terminal_error,
                    accepted=candidate.accepted,
                    reason=candidate.reason,
                )
            )
        ranked.sort(key=lambda item: (item.accepted, item.valid, item.score), reverse=True)
        return ranked

    def select_best(
        self,
        candidates: list[ProposalCandidate],
        evaluator=None,
    ) -> ProposalCandidate | None:
        ranked = self.rerank(candidates, evaluator=evaluator)
        return ranked[0] if ranked else None

    def _dedupe(self, candidates: list[ProposalCandidate]) -> list[ProposalCandidate]:
        best_by_key: dict[str, ProposalCandidate] = {}
        for candidate in candidates:
            key = self._canonical_key(candidate.body_tokens)
            previous = best_by_key.get(key)
            if previous is None or candidate.score > previous.score:
                best_by_key[key] = candidate
        return list(best_by_key.values())

    def _canonical_key(self, tokens: tuple[str, ...]) -> str:
        try:
            parsed = self.parser.parse(tokens)
        except ParseError:
            return "raw:" + " ".join(tokens)
        return parsed.canonical
