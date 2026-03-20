from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..language.grammar import GrammarState, RPNGrammar
from .sampler import rank_tokens


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_width: int = 4
    per_node_top_k: int = 3
    max_steps: int = 32
    length_penalty: float = 0.03


@dataclass(frozen=True)
class BeamCandidate:
    body_tokens: tuple[str, ...]
    score: float
    valid: bool
    terminal_error: str | None


class BeamSearch:
    def __init__(
        self,
        grammar: RPNGrammar,
        generator,
        config: BeamSearchConfig | None = None,
        token_filter: Callable[[tuple[str, ...]], tuple[str, ...]] | None = None,
        score_adjuster: Callable[[GrammarState, dict[str, float], tuple[str, ...]], dict[str, float]] | None = None,
    ) -> None:
        self.grammar = grammar
        self.generator = generator
        self.config = config or BeamSearchConfig()
        self.token_filter = token_filter
        self.score_adjuster = score_adjuster

    def search(self) -> list[BeamCandidate]:
        beams = [(self.grammar.initial_state(), 0.0, 0.0)]
        completed: list[BeamCandidate] = []
        for _ in range(self.config.max_steps):
            next_beams = []
            for state, raw_score, score in beams:
                if state.finished:
                    completed.append(
                        BeamCandidate(
                            body_tokens=state.body_tokens,
                            score=score,
                            valid=self.grammar.is_valid_terminal(state),
                            terminal_error=state.terminal_error,
                        )
                    )
                    continue
                valid_tokens = self._filter_tokens(self.grammar.valid_next_tokens(state))
                scores = self.generator.score_tokens(state, valid_tokens)
                if self.score_adjuster is not None:
                    scores = self.score_adjuster(state, scores, valid_tokens)
                for token, token_score in rank_tokens(
                    valid_tokens,
                    scores,
                    top_k=self.config.per_node_top_k,
                ):
                    next_state = self.grammar.step(state, token)
                    next_raw_score = raw_score + token_score
                    next_score = self._rank_score(next_state.body_tokens, next_raw_score)
                    next_beams.append((next_state, next_raw_score, next_score))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[2], reverse=True)
            beams = next_beams[: self.config.beam_width]
        completed.extend(
            BeamCandidate(
                body_tokens=state.body_tokens,
                score=score,
                valid=self.grammar.is_valid_terminal(state),
                terminal_error=state.terminal_error,
            )
            for state, _, score in beams
        )
        completed.sort(key=lambda candidate: candidate.score, reverse=True)
        return completed

    def _filter_tokens(self, valid_tokens: tuple[str, ...]) -> tuple[str, ...]:
        if self.token_filter is None:
            return valid_tokens
        filtered = self.token_filter(valid_tokens)
        if filtered:
            return filtered
        if "<EOS>" in valid_tokens:
            return ("<EOS>",)
        return valid_tokens

    def _rank_score(self, body_tokens: tuple[str, ...], raw_score: float) -> float:
        length = max(len(body_tokens), 1)
        average_score = raw_score / length
        return average_score - self.config.length_penalty * length
