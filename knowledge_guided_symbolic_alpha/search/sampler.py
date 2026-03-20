from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..language.grammar import GrammarState, RPNGrammar


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None
    logit_clip: float = 20.0
    max_steps: int = 32
    seed: int = 7


@dataclass(frozen=True)
class SampledFormula:
    body_tokens: tuple[str, ...]
    logprob: float
    valid: bool
    terminal_error: str | None


def rank_tokens(
    valid_tokens: tuple[str, ...],
    scores: dict[str, float],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    ranked = sorted(
        ((token, float(scores.get(token, 0.0))) for token in valid_tokens),
        key=lambda item: item[1],
        reverse=True,
    )
    if top_k is not None:
        return ranked[:top_k]
    return ranked


class FormulaSampler:
    def __init__(
        self,
        grammar: RPNGrammar,
        generator,
        config: SamplingConfig | None = None,
        token_filter: Callable[[tuple[str, ...]], tuple[str, ...]] | None = None,
        score_adjuster: Callable[[GrammarState, dict[str, float], tuple[str, ...]], dict[str, float]] | None = None,
    ) -> None:
        self.grammar = grammar
        self.generator = generator
        self.config = config or SamplingConfig()
        self.token_filter = token_filter
        self.score_adjuster = score_adjuster
        self.rng = np.random.default_rng(self.config.seed)

    def sample(self) -> SampledFormula:
        state = self.grammar.initial_state()
        total_logprob = 0.0
        steps = 0
        while not state.finished and steps < self.config.max_steps:
            valid_tokens = self._filter_tokens(self.grammar.valid_next_tokens(state))
            token, logprob = self._sample_next_token(state, valid_tokens)
            total_logprob += logprob
            state = self.grammar.step(state, token)
            steps += 1
        if not state.finished:
            state = self.grammar.step(state, "<EOS>")
        return SampledFormula(
            body_tokens=state.body_tokens,
            logprob=float(total_logprob),
            valid=self.grammar.is_valid_terminal(state),
            terminal_error=state.terminal_error,
        )

    def _sample_next_token(
        self,
        state: GrammarState,
        valid_tokens: tuple[str, ...],
    ) -> tuple[str, float]:
        scores = self.generator.score_tokens(state, valid_tokens)
        if self.score_adjuster is not None:
            scores = self.score_adjuster(state, scores, valid_tokens)
        ranked = rank_tokens(valid_tokens, scores, top_k=self.config.top_k)
        tokens = [token for token, _ in ranked]
        logits = np.array([score for _, score in ranked], dtype=float)
        logits = np.clip(logits, -self.config.logit_clip, self.config.logit_clip)
        if self.config.temperature <= 0.0 or len(tokens) == 1:
            return tokens[0], 0.0
        logits = logits / self.config.temperature
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if not np.isfinite(probs_sum) or probs_sum <= 0.0:
            return tokens[0], 0.0
        probs = probs / probs_sum
        index = int(self.rng.choice(len(tokens), p=probs))
        return tokens[index], float(np.log(probs[index]))

    def _filter_tokens(self, valid_tokens: tuple[str, ...]) -> tuple[str, ...]:
        if self.token_filter is None:
            return valid_tokens
        filtered = self.token_filter(valid_tokens)
        if filtered:
            return filtered
        if "<EOS>" in valid_tokens:
            return ("<EOS>",)
        return valid_tokens
