from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..language import EOS_TOKEN, GrammarState, RPNGrammar


@dataclass(frozen=True)
class TSLTransition:
    state: GrammarState
    action: str
    next_state: GrammarState
    done: bool


class TreeStructuredLanguageMDP:
    def __init__(
        self,
        grammar: RPNGrammar,
        token_filter: Callable[[tuple[str, ...]], tuple[str, ...]] | None = None,
    ) -> None:
        self.grammar = grammar
        self.token_filter = token_filter

    def reset(self) -> GrammarState:
        return self.grammar.initial_state()

    def valid_actions(self, state: GrammarState) -> tuple[str, ...]:
        actions = self.grammar.valid_next_tokens(state)
        if self.token_filter is None:
            return actions
        filtered = self.token_filter(actions)
        if filtered:
            return filtered
        if EOS_TOKEN in actions:
            return (EOS_TOKEN,)
        return actions

    def step(self, state: GrammarState, action: str) -> TSLTransition:
        next_state = self.grammar.step(state, action)
        return TSLTransition(
            state=state,
            action=action,
            next_state=next_state,
            done=next_state.finished,
        )

    def is_terminal(self, state: GrammarState) -> bool:
        return state.finished

    def terminal_tokens(self, state: GrammarState) -> tuple[str, ...]:
        return state.body_tokens
