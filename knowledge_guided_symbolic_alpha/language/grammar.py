from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.operator_registry import OPERATOR_REGISTRY, get_operator
from .semantic_rules import ExpressionInfo, SemanticError, SemanticRuleEngine
from .tokens import BODY_TOKENS, EOS_TOKEN


class GrammarError(ValueError):
    """Raised when a grammar transition is invalid."""


@dataclass(frozen=True)
class GrammarState:
    body_tokens: tuple[str, ...] = field(default_factory=tuple)
    stack: tuple[ExpressionInfo, ...] = field(default_factory=tuple)
    finished: bool = False
    terminal_error: str | None = None


class RPNGrammar:
    def __init__(
        self,
        max_length: int = 15,
        min_length: int = 3,
        force_eos_on_empty_mask: bool = True,
        rule_engine: SemanticRuleEngine | None = None,
    ) -> None:
        if min_length < 1 or max_length < min_length:
            raise ValueError("Require 1 <= min_length <= max_length.")
        self.max_length = max_length
        self.min_length = min_length
        self.force_eos_on_empty_mask = force_eos_on_empty_mask
        self.rule_engine = rule_engine or SemanticRuleEngine()

    def initial_state(self) -> GrammarState:
        return GrammarState()

    def valid_next_tokens(self, state: GrammarState) -> tuple[str, ...]:
        if state.finished:
            return ()
        candidates: list[str] = []
        if len(state.body_tokens) < self.max_length:
            for token in BODY_TOKENS:
                try:
                    next_state = self._apply_body_token(state, token)
                except GrammarError:
                    continue
                remaining_slots = self.max_length - len(next_state.body_tokens)
                if self._can_finish_within_budget(next_state, remaining_slots):
                    candidates.append(token)
        if self._can_end(state):
            candidates.append(EOS_TOKEN)
        if not candidates and self.force_eos_on_empty_mask:
            return (EOS_TOKEN,)
        return tuple(candidates)

    def step(self, state: GrammarState, token: str) -> GrammarState:
        if state.finished:
            raise GrammarError("Cannot step a finished grammar state.")
        if token == EOS_TOKEN:
            if self._can_end(state):
                return GrammarState(
                    body_tokens=state.body_tokens,
                    stack=state.stack,
                    finished=True,
                    terminal_error=None,
                )
            return GrammarState(
                body_tokens=state.body_tokens,
                stack=state.stack,
                finished=True,
                terminal_error="forced_eos",
            )
        return self._apply_body_token(state, token)

    def is_valid_terminal(self, state: GrammarState) -> bool:
        return state.finished and state.terminal_error is None and self._can_end(state)

    def _apply_body_token(self, state: GrammarState, token: str) -> GrammarState:
        if token not in BODY_TOKENS:
            raise GrammarError(f"{token!r} is not a valid body token.")
        if len(state.body_tokens) >= self.max_length:
            raise GrammarError("Maximum body length reached.")
        stack = list(state.stack)
        if token in OPERATOR_REGISTRY:
            spec = get_operator(token)
            if len(stack) < spec.arity:
                raise GrammarError(f"{token} needs {spec.arity} operands, got stack depth {len(stack)}.")
            operands = tuple(stack[-spec.arity :])
            del stack[-spec.arity :]
            try:
                stack.append(self.rule_engine.apply_operator(token, operands))
            except SemanticError as exc:
                raise GrammarError(str(exc)) from exc
        else:
            try:
                stack.append(self.rule_engine.make_feature(token))
            except KeyError as exc:
                raise GrammarError(str(exc)) from exc
        return GrammarState(body_tokens=state.body_tokens + (token,), stack=tuple(stack))

    def _can_end(self, state: GrammarState) -> bool:
        return (
            len(state.body_tokens) >= self.min_length
            and len(state.stack) == 1
            and not state.stack[0].needs_macro_delay
        )

    def _can_finish_within_budget(self, state: GrammarState, remaining_slots: int) -> bool:
        if remaining_slots < 0 or not state.stack:
            return False
        pending_delay = sum(1 for expr in state.stack if expr.needs_macro_delay)
        lower_bound = pending_delay + max(0, len(state.stack) - 1)
        lower_bound = max(lower_bound, self.min_length - len(state.body_tokens))
        return lower_bound <= remaining_slots or self._can_end(state)
