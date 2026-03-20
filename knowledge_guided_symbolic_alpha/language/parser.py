from __future__ import annotations

from dataclasses import dataclass

from ..domain.operator_registry import OPERATOR_REGISTRY, get_operator
from ..domain.unit_system import Unit
from .ast import ExpressionNode
from .canonicalizer import canonical_key, canonicalize, to_rpn_tokens
from .semantic_rules import ExpressionInfo, SemanticError, SemanticRuleEngine
from .tokens import is_special_token


class ParseError(ValueError):
    """Raised when an RPN formula is syntactically or semantically invalid."""


@dataclass(frozen=True)
class ParsedExpression:
    tokens: tuple[str, ...]
    root: ExpressionNode
    canonical: str
    canonical_rpn: tuple[str, ...]
    unit: Unit


class RPNParser:
    def __init__(self, rule_engine: SemanticRuleEngine | None = None) -> None:
        self.rule_engine = rule_engine or SemanticRuleEngine()

    def parse(self, tokens: list[str] | tuple[str, ...]) -> ParsedExpression:
        body = tuple(tokens)
        if not body:
            raise ParseError("Formula body cannot be empty.")
        stack: list[ExpressionInfo] = []
        for token in body:
            if is_special_token(token):
                raise ParseError(f"Special token {token!r} is not allowed in the formula body.")
            if token in OPERATOR_REGISTRY:
                stack = self._apply_operator(token, stack)
                continue
            try:
                stack.append(self.rule_engine.make_feature(token))
            except KeyError as exc:
                raise ParseError(str(exc)) from exc
        if len(stack) != 1:
            raise ParseError(f"Formula must reduce to one expression, got stack depth {len(stack)}.")
        summary = stack[0]
        if summary.needs_macro_delay:
            raise ParseError("Formula terminates with a macro feature that still requires delay.")
        canonical_root = canonicalize(summary.node)
        return ParsedExpression(
            tokens=body,
            root=summary.node,
            canonical=canonical_key(canonical_root),
            canonical_rpn=to_rpn_tokens(canonical_root),
            unit=summary.unit,
        )

    def parse_text(self, expression: str) -> ParsedExpression:
        return self.parse(expression.split())

    def _apply_operator(self, token: str, stack: list[ExpressionInfo]) -> list[ExpressionInfo]:
        spec = get_operator(token)
        if len(stack) < spec.arity:
            raise ParseError(f"{token} needs {spec.arity} operands, got stack depth {len(stack)}.")
        operands = tuple(stack[-spec.arity :])
        del stack[-spec.arity :]
        try:
            stack.append(self.rule_engine.apply_operator(token, operands))
        except SemanticError as exc:
            raise ParseError(str(exc)) from exc
        return stack
