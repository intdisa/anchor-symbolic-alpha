from __future__ import annotations

from .ast import ExpressionNode
from ..domain.operator_registry import OPERATOR_REGISTRY


_COMMUTATIVE_OPERATORS = frozenset(
    name for name, spec in OPERATOR_REGISTRY.items() if spec.commutative
)


def canonicalize(node: ExpressionNode) -> ExpressionNode:
    if node.is_leaf:
        return node
    children = tuple(canonicalize(child) for child in node.children)
    if node.token in _COMMUTATIVE_OPERATORS:
        children = tuple(sorted(children, key=canonical_key))
    return ExpressionNode(token=node.token, children=children)


def canonical_key(node: ExpressionNode) -> str:
    node = canonicalize(node)
    if node.is_leaf:
        return node.token
    return f"{node.token}(" + ",".join(canonical_key(child) for child in node.children) + ")"


def to_rpn_tokens(node: ExpressionNode) -> tuple[str, ...]:
    if node.is_leaf:
        return (node.token,)
    tokens: list[str] = []
    for child in node.children:
        tokens.extend(to_rpn_tokens(child))
    tokens.append(node.token)
    return tuple(tokens)
