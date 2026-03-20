from __future__ import annotations

from dataclasses import dataclass

from ..domain.feature_registry import get_feature
from ..domain.operator_registry import get_operator
from ..domain.unit_system import Unit, UnitError, infer_binary_unit, infer_unary_unit
from .ast import ExpressionNode
from .canonicalizer import canonical_key


class SemanticError(ValueError):
    """Raised when a token sequence violates financial or language constraints."""


@dataclass(frozen=True)
class ExpressionInfo:
    node: ExpressionNode
    unit: Unit
    needs_macro_delay: bool
    canonical: str


class SemanticRuleEngine:
    """Shared semantic protocol for both the parser and grammar mask."""

    def make_feature(self, name: str) -> ExpressionInfo:
        feature = get_feature(name)
        node = ExpressionNode(token=feature.name)
        return ExpressionInfo(
            node=node,
            unit=feature.unit,
            needs_macro_delay=feature.requires_delay,
            canonical=feature.name,
        )

    def apply_operator(
        self,
        operator_name: str,
        operands: tuple[ExpressionInfo, ...],
    ) -> ExpressionInfo:
        spec = get_operator(operator_name)
        if len(operands) != spec.arity:
            raise SemanticError(
                f"{operator_name} expects {spec.arity} operands, got {len(operands)}."
            )
        if spec.arity == 1:
            return self._apply_unary(operator_name, operands[0])
        return self._apply_binary(operator_name, operands[0], operands[1])

    def _apply_unary(self, operator_name: str, operand: ExpressionInfo) -> ExpressionInfo:
        if operand.needs_macro_delay and operator_name != "DELAY_1":
            raise SemanticError("Macro features must pass through DELAY_1 before further use.")
        if operand.node.token == operator_name:
            raise SemanticError(f"Adjacent repeated unary operator {operator_name} is forbidden.")
        try:
            unit = infer_unary_unit(operator_name, operand.unit)
        except UnitError as exc:
            raise SemanticError(str(exc)) from exc
        node = ExpressionNode(token=operator_name, children=(operand.node,))
        return ExpressionInfo(
            node=node,
            unit=unit,
            needs_macro_delay=False,
            canonical=canonical_key(node),
        )

    def _apply_binary(
        self,
        operator_name: str,
        lhs: ExpressionInfo,
        rhs: ExpressionInfo,
    ) -> ExpressionInfo:
        if lhs.needs_macro_delay or rhs.needs_macro_delay:
            raise SemanticError("Macro features must pass through DELAY_1 before binary operators.")
        if lhs.canonical == rhs.canonical:
            raise SemanticError(f"Degenerate binary form {operator_name}({lhs.canonical}, {rhs.canonical}).")
        try:
            unit = infer_binary_unit(operator_name, lhs.unit, rhs.unit)
        except UnitError as exc:
            raise SemanticError(str(exc)) from exc
        node = ExpressionNode(token=operator_name, children=(lhs.node, rhs.node))
        return ExpressionInfo(
            node=node,
            unit=unit,
            needs_macro_delay=False,
            canonical=canonical_key(node),
        )
