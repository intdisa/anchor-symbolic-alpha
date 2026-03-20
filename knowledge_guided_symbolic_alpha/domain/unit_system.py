from __future__ import annotations

from enum import StrEnum


class Unit(StrEnum):
    PRICE = "price"
    VOLUME = "volume"
    MACRO = "macro"
    DIMENSIONLESS = "dimensionless"
    RATIO = "ratio"


class UnitError(ValueError):
    """Raised when an operator violates the unit protocol."""


_DIMENSIONLESS_LIKE = frozenset({Unit.DIMENSIONLESS, Unit.RATIO})


def infer_unary_unit(operator_name: str, operand: Unit) -> Unit:
    if operator_name in {"NEG", "ABS", "DELAY_1", "DELTA_1", "TS_MEAN_5", "TS_STD_5"}:
        return operand
    if operator_name in {"RANK", "LOG1P_ABS"}:
        return Unit.DIMENSIONLESS
    raise UnitError(f"Unknown unary operator {operator_name!r}.")


def infer_binary_unit(operator_name: str, lhs: Unit, rhs: Unit) -> Unit:
    if operator_name in {"ADD", "SUB"}:
        if lhs != rhs:
            raise UnitError(f"{operator_name} requires matching units, got {lhs} and {rhs}.")
        return lhs
    if operator_name == "MUL":
        if lhs in _DIMENSIONLESS_LIKE:
            return rhs
        if rhs in _DIMENSIONLESS_LIKE:
            return lhs
        raise UnitError(f"MUL requires one dimensionless-like input, got {lhs} and {rhs}.")
    if operator_name == "DIV":
        if rhs in _DIMENSIONLESS_LIKE:
            return lhs
        if lhs == rhs:
            return Unit.RATIO
        raise UnitError(f"DIV requires compatible units, got {lhs} and {rhs}.")
    if operator_name == "CORR_5":
        return Unit.DIMENSIONLESS
    raise UnitError(f"Unknown binary operator {operator_name!r}.")
