from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    family: str
    risk_level: str
    commutative: bool = False


OPERATOR_REGISTRY: dict[str, OperatorSpec] = {
    "NEG": OperatorSpec(name="NEG", arity=1, family="unary", risk_level="low"),
    "ABS": OperatorSpec(name="ABS", arity=1, family="unary", risk_level="low"),
    "RANK": OperatorSpec(name="RANK", arity=1, family="unary", risk_level="low"),
    "DELAY_1": OperatorSpec(name="DELAY_1", arity=1, family="unary", risk_level="low"),
    "DELTA_1": OperatorSpec(name="DELTA_1", arity=1, family="unary", risk_level="medium"),
    "TS_MEAN_5": OperatorSpec(name="TS_MEAN_5", arity=1, family="rolling", risk_level="low"),
    "TS_STD_5": OperatorSpec(name="TS_STD_5", arity=1, family="rolling", risk_level="medium"),
    "ADD": OperatorSpec(name="ADD", arity=2, family="binary", risk_level="medium", commutative=True),
    "SUB": OperatorSpec(name="SUB", arity=2, family="binary", risk_level="medium"),
    "MUL": OperatorSpec(name="MUL", arity=2, family="binary", risk_level="high", commutative=True),
    "DIV": OperatorSpec(name="DIV", arity=2, family="binary", risk_level="high"),
    "CORR_5": OperatorSpec(
        name="CORR_5",
        arity=2,
        family="paired_rolling",
        risk_level="medium",
        commutative=True,
    ),
}


def get_operator(name: str) -> OperatorSpec:
    try:
        return OPERATOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown operator {name!r}.") from exc
