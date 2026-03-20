from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..domain.unit_system import Unit
from ..language.ast import ExpressionNode
from ..language.parser import ParseError, ParsedExpression, RPNParser


class EvaluationError(ValueError):
    """Raised when a parsed formula cannot produce a valid signal."""


@dataclass(frozen=True)
class EvaluatedFormula:
    parsed: ParsedExpression
    signal: pd.Series
    unit: Unit


class FormulaEvaluator:
    def __init__(self, parser: RPNParser | None = None) -> None:
        self.parser = parser or RPNParser()

    def evaluate(
        self,
        formula: str | list[str] | tuple[str, ...],
        data: pd.DataFrame,
    ) -> EvaluatedFormula:
        try:
            parsed = (
                self.parser.parse_text(formula)
                if isinstance(formula, str)
                else self.parser.parse(formula)
            )
        except ParseError as exc:
            raise EvaluationError(str(exc)) from exc
        signal = self._evaluate_node(parsed.root, data).replace([np.inf, -np.inf], np.nan)
        if signal.dropna().empty:
            raise EvaluationError("Formula evaluation produced only NaN values.")
        if float(signal.dropna().std(ddof=0)) == 0.0:
            raise EvaluationError("Formula evaluation produced a zero-variance signal.")
        signal.name = parsed.canonical
        return EvaluatedFormula(parsed=parsed, signal=signal, unit=parsed.unit)

    def _evaluate_node(self, node: ExpressionNode, data: pd.DataFrame) -> pd.Series:
        if node.is_leaf:
            if node.token not in data.columns:
                raise EvaluationError(f"Feature column {node.token!r} is missing from the input frame.")
            return pd.to_numeric(data[node.token], errors="coerce").astype(float)
        children = [self._evaluate_node(child, data) for child in node.children]
        if node.token == "NEG":
            return -children[0]
        if node.token == "ABS":
            return children[0].abs()
        if node.token == "RANK":
            return children[0].rank(pct=True)
        if node.token == "DELAY_1":
            return children[0].shift(1)
        if node.token == "DELTA_1":
            return children[0].diff()
        if node.token == "TS_MEAN_5":
            return children[0].rolling(window=5, min_periods=5).mean()
        if node.token == "TS_STD_5":
            return children[0].rolling(window=5, min_periods=5).std()
        if node.token == "ADD":
            return children[0] + children[1]
        if node.token == "SUB":
            return children[0] - children[1]
        if node.token == "MUL":
            return children[0] * children[1]
        if node.token == "DIV":
            denominator = children[1].replace(0.0, np.nan)
            return children[0] / denominator
        if node.token == "CORR_5":
            return children[0].rolling(window=5, min_periods=5).corr(children[1])
        raise EvaluationError(f"Unsupported operator {node.token!r}.")
