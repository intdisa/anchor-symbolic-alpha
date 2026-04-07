from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..domain.unit_system import Unit
from ..language.ast import ExpressionNode
from ..language.parser import ParseError, ParsedExpression, RPNParser
from .evaluator import EvaluationError


@dataclass(frozen=True)
class CrossSectionalEvaluatedFormula:
    parsed: ParsedExpression
    signal: pd.Series
    unit: Unit


class CrossSectionalFormulaEvaluator:
    def __init__(
        self,
        parser: RPNParser | None = None,
        *,
        entity_column: str = "permno",
        time_column: str = "date",
        cross_section_column: str = "date",
    ) -> None:
        self.parser = parser or RPNParser()
        self.entity_column = entity_column
        self.time_column = time_column
        self.cross_section_column = cross_section_column

    def evaluate(
        self,
        formula: str | list[str] | tuple[str, ...],
        data: pd.DataFrame,
    ) -> CrossSectionalEvaluatedFormula:
        try:
            parsed = (
                self.parser.parse_text(formula)
                if isinstance(formula, str)
                else self.parser.parse(formula)
            )
        except ParseError as exc:
            raise EvaluationError(str(exc)) from exc

        prepared = self._prepare_panel(data)
        signal = self._evaluate_node(parsed.root, prepared)
        signal = signal.replace([np.inf, -np.inf], np.nan)
        signal = signal.loc[prepared.index]
        if signal.dropna().empty:
            raise EvaluationError("Formula evaluation produced only NaN values.")
        if float(signal.dropna().std(ddof=0)) == 0.0:
            raise EvaluationError("Formula evaluation produced a zero-variance signal.")
        restored = pd.Series(signal.to_numpy(), index=prepared["_row_order"]).sort_index()
        restored.index = data.index
        restored.name = parsed.canonical
        return CrossSectionalEvaluatedFormula(parsed=parsed, signal=restored, unit=parsed.unit)

    def _prepare_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {self.entity_column, self.time_column, self.cross_section_column}
        missing = required.difference(data.columns)
        if missing:
            raise EvaluationError(f"Missing panel columns: {sorted(missing)!r}.")
        prepared = data.copy()
        prepared["_row_order"] = np.arange(len(prepared))
        prepared[self.time_column] = pd.to_datetime(prepared[self.time_column], errors="coerce")
        prepared[self.cross_section_column] = pd.to_datetime(prepared[self.cross_section_column], errors="coerce")
        prepared = prepared.sort_values([self.entity_column, self.time_column, "_row_order"]).reset_index(drop=True)
        prepared.index = prepared["_row_order"]
        return prepared

    def _evaluate_node(self, node: ExpressionNode, data: pd.DataFrame) -> pd.Series:
        if node.is_leaf:
            if node.token not in data.columns:
                raise EvaluationError(f"Feature column {node.token!r} is missing from the input frame.")
            return pd.to_numeric(data[node.token], errors="coerce").astype(float)

        children = [self._evaluate_node(child, data) for child in node.children]
        entity_groups = data[self.entity_column]
        cross_section_groups = data[self.cross_section_column]

        if node.token == "NEG":
            return -children[0]
        if node.token == "ABS":
            return children[0].abs()
        if node.token == "RANK":
            return children[0].groupby(cross_section_groups, sort=False).rank(pct=True)
        if node.token == "DELAY_1":
            return children[0].groupby(entity_groups, sort=False).shift(1)
        if node.token == "DELTA_1":
            return children[0].groupby(entity_groups, sort=False).diff()
        if node.token == "TS_MEAN_5":
            return self._grouped_rolling(children[0], entity_groups, "mean")
        if node.token == "TS_STD_5":
            return self._grouped_rolling(children[0], entity_groups, "std")
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
            return self._grouped_corr(children[0], children[1], entity_groups)
        raise EvaluationError(f"Unsupported operator {node.token!r}.")

    @staticmethod
    def _grouped_rolling(series: pd.Series, groups: pd.Series, method: str) -> pd.Series:
        frame = pd.DataFrame({"group": groups.to_numpy(), "value": series.to_numpy()}, index=series.index)
        if method == "mean":
            values = (
                frame.groupby("group", sort=False)["value"]
                .rolling(window=5, min_periods=5)
                .mean()
                .reset_index(level=0, drop=True)
            )
        elif method == "std":
            values = (
                frame.groupby("group", sort=False)["value"]
                .rolling(window=5, min_periods=5)
                .std()
                .reset_index(level=0, drop=True)
            )
        else:
            raise ValueError(f"Unsupported rolling method {method!r}.")
        values.index = series.index
        return values

    @staticmethod
    def _grouped_corr(left: pd.Series, right: pd.Series, groups: pd.Series) -> pd.Series:
        frame = pd.DataFrame(
            {"group": groups.to_numpy(), "left": left.to_numpy(), "right": right.to_numpy()},
            index=left.index,
        )
        corr = (
            frame.groupby("group", sort=False)[["left", "right"]]
            .apply(lambda item: item["left"].rolling(window=5, min_periods=5).corr(item["right"]))
            .reset_index(level=0, drop=True)
        )
        corr.index = left.index
        return corr
