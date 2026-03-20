from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cross_sectional_evaluator import (
    CrossSectionalEvaluatedFormula,
    CrossSectionalFormulaEvaluator,
)
from .cross_sectional_metrics import (
    cross_sectional_ic_summary,
    cross_sectional_risk_summary,
    cross_sectional_stability_summary,
)
from .evaluator import EvaluatedFormula, FormulaEvaluator
from .ic_metrics import ic_summary
from .risk_metrics import risk_summary


@dataclass(frozen=True)
class EvaluatedMetrics:
    evaluated: EvaluatedFormula | CrossSectionalEvaluatedFormula
    metrics: dict[str, float]
    cross_sectional: bool


def is_cross_sectional_frame(data: pd.DataFrame) -> bool:
    return {"date", "permno"}.issubset(data.columns)


def resolve_formula_evaluator(
    data: pd.DataFrame,
    evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
) -> FormulaEvaluator | CrossSectionalFormulaEvaluator:
    if is_cross_sectional_frame(data):
        if isinstance(evaluator, CrossSectionalFormulaEvaluator):
            return evaluator
        return CrossSectionalFormulaEvaluator()
    if isinstance(evaluator, FormulaEvaluator):
        return evaluator
    return FormulaEvaluator()


def evaluate_formula_metrics(
    formula: str | list[str] | tuple[str, ...],
    data: pd.DataFrame,
    target: pd.Series,
    evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
) -> EvaluatedMetrics:
    resolved_evaluator = resolve_formula_evaluator(data, evaluator=evaluator)
    evaluated = resolved_evaluator.evaluate(formula, data)
    if isinstance(resolved_evaluator, CrossSectionalFormulaEvaluator):
        risk_target = _cross_sectional_return_target(data, target)
        metrics = cross_sectional_ic_summary(evaluated.signal, target, data["date"])
        metrics.update(
            cross_sectional_risk_summary(
                evaluated.signal,
                risk_target,
                data["date"],
                data["permno"],
            )
        )
        metrics.update(
            cross_sectional_stability_summary(
                evaluated.signal,
                risk_target,
                data["date"],
                data["permno"],
            )
        )
        metrics["cross_sectional"] = 1.0
        return EvaluatedMetrics(evaluated=evaluated, metrics=metrics, cross_sectional=True)
    metrics = ic_summary(evaluated.signal, target)
    metrics.update(risk_summary(evaluated.signal, target))
    metrics["cross_sectional"] = 0.0
    return EvaluatedMetrics(evaluated=evaluated, metrics=metrics, cross_sectional=False)


def score_signal_metrics(
    signal: pd.Series,
    data: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    if is_cross_sectional_frame(data):
        risk_target = _cross_sectional_return_target(data, target)
        metrics = cross_sectional_ic_summary(signal, target, data["date"])
        metrics.update(cross_sectional_risk_summary(signal, risk_target, data["date"], data["permno"]))
        metrics.update(cross_sectional_stability_summary(signal, risk_target, data["date"], data["permno"]))
        metrics["cross_sectional"] = 1.0
        return metrics
    metrics = ic_summary(signal, target)
    metrics.update(risk_summary(signal, target))
    metrics["cross_sectional"] = 0.0
    return metrics


def _cross_sectional_return_target(data: pd.DataFrame, target: pd.Series) -> pd.Series:
    if "TARGET_RET_1" in data.columns:
        return data["TARGET_RET_1"]
    return target
