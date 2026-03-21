from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorRule:
    name: str
    features: tuple[str, ...]
    relation: str
    description: str


PRIOR_RULES: tuple[PriorRule, ...] = (
    PriorRule(
        name="cash_quality_anchor",
        features=("CASH_RATIO_Q", "PROFITABILITY_Q"),
        relation="positive",
        description="Cash-rich firms with stronger profitability tend to rank better cross-sectionally.",
    ),
    PriorRule(
        name="quality_over_leverage",
        features=("PROFITABILITY_Q", "LEVERAGE_Q"),
        relation="negative",
        description="Higher leverage usually weakens the quality signal from profitability.",
    ),
    PriorRule(
        name="efficiency_quality_pair",
        features=("SALES_TO_ASSETS_Q", "PROFITABILITY_Q"),
        relation="positive",
        description="Operating efficiency combined with profitability can strengthen quality-style selection.",
    ),
)
