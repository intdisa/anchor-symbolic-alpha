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
        name="gold_dxy_inverse",
        features=("GOLD_CLOSE", "DXY"),
        relation="negative",
        description="Gold often weakens when the U.S. dollar strengthens.",
    ),
    PriorRule(
        name="gold_real_rate_inverse",
        features=("GOLD_CLOSE", "TNX"),
        relation="negative",
        description="Higher rates typically pressure non-yielding safe-haven assets.",
    ),
    PriorRule(
        name="gold_risk_off_support",
        features=("GOLD_CLOSE", "VIX"),
        relation="positive",
        description="Gold can strengthen during risk-off volatility regimes.",
    ),
)
