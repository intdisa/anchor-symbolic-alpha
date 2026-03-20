from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    description: str
    triggers: tuple[str, ...]


REGIME_REGISTRY: dict[str, RegimeSpec] = {
    "HIGH_VOLATILITY": RegimeSpec(
        name="HIGH_VOLATILITY",
        description="Risk-off state dominated by volatility shocks.",
        triggers=("VIX",),
    ),
    "RATE_HIKING": RegimeSpec(
        name="RATE_HIKING",
        description="Policy tightening regime dominated by rising yields.",
        triggers=("TNX",),
    ),
    "INFLATION_SHOCK": RegimeSpec(
        name="INFLATION_SHOCK",
        description="Inflation surprise regime dominated by CPI prints.",
        triggers=("CPI",),
    ),
    "USD_STRENGTH": RegimeSpec(
        name="USD_STRENGTH",
        description="Broad dollar strength regime.",
        triggers=("DXY",),
    ),
}
