from __future__ import annotations

from dataclasses import dataclass
import re

from .unit_system import Unit


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    frequency: str
    category: str
    unit: Unit
    is_macro: bool
    is_micro: bool
    requires_delay: bool


def _feature(
    name: str,
    category: str,
    unit: Unit,
    *,
    frequency: str = "daily",
    is_macro: bool,
    is_micro: bool,
    requires_delay: bool = False,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        frequency=frequency,
        category=category,
        unit=unit,
        is_macro=is_macro,
        is_micro=is_micro,
        requires_delay=requires_delay,
    )


def _us_equities_panel_features() -> dict[str, FeatureSpec]:
    specs = {
        "RET_1": ("return_1d", Unit.DIMENSIONLESS),
        "RET_5": ("return_5d", Unit.DIMENSIONLESS),
        "RET_20": ("return_20d", Unit.DIMENSIONLESS),
        "VOLATILITY_20": ("volatility_20d", Unit.DIMENSIONLESS),
        "TURNOVER_20": ("turnover_20d", Unit.DIMENSIONLESS),
        "DOLLAR_VOLUME_20": ("dollar_volume_20d", Unit.DIMENSIONLESS),
        "AMIHUD_20": ("amihud_20d", Unit.DIMENSIONLESS),
        "PRICE_TO_252_HIGH": ("distance_to_252_high", Unit.DIMENSIONLESS),
        "SIZE_LOG_MCAP": ("log_market_cap", Unit.DIMENSIONLESS),
        "BOOK_TO_MARKET_Q": ("book_to_market_quarterly", Unit.RATIO),
        "BOOK_TO_MARKET_A": ("book_to_market_annual", Unit.RATIO),
        "PROFITABILITY_Q": ("profitability_quarterly", Unit.DIMENSIONLESS),
        "PROFITABILITY_A": ("profitability_annual", Unit.DIMENSIONLESS),
        "ASSET_GROWTH_A": ("asset_growth_annual", Unit.DIMENSIONLESS),
        "LEVERAGE_Q": ("leverage_quarterly", Unit.RATIO),
        "LEVERAGE_A": ("leverage_annual", Unit.RATIO),
        "CASH_RATIO_Q": ("cash_ratio_quarterly", Unit.RATIO),
        "SALES_TO_ASSETS_Q": ("sales_to_assets_quarterly", Unit.RATIO),
    }
    return {
        name: _feature(name, category, unit, is_macro=False, is_micro=True)
        for name, (category, unit) in specs.items()
    }


FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    **_us_equities_panel_features(),
    "CPI": _feature(
        "CPI",
        "inflation",
        Unit.MACRO,
        frequency="monthly",
        is_macro=True,
        is_micro=False,
        requires_delay=True,
    ),
    "TNX": _feature(
        "TNX",
        "yield",
        Unit.MACRO,
        is_macro=True,
        is_micro=False,
        requires_delay=True,
    ),
    "VIX": _feature(
        "VIX",
        "volatility",
        Unit.DIMENSIONLESS,
        is_macro=True,
        is_micro=False,
    ),
    "DXY": _feature(
        "DXY",
        "dollar_index",
        Unit.MACRO,
        is_macro=True,
        is_micro=False,
        requires_delay=True,
    ),
}

_GENERIC_BENCHMARK_FEATURE_RE = re.compile(r"X\d+$")


def is_generic_benchmark_feature(name: str) -> bool:
    return bool(_GENERIC_BENCHMARK_FEATURE_RE.fullmatch(name))


def make_generic_benchmark_feature(name: str) -> FeatureSpec:
    if not is_generic_benchmark_feature(name):
        raise KeyError(f"Unknown feature {name!r}.")
    return _feature(
        name,
        "benchmark_generic",
        Unit.DIMENSIONLESS,
        frequency="generic",
        is_macro=False,
        is_micro=True,
        requires_delay=False,
    )


def get_feature(name: str) -> FeatureSpec:
    try:
        return FEATURE_REGISTRY[name]
    except KeyError as exc:
        if is_generic_benchmark_feature(name):
            return make_generic_benchmark_feature(name)
        raise KeyError(f"Unknown feature {name!r}.") from exc
