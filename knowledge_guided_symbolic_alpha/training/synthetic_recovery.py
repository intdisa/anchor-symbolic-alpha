from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation import FactorPool, FormulaEvaluator
from ..evaluation.pool_scoring import evaluate_formula_record


@dataclass(frozen=True)
class SyntheticRecoveryExample:
    dataset_name: str
    role: str
    regime: str
    allowed_features: frozenset[str]
    formula_tokens: tuple[str, ...]
    frame: pd.DataFrame
    target: pd.Series
    validation_frame: pd.DataFrame
    validation_target: pd.Series
    pool: FactorPool


class SyntheticRecoveryDatasetBuilder:
    def __init__(self, seed: int = 7, length: int = 96) -> None:
        self.seed = seed
        self.length = length
        self.evaluator = FormulaEvaluator()
        self.templates = {
            "quality_solvency": (
                ("CASH_RATIO_Q", "RANK", "PROFITABILITY_Q", "RANK", "ADD"),
                ("PROFITABILITY_Q", "RANK", "LEVERAGE_Q", "RANK", "SUB"),
                ("CASH_RATIO_Q", "RANK"),
            ),
            "efficiency_growth": (
                ("SALES_TO_ASSETS_Q", "RANK"),
                ("ASSET_GROWTH_A", "RANK", "NEG"),
                ("SALES_TO_ASSETS_Q", "RANK", "ASSET_GROWTH_A", "RANK", "SUB"),
            ),
            "valuation_size": (
                ("BOOK_TO_MARKET_Q", "RANK"),
                ("BOOK_TO_MARKET_A", "RANK"),
                ("BOOK_TO_MARKET_Q", "RANK", "SIZE_LOG_MCAP", "RANK", "SUB"),
            ),
            "short_horizon_flow": (
                ("RET_1", "NEG"),
                ("RET_5", "NEG"),
                ("VOLATILITY_20", "TS_MEAN_5", "NEG"),
                ("AMIHUD_20", "RANK", "NEG"),
            ),
        }

    def build(self, num_examples: int) -> list[SyntheticRecoveryExample]:
        rng = np.random.default_rng(self.seed)
        examples: list[SyntheticRecoveryExample] = []
        roles = tuple(self.templates)
        regimes = ("BALANCED", "HIGH_VOLATILITY", "USD_STRENGTH")
        attempts = 0
        while len(examples) < num_examples and attempts < num_examples * 20:
            attempts += 1
            role = roles[int(rng.integers(len(roles)))]
            regime = regimes[int(rng.integers(len(regimes)))]
            frame = self._build_frame(regime, rng)
            allowed_features = self._allowed_features(role)
            template = self.templates[role][int(rng.integers(len(self.templates[role])))]
            formula_tokens = tuple(template)

            split = int(self.length * 0.75)
            train_frame = frame.iloc[:split].copy()
            valid_frame = frame.iloc[split:].copy()
            target = self._target_from_formula(formula_tokens, frame)
            train_target = target.iloc[:split].copy()
            valid_target = target.iloc[split:].copy()
            if train_target.abs().sum() == 0.0 or valid_target.abs().sum() == 0.0:
                continue

            try:
                evaluate_formula_record(formula_tokens, train_frame, train_target, role=role)
            except Exception:
                continue

            pool = self._build_pool(role, train_frame, train_target)
            examples.append(
                SyntheticRecoveryExample(
                    dataset_name="us_equities",
                    role=role,
                    regime=regime,
                    allowed_features=allowed_features,
                    formula_tokens=formula_tokens,
                    frame=train_frame,
                    target=train_target,
                    validation_frame=valid_frame,
                    validation_target=valid_target,
                    pool=pool,
                )
            )
        if len(examples) < num_examples:
            raise RuntimeError(f"Only generated {len(examples)} synthetic examples out of requested {num_examples}.")
        return examples

    def _build_frame(self, regime: str, rng: np.random.Generator) -> pd.DataFrame:
        index = pd.date_range("2018-01-01", periods=self.length, freq="D")
        regime_scale = {
            "BALANCED": 1.0,
            "HIGH_VOLATILITY": 1.8,
            "USD_STRENGTH": 1.2,
        }[regime]

        ret_1 = pd.Series(rng.normal(0.0004, 0.012 * regime_scale, size=self.length), index=index)
        ret_5 = (1.0 + ret_1).rolling(5, min_periods=1).apply(np.prod, raw=True) - 1.0
        ret_20 = (1.0 + ret_1).rolling(20, min_periods=1).apply(np.prod, raw=True) - 1.0
        volatility_20 = ret_1.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)

        turnover_20 = pd.Series(0.02 + np.abs(rng.normal(0.0, 0.008 * regime_scale, size=self.length)), index=index)
        dollar_volume_20 = pd.Series(
            1_000_000.0 * (1.0 + np.abs(rng.normal(0.0, 0.20 * regime_scale, size=self.length))),
            index=index,
        )
        amihud_20 = (ret_1.abs() / dollar_volume_20.replace(0.0, np.nan)).rolling(20, min_periods=1).mean().fillna(0.0)
        price_to_252_high = pd.Series(np.minimum(0.0, np.cumsum(ret_1.values) / 10.0), index=index)

        size_log_mcap = pd.Series(9.0 + np.cumsum(rng.normal(0.0, 0.01, size=self.length)), index=index)
        book_to_market_q = pd.Series(0.4 + np.cumsum(rng.normal(0.0, 0.005, size=self.length)), index=index)
        book_to_market_a = pd.Series(0.5 + np.cumsum(rng.normal(0.0, 0.004, size=self.length)), index=index)
        profitability_q = pd.Series(0.08 + np.cumsum(rng.normal(0.0, 0.002, size=self.length)), index=index)
        profitability_a = pd.Series(0.09 + np.cumsum(rng.normal(0.0, 0.0015, size=self.length)), index=index)
        asset_growth_a = pd.Series(0.04 + np.cumsum(rng.normal(0.0, 0.002, size=self.length)), index=index)
        leverage_q = pd.Series(0.45 + np.cumsum(rng.normal(0.0, 0.003, size=self.length)), index=index)
        leverage_a = pd.Series(0.48 + np.cumsum(rng.normal(0.0, 0.002, size=self.length)), index=index)
        cash_ratio_q = pd.Series(0.12 + np.cumsum(rng.normal(0.0, 0.002, size=self.length)), index=index)
        sales_to_assets_q = pd.Series(0.30 + np.cumsum(rng.normal(0.0, 0.003, size=self.length)), index=index)

        frame = pd.DataFrame(
            {
                "RET_1": ret_1,
                "RET_5": ret_5,
                "RET_20": ret_20,
                "VOLATILITY_20": volatility_20,
                "TURNOVER_20": turnover_20,
                "DOLLAR_VOLUME_20": dollar_volume_20,
                "AMIHUD_20": amihud_20,
                "PRICE_TO_252_HIGH": price_to_252_high,
                "SIZE_LOG_MCAP": size_log_mcap,
                "BOOK_TO_MARKET_Q": book_to_market_q,
                "BOOK_TO_MARKET_A": book_to_market_a,
                "PROFITABILITY_Q": profitability_q,
                "PROFITABILITY_A": profitability_a,
                "ASSET_GROWTH_A": asset_growth_a,
                "LEVERAGE_Q": leverage_q,
                "LEVERAGE_A": leverage_a,
                "CASH_RATIO_Q": cash_ratio_q,
                "SALES_TO_ASSETS_Q": sales_to_assets_q,
                "CPI": 100.0 + np.cumsum(rng.normal(0.02, 0.01, size=self.length)),
                "TNX": 2.0 + np.cumsum(rng.normal(0.0, 0.015 * regime_scale, size=self.length)),
                "VIX": np.clip(18.0 + rng.normal(0.0, 2.0 * regime_scale, size=self.length).cumsum(), 8.0, None),
                "DXY": 100.0 + np.cumsum(rng.normal(0.0, 0.12 * regime_scale, size=self.length)),
            },
            index=index,
        )
        return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _target_from_formula(self, formula_tokens: tuple[str, ...], frame: pd.DataFrame) -> pd.Series:
        signal = self.evaluator.evaluate(formula_tokens, frame).signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ranked = signal.rank(pct=True).sub(0.5).fillna(0.0)
        return ranked.shift(-1).fillna(0.0)

    def _allowed_features(self, role: str) -> frozenset[str]:
        if role == "quality_solvency":
            return frozenset({"CASH_RATIO_Q", "PROFITABILITY_Q", "PROFITABILITY_A", "LEVERAGE_Q", "LEVERAGE_A", "SIZE_LOG_MCAP"})
        if role == "efficiency_growth":
            return frozenset({"SALES_TO_ASSETS_Q", "ASSET_GROWTH_A", "PROFITABILITY_A", "SIZE_LOG_MCAP"})
        if role == "valuation_size":
            return frozenset({"BOOK_TO_MARKET_Q", "BOOK_TO_MARKET_A", "SIZE_LOG_MCAP"})
        if role == "short_horizon_flow":
            return frozenset({"RET_1", "RET_5", "VOLATILITY_20", "TURNOVER_20", "DOLLAR_VOLUME_20", "AMIHUD_20", "PRICE_TO_252_HIGH"})
        return frozenset(self._all_features())

    def _all_features(self) -> tuple[str, ...]:
        return (
            "RET_1",
            "RET_5",
            "RET_20",
            "VOLATILITY_20",
            "TURNOVER_20",
            "DOLLAR_VOLUME_20",
            "AMIHUD_20",
            "PRICE_TO_252_HIGH",
            "SIZE_LOG_MCAP",
            "BOOK_TO_MARKET_Q",
            "BOOK_TO_MARKET_A",
            "PROFITABILITY_Q",
            "PROFITABILITY_A",
            "ASSET_GROWTH_A",
            "LEVERAGE_Q",
            "LEVERAGE_A",
            "CASH_RATIO_Q",
            "SALES_TO_ASSETS_Q",
            "CPI",
            "TNX",
            "VIX",
            "DXY",
        )

    def _build_pool(self, role: str, frame: pd.DataFrame, target: pd.Series) -> FactorPool:
        pool = FactorPool(max_size=8)
        if role == "short_horizon_flow":
            baseline_formula = ("CASH_RATIO_Q", "RANK")
            pool.add(evaluate_formula_record(baseline_formula, frame, target, role="quality_solvency"))
        elif role in {"efficiency_growth", "valuation_size"}:
            baseline_formula = ("RET_1", "NEG")
            pool.add(evaluate_formula_record(baseline_formula, frame, target, role="short_horizon_flow"))
        return pool
