from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation import FormulaEvaluator
from ..evaluation import FactorPool
from ..evaluation.pool_scoring import evaluate_formula_record


DATASET_PREFIX = {
    "gold": "GOLD",
    "crude_oil": "CRUDE_OIL",
    "sp500": "SP500",
}


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
            "target_price": (
                ("PREFIX_CLOSE", "DELTA_1", "NEG"),
                ("PREFIX_GAP_RET", "NEG"),
                ("PREFIX_OC_RET", "NEG"),
                ("PREFIX_LOW", "PREFIX_CLOSE", "SUB"),
            ),
            "target_flow_vol": (
                ("PREFIX_REALIZED_VOL_5", "TS_MEAN_5", "NEG"),
                ("PREFIX_VOLUME_ZSCORE_20", "DELTA_1", "NEG"),
                ("PREFIX_HL_SPREAD", "PREFIX_REALIZED_VOL_5", "DELTA_1", "DIV"),
            ),
            "target_flow_gap": (
                ("PREFIX_GAP_RET", "NEG"),
                ("PREFIX_OC_RET", "NEG"),
                ("PREFIX_GAP_RET", "PREFIX_OC_RET", "SUB", "NEG"),
                ("PREFIX_OC_RET", "DELTA_1", "NEG"),
            ),
            "context": (
                ("VIX", "DELTA_1", "NEG"),
                ("DXY", "DELAY_1", "NEG"),
                ("TNX", "DELAY_1", "DXY", "DELAY_1", "CORR_5"),
            ),
        }

    def build(self, num_examples: int) -> list[SyntheticRecoveryExample]:
        rng = np.random.default_rng(self.seed)
        examples: list[SyntheticRecoveryExample] = []
        roles = tuple(self.templates)
        dataset_names = tuple(DATASET_PREFIX)
        regimes = ("BALANCED", "HIGH_VOLATILITY", "USD_STRENGTH")
        attempts = 0
        while len(examples) < num_examples and attempts < num_examples * 20:
            attempts += 1
            dataset_name = dataset_names[int(rng.integers(len(dataset_names)))]
            role = roles[int(rng.integers(len(roles)))]
            regime = regimes[int(rng.integers(len(regimes)))]
            frame = self._build_frame(dataset_name, regime, rng)
            allowed_features = self._allowed_features(dataset_name, role)
            template = self.templates[role][int(rng.integers(len(self.templates[role])))]
            formula_tokens = self._render_template(template, DATASET_PREFIX[dataset_name])

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

            pool = self._build_pool(dataset_name, role, train_frame, train_target)
            examples.append(
                SyntheticRecoveryExample(
                    dataset_name=dataset_name,
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

    def _build_frame(self, dataset_name: str, regime: str, rng: np.random.Generator) -> pd.DataFrame:
        index = pd.date_range("2018-01-01", periods=self.length, freq="D")
        frame = pd.DataFrame(index=index)
        regime_scale = {
            "BALANCED": 1.0,
            "HIGH_VOLATILITY": 1.8,
            "USD_STRENGTH": 1.2,
        }[regime]
        for name, prefix in DATASET_PREFIX.items():
            asset_scale = 1.0 + 0.2 * list(DATASET_PREFIX).index(name)
            ret = rng.normal(0.0005, 0.01 * regime_scale * asset_scale, size=self.length)
            close = 100.0 * np.exp(np.cumsum(ret))
            gap = rng.normal(0.0, 0.006 * regime_scale * asset_scale, size=self.length)
            open_ = np.roll(close, 1) * (1.0 + gap)
            open_[0] = close[0]
            intraday = rng.normal(0.0, 0.005 * regime_scale, size=self.length)
            close = open_ * (1.0 + intraday)
            high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004 * regime_scale, size=self.length)))
            low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004 * regime_scale, size=self.length)))
            volume = np.abs(1_000_000.0 * (1.0 + rng.normal(0.0, 0.12 * regime_scale, size=self.length)))
            ret1 = pd.Series(close, index=index).pct_change().fillna(0.0)
            volume_series = pd.Series(volume, index=index)

            frame[f"{prefix}_CLOSE"] = close
            frame[f"{prefix}_OPEN"] = open_
            frame[f"{prefix}_HIGH"] = high
            frame[f"{prefix}_LOW"] = low
            frame[f"{prefix}_VOLUME"] = volume
            frame[f"{prefix}_HL_SPREAD"] = (pd.Series(high, index=index) - pd.Series(low, index=index)).div(pd.Series(close, index=index)).fillna(0.0)
            frame[f"{prefix}_OC_RET"] = (pd.Series(close, index=index) - pd.Series(open_, index=index)).div(pd.Series(open_, index=index)).fillna(0.0)
            frame[f"{prefix}_GAP_RET"] = pd.Series(open_, index=index).div(pd.Series(close, index=index).shift(1)).sub(1.0).fillna(0.0)
            frame[f"{prefix}_REALIZED_VOL_5"] = ret1.rolling(5, min_periods=1).std(ddof=0).fillna(0.0)
            frame[f"{prefix}_REALIZED_VOL_20"] = ret1.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
            rolling_mean = volume_series.rolling(20, min_periods=1).mean()
            rolling_std = volume_series.rolling(20, min_periods=1).std(ddof=0).replace(0.0, np.nan)
            frame[f"{prefix}_VOLUME_ZSCORE_20"] = volume_series.sub(rolling_mean).div(rolling_std).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        frame["CPI"] = 100.0 + np.cumsum(rng.normal(0.02, 0.01, size=self.length))
        frame["TNX"] = 2.0 + np.cumsum(rng.normal(0.0, 0.015 * regime_scale, size=self.length))
        frame["VIX"] = np.clip(18.0 + rng.normal(0.0, 2.0 * regime_scale, size=self.length).cumsum(), 8.0, None)
        frame["DXY"] = 100.0 + np.cumsum(rng.normal(0.0, 0.12 * regime_scale, size=self.length))
        return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _target_from_formula(self, formula_tokens: tuple[str, ...], frame: pd.DataFrame) -> pd.Series:
        signal = self.evaluator.evaluate(formula_tokens, frame).signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ranked = signal.rank(pct=True).sub(0.5).fillna(0.0)
        return ranked.shift(-1).fillna(0.0)

    def _allowed_features(self, dataset_name: str, role: str) -> frozenset[str]:
        prefix = DATASET_PREFIX[dataset_name]
        if role == "target_price":
            return frozenset(
                {
                    f"{prefix}_CLOSE",
                    f"{prefix}_OPEN",
                    f"{prefix}_HIGH",
                    f"{prefix}_LOW",
                    f"{prefix}_OC_RET",
                    f"{prefix}_GAP_RET",
                }
            )
        if role == "target_flow_vol":
            return frozenset(
                {
                    f"{prefix}_VOLUME",
                    f"{prefix}_HL_SPREAD",
                    f"{prefix}_REALIZED_VOL_5",
                    f"{prefix}_REALIZED_VOL_20",
                    f"{prefix}_VOLUME_ZSCORE_20",
                }
            )
        if role == "target_flow_gap":
            return frozenset(
                {
                    f"{prefix}_GAP_RET",
                    f"{prefix}_OC_RET",
                    f"{prefix}_HL_SPREAD",
                }
            )
        return frozenset(frame_feature for frame_feature in self._all_features() if not frame_feature.startswith(f"{prefix}_"))

    def _all_features(self) -> tuple[str, ...]:
        features: list[str] = []
        for prefix in DATASET_PREFIX.values():
            features.extend(
                [
                    f"{prefix}_CLOSE",
                    f"{prefix}_OPEN",
                    f"{prefix}_HIGH",
                    f"{prefix}_LOW",
                    f"{prefix}_VOLUME",
                    f"{prefix}_HL_SPREAD",
                    f"{prefix}_OC_RET",
                    f"{prefix}_GAP_RET",
                    f"{prefix}_REALIZED_VOL_5",
                    f"{prefix}_REALIZED_VOL_20",
                    f"{prefix}_VOLUME_ZSCORE_20",
                ]
            )
        features.extend(["CPI", "TNX", "VIX", "DXY"])
        return tuple(features)

    def _build_pool(self, dataset_name: str, role: str, frame: pd.DataFrame, target: pd.Series) -> FactorPool:
        pool = FactorPool(max_size=8)
        prefix = DATASET_PREFIX[dataset_name]
        if role.startswith("target_flow"):
            baseline_formula = (f"{prefix}_CLOSE", "DELTA_1", "NEG")
            pool.add(evaluate_formula_record(baseline_formula, frame, target, role="target_price"))
        elif role == "context":
            baseline_formula = (f"{prefix}_GAP_RET", "NEG")
            pool.add(evaluate_formula_record(baseline_formula, frame, target, role="target_flow_gap"))
        return pool

    def _render_template(self, template: tuple[str, ...], prefix: str) -> tuple[str, ...]:
        rendered: list[str] = []
        for token in template:
            rendered.append(token.replace("PREFIX_", f"{prefix}_"))
        return tuple(rendered)
