from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation import FormulaEvaluator, ic_summary
from ..evaluation.cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from ..evaluation.cross_sectional_metrics import (
    cross_sectional_ic_summary,
    cross_sectional_long_short_returns,
    cross_sectional_risk_summary,
)
from ..evaluation.panel_dispatch import is_cross_sectional_frame
from .portfolio import PortfolioConfig, portfolio_returns, portfolio_summary
from .report import FoldReport, WalkForwardReport
from .signal_fusion import SignalFusionConfig, fuse_signals
from ..evaluation.risk_metrics import annual_return, max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int = 756
    test_size: int = 252
    step_size: int = 126
    top_k: int = 3


class WalkForwardBacktester:
    def __init__(
        self,
        evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator | None = None,
        signal_fusion_config: SignalFusionConfig | None = None,
        portfolio_config: PortfolioConfig | None = None,
    ) -> None:
        self.evaluator = evaluator or FormulaEvaluator()
        self.signal_fusion_config = signal_fusion_config or SignalFusionConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()

    def run(
        self,
        formulas: list[str] | list[tuple[str, ...]],
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...] | list[str],
        target_column: str,
        return_column: str,
        config: WalkForwardConfig | None = None,
    ) -> WalkForwardReport:
        config = config or WalkForwardConfig()
        folds: list[FoldReport] = []
        collected_returns: list[pd.Series] = []
        cross_sectional = is_cross_sectional_frame(frame)
        evaluator = self._resolve_evaluator(cross_sectional)
        fold_slices = self._fold_slices(frame, config, cross_sectional)
        for fold_index, (train, test) in enumerate(fold_slices):
            ranked_formulas = self._rank_formulas(formulas, train, feature_columns, target_column, evaluator=evaluator)
            if not ranked_formulas:
                continue
            selected = ranked_formulas[: config.top_k]
            weights = self._weights_from_rank_ic(selected)
            test_signals = {}
            for formula, canonical, _ in selected:
                evaluation_frame = test if cross_sectional else test[list(feature_columns)]
                evaluated = evaluator.evaluate(formula, evaluation_frame)
                test_signals[evaluated.parsed.canonical] = evaluated.signal
            signal_frame = pd.DataFrame(test_signals).dropna(how="all")
            if signal_frame.empty:
                continue
            if cross_sectional:
                fused = self._fuse_cross_sectional_signals(signal_frame, test["date"], weights)
                aligned_returns = test[return_column].reindex(fused.index)
                metrics = cross_sectional_risk_summary(fused, aligned_returns, test["date"], test["permno"])
                test_ic = cross_sectional_ic_summary(fused, test[target_column].reindex(fused.index), test["date"])
                returns, _ = cross_sectional_long_short_returns(
                    fused,
                    aligned_returns,
                    test["date"],
                    test["permno"],
                )
            else:
                fused = fuse_signals(signal_frame, weights=weights, config=self.signal_fusion_config)
                aligned_returns = test[return_column].reindex(fused.index)
                metrics = portfolio_summary(fused, aligned_returns, config=self.portfolio_config)
                test_ic = ic_summary(fused, test[target_column].reindex(fused.index))
                returns = portfolio_returns(fused, aligned_returns, config=self.portfolio_config)
            collected_returns.append(returns)
            folds.append(
                FoldReport(
                    fold_index=fold_index,
                    train_start=train.index[0],
                    train_end=train.index[-1],
                    test_start=test.index[0],
                    test_end=test.index[-1],
                    selected_formulas=tuple(canonical for canonical in weights),
                    weights=weights,
                    train_rank_ic=float(np.mean([metrics_["rank_ic"] for _, _, metrics_ in selected])),
                    test_rank_ic=float(test_ic["rank_ic"]),
                    metrics=metrics,
                )
            )
        combined_returns = pd.concat(collected_returns).sort_index() if collected_returns else pd.Series(dtype=float)
        aggregate = self._aggregate_metrics(folds, combined_returns)
        return WalkForwardReport(folds=folds, aggregate_metrics=aggregate, returns=combined_returns)

    def _fold_slices(
        self,
        frame: pd.DataFrame,
        config: WalkForwardConfig,
        cross_sectional: bool,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        if not cross_sectional:
            total_length = len(frame)
            return [
                (
                    frame.iloc[start : start + config.train_size],
                    frame.iloc[start + config.train_size : start + config.train_size + config.test_size],
                )
                for start in range(0, total_length - config.train_size - config.test_size + 1, config.step_size)
            ]
        normalized_dates = pd.to_datetime(frame["date"])
        dates = pd.Index(sorted(normalized_dates.dropna().unique()))
        slices: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for start in range(0, len(dates) - config.train_size - config.test_size + 1, config.step_size):
            train_dates = dates[start : start + config.train_size]
            test_dates = dates[start + config.train_size : start + config.train_size + config.test_size]
            train = frame[normalized_dates.isin(train_dates)].reset_index(drop=True)
            test = frame[normalized_dates.isin(test_dates)].reset_index(drop=True)
            slices.append((train, test))
        return slices

    def _rank_formulas(
        self,
        formulas: list[str] | list[tuple[str, ...]],
        train: pd.DataFrame,
        feature_columns: tuple[str, ...] | list[str],
        target_column: str,
        *,
        evaluator: FormulaEvaluator | CrossSectionalFormulaEvaluator,
    ) -> list[tuple[str | tuple[str, ...], str, dict[str, float]]]:
        ranked = []
        for formula in formulas:
            try:
                evaluation_frame = train if isinstance(evaluator, CrossSectionalFormulaEvaluator) else train[list(feature_columns)]
                evaluated = evaluator.evaluate(formula, evaluation_frame)
            except Exception:
                continue
            if isinstance(evaluator, CrossSectionalFormulaEvaluator):
                metrics = cross_sectional_ic_summary(evaluated.signal, train[target_column], train["date"])
            else:
                metrics = ic_summary(evaluated.signal, train[target_column])
            if np.isfinite(metrics["rank_ic"]):
                ranked.append((formula, evaluated.parsed.canonical, metrics))
        ranked.sort(key=lambda item: abs(item[2]["rank_ic"]), reverse=True)
        return ranked

    def _weights_from_rank_ic(self, selected: list[tuple[str | tuple[str, ...], str, dict[str, float]]]) -> dict[str, float]:
        raw = {}
        for _, canonical, metrics in selected:
            raw[canonical] = float(metrics["rank_ic"])
        denominator = sum(abs(value) for value in raw.values())
        if denominator == 0.0:
            denominator = max(len(raw), 1)
            return {key: 1.0 / denominator for key in raw}
        return {key: value / denominator for key, value in raw.items()}

    def _resolve_evaluator(
        self,
        cross_sectional: bool,
    ) -> FormulaEvaluator | CrossSectionalFormulaEvaluator:
        if cross_sectional:
            if isinstance(self.evaluator, CrossSectionalFormulaEvaluator):
                return self.evaluator
            return CrossSectionalFormulaEvaluator()
        if isinstance(self.evaluator, FormulaEvaluator):
            return self.evaluator
        return FormulaEvaluator()

    def _fuse_cross_sectional_signals(
        self,
        signal_frame: pd.DataFrame,
        dates: pd.Series,
        weights: dict[str, float],
    ) -> pd.Series:
        processed = signal_frame.copy()
        grouped_dates = pd.to_datetime(dates.reindex(processed.index))
        for column in processed.columns:
            series = processed[column].replace([np.inf, -np.inf], np.nan)
            mean = series.groupby(grouped_dates).transform("mean")
            std = series.groupby(grouped_dates).transform(lambda values: values.std(ddof=0))
            standardized = (series - mean) / std.replace(0.0, np.nan)
            processed[column] = standardized.clip(
                -self.signal_fusion_config.clip_zscore,
                self.signal_fusion_config.clip_zscore,
            ).fillna(0.0)
        weight_vector = np.array([weights.get(column, 0.0) for column in processed.columns], dtype=float)
        if not np.isfinite(weight_vector).all() or np.allclose(weight_vector, 0.0):
            weight_vector = np.ones(len(processed.columns), dtype=float)
        weight_vector = weight_vector / np.sum(np.abs(weight_vector))
        fused = processed.to_numpy() @ weight_vector
        return pd.Series(fused, index=processed.index, name="fused_signal")

    def _aggregate_metrics(
        self,
        folds: list[FoldReport],
        returns: pd.Series,
    ) -> dict[str, float]:
        if returns.empty:
            return {
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
                "annual_return": float("nan"),
                "turnover": float("nan"),
                "mean_test_rank_ic": float("nan"),
                "fold_count": 0.0,
            }
        turnover = float(np.mean([fold.metrics["turnover"] for fold in folds])) if folds else float("nan")
        mean_test_rank_ic = float(np.mean([fold.test_rank_ic for fold in folds])) if folds else float("nan")
        return {
            "sharpe": sharpe_ratio(returns),
            "max_drawdown": max_drawdown(returns),
            "annual_return": annual_return(returns),
            "turnover": turnover,
            "mean_test_rank_ic": mean_test_rank_ic,
            "fold_count": float(len(folds)),
        }
