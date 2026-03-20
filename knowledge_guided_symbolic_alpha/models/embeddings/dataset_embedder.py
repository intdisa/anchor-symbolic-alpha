from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetEmbedding:
    vector: tuple[float, ...]
    token_biases: dict[str, float]


class DatasetEmbedder:
    VECTOR_DIM = 8

    def embed(
        self,
        frame: pd.DataFrame,
        allowed_features: frozenset[str],
        target: pd.Series | None = None,
        regime: str | None = None,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> DatasetEmbedding:
        feature_scores = self._feature_scores(frame, allowed_features, target, validation_data, validation_target)
        token_biases = self._token_biases(feature_scores, regime)
        vector = self._vector(frame, allowed_features, feature_scores, regime)
        return DatasetEmbedding(vector=vector, token_biases=token_biases)

    def _feature_scores(
        self,
        frame: pd.DataFrame,
        allowed_features: frozenset[str],
        target: pd.Series | None,
        validation_data: pd.DataFrame | None,
        validation_target: pd.Series | None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for feature in allowed_features:
            if feature not in frame.columns:
                continue
            train_score = self._series_score(frame[feature], target)
            valid_score = 0.0
            if validation_data is not None and validation_target is not None and feature in validation_data.columns:
                valid_score = self._series_score(validation_data[feature], validation_target)
            scores[feature] = float(0.7 * train_score + 0.3 * valid_score)
        return scores

    def _series_score(self, series: pd.Series, target: pd.Series | None) -> float:
        if target is None:
            clean = series.replace([np.inf, -np.inf], np.nan).dropna()
            if clean.empty:
                return 0.0
            return float(min(abs(clean.std(ddof=0)), 1.0))
        aligned = pd.concat([series, target], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if aligned.shape[0] < 8:
            return 0.0
        ranked = aligned.rank()
        corr = ranked.iloc[:, 0].corr(ranked.iloc[:, 1], method="pearson")
        if corr is None or not np.isfinite(corr):
            return 0.0
        return float(abs(corr))

    def _token_biases(self, feature_scores: dict[str, float], regime: str | None) -> dict[str, float]:
        biases: dict[str, float] = {}
        if not feature_scores:
            return biases

        ranked_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
        top_features = ranked_features[: min(6, len(ranked_features))]
        max_score = max(score for _, score in top_features) or 1.0
        for feature, score in top_features:
            normalized = score / max_score if max_score > 0.0 else 0.0
            biases[feature] = 0.05 + 0.25 * normalized

        has_gap = any("GAP_RET" in feature for feature, _ in top_features)
        has_intraday = any("OC_RET" in feature or "HL_SPREAD" in feature for feature, _ in top_features)
        has_vol = any("REALIZED_VOL" in feature or "VOLUME" in feature for feature, _ in top_features)
        has_short_horizon = any(feature.startswith("RET_") or feature == "PRICE_TO_252_HIGH" for feature, _ in top_features)
        has_liquidity = any(
            any(tag in feature for tag in ("VOLATILITY_", "TURNOVER_", "DOLLAR_VOLUME_", "AMIHUD_"))
            for feature, _ in top_features
        )
        has_fundamentals = any(
            any(
                tag in feature
                for tag in (
                    "BOOK_TO_MARKET",
                    "PROFITABILITY",
                    "ASSET_GROWTH",
                    "LEVERAGE",
                    "CASH_RATIO",
                    "SALES_TO_ASSETS",
                    "SIZE_LOG_MCAP",
                )
            )
            for feature, _ in top_features
        )

        if has_gap:
            biases["NEG"] = biases.get("NEG", 0.0) + 0.12
            biases["DELAY_1"] = biases.get("DELAY_1", 0.0) + 0.08
            biases["RANK"] = biases.get("RANK", 0.0) + 0.06
        if has_intraday:
            biases["SUB"] = biases.get("SUB", 0.0) + 0.08
            biases["DELTA_1"] = biases.get("DELTA_1", 0.0) + 0.08
            biases["NEG"] = biases.get("NEG", 0.0) + 0.04
        if has_vol:
            biases["TS_STD_5"] = biases.get("TS_STD_5", 0.0) + 0.10
            biases["TS_MEAN_5"] = biases.get("TS_MEAN_5", 0.0) + 0.08
            biases["CORR_5"] = biases.get("CORR_5", 0.0) + 0.06
        if has_short_horizon:
            biases["NEG"] = biases.get("NEG", 0.0) + 0.10
            biases["RANK"] = biases.get("RANK", 0.0) + 0.08
            biases["SUB"] = biases.get("SUB", 0.0) + 0.06
        if has_liquidity:
            biases["RANK"] = biases.get("RANK", 0.0) + 0.08
            biases["CORR_5"] = biases.get("CORR_5", 0.0) + 0.05
            biases["TS_STD_5"] = biases.get("TS_STD_5", 0.0) + 0.08
        if has_fundamentals:
            biases["RANK"] = biases.get("RANK", 0.0) + 0.12
            biases["NEG"] = biases.get("NEG", 0.0) + 0.06
            biases["DELAY_1"] = biases.get("DELAY_1", 0.0) + 0.04

        if regime == "HIGH_VOLATILITY":
            biases["TS_STD_5"] = biases.get("TS_STD_5", 0.0) + 0.06
            biases["CORR_5"] = biases.get("CORR_5", 0.0) + 0.04
        elif regime == "USD_STRENGTH":
            biases["DELAY_1"] = biases.get("DELAY_1", 0.0) + 0.04
            biases["RANK"] = biases.get("RANK", 0.0) + 0.03
        return biases

    def _vector(
        self,
        frame: pd.DataFrame,
        allowed_features: frozenset[str],
        feature_scores: dict[str, float],
        regime: str | None,
    ) -> tuple[float, ...]:
        top_scores = sorted(feature_scores.values(), reverse=True)
        mean_top = float(np.mean(top_scores[:3])) if top_scores else 0.0
        max_score = float(top_scores[0]) if top_scores else 0.0
        gap_strength = float(
            np.mean([score for feature, score in feature_scores.items() if "GAP_RET" in feature]) if any("GAP_RET" in feature for feature in feature_scores) else 0.0
        )
        vol_strength = float(
            np.mean([score for feature, score in feature_scores.items() if "REALIZED_VOL" in feature or "VOLUME" in feature]) if any(("REALIZED_VOL" in feature or "VOLUME" in feature) for feature in feature_scores) else 0.0
        )
        price_strength = float(
            np.mean([score for feature, score in feature_scores.items() if any(tag in feature for tag in ("CLOSE", "OPEN", "HIGH", "LOW", "OC_RET"))]) if any(any(tag in feature for tag in ("CLOSE", "OPEN", "HIGH", "LOW", "OC_RET")) for feature in feature_scores) else 0.0
        )
        if gap_strength == 0.0:
            gap_strength = float(
                np.mean(
                    [
                        score
                        for feature, score in feature_scores.items()
                        if feature.startswith("RET_") or feature == "PRICE_TO_252_HIGH"
                    ]
                )
                if any(feature.startswith("RET_") or feature == "PRICE_TO_252_HIGH" for feature in feature_scores)
                else 0.0
            )
        if vol_strength == 0.0:
            vol_strength = float(
                np.mean(
                    [
                        score
                        for feature, score in feature_scores.items()
                        if any(tag in feature for tag in ("VOLATILITY_", "TURNOVER_", "DOLLAR_VOLUME_", "AMIHUD_"))
                    ]
                )
                if any(any(tag in feature for tag in ("VOLATILITY_", "TURNOVER_", "DOLLAR_VOLUME_", "AMIHUD_")) for feature in feature_scores)
                else 0.0
            )
        if price_strength == 0.0:
            price_strength = float(
                np.mean(
                    [
                        score
                        for feature, score in feature_scores.items()
                        if any(
                            tag in feature
                            for tag in (
                                "RET_",
                                "PRICE_TO_252_HIGH",
                                "BOOK_TO_MARKET",
                                "PROFITABILITY",
                                "ASSET_GROWTH",
                                "LEVERAGE",
                                "CASH_RATIO",
                                "SALES_TO_ASSETS",
                                "SIZE_LOG_MCAP",
                            )
                        )
                    ]
                )
                if any(
                    any(
                        tag in feature
                        for tag in (
                            "RET_",
                            "PRICE_TO_252_HIGH",
                            "BOOK_TO_MARKET",
                            "PROFITABILITY",
                            "ASSET_GROWTH",
                            "LEVERAGE",
                            "CASH_RATIO",
                            "SALES_TO_ASSETS",
                            "SIZE_LOG_MCAP",
                        )
                    )
                    for feature in feature_scores
                )
                else 0.0
            )
        regime_code = {
            "BALANCED": 0.0,
            "HIGH_VOLATILITY": 1.0,
            "RATE_HIKING": 0.5,
            "INFLATION_SHOCK": 0.75,
            "USD_STRENGTH": 0.25,
        }.get(regime or "BALANCED", 0.0)
        return (
            float(np.log1p(len(frame)) / 10.0),
            float(len(allowed_features) / max(len(frame.columns), 1)),
            mean_top,
            max_score,
            gap_strength,
            vol_strength,
            price_strength,
            regime_code,
        )
