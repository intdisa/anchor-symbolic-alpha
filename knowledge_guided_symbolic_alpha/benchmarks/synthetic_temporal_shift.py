from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..generation import FormulaCandidate
from ..evaluation.panel_dispatch import evaluate_formula_metrics


@dataclass(frozen=True)
class SyntheticSelectorBenchmark:
    frame: pd.DataFrame
    target: pd.Series
    true_formula: str
    spurious_formula: str
    candidate_formulas: list[FormulaCandidate]


def naive_rank_ic_selection(
    candidates: list[FormulaCandidate],
    frame: pd.DataFrame,
    target: pd.Series,
) -> str:
    best_formula = ""
    best_rank_ic = float("-inf")
    for candidate in candidates:
        try:
            metrics = evaluate_formula_metrics(candidate.formula, frame, target).metrics
        except Exception:
            continue
        rank_ic = float(metrics.get("rank_ic", float("-inf")))
        if np.isfinite(rank_ic) and rank_ic > best_rank_ic:
            best_rank_ic = rank_ic
            best_formula = candidate.formula
    return best_formula


def generate_synthetic_temporal_shift_panel(
    num_dates: int = 72,
    num_entities: int = 80,
    seed: int = 7,
) -> SyntheticSelectorBenchmark:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=num_dates, freq="ME")
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for date_index, date in enumerate(dates):
        quality_regime = 1.0 + 0.10 * np.sin(date_index / 8.0)
        if date_index < num_dates // 3:
            flow_regime = 1.00
        elif date_index < 2 * num_dates // 3:
            flow_regime = 0.20
        else:
            flow_regime = -0.50
        for entity in range(num_entities):
            latent_quality = rng.normal()
            latent_balance = 0.8 * latent_quality + rng.normal(scale=0.6)
            latent_flow = rng.normal()
            cash_ratio = 0.85 * latent_quality + rng.normal(scale=0.30)
            profitability_q = 0.90 * latent_balance + rng.normal(scale=0.25)
            profitability_a = 0.70 * latent_balance + rng.normal(scale=0.35)
            ret_1 = latent_flow + rng.normal(scale=0.25)
            asset_growth = -0.20 * latent_quality + rng.normal(scale=0.80)
            target = (
                0.05 * quality_regime * latent_quality
                + 0.04 * quality_regime * latent_balance
                - flow_regime * ret_1
                + 0.02 * asset_growth
                + rng.normal(scale=0.25)
            )
            rows.append(
                {
                    "date": date,
                    "permno": entity + 10000,
                    "CASH_RATIO_Q": cash_ratio,
                    "PROFITABILITY_Q": profitability_q,
                    "PROFITABILITY_A": profitability_a,
                    "RET_1": ret_1,
                    "ASSET_GROWTH_A": asset_growth,
                    "TARGET_XS_RET_1": target,
                    "TARGET_RET_1": target,
                }
            )
    frame = pd.DataFrame(rows)
    target = frame["TARGET_XS_RET_1"].copy()
    true_formula = "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"
    spurious_formula = "RET_1 NEG"
    return SyntheticSelectorBenchmark(
        frame=frame,
        target=target,
        true_formula=true_formula,
        spurious_formula=spurious_formula,
        candidate_formulas=build_selector_benchmark_candidates(true_formula, spurious_formula),
    )


def build_selector_benchmark_candidates(
    true_formula: str = "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD",
    spurious_formula: str = "RET_1 NEG",
) -> list[FormulaCandidate]:
    return [
        FormulaCandidate(formula=true_formula, source="stable_anchor", role="quality_solvency"),
        FormulaCandidate(formula=spurious_formula, source="spurious_flow", role="short_horizon_flow"),
        FormulaCandidate(
            formula="CASH_RATIO_Q RANK PROFITABILITY_A RANK ADD",
            source="anchor_variant",
            role="quality_solvency",
        ),
        FormulaCandidate(
            formula="ASSET_GROWTH_A RANK NEG",
            source="noise_challenger",
            role="efficiency_growth",
        ),
    ]
