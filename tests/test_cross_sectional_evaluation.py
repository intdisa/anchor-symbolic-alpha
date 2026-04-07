from __future__ import annotations

import pandas as pd

from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_evaluator import (
    CrossSectionalFormulaEvaluator,
)
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_metrics import (
    cross_sectional_ic_summary,
    cross_sectional_rank_ic,
    cross_sectional_risk_summary,
    cross_sectional_stability_summary,
)
from knowledge_guided_symbolic_alpha.evaluation.panel_dispatch import score_signal_metrics


def _make_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-06",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-07",
                ]
            ),
            "permno": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "RET_1": [0.01, 0.03, 0.02, -0.01, 0.04, 0.02, -0.02, 0.01, 0.03, -0.02],
            "SIZE_LOG_MCAP": [10, 20, 10, 20, 10, 20, 10, 20, 10, 20],
            "TARGET_RET_1": [0.02, -0.01, 0.03, -0.02, 0.04, -0.01, -0.02, 0.02, 0.03, -0.03],
            "TARGET_XS_RET_1": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, -0.01, 0.01, 0.02, -0.02],
        }
    )


def test_cross_sectional_evaluator_uses_entity_time_rolls_and_date_ranks() -> None:
    panel = _make_panel()
    evaluator = CrossSectionalFormulaEvaluator()

    delay_signal = evaluator.evaluate("RET_1 DELAY_1", panel).signal
    rank_signal = evaluator.evaluate("SIZE_LOG_MCAP RANK", panel).signal

    assert pd.isna(delay_signal.iloc[0])
    assert pd.isna(delay_signal.iloc[1])
    assert delay_signal.iloc[2] == panel.loc[0, "RET_1"]
    assert delay_signal.iloc[3] == panel.loc[1, "RET_1"]
    assert rank_signal.iloc[0] == 0.5
    assert rank_signal.iloc[1] == 1.0
    assert rank_signal.iloc[2] == 0.5
    assert rank_signal.iloc[3] == 1.0


def test_cross_sectional_metrics_compute_datewise_rank_ic() -> None:
    panel = _make_panel()
    signal = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0], index=panel.index, dtype=float)

    rank_ic = cross_sectional_rank_ic(signal, panel["TARGET_XS_RET_1"], panel["date"])
    summary = cross_sectional_ic_summary(signal, panel["TARGET_XS_RET_1"], panel["date"])

    assert rank_ic > 0.0
    assert summary["rank_ic"] > 0.0
    assert "rank_icir" in summary


def test_cross_sectional_risk_summary_builds_long_short_portfolio() -> None:
    panel = _make_panel()
    signal = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0], index=panel.index, dtype=float)

    metrics = cross_sectional_risk_summary(
        signal,
        panel["TARGET_XS_RET_1"],
        panel["date"],
        panel["permno"],
        quantile=0.5,
        min_long_count=1,
        min_short_count=1,
    )

    assert metrics["annual_return"] != 0.0
    assert metrics["turnover"] >= 0.0


def test_cross_sectional_stability_summary_reports_window_metrics() -> None:
    panel = _make_panel()
    signal = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0], index=panel.index, dtype=float)

    metrics = cross_sectional_stability_summary(
        signal,
        panel["TARGET_XS_RET_1"],
        panel["date"],
        panel["permno"],
        quantile=0.5,
        window_count=3,
        min_long_count=1,
        min_short_count=1,
    )

    assert "rank_ic_window_min" in metrics
    assert "ls_return_window_min" in metrics
    assert "stability_score" in metrics


def test_panel_dispatch_uses_raw_returns_for_cross_sectional_risk_metrics() -> None:
    panel = _make_panel()
    signal = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0], index=panel.index, dtype=float)

    expected = cross_sectional_risk_summary(
        signal,
        panel["TARGET_RET_1"],
        panel["date"],
        panel["permno"],
        quantile=0.2,
    )
    scored = score_signal_metrics(signal, panel, panel["TARGET_XS_RET_1"])

    assert scored["annual_return"] == expected["annual_return"]
    assert scored["turnover"] == expected["turnover"]



def test_cross_sectional_risk_summary_supports_value_weighting() -> None:
    panel = _make_panel()
    signal = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0], index=panel.index, dtype=float)
    size_proxy = pd.Series([1, 9, 1, 9, 1, 9, 1, 9, 1, 9], index=panel.index, dtype=float)

    metrics = cross_sectional_risk_summary(
        signal,
        panel["TARGET_XS_RET_1"],
        panel["date"],
        panel["permno"],
        quantile=0.5,
        weight_scheme="value",
        size_proxy=size_proxy,
        min_long_count=1,
        min_short_count=1,
    )

    assert metrics["annual_return"] != 0.0
    assert metrics["turnover"] >= 0.0


def test_cross_sectional_long_short_fuse_zeros_out_sparse_days() -> None:
    panel = _make_panel()
    signal = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 0], index=panel.index, dtype=float)

    metrics = cross_sectional_risk_summary(
        signal,
        panel["TARGET_XS_RET_1"],
        panel["date"],
        panel["permno"],
        quantile=0.2,
    )

    assert metrics["annual_return"] == 0.0
    assert metrics["turnover"] == 0.0
