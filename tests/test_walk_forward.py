import numpy as np
import pandas as pd

from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester, WalkForwardConfig


def make_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    vix = pd.Series(20 - 0.05 * np.arange(len(index)), index=index)
    gold_close = pd.Series(100 + np.cumsum(-vix.diff().fillna(0.0)), index=index)
    gold_volume = pd.Series(1000 + 20 * np.sin(np.arange(len(index)) / 3), index=index)
    cpi = pd.Series(100 + 0.01 * np.arange(len(index)), index=index)
    tnx = pd.Series(1.5 + 0.002 * np.arange(len(index)), index=index)
    dxy = pd.Series(100 + 0.01 * np.arange(len(index)), index=index)
    frame = pd.DataFrame(
        {
            "GOLD_CLOSE": gold_close,
            "GOLD_VOLUME": gold_volume,
            "CPI": cpi,
            "TNX": tnx,
            "VIX": vix,
            "DXY": dxy,
        }
    )
    frame["GOLD_RET_1"] = frame["GOLD_CLOSE"].pct_change().fillna(0.0)
    frame["TARGET_GOLD_FWD_RET_1"] = frame["GOLD_RET_1"].shift(-1).fillna(0.0)
    return frame


def test_walk_forward_backtester_runs_smoke() -> None:
    frame = make_frame()
    backtester = WalkForwardBacktester()
    report = backtester.run(
        formulas=["VIX DELTA_1 NEG", "GOLD_CLOSE DELTA_1"],
        frame=frame,
        feature_columns=("GOLD_CLOSE", "GOLD_VOLUME", "CPI", "TNX", "VIX", "DXY"),
        target_column="TARGET_GOLD_FWD_RET_1",
        return_column="GOLD_RET_1",
        config=WalkForwardConfig(train_size=40, test_size=20, step_size=20, top_k=1),
    )
    assert report.folds
    assert "sharpe" in report.aggregate_metrics


def make_cross_sectional_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    rows = []
    for idx, date in enumerate(dates):
        for permno, profitability, cash_ratio in [(1, 1.0, 2.0), (2, 2.0, 1.0), (3, 3.0, 0.5)]:
            target_ret = 0.01 * profitability - 0.005 * cash_ratio + 0.0005 * idx
            rows.append(
                {
                    "date": date,
                    "permno": permno,
                    "PROFITABILITY_Q": profitability,
                    "CASH_RATIO_Q": cash_ratio,
                    "TARGET_RET_1": target_ret,
                }
            )
    frame = pd.DataFrame(rows)
    frame["TARGET_XS_RET_1"] = frame["TARGET_RET_1"] - frame.groupby("date")["TARGET_RET_1"].transform("mean")
    return frame


def test_walk_forward_backtester_runs_cross_sectional_smoke() -> None:
    frame = make_cross_sectional_frame()
    backtester = WalkForwardBacktester()
    report = backtester.run(
        formulas=["PROFITABILITY_Q RANK", "CASH_RATIO_Q RANK"],
        frame=frame,
        feature_columns=("PROFITABILITY_Q", "CASH_RATIO_Q"),
        target_column="TARGET_XS_RET_1",
        return_column="TARGET_RET_1",
        config=WalkForwardConfig(train_size=40, test_size=20, step_size=20, top_k=1),
    )
    assert report.folds
    assert np.isfinite(report.aggregate_metrics["sharpe"])
    assert report.aggregate_metrics["mean_test_rank_ic"] > 0.0
