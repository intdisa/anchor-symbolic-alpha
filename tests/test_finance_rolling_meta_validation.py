from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_finance_rolling_meta_validation.py"
SPEC = importlib.util.spec_from_file_location("run_finance_rolling_meta_validation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_rolling_windows_splits_each_window_into_calibration_and_meta() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    frame = pd.DataFrame({"date": dates.repeat(2), "TARGET_XS_RET_1": range(24)})
    target = frame["TARGET_XS_RET_1"].astype(float)

    windows = MODULE.build_rolling_windows(frame, target, window_count=3)

    assert len(windows) == 3
    assert all(not window.calibration_frame.empty for window in windows)
    assert all(not window.meta_frame.empty for window in windows)
    assert all(window.calibration_frame["date"].max() < window.meta_frame["date"].min() for window in windows)


def test_coarse_and_fine_config_grids_are_deduplicated() -> None:
    coarse = MODULE.coarse_config_grid()
    fine = MODULE.fine_neighbors([MODULE.BASE_CONFIG])

    assert len(coarse) == len({MODULE.config_key(cfg) for cfg in coarse})
    assert MODULE.config_key(MODULE.BASE_CONFIG) in {MODULE.config_key(cfg) for cfg in coarse}
    assert MODULE.config_key(MODULE.BASE_CONFIG) in {MODULE.config_key(cfg) for cfg in fine}
    assert len({MODULE.config_key(cfg) for cfg in fine}) == len(fine)
    assert len(coarse) >= 20
    assert len(fine) > 1



def test_selection_score_prefers_rank_consistency_over_local_sharpe_spike() -> None:
    stable = MODULE.selection_score(
        calibration_metrics={"rank_ic": 0.012},
        meta_metrics={"rank_ic": 0.011, "turnover": 0.02},
        calibration_candidate_rank_ics=[0.03, 0.02, 0.01, -0.01],
        meta_candidate_rank_ics=[0.028, 0.019, 0.011, -0.009],
    )
    spiky = MODULE.selection_score(
        calibration_metrics={"rank_ic": 0.030},
        meta_metrics={"rank_ic": 0.005, "turnover": 0.02},
        calibration_candidate_rank_ics=[0.05, 0.03, 0.01, -0.01],
        meta_candidate_rank_ics=[-0.02, 0.04, 0.00, 0.01],
    )

    assert stable["rank_consistency"] > spiky["rank_consistency"]
    assert stable["rank_ic_retention"] > spiky["rank_ic_retention"]
    assert stable["meta_score"] > spiky["meta_score"]
    assert 0.0 <= stable["rank_ic_retention"] <= 1.0
    assert 0.0 <= spiky["rank_ic_retention"] <= 1.0
