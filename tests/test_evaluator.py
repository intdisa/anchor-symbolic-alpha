import pandas as pd
import pytest

from knowledge_guided_symbolic_alpha.evaluation import (
    AdmissionPolicy,
    EvaluationError,
    FactorPool,
    FormulaEvaluator,
)


def make_frame() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    diff = pd.Series([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], index=index, dtype=float)
    gold_close = 100.0 + diff.cumsum()
    gold_volume = pd.Series([10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18], index=index, dtype=float)
    cpi = pd.Series([100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103], index=index, dtype=float)
    tnx = pd.Series([2.0, 2.1, 2.2, 2.2, 2.3, 2.5, 2.6, 2.6, 2.8, 2.9, 3.0, 3.1], index=index, dtype=float)
    vix = pd.Series([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7], index=index, dtype=float)
    dxy = pd.Series([100, 100, 101, 101, 102, 102, 103, 103, 104, 104, 105, 105], index=index, dtype=float)
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
    target = diff.shift(-1).rename("target")
    return frame, target


def test_evaluator_runs_valid_formula() -> None:
    frame, _ = make_frame()
    evaluator = FormulaEvaluator()
    evaluated = evaluator.evaluate("CPI DELAY_1 VIX MUL", frame)
    assert evaluated.signal.dropna().shape[0] > 0
    assert evaluated.signal.name == evaluated.parsed.canonical


def test_evaluator_rejects_zero_variance_signal() -> None:
    frame, _ = make_frame()
    frame["GOLD_CLOSE"] = 100.0
    evaluator = FormulaEvaluator()
    with pytest.raises(EvaluationError):
        evaluator.evaluate("GOLD_CLOSE", frame)


def test_admission_accepts_first_formula_and_rejects_duplicate_and_high_corr() -> None:
    frame, target = make_frame()
    policy = AdmissionPolicy(min_abs_rank_ic=0.1, max_correlation=0.95)
    pool = FactorPool(max_size=4)

    first = policy.screen("GOLD_CLOSE DELTA_1", frame, target, pool)
    assert first.accepted
    assert len(pool) == 1

    duplicate = policy.screen("GOLD_CLOSE DELTA_1", frame, target, pool)
    assert not duplicate.accepted
    assert duplicate.reason == "duplicate_canonical"

    high_corr = policy.screen("GOLD_CLOSE DELTA_1 ABS", frame, target, pool)
    assert not high_corr.accepted
    assert high_corr.reason == "correlation_check"
