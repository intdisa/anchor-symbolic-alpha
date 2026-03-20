import numpy as np
import pandas as pd

from knowledge_guided_symbolic_alpha.evaluation import (
    AdmissionPolicy,
    FactorPool,
    FormulaEvaluator,
    preview_candidate_on_dataset,
    score_pool_on_dataset,
)
from knowledge_guided_symbolic_alpha.evaluation.factor_pool import FactorRecord
from knowledge_guided_symbolic_alpha.evaluation.ic_metrics import ic_summary
from knowledge_guided_symbolic_alpha.evaluation.risk_metrics import risk_summary


def make_frame() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    vix = pd.Series(20 - 0.08 * np.arange(len(index)), index=index)
    gold_close = pd.Series(100 + np.cumsum(-vix.diff().fillna(0.0)), index=index)
    gold_volume = pd.Series(1000 + 10 * np.sin(np.arange(len(index)) / 4.0), index=index)
    cpi = pd.Series(100.0, index=index)
    tnx = pd.Series(1.5 + 0.001 * np.arange(len(index)), index=index)
    dxy = pd.Series(100 + 0.005 * np.arange(len(index)), index=index)
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
    target = frame["GOLD_CLOSE"].pct_change().shift(-1).fillna(0.0)
    return frame.iloc[:-1], target.iloc[:-1]


def make_record(tokens: tuple[str, ...], frame: pd.DataFrame, target: pd.Series) -> FactorRecord:
    evaluated = FormulaEvaluator().evaluate(tokens, frame)
    metrics = ic_summary(evaluated.signal, target)
    metrics.update(risk_summary(evaluated.signal, target))
    return FactorRecord(tokens=tokens, canonical=evaluated.parsed.canonical, signal=evaluated.signal, metrics=metrics)


def test_score_pool_on_dataset_skips_records_that_fail_on_rescore() -> None:
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(make_record(("GOLD_CLOSE", "DELTA_1", "NEG"), frame, target))
    pool.add(
        FactorRecord(
            tokens=("CPI", "DELAY_1"),
            canonical="DELAY_1(CPI)",
            signal=frame["CPI"],
            metrics={"rank_ic": 0.1},
        )
    )

    score = score_pool_on_dataset(pool, frame, target)
    assert np.isfinite(score)


def test_preview_candidate_on_dataset_returns_positive_gain_for_signal() -> None:
    frame, target = make_frame()
    preview = preview_candidate_on_dataset(
        ("VIX", "RANK"),
        FactorPool(max_size=4),
        frame,
        target,
        min_abs_rank_ic=0.0,
    )
    assert preview.record is not None
    assert preview.accepted
    assert preview.marginal_gain > 0.0
    assert np.isfinite(preview.trade_proxy_gain)


def test_factor_pool_rewards_target_role_diversity_bonus() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    price_record = FactorRecord(
        tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
        canonical="price",
        signal=pd.Series([1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5], index=index),
        metrics={"rank_ic": 0.05, "turnover": 1.0, "max_drawdown": -0.4},
        role="target_price",
    )
    flow_record = FactorRecord(
        tokens=("GOLD_REALIZED_VOL_20", "DELTA_1", "GOLD_HL_SPREAD", "DELTA_1", "DIV"),
        canonical="flow",
        signal=pd.Series([3.0, 2.0, 3.5, 2.5, 4.0, 3.0, 4.5, 3.5], index=index),
        metrics={"rank_ic": 0.03, "turnover": 0.4, "max_drawdown": -0.2},
        role="target_flow",
    )
    second_price_record = FactorRecord(
        tokens=("GOLD_GAP_RET", "RANK"),
        canonical="price_2",
        signal=pd.Series([3.0, 2.0, 3.5, 2.5, 4.0, 3.0, 4.5, 3.5], index=index),
        metrics={"rank_ic": 0.03, "turnover": 0.4, "max_drawdown": -0.2},
        role="target_price",
    )
    context_record = FactorRecord(
        tokens=("TNX", "DELAY_1", "RANK"),
        canonical="context",
        signal=pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], index=index),
        metrics={"rank_ic": 0.03, "turnover": 0.2, "max_drawdown": -0.1},
        role="context",
    )
    pool = FactorPool(max_size=4)

    price_only = pool.score_with(price_record)
    pool.add(price_record)
    with_flow = pool.score_with(flow_record)
    with_second_price = pool.score_with(second_price_record)
    with_context = pool.score_with(context_record)

    assert price_only > 0.0
    assert with_flow > with_second_price
    assert with_flow > with_context


def test_factor_pool_trade_proxy_prefers_low_turnover_flow_over_context() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    price_record = FactorRecord(
        tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
        canonical="price",
        signal=pd.Series([1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5], index=index),
        metrics={"rank_ic": 0.05, "sharpe": 0.2, "annual_return": 0.04, "turnover": 1.0, "max_drawdown": -0.4},
        role="target_price",
    )
    flow_record = FactorRecord(
        tokens=("GOLD_REALIZED_VOL_20", "DELTA_1", "GOLD_HL_SPREAD", "DELTA_1", "DIV"),
        canonical="flow",
        signal=pd.Series([3.0, 2.0, 3.5, 2.5, 4.0, 3.0, 4.5, 3.5], index=index),
        metrics={"rank_ic": 0.03, "sharpe": 0.35, "annual_return": 0.05, "turnover": 0.35, "max_drawdown": -0.15},
        role="target_flow",
    )
    context_record = FactorRecord(
        tokens=("TNX", "SP500_OC_RET", "MUL"),
        canonical="context",
        signal=pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], index=index),
        metrics={"rank_ic": 0.05, "sharpe": -0.2, "annual_return": -0.03, "turnover": 1.1, "max_drawdown": -0.7},
        role="context",
    )
    pool = FactorPool(max_size=4)
    pool.add(price_record)

    with_flow = pool.trade_proxy_with(flow_record)
    with_context = pool.trade_proxy_with(context_record)

    assert with_flow > with_context


def test_preview_candidate_on_dataset_supports_cross_sectional_panels() -> None:
    panel = pd.DataFrame(
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
            "RET_1": [0.02, -0.01, 0.03, -0.02, 0.01, 0.00, -0.01, 0.02, 0.02, -0.03],
            "TARGET_XS_RET_1": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, -0.01, 0.01, 0.02, -0.02],
        }
    )
    preview = preview_candidate_on_dataset(
        ("RET_1", "NEG"),
        FactorPool(max_size=4),
        panel,
        panel["TARGET_XS_RET_1"],
        min_abs_rank_ic=0.0,
    )

    assert preview.record is not None
    assert "rank_ic" in preview.record.metrics
    assert "sharpe" in preview.record.metrics


def test_cross_sectional_trend_preview_prefers_same_role_replacement(monkeypatch) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "permno": [1, 2, 1, 2],
            "TARGET_XS_RET_1": [0.01, -0.01, 0.02, -0.02],
        }
    )
    existing_signal = pd.Series([1.0, 2.0, 1.0, 2.0], index=panel.index)
    candidate_signal = pd.Series([1.0, 2.0, 1.1, 2.1], index=panel.index)
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="RANK(PROFITABILITY_Q)",
            signal=existing_signal,
            metrics={"rank_ic": 0.01, "turnover": 0.01, "sharpe": 0.1, "annual_return": 0.01, "max_drawdown": -0.1},
            role="trend_structure",
        )
    )

    def fake_evaluate(formula, data, target, evaluator=None):
        del data, target, evaluator
        tokens = tuple(formula) if isinstance(formula, (list, tuple)) else tuple(str(formula).split())
        if tokens == ("PROFITABILITY_Q", "RANK"):
            return type(
                "Result",
                (),
                {
                    "evaluated": type(
                        "Evaluated",
                        (),
                        {
                            "parsed": type("Parsed", (), {"tokens": ("PROFITABILITY_Q", "RANK"), "canonical": "RANK(PROFITABILITY_Q)"})(),
                            "signal": existing_signal,
                        },
                    )(),
                    "metrics": {
                        "rank_ic": 0.01,
                        "turnover": 0.01,
                        "sharpe": 0.1,
                        "annual_return": 0.01,
                        "max_drawdown": -0.1,
                    },
                },
            )()
        return type(
            "Result",
            (),
            {
                "evaluated": type(
                    "Evaluated",
                    (),
                    {
                        "parsed": type("Parsed", (), {"tokens": ("CASH_RATIO_Q", "RANK"), "canonical": "RANK(CASH_RATIO_Q)"})(),
                        "signal": candidate_signal,
                    },
                )(),
                "metrics": {
                    "rank_ic": 0.02,
                    "turnover": 0.01,
                    "sharpe": 0.2,
                    "annual_return": 0.02,
                    "max_drawdown": -0.05,
                },
            },
        )()

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.pool_scoring.evaluate_formula_metrics",
        fake_evaluate,
    )

    preview = preview_candidate_on_dataset(
        ("CASH_RATIO_Q", "RANK"),
        pool,
        panel,
        panel["TARGET_XS_RET_1"],
        role="trend_structure",
    )

    assert preview.accepted
    assert preview.reason in {"replaced", "replaced_baseline"}
    assert preview.replaced_canonical == "RANK(PROFITABILITY_Q)"


def test_cross_sectional_trend_admission_prefers_same_role_replacement(monkeypatch) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "permno": [1, 2, 1, 2],
            "TARGET_XS_RET_1": [0.01, -0.01, 0.02, -0.02],
        }
    )
    existing_signal = pd.Series([1.0, 2.0, 1.0, 2.0], index=panel.index)
    candidate_signal = pd.Series([1.0, 2.0, 1.1, 2.1], index=panel.index)
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="RANK(PROFITABILITY_Q)",
            signal=existing_signal,
            metrics={"rank_ic": 0.01, "turnover": 0.01, "sharpe": 0.1, "annual_return": 0.01, "max_drawdown": -0.1},
            role="trend_structure",
        )
    )

    def fake_evaluate(formula, data, target, evaluator=None):
        del data, target, evaluator
        tokens = tuple(formula) if isinstance(formula, (list, tuple)) else tuple(str(formula).split())
        if tokens == ("PROFITABILITY_Q", "RANK"):
            return type(
                "Result",
                (),
                {
                    "evaluated": type(
                        "Evaluated",
                        (),
                        {
                            "parsed": type("Parsed", (), {"tokens": ("PROFITABILITY_Q", "RANK"), "canonical": "RANK(PROFITABILITY_Q)"})(),
                            "signal": existing_signal,
                        },
                    )(),
                    "metrics": {
                        "rank_ic": 0.01,
                        "turnover": 0.01,
                        "sharpe": 0.1,
                        "annual_return": 0.01,
                        "max_drawdown": -0.1,
                    },
                },
            )()
        return type(
            "Result",
            (),
            {
                "evaluated": type(
                    "Evaluated",
                    (),
                    {
                        "parsed": type("Parsed", (), {"tokens": ("CASH_RATIO_Q", "RANK"), "canonical": "RANK(CASH_RATIO_Q)"})(),
                        "signal": candidate_signal,
                    },
                )(),
                "metrics": {
                    "rank_ic": 0.02,
                    "turnover": 0.01,
                    "sharpe": 0.2,
                    "annual_return": 0.02,
                    "max_drawdown": -0.05,
                },
            },
        )()

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.admission.evaluate_formula_metrics",
        fake_evaluate,
    )

    decision = AdmissionPolicy().screen(
        ("CASH_RATIO_Q", "RANK"),
        panel,
        panel["TARGET_XS_RET_1"],
        pool,
        role="trend_structure",
    )

    assert decision.accepted
    assert decision.reason in {"replaced", "replaced_baseline"}
    assert [record.canonical for record in pool.records] == ["RANK(CASH_RATIO_Q)"]


def test_cross_sectional_trend_replacement_can_bypass_fast_ic_for_stronger_baseline(monkeypatch) -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "permno": [1, 2, 1, 2],
            "TARGET_XS_RET_1": [0.01, -0.01, 0.02, -0.02],
        }
    )
    existing_signal = pd.Series([1.0, 2.0, 1.0, 2.0], index=panel.index)
    candidate_signal = pd.Series([1.0, 2.0, 1.2, 2.2], index=panel.index)
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("PROFITABILITY_Q", "RANK"),
            canonical="RANK(PROFITABILITY_Q)",
            signal=existing_signal,
            metrics={
                "rank_ic": 0.018,
                "turnover": 0.02,
                "sharpe": 0.15,
                "annual_return": 0.01,
                "max_drawdown": -0.20,
                "stability_score": 0.002,
            },
            role="trend_structure",
        )
    )

    def fake_preview_eval(formula, data, target, evaluator=None):
        del data, target, evaluator
        tokens = tuple(formula) if isinstance(formula, (list, tuple)) else tuple(str(formula).split())
        if tokens == ("PROFITABILITY_Q", "RANK"):
            return type(
                "Result",
                (),
                {
                    "evaluated": type(
                        "Evaluated",
                        (),
                        {
                            "parsed": type("Parsed", (), {"tokens": tokens, "canonical": "RANK(PROFITABILITY_Q)"})(),
                            "signal": existing_signal,
                        },
                    )(),
                    "metrics": {
                        "rank_ic": 0.018,
                        "turnover": 0.02,
                        "sharpe": 0.15,
                        "annual_return": 0.01,
                        "max_drawdown": -0.20,
                        "stability_score": 0.002,
                    },
                },
            )()
        if tokens == ("CASH_RATIO_Q", "RANK"):
            return type(
                "Result",
                (),
                {
                    "evaluated": type(
                        "Evaluated",
                        (),
                        {
                            "parsed": type("Parsed", (), {"tokens": tokens, "canonical": "RANK(CASH_RATIO_Q)"})(),
                            "signal": candidate_signal,
                        },
                    )(),
                    "metrics": {
                        "rank_ic": 0.004,
                        "turnover": 0.005,
                        "sharpe": 0.80,
                        "annual_return": 0.08,
                        "max_drawdown": -0.05,
                        "stability_score": 0.001,
                    },
                },
            )()
        raise AssertionError(f"unexpected formula {tokens}")

    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.pool_scoring.evaluate_formula_metrics",
        fake_preview_eval,
    )
    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.evaluation.admission.evaluate_formula_metrics",
        fake_preview_eval,
    )

    preview = preview_candidate_on_dataset(
        ("CASH_RATIO_Q", "RANK"),
        pool.copy(),
        panel,
        panel["TARGET_XS_RET_1"],
        role="trend_structure",
    )
    decision = AdmissionPolicy().screen(
        ("CASH_RATIO_Q", "RANK"),
        panel,
        panel["TARGET_XS_RET_1"],
        pool,
        role="trend_structure",
    )

    assert preview.accepted
    assert preview.reason == "replaced_baseline"
    assert decision.accepted
    assert decision.reason == "replaced_baseline"
    assert [record.canonical for record in pool.records] == ["RANK(CASH_RATIO_Q)"]
