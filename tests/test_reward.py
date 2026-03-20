import pandas as pd

from knowledge_guided_symbolic_alpha.evaluation import AdmissionDecision, AdmissionPolicy, FactorPool
from knowledge_guided_symbolic_alpha.evaluation.factor_pool import FactorRecord
from knowledge_guided_symbolic_alpha.training import PoolRewardShaper


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


def test_reward_shaper_uses_delta_pool_score_and_commit_flag() -> None:
    frame, target = make_frame()
    policy = AdmissionPolicy(min_abs_rank_ic=0.1, max_correlation=0.95)
    shaper = PoolRewardShaper(admission_policy=policy)
    pool = FactorPool(max_size=4)

    outcome = shaper.shape("GOLD_CLOSE DELTA_1", frame, target, pool, commit=False)
    assert outcome.decision.accepted
    assert outcome.components["delta_pool_score"] > 0.0
    assert len(pool) == 0

    committed = shaper.shape("GOLD_CLOSE DELTA_1", frame, target, pool, commit=True)
    assert committed.decision.accepted
    assert len(pool) == 1


def test_reward_shaper_penalizes_invalid_formula() -> None:
    frame, target = make_frame()
    shaper = PoolRewardShaper()
    pool = FactorPool(max_size=4)

    outcome = shaper.shape("GOLD_CLOSE GOLD_VOLUME ADD", frame, target, pool, commit=False)
    assert outcome.decision.candidate is None
    assert outcome.clipped_reward < 0.0


def test_reward_shaper_adds_trade_proxy_bonus_for_target_flow() -> None:
    class DummyPolicy:
        def screen(self, formula, data, target, pool, role=None):
            del formula, data, target, pool
            record = FactorRecord(
                tokens=("GOLD_REALIZED_VOL_20", "DELTA_1"),
                canonical="DELTA_1(GOLD_REALIZED_VOL_20)",
                signal=pd.Series([1.0, 2.0, 3.0]),
                metrics={"rank_ic": 0.03, "max_corr": 0.1, "turnover": 0.4, "max_drawdown": -0.2},
                role=role,
            )
            return AdmissionDecision(
                True,
                "accepted",
                record,
                0.01,
                trade_proxy_gain=0.02,
                baseline_trade_proxy=0.0,
                new_trade_proxy=0.02,
            )

    frame, target = make_frame()
    shaper = PoolRewardShaper(admission_policy=DummyPolicy())  # type: ignore[arg-type]
    outcome = shaper.shape(("GOLD_REALIZED_VOL_20", "DELTA_1"), frame, target, FactorPool(max_size=4), role="target_flow")

    assert outcome.components["trade_proxy_bonus"] > 0.0
    assert outcome.reward > outcome.components["delta_pool_score"]
