from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from knowledge_guided_symbolic_alpha.selection import FormulaEvaluationCache


def test_formula_evaluation_cache_reuses_formula_split_results() -> None:
    calls: list[tuple[str, int]] = []

    def fake_evaluator(formula: str, frame: pd.DataFrame, target: pd.Series):
        calls.append((formula, len(frame)))
        signal = pd.Series(range(len(frame)), index=frame.index, dtype="float32")
        return SimpleNamespace(
            metrics={"rank_ic": float(len(frame)), "sharpe": 1.0, "turnover": 0.01},
            evaluated=SimpleNamespace(signal=signal),
        )

    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4, freq="D")})
    target = pd.Series([0.1, 0.2, 0.3, 0.4], index=frame.index)
    cache = FormulaEvaluationCache(evaluator=fake_evaluator)

    first = cache.get("A", frame, target, slice_count=2, context_key="split_a")
    second = cache.get("A", frame, target, slice_count=2, context_key="split_a")
    other = cache.get("A", frame, target, slice_count=2, context_key="split_b")

    assert first.full_metrics["rank_ic"] == 4.0
    assert second.full_metrics == first.full_metrics
    assert other.full_metrics == first.full_metrics
    assert len(calls) == 6
    assert cache.stats() == {"entries": 2, "hits": 1, "misses": 2}
