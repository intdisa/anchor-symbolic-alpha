from __future__ import annotations

import pandas as pd

from knowledge_guided_symbolic_alpha.domain.feature_registry import get_feature, is_generic_benchmark_feature
from knowledge_guided_symbolic_alpha.evaluation.panel_dispatch import evaluate_formula_metrics
from knowledge_guided_symbolic_alpha.language.parser import RPNParser


def test_generic_benchmark_feature_overlay_supports_x_tokens() -> None:
    feature = get_feature("X3")
    assert is_generic_benchmark_feature("X3") is True
    assert feature.name == "X3"
    assert feature.frequency == "generic"


def test_parser_and_evaluator_accept_generic_benchmark_features() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "X0": [0.2, 0.1, -0.1, 0.3, 0.2, -0.2],
            "X1": [0.1, -0.1, 0.0, 0.2, -0.2, 0.1],
            "TARGET_RET_1": [0.3, 0.0, -0.2, 0.4, 0.1, -0.1],
        }
    )
    parser = RPNParser()
    parsed = parser.parse_text("X0 X1 ADD")
    metrics = evaluate_formula_metrics(" ".join(parsed.canonical_rpn), frame, frame["TARGET_RET_1"]).metrics

    assert parsed.canonical_rpn == ("X0", "X1", "ADD")
    assert "rank_ic" in metrics
