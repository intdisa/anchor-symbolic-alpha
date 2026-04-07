from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..domain.feature_registry import FEATURE_REGISTRY
from ..generation import FormulaCandidate


def micro_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name, spec in FEATURE_REGISTRY.items()
        if spec.is_micro and name in frame.columns
    ]


def rank_transform_frame(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    *,
    date_column: str = "date",
    prefix: str = "R_",
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    columns = feature_columns or micro_feature_columns(frame)
    ranked = pd.DataFrame(index=frame.index)
    variable_map: dict[str, tuple[str, ...]] = {}
    grouped = frame.groupby(date_column, sort=False)
    for feature in columns:
        ranked_name = f"{prefix}{feature}"
        ranked[ranked_name] = grouped[feature].rank(method="average", pct=True)
        variable_map[ranked_name] = (feature, "RANK")
    return ranked, variable_map


def formula_features(formula: str) -> tuple[str, ...]:
    return tuple(token for token in str(formula).split() if token in FEATURE_REGISTRY)


def formula_feature_families(formula: str) -> tuple[str, ...]:
    families = {FEATURE_REGISTRY[token].category for token in formula_features(formula)}
    return tuple(sorted(families))


def token_overlap(left_formula: str, right_formula: str) -> float:
    left = left_formula.split()
    right = right_formula.split()
    if not left or not right:
        return 0.0
    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = sum(min(left_counts[token], right_counts[token]) for token in set(left_counts) | set(right_counts))
    total = sum(left_counts.values()) + sum(right_counts.values()) - overlap
    return 0.0 if total <= 0 else float(overlap / total)


def feature_family_overlap(left_formula: str, right_formula: str) -> float:
    left = set(formula_feature_families(left_formula))
    right = set(formula_feature_families(right_formula))
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def build_candidate_pool_from_runs(raw_runs: list[dict[str, object]]) -> list[FormulaCandidate]:
    seen: set[str] = set()
    candidates: list[FormulaCandidate] = []
    for run in raw_runs:
        for formula in run.get("candidate_records", []):
            if not formula or formula in seen:
                continue
            seen.add(formula)
            candidates.append(FormulaCandidate(formula=str(formula), source="finance_candidate_pool", role="finance"))
    return candidates


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
