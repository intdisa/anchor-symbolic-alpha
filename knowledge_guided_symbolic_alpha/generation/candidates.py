from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormulaCandidate:
    formula: str
    source: str
    role: str | None = None


def merge_formula_candidates(*candidate_groups: list[FormulaCandidate]) -> list[FormulaCandidate]:
    ordered: list[FormulaCandidate] = []
    seen: set[str] = set()
    for group in candidate_groups:
        for candidate in group:
            normalized = str(candidate.formula).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(
                FormulaCandidate(
                    formula=normalized,
                    source=str(candidate.source),
                    role=None if candidate.role is None else str(candidate.role),
                )
            )
    return ordered
