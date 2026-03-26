from __future__ import annotations

from dataclasses import dataclass

from .candidates import FormulaCandidate, merge_formula_candidates


@dataclass(frozen=True)
class AnchorGenerationSummary:
    candidates: list[FormulaCandidate]
    champion_records: tuple[str, ...]
    final_records: tuple[str, ...]
    candidate_records: tuple[str, ...]


def build_anchor_generation_summary(summary) -> AnchorGenerationSummary:
    champion_candidates = [
        FormulaCandidate(formula=str(formula), source="champion_records")
        for formula in getattr(summary, "champion_records", ())
    ]
    final_candidates = [
        FormulaCandidate(formula=str(formula), source="final_records")
        for formula in getattr(summary, "final_records", ())
    ]
    proposed_candidates = list(getattr(summary, "candidate_record_summaries", ()))
    if not proposed_candidates:
        role_lookup = {}
        for episode in getattr(summary, "history", ()):
            episode_formula = str(getattr(episode, "formula", "")).strip()
            if episode_formula and episode_formula not in role_lookup:
                role_lookup[episode_formula] = getattr(episode, "selected_agent", None)
        for formula in getattr(summary, "candidate_records", ()):
            proposed_candidates.append(
                FormulaCandidate(
                    formula=str(formula),
                    source="candidate_records",
                    role=role_lookup.get(str(formula)),
                )
            )
    merged = merge_formula_candidates(champion_candidates, final_candidates, proposed_candidates)
    return AnchorGenerationSummary(
        candidates=merged,
        champion_records=tuple(str(item) for item in getattr(summary, "champion_records", ()) if str(item)),
        final_records=tuple(str(item) for item in getattr(summary, "final_records", ()) if str(item)),
        candidate_records=tuple(str(item.formula) for item in merged),
    )
