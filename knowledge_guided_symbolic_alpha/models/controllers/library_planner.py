from __future__ import annotations

from dataclasses import dataclass

from ...envs.common_knowledge import CommonKnowledgeState


@dataclass(frozen=True)
class PlannerDecision:
    ordered_skills: tuple[str, ...]
    selected_slot: int | None
    budget_multiplier: float
    stop: bool
    rationale: str
    skill_scores: dict[str, float]


class LibraryPlanner:
    def __init__(
        self,
        skill_names: tuple[str, ...],
        max_shortlist: int = 3,
        base_skill_weights: dict[str, float] | None = None,
    ) -> None:
        self.skill_names = skill_names
        self.max_shortlist = max_shortlist
        self.base_skill_weights = base_skill_weights or {}

    def plan(self, state: CommonKnowledgeState) -> PlannerDecision:
        gap_strength = state.dataset_embedding[4] if len(state.dataset_embedding) > 4 else 0.0
        vol_strength = state.dataset_embedding[5] if len(state.dataset_embedding) > 5 else 0.0
        price_strength = state.dataset_embedding[6] if len(state.dataset_embedding) > 6 else 0.0
        high_vol = 1.0 if state.regime == "HIGH_VOLATILITY" else 0.0
        usd_strength = 1.0 if state.regime == "USD_STRENGTH" else 0.0
        us_equities_slow_skills = ("quality_solvency", "efficiency_growth", "valuation_size")
        occupied_us_equities_slow = tuple(skill for skill in us_equities_slow_skills if skill in state.occupied_skills)

        skill_scores: dict[str, float] = {}
        for skill in self.skill_names:
            score = float(self.base_skill_weights.get(skill, 0.0))
            if skill in state.missing_skills:
                score += 0.35
            if skill in state.occupied_skills:
                score -= 0.08
            if state.dataset_name == "us_equities" and state.pool_size == 0:
                if skill == "quality_solvency":
                    score += 0.32
                elif skill == "efficiency_growth":
                    score += 0.12
                elif skill == "valuation_size":
                    score += 0.04
                elif skill == "short_horizon_flow":
                    score -= 0.10
            if skill == "short_horizon_flow":
                score += 0.70 * gap_strength + 0.80 * vol_strength + 0.25 * high_vol
            elif skill == "quality_solvency":
                score += 0.10 * price_strength + 0.12 * state.pool_trade_proxy
            elif skill == "efficiency_growth":
                score += 0.08 * price_strength + 0.05 * state.pool_trade_proxy
            elif skill == "valuation_size":
                score += 0.05 * price_strength + 0.06 * state.redundancy
            elif skill == "price_structure":
                score += 0.60 * price_strength + 0.20 * gap_strength
            elif skill == "reversal_gap":
                score += 0.90 * gap_strength + 0.25 * high_vol
            elif skill == "intraday_imbalance":
                score += 0.55 * price_strength + 0.25 * gap_strength
            elif skill == "volatility_liquidity":
                score += 0.95 * vol_strength + 0.30 * high_vol
            elif skill == "trend_structure":
                score += 0.75 * price_strength + 0.10 * state.pool_trade_proxy
            elif skill == "cross_asset_context":
                score += 0.40 * state.redundancy + 0.20 * usd_strength
            elif skill == "regime_filter":
                score += 0.45 * high_vol + 0.20 * usd_strength + 0.15 * state.redundancy
            if state.pool_size >= state.max_pool_size:
                score += 0.05
            if state.dataset_name == "us_equities":
                if (
                    skill == "efficiency_growth"
                    and "quality_solvency" in state.occupied_skills
                    and "efficiency_growth" in state.missing_skills
                ):
                    score += 0.18 + 0.20 * max(0.0, state.pool_trade_proxy)
                if (
                    skill == "valuation_size"
                    and any(parent in state.occupied_skills for parent in ("quality_solvency", "efficiency_growth"))
                    and "valuation_size" in state.missing_skills
                ):
                    score += 0.10 + 0.05 * max(0.0, state.pool_trade_proxy)
                if (
                    skill in {"efficiency_growth", "valuation_size"}
                    and len(occupied_us_equities_slow) == 1
                    and skill in state.missing_skills
                ):
                    score += 0.12
                if (
                    skill == "short_horizon_flow"
                    and any(parent in state.occupied_skills for parent in us_equities_slow_skills)
                    and "short_horizon_flow" in state.missing_skills
                ):
                    score += 0.06 + 0.05 * max(0.0, state.validation_pool_score)
                if skill == "short_horizon_flow" and 0 < len(occupied_us_equities_slow) < 2:
                    score -= 0.22
            else:
                if (
                    skill == "trend_structure"
                    and "short_horizon_flow" in state.occupied_skills
                    and "trend_structure" in state.missing_skills
                ):
                    score += 0.18 + 0.20 * max(0.0, state.pool_trade_proxy)
                if (
                    skill == "short_horizon_flow"
                    and "trend_structure" in state.occupied_skills
                    and "short_horizon_flow" in state.missing_skills
                ):
                    score += 0.10 + 0.10 * max(0.0, state.validation_pool_score)
            skill_scores[skill] = score

        ranked_skills = tuple(
            skill
            for skill, _ in sorted(
                skill_scores.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        )
        stage_eligible = self._us_equities_stage_eligible_skills(state)
        if stage_eligible is not None:
            ranked_skills = tuple(skill for skill in ranked_skills if skill in stage_eligible)
        ordered_skills = ranked_skills[: self.max_shortlist]
        ordered_skills = self._inject_paired_shortlist(ordered_skills, skill_scores, state)
        if not ordered_skills:
            ordered_skills = self.skill_names[:1]
        budget_multiplier = 1.20 if ordered_skills[0] in state.missing_skills else 1.0
        rationale = f"lead_skill={ordered_skills[0]} missing={','.join(state.missing_skills) or 'none'}"
        return PlannerDecision(
            ordered_skills=ordered_skills,
            selected_slot=state.pool_size if state.pool_size < state.max_pool_size else 0,
            budget_multiplier=budget_multiplier,
            stop=False,
            rationale=rationale,
            skill_scores=skill_scores,
        )

    def _us_equities_stage_eligible_skills(self, state: CommonKnowledgeState) -> tuple[str, ...] | None:
        if state.dataset_name != "us_equities":
            return None
        slow_skills = tuple(
            skill
            for skill in ("quality_solvency", "efficiency_growth", "valuation_size")
            if skill in self.skill_names
        )
        if not slow_skills:
            return None
        occupied_slow = tuple(skill for skill in slow_skills if skill in state.occupied_skills)
        if len(occupied_slow) == 0:
            return ("quality_solvency",) if "quality_solvency" in slow_skills else slow_skills
        if len(occupied_slow) == 1:
            if (
                "quality_solvency" in state.occupied_skills
                and "efficiency_growth" in slow_skills
                and "efficiency_growth" in state.missing_skills
            ):
                return ("efficiency_growth",)
            missing_slow = tuple(skill for skill in slow_skills if skill not in state.occupied_skills)
            if missing_slow:
                return missing_slow
        return None

    def _inject_paired_shortlist(
        self,
        ordered_skills: tuple[str, ...],
        skill_scores: dict[str, float],
        state: CommonKnowledgeState,
    ) -> tuple[str, ...]:
        pair = ("short_horizon_flow", "price_structure")
        if not all(skill in self.skill_names for skill in pair):
            return ordered_skills
        if state.dataset_name == "us_equities" and state.pool_size == 0:
            return ordered_skills
        if any(skill in state.occupied_skills for skill in pair):
            return ordered_skills
        if state.pool_size > 1 and not any(skill in state.missing_skills for skill in pair):
            return ordered_skills
        shortlist = list(ordered_skills)
        ranked_skills = [
            skill
            for skill, _ in sorted(skill_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        ]
        for skill in pair:
            if skill in shortlist:
                continue
            replacement_index = next(
                (
                    index
                    for index in range(len(shortlist) - 1, -1, -1)
                    if shortlist[index] not in pair
                ),
                None,
            )
            if replacement_index is not None:
                shortlist[replacement_index] = skill
            elif len(shortlist) < self.max_shortlist:
                shortlist.append(skill)
        ordered = [skill for skill in ranked_skills if skill in set(shortlist)]
        return tuple(dict.fromkeys(ordered[: self.max_shortlist]))
