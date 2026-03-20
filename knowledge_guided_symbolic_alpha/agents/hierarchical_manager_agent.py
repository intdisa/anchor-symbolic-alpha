from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..envs.common_knowledge import CommonKnowledgeEncoder
from ..evaluation.distributional_critic import DistributionalCollectionCritic
from ..evaluation.factor_pool import FactorPool
from ..evaluation.panel_dispatch import is_cross_sectional_frame
from ..evaluation.role_profiles import normalize_role
from ..memory import ExperienceMemory
from ..models.controllers import LibraryPlanner, RegimeController
from ..training.curriculum import CurriculumStage
from ..training.reward_shaping import PoolRewardShaper
from .base import AgentProposal, BaseRoleAgent
from .reviewer_agent import ReviewOutcome, ReviewerAgent


@dataclass(frozen=True)
class HierarchicalManagerStep:
    selected_agent: str
    regime: str
    reward: float
    accepted: bool
    decision_reason: str
    review_reason: str
    pool_size: int


@dataclass(frozen=True)
class _SkillCandidate:
    skill_name: str
    proposal: AgentProposal
    normalized_role: str | None
    selection_score: float
    planner_score: float
    expected_gain: float
    risk_adjusted_gain: float
    walk_forward_proxy_gain: float
    critic_accepted: bool
    validation_bootstrap_eligible: bool
    review_reason: str
    review_approved: bool
    decision_reason: str


@dataclass(frozen=True)
class _PairSelectionPlan:
    anchor: _SkillCandidate
    follower: _SkillCandidate
    pair_score: float
    final_walk_forward_proxy: float


class HierarchicalManagerAgent:
    def __init__(
        self,
        agents: dict[str, BaseRoleAgent],
        planner: LibraryPlanner,
        reviewer_agent: ReviewerAgent | None = None,
        regime_controller: RegimeController | None = None,
        reward_shaper: PoolRewardShaper | None = None,
        critic: DistributionalCollectionCritic | None = None,
        experience_memory: ExperienceMemory | None = None,
        common_knowledge_encoder: CommonKnowledgeEncoder | None = None,
        warm_start_seed_limit: int = 4,
        rejection_cooldown_threshold: int = 3,
        cooldown_steps: int = 2,
        bootstrap_anchor_skill: str | None = None,
        allow_validation_backed_bootstrap: bool = True,
        allow_validation_backed_replacement: bool = True,
        allow_validation_backed_upgrade: bool = True,
        enforce_flow_residual_gate: bool = True,
    ) -> None:
        if not agents:
            raise ValueError("HierarchicalManagerAgent requires at least one skill agent.")
        self.agents = dict(agents)
        self.agent_names = tuple(self.agents.keys())
        self.planner = planner
        self.reviewer_agent = reviewer_agent or ReviewerAgent()
        self.regime_controller = regime_controller or RegimeController()
        self.reward_shaper = reward_shaper or PoolRewardShaper()
        self.critic = critic or DistributionalCollectionCritic(self.reward_shaper)
        self.experience_memory = experience_memory or ExperienceMemory()
        self.common_knowledge_encoder = common_knowledge_encoder or CommonKnowledgeEncoder()
        self.selection_counts = {name: 0 for name in self.agent_names}
        self.accepted_counts = {name: 0 for name in self.agent_names}
        self.rejection_streaks = {name: 0 for name in self.agent_names}
        self.cooldown_remaining = {name: 0 for name in self.agent_names}
        self.warm_start_seed_limit = warm_start_seed_limit
        self.rejection_cooldown_threshold = rejection_cooldown_threshold
        self.cooldown_steps = cooldown_steps
        self._warm_started = False
        for agent in self.agents.values():
            agent.experience_memory = self.experience_memory
        fallback_anchor = next(
            (name for name in self.agent_names if normalize_role(name) == "target_flow"),
            self.agent_names[0],
        )
        self.explicit_bootstrap_anchor = bootstrap_anchor_skill in self.agents if bootstrap_anchor_skill is not None else False
        self.bootstrap_anchor_skill = (
            bootstrap_anchor_skill if bootstrap_anchor_skill in self.agents else fallback_anchor
        )
        self.allow_validation_backed_bootstrap = allow_validation_backed_bootstrap
        self.allow_validation_backed_replacement = allow_validation_backed_replacement
        self.allow_validation_backed_upgrade = allow_validation_backed_upgrade
        self.enforce_flow_residual_gate = enforce_flow_residual_gate

    def apply_curriculum(self, stage: CurriculumStage) -> None:
        for agent in self.agents.values():
            agent.apply_curriculum(stage)

    def warm_start(
        self,
        pool: FactorPool,
        data: pd.DataFrame,
        target: pd.Series,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> None:
        if self._warm_started:
            return
        bootstrap_pool = pool.copy()
        regime = self.regime_controller.infer(data).regime
        common_state = self.common_knowledge_encoder.encode(
            dataset_name="warm_start",
            regime=regime,
            data=data,
            pool=bootstrap_pool,
            skill_names=self.agent_names,
            target=target,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        ordered_skills = self._candidate_skill_order(self.planner.plan(common_state).ordered_skills, bootstrap_mode=True)
        for skill_name in ordered_skills:
            agent = self.agents[skill_name]
            agent.set_context(regime)
            skill_bootstrap_pool = (
                bootstrap_pool
                if skill_name == self.bootstrap_anchor_skill
                else FactorPool(max_size=bootstrap_pool.max_size)
            )
            for tokens in agent.seed_formulas[: self.warm_start_seed_limit]:
                outcome = self.reward_shaper.shape(tokens, data, target, skill_bootstrap_pool, commit=True, role=skill_name)
                bootstrap_reward = outcome.clipped_reward
                if (
                    skill_name != self.bootstrap_anchor_skill
                    and not outcome.decision.accepted
                    and outcome.decision.candidate is not None
                ):
                    bootstrap_reward = max(
                        0.05,
                        0.40 * abs(float(outcome.decision.candidate.metrics.get("rank_ic", 0.0))),
                    )
                agent.observe(tokens, bootstrap_reward, outcome.decision.accepted)
                self.experience_memory.record(
                    regime,
                    skill_name,
                    tokens,
                    bootstrap_reward,
                    outcome.decision.accepted,
                    "bootstrap_seed" if bootstrap_reward != outcome.clipped_reward else outcome.decision.reason,
                )
        self._warm_started = True

    def run_step(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        commit: bool = True,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> HierarchicalManagerStep:
        regime = self.regime_controller.infer(data).regime
        self._decay_cooldowns()
        common_state = self.common_knowledge_encoder.encode(
            dataset_name="runtime",
            regime=regime,
            data=data,
            pool=pool,
            skill_names=self.agent_names,
            target=target,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        planner_decision = self.planner.plan(common_state)
        ordered_skills = self._candidate_skill_order(planner_decision.ordered_skills)
        candidates: list[_SkillCandidate] = []
        for skill_name in ordered_skills:
            if not self._route_b_stage_allows_skill(skill_name, data):
                continue
            agent = self.agents[skill_name]
            agent.set_context(regime)
            proposal = agent.propose(
                data=data,
                target=target,
                pool=pool,
                reward_shaper=self.reward_shaper,
                validation_data=validation_data,
                validation_target=validation_target,
            )
            estimate = self.critic.estimate(
                proposal.body_tokens,
                data,
                target,
                pool,
                role=skill_name,
                validation_data=validation_data,
                validation_target=validation_target,
            )
            preview_outcome = self.reward_shaper.shape(
                proposal.body_tokens,
                data,
                target,
                pool,
                commit=False,
                role=skill_name,
            )
            review = self.reviewer_agent.review(preview_outcome.decision, pool, role=skill_name)
            if (
                not review.approved
                and self._validation_backed_preview_allowed(
                    skill_name,
                    estimate,
                    preview_outcome,
                    data,
                    pool,
                )
            ):
                review = ReviewOutcome(True, "validation_preview_accept")
            validation_bootstrap_eligible = (
                self._validation_backed_bootstrap_allowed(pool, estimate, data)
                if is_cross_sectional_frame(data)
                else False
            )
            selection_score = (
                estimate.risk_adjusted_gain
                + 0.35 * estimate.walk_forward_proxy_gain
                + 0.20 * planner_decision.skill_scores.get(skill_name, 0.0)
                + (0.20 if review.approved else -0.20)
            )
            if is_cross_sectional_frame(data) and len(pool.records) == 0:
                if skill_name == self.bootstrap_anchor_skill and validation_bootstrap_eligible:
                    selection_score += 1.0
                elif normalize_role(skill_name) == "target_flow":
                    selection_score -= 0.04
            selection_score += self._route_b_slow_family_bonus(skill_name, data)
            selection_score += self._cross_sectional_selection_bonus(estimate, data)
            flow_residual_allowed, flow_residual_reason = self._cross_sectional_flow_residual_status(
                pool,
                skill_name,
                estimate,
                data,
            )
            if (
                is_cross_sectional_frame(data)
                and len(pool.records) > 0
                and normalize_role(skill_name) == "target_flow"
                and not flow_residual_allowed
            ):
                selection_score -= 0.08
            if estimate.accepted:
                selection_score += 0.05
            candidates.append(
                _SkillCandidate(
                    skill_name=skill_name,
                    proposal=proposal,
                    normalized_role=normalize_role(skill_name),
                    selection_score=float(selection_score),
                    planner_score=float(planner_decision.skill_scores.get(skill_name, 0.0)),
                    expected_gain=float(estimate.expected_gain),
                    risk_adjusted_gain=float(estimate.risk_adjusted_gain),
                    walk_forward_proxy_gain=float(estimate.walk_forward_proxy_gain),
                    critic_accepted=bool(estimate.accepted),
                    validation_bootstrap_eligible=bool(validation_bootstrap_eligible),
                    review_reason=review.reason,
                    review_approved=review.approved,
                    decision_reason=estimate.reason,
                )
            )

        if not candidates:
            return HierarchicalManagerStep(
                selected_agent="none",
                regime=regime,
                reward=0.0,
                accepted=False,
                decision_reason="planner_empty",
                review_reason="planner_empty",
                pool_size=len(pool),
            )

        selected = self._select_bootstrap_anchor_candidate(candidates, pool, data) or max(candidates, key=self._candidate_sort_key)
        paired_anchor = self._select_paired_anchor(
            candidates,
            data,
            target,
            pool,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        if paired_anchor is not None:
            selected = paired_anchor
        commit_pool = pool
        selected_estimate = self.critic.estimate(
            selected.proposal.body_tokens,
            data,
            target,
            pool,
            role=selected.skill_name,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        override_pool = self._override_commit_pool(selected, pool)
        allow_commit_override = not (
            is_cross_sectional_frame(data) and selected.normalized_role == "target_flow"
        )
        if override_pool is not None and allow_commit_override:
            override_estimate = self.critic.estimate(
                selected.proposal.body_tokens,
                data,
                target,
                override_pool,
                role=selected.skill_name,
                validation_data=validation_data,
                validation_target=validation_target,
            )
            if self._prefer_commit_override(selected, selected_estimate, override_estimate):
                commit_pool = override_pool
                selected_estimate = override_estimate
        selected_flow_allowed, selected_flow_reason = self._cross_sectional_flow_residual_status(
            commit_pool,
            selected.skill_name,
            selected_estimate,
            data,
        )
        commit_allowed = (
            selected.review_approved
            and (
                (len(commit_pool.records) == 0 and selected_estimate.accepted)
                or (
                    self._bootstrap_commit_allowed(commit_pool, selected)
                    and (
                        not (
                            is_cross_sectional_frame(data)
                            and selected.normalized_role == "target_flow"
                            and len(commit_pool.records) > 0
                        )
                        or selected_flow_allowed
                    )
                )
                or (self._validation_backed_bootstrap_allowed(commit_pool, selected_estimate, data))
                or (
                    len(commit_pool.records) > 0
                    and selected_estimate.risk_adjusted_gain > 0.0
                    and selected_estimate.walk_forward_proxy_gain >= 0.0
                    and selected_flow_allowed
                )
            )
        )
        if commit_allowed and commit:
            final_outcome = self.reward_shaper.shape(
                selected.proposal.body_tokens,
                data,
                target,
                commit_pool,
                commit=True,
                role=selected.skill_name,
            )
            accepted = final_outcome.decision.accepted
            final_reward = float(final_outcome.clipped_reward)
            decision_reason = final_outcome.decision.reason
            if (
                not accepted
                and final_outcome.decision.reason in {"full_validation", "fast_ic_screen"}
                and self._validation_backed_bootstrap_allowed(commit_pool, selected_estimate, data)
            ):
                accepted = self._apply_validation_backed_commit(commit_pool, final_outcome)
                if accepted:
                    final_reward = float(max(final_reward, selected_estimate.risk_adjusted_gain))
                    decision_reason = "validation_backed_commit"
            if (
                not accepted
                and final_outcome.decision.reason in {"full_validation", "fast_ic_screen", "replacement_check"}
                and self._validation_backed_replacement_allowed(
                    commit_pool,
                    selected.skill_name,
                    selected_estimate,
                    data,
                )
            ):
                accepted = self._apply_validation_backed_replacement(commit_pool, selected_estimate)
                if accepted:
                    final_reward = float(max(final_reward, selected_estimate.risk_adjusted_gain))
                    decision_reason = "validation_backed_replacement"
            if (
                not accepted
                and final_outcome.decision.reason in {"full_validation", "fast_ic_screen", "replacement_check"}
                and self._validation_backed_upgrade_allowed(
                    commit_pool,
                    selected.skill_name,
                    selected_estimate,
                    data,
                )
            ):
                accepted = self._apply_validation_backed_upgrade(commit_pool, selected_estimate)
                if accepted:
                    final_reward = float(max(final_reward, selected_estimate.risk_adjusted_gain))
                    decision_reason = "validation_backed_upgrade"
            if accepted and commit_pool is not pool:
                pool.records = list(commit_pool.records)
        else:
            final_outcome = self.reward_shaper.shape(
                selected.proposal.body_tokens,
                data,
                target,
                commit_pool,
                commit=False,
                role=selected.skill_name,
            )
            accepted = False
            final_reward = float(min(final_outcome.clipped_reward, selected_estimate.risk_adjusted_gain) - 0.10)
            if not selected.review_approved:
                decision_reason = selected.review_reason
            elif len(commit_pool.records) == 0 and not selected_estimate.accepted:
                decision_reason = "critic_bootstrap_reject"
            elif selected_estimate.risk_adjusted_gain <= 0.0:
                decision_reason = "critic_negative_gain"
            elif (
                is_cross_sectional_frame(data)
                and selected.normalized_role == "target_flow"
                and not selected_flow_allowed
            ):
                decision_reason = selected_flow_reason or "critic_negative_trade_proxy"
            else:
                decision_reason = "critic_negative_walk_forward_proxy"

        for candidate in candidates:
            candidate_reward = final_reward if candidate.skill_name == selected.skill_name else candidate.risk_adjusted_gain
            candidate_accepted = accepted if candidate.skill_name == selected.skill_name else False
            self.agents[candidate.skill_name].observe(candidate.proposal.body_tokens, candidate_reward, candidate_accepted)
            self.experience_memory.record(
                regime,
                candidate.skill_name,
                candidate.proposal.body_tokens,
                candidate_reward,
                candidate_accepted,
                decision_reason if candidate.skill_name == selected.skill_name else candidate.decision_reason,
            )

        self._update_routing_state(selected.skill_name, accepted, decision_reason)
        self.selection_counts[selected.skill_name] += 1
        if accepted:
            self.accepted_counts[selected.skill_name] += 1
        return HierarchicalManagerStep(
            selected_agent=selected.skill_name,
            regime=regime,
            reward=float(final_reward),
            accepted=accepted,
            decision_reason=decision_reason,
            review_reason=selected.review_reason,
            pool_size=len(pool),
        )

    def _candidate_skill_order(self, ordered_skills: tuple[str, ...], bootstrap_mode: bool = False) -> tuple[str, ...]:
        available = [
            skill
            for skill in ordered_skills
            if self.cooldown_remaining.get(skill, 0) <= 0
        ]
        if not available:
            available = list(ordered_skills)
        anchored_skills: list[str] = []
        if (
            self.explicit_bootstrap_anchor
            and self.bootstrap_anchor_skill in self.agent_names
            and self.accepted_counts.get(self.bootstrap_anchor_skill, 0) == 0
            and self.cooldown_remaining.get(self.bootstrap_anchor_skill, 0) <= 0
        ):
            anchored_skills.append(self.bootstrap_anchor_skill)
        bootstrap_skills = [
            skill
            for skill in self.agent_names
            if skill != self.bootstrap_anchor_skill
            and self.accepted_counts.get(skill, 0) == 0
            and self.cooldown_remaining.get(skill, 0) <= 0
        ]
        if bootstrap_mode or bootstrap_skills:
            for skill in anchored_skills + bootstrap_skills:
                if skill in available:
                    available.remove(skill)
            available = anchored_skills + bootstrap_skills + available
        return tuple(dict.fromkeys(available))

    def _route_b_stage_allows_skill(self, skill_name: str, data: pd.DataFrame) -> bool:
        if not is_cross_sectional_frame(data) or self.bootstrap_anchor_skill != "quality_solvency":
            return True
        slow_skills = tuple(
            skill
            for skill in ("quality_solvency", "efficiency_growth", "valuation_size")
            if skill in self.agents
        )
        if not slow_skills:
            return True
        accepted_slow = tuple(skill for skill in slow_skills if self.accepted_counts.get(skill, 0) > 0)
        if len(accepted_slow) == 0:
            return skill_name == "quality_solvency"
        if len(accepted_slow) == 1:
            if (
                self.accepted_counts.get("quality_solvency", 0) > 0
                and "efficiency_growth" in slow_skills
                and self.accepted_counts.get("efficiency_growth", 0) == 0
                and self.cooldown_remaining.get("efficiency_growth", 0) <= 0
            ):
                return skill_name == "efficiency_growth"
            if (
                "valuation_size" in slow_skills
                and self.accepted_counts.get("valuation_size", 0) == 0
                and self.cooldown_remaining.get("valuation_size", 0) <= 0
            ):
                return skill_name == "valuation_size"
            return skill_name in {skill for skill in slow_skills if self.accepted_counts.get(skill, 0) == 0}
        return True

    def _candidate_sort_key(self, candidate: _SkillCandidate) -> tuple[float, float, float, float, float]:
        return (
            float(int(candidate.review_approved)),
            float(candidate.walk_forward_proxy_gain),
            float(candidate.selection_score),
            float(candidate.risk_adjusted_gain),
            float(candidate.planner_score),
        )

    def _select_bootstrap_anchor_candidate(
        self,
        candidates: list[_SkillCandidate],
        pool: FactorPool,
        data: pd.DataFrame,
    ) -> _SkillCandidate | None:
        if not is_cross_sectional_frame(data) or len(pool.records) != 0:
            return None
        for candidate in candidates:
            if (
                candidate.skill_name == self.bootstrap_anchor_skill
                and candidate.review_approved
                and candidate.validation_bootstrap_eligible
            ):
                return candidate
        return None

    def _select_paired_anchor(
        self,
        candidates: list[_SkillCandidate],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> _SkillCandidate | None:
        if (
            is_cross_sectional_frame(data)
            and len(pool.records) == 0
            and self.explicit_bootstrap_anchor
            and self.accepted_counts.get(self.bootstrap_anchor_skill, 0) == 0
        ):
            return None
        flow_candidates = [candidate for candidate in candidates if candidate.normalized_role == "target_flow"]
        price_candidates = [candidate for candidate in candidates if candidate.normalized_role == "target_price"]
        if not flow_candidates or not price_candidates:
            return None
        best_single = max(candidates, key=self._candidate_sort_key)
        best_flow = max(flow_candidates, key=self._candidate_sort_key)
        best_price = max(price_candidates, key=self._candidate_sort_key)
        plans: list[_PairSelectionPlan] = []
        for anchor, follower in ((best_flow, best_price), (best_price, best_flow)):
            plan = self._evaluate_pair_plan(
                anchor,
                follower,
                data,
                target,
                pool,
                validation_data=validation_data,
                validation_target=validation_target,
            )
            if plan is not None:
                plans.append(plan)
        if not plans:
            return None
        best_plan = max(plans, key=lambda plan: (plan.final_walk_forward_proxy, plan.pair_score))
        if (
            best_plan.final_walk_forward_proxy >= best_single.walk_forward_proxy_gain + 0.002
            or best_plan.pair_score >= best_single.selection_score + 0.05
        ):
            return best_plan.anchor
        return None

    def _override_commit_pool(self, candidate: _SkillCandidate, pool: FactorPool) -> FactorPool | None:
        if candidate.normalized_role == "target_price":
            conflicting_role = "target_flow"
        elif candidate.normalized_role == "target_flow":
            conflicting_role = "target_price"
        else:
            conflicting_role = None
        if not any(normalize_role(record.role) == conflicting_role for record in pool.records):
            if candidate.normalized_role != "target_price":
                return None
            same_role_indices = [
                index
                for index, record in enumerate(pool.records)
                if normalize_role(record.role) == candidate.normalized_role
            ]
            if not same_role_indices:
                return None
            weakest_index = min(
                same_role_indices,
                key=lambda index: float(pool.records[index].metrics.get("rank_ic", 0.0)),
            )
            override_pool = pool.copy()
            override_pool.records = [
                record
                for index, record in enumerate(override_pool.records)
                if index != weakest_index
            ]
            return override_pool
        override_pool = pool.copy()
        override_pool.records = [
            record
            for record in override_pool.records
            if normalize_role(record.role) != conflicting_role
        ]
        if len(override_pool.records) == len(pool.records):
            return None
        return override_pool

    def _prefer_commit_override(
        self,
        candidate: _SkillCandidate,
        base_estimate,
        override_estimate,
    ) -> bool:
        if candidate.normalized_role == "target_price":
            if override_estimate.accepted and not base_estimate.accepted:
                return True
            return (
                override_estimate.accepted
                and override_estimate.new_walk_forward_proxy
                >= max(base_estimate.new_walk_forward_proxy, base_estimate.baseline_walk_forward_proxy) + 0.002
                and override_estimate.risk_adjusted_gain >= base_estimate.risk_adjusted_gain - 0.01
            )
        if candidate.normalized_role == "target_flow":
            if override_estimate.accepted and not base_estimate.accepted:
                return True
            return (
                override_estimate.accepted
                and override_estimate.trade_proxy_gain >= base_estimate.trade_proxy_gain + 5e-4
                and override_estimate.risk_adjusted_gain >= base_estimate.risk_adjusted_gain - 0.01
            )
        return False

    def _evaluate_pair_plan(
        self,
        anchor: _SkillCandidate,
        follower: _SkillCandidate,
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> _PairSelectionPlan | None:
        if not anchor.review_approved or not anchor.critic_accepted:
            return None
        pair_pool = pool.copy()
        anchor_outcome = self.reward_shaper.shape(
            anchor.proposal.body_tokens,
            data,
            target,
            pair_pool,
            commit=True,
            role=anchor.skill_name,
        )
        if not anchor_outcome.decision.accepted:
            return None
        follower_estimate = self.critic.estimate(
            follower.proposal.body_tokens,
            data,
            target,
            pair_pool,
            role=follower.skill_name,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        follower_preview = self.reward_shaper.shape(
            follower.proposal.body_tokens,
            data,
            target,
            pair_pool,
            commit=False,
            role=follower.skill_name,
        )
        follower_review = self.reviewer_agent.review(
            follower_preview.decision,
            pair_pool,
            role=follower.skill_name,
        )
        if not follower_review.approved or not follower_estimate.accepted:
            return None
        pair_score = (
            anchor.selection_score
            + 0.55 * max(0.0, follower_estimate.risk_adjusted_gain)
            + 0.80 * max(0.0, follower_estimate.walk_forward_proxy_gain)
            + 0.60 * follower_estimate.new_walk_forward_proxy
        )
        if anchor.normalized_role == "target_price" and follower.normalized_role == "target_flow":
            pair_score += 0.03
        return _PairSelectionPlan(
            anchor=anchor,
            follower=follower,
            pair_score=float(pair_score),
            final_walk_forward_proxy=float(follower_estimate.new_walk_forward_proxy),
        )

    def _bootstrap_commit_allowed(self, pool: FactorPool, candidate: _SkillCandidate) -> bool:
        if len(pool.records) >= 2:
            return False
        if candidate.skill_name == self.bootstrap_anchor_skill:
            return False
        if self.accepted_counts.get(candidate.skill_name, 0) > 0:
            return False
        return candidate.critic_accepted and candidate.expected_gain > -0.02 and candidate.walk_forward_proxy_gain > -0.02

    def _validation_backed_bootstrap_allowed(
        self,
        pool: FactorPool,
        estimate,
        data: pd.DataFrame,
    ) -> bool:
        if not self.allow_validation_backed_bootstrap:
            return False
        if len(pool.records) != 0:
            return False
        if not is_cross_sectional_frame(data):
            return False
        preview = estimate.preview
        if preview is None or preview.record is None:
            return False
        if preview.accepted and estimate.validation_gain > 0.0 and estimate.walk_forward_proxy_gain >= 0.0:
            return True
        metrics = preview.record.metrics
        sharpe = float(metrics.get("sharpe", 0.0))
        annual_return = float(metrics.get("annual_return", 0.0))
        turnover = float(metrics.get("turnover", float("inf")))
        drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        return (
            normalize_role(preview.record.role) == "target_price"
            and estimate.new_walk_forward_proxy >= estimate.baseline_walk_forward_proxy + 0.01
            and sharpe > 0.0
            and annual_return > 0.0
            and turnover < 0.10
            and drawdown < 0.35
        )

    def _validation_backed_preview_allowed(
        self,
        skill_name: str,
        estimate,
        preview_outcome,
        data: pd.DataFrame,
        pool: FactorPool,
    ) -> bool:
        if preview_outcome.decision.candidate is None:
            return False
        if preview_outcome.decision.reason == "full_validation":
            return self._validation_backed_bootstrap_allowed(pool, estimate, data)
        if preview_outcome.decision.reason == "fast_ic_screen" and len(pool.records) == 0:
            return self._validation_backed_bootstrap_allowed(pool, estimate, data)
        if not is_cross_sectional_frame(data):
            return False
        if normalize_role(skill_name) != "target_price":
            return False
        if not self.allow_validation_backed_replacement:
            return False
        if self._validation_backed_upgrade_allowed(pool, skill_name, estimate, data):
            return True
        if preview_outcome.decision.reason not in {"replaced_baseline", "replaced"}:
            preview = estimate.preview
            if preview is None or preview.reason not in {"replaced_baseline", "replaced"}:
                return False
        if estimate.preview is None or not estimate.preview.accepted:
            return False
        preview = estimate.preview
        if preview.reason in {"replaced_baseline", "replaced"}:
            return (
                estimate.trade_proxy_gain > -0.01
                and estimate.walk_forward_proxy_gain > 0.0
                and estimate.new_walk_forward_proxy >= estimate.baseline_walk_forward_proxy + 0.002
            )
        if estimate.trade_proxy_gain <= 0.0 or estimate.walk_forward_proxy_gain <= 0.0:
            return False
        return True

    def _cross_sectional_flow_residual_allowed(
        self,
        pool: FactorPool,
        skill_name: str,
        estimate,
        data: pd.DataFrame,
    ) -> bool:
        return self._cross_sectional_flow_residual_status(pool, skill_name, estimate, data)[0]

    def _cross_sectional_flow_residual_status(
        self,
        pool: FactorPool,
        skill_name: str,
        estimate,
        data: pd.DataFrame,
    ) -> tuple[bool, str | None]:
        if not is_cross_sectional_frame(data):
            return True, None
        if normalize_role(skill_name) != "target_flow":
            return True, None
        if not self.enforce_flow_residual_gate:
            return True, None
        if not pool.records:
            if skill_name == self.bootstrap_anchor_skill:
                return True, None
            return False, "critic_anchor_bootstrap_only"
        has_price_baseline = any(normalize_role(record.role) == "target_price" for record in pool.records)
        if not has_price_baseline:
            return True, None
        if estimate.trade_proxy_gain <= 0.0:
            return False, "critic_negative_trade_proxy"
        if (
            estimate.walk_forward_proxy_gain <= 0.0
            or estimate.new_walk_forward_proxy < estimate.baseline_walk_forward_proxy + 0.002
        ):
            return False, "critic_negative_walk_forward_proxy"
        preview = estimate.preview
        if preview is None or preview.record is None:
            return False, "critic_unstable_flow_residual"
        metrics = preview.record.metrics
        stability = float(metrics.get("stability_score", float("nan")))
        rank_positive = float(metrics.get("rank_ic_window_positive_frac", 0.0))
        ls_positive = float(metrics.get("ls_return_window_positive_frac", 0.0))
        turnover = float(metrics.get("turnover", float("inf")))
        if (
            not np.isfinite(stability)
            or stability < 0.0
            or rank_positive < 0.75
            or ls_positive < 0.75
            or turnover > 0.50
        ):
            return False, "critic_unstable_flow_residual"
        return True, None

    def _apply_validation_backed_commit(
        self,
        pool: FactorPool,
        outcome,
    ) -> bool:
        record = outcome.decision.candidate
        if record is None:
            return False
        if record.canonical in pool.canonicals():
            return False
        if len(pool.records) >= pool.max_size:
            return False
        pool.add(record)
        return True

    def _validation_backed_replacement_allowed(
        self,
        pool: FactorPool,
        skill_name: str,
        estimate,
        data: pd.DataFrame,
    ) -> bool:
        if not self.allow_validation_backed_replacement:
            return False
        if not is_cross_sectional_frame(data):
            return False
        if normalize_role(skill_name) != "target_price":
            return False
        if len(pool.records) == 0:
            return False
        preview = estimate.preview
        if preview is None or preview.record is None or not preview.accepted:
            return False
        if preview.reason not in {"replaced_baseline", "replaced"}:
            return False
        if preview.replaced_canonical is None:
            return False
        if estimate.trade_proxy_gain <= -0.01:
            return False
        if estimate.walk_forward_proxy_gain <= 0.0:
            return False
        if estimate.new_walk_forward_proxy < estimate.baseline_walk_forward_proxy:
            return False
        return True

    def _validation_backed_upgrade_allowed(
        self,
        pool: FactorPool,
        skill_name: str,
        estimate,
        data: pd.DataFrame,
    ) -> bool:
        if not self.allow_validation_backed_upgrade:
            return False
        if not is_cross_sectional_frame(data):
            return False
        if normalize_role(skill_name) != "target_price":
            return False
        replace_index = self._target_price_replace_index(pool)
        if replace_index is None:
            return False
        preview = estimate.preview
        if preview is None or preview.record is None:
            return False
        if preview.record.canonical == pool.records[replace_index].canonical:
            return False
        metrics = preview.record.metrics
        sharpe = float(metrics.get("sharpe", 0.0))
        annual_return = float(metrics.get("annual_return", 0.0))
        turnover = float(metrics.get("turnover", float("inf")))
        drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        if (
            sharpe <= 0.0
            or annual_return <= 0.0
            or turnover >= 0.10
            or drawdown >= 0.35
        ):
            return False
        if estimate.new_walk_forward_proxy < estimate.baseline_walk_forward_proxy + 0.01:
            return False
        if estimate.walk_forward_proxy_gain <= -0.01:
            return False
        if estimate.trade_proxy_gain <= -0.02:
            return False
        return True

    def _apply_validation_backed_replacement(
        self,
        pool: FactorPool,
        estimate,
    ) -> bool:
        preview = estimate.preview
        if preview is None or preview.record is None or preview.replaced_canonical is None:
            return False
        replace_index = next(
            (
                index
                for index, record in enumerate(pool.records)
                if record.canonical == preview.replaced_canonical
            ),
            None,
        )
        if replace_index is None:
            return False
        if preview.record.canonical in pool.canonicals() and pool.records[replace_index].canonical != preview.record.canonical:
            return False
        pool.replace(replace_index, preview.record)
        return True

    def _apply_validation_backed_upgrade(
        self,
        pool: FactorPool,
        estimate,
    ) -> bool:
        preview = estimate.preview
        if preview is None or preview.record is None:
            return False
        replace_index = self._target_price_replace_index(pool)
        if replace_index is None:
            return False
        if preview.record.canonical in pool.canonicals() and pool.records[replace_index].canonical != preview.record.canonical:
            return False
        pool.replace(replace_index, preview.record)
        return True

    @staticmethod
    def _target_price_replace_index(pool: FactorPool) -> int | None:
        candidates = [
            index
            for index, record in enumerate(pool.records)
            if normalize_role(record.role) == "target_price"
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda index: float(pool.records[index].metrics.get("rank_ic", 0.0)))

    def _cross_sectional_selection_bonus(self, estimate, data: pd.DataFrame) -> float:
        if not is_cross_sectional_frame(data):
            return 0.0
        preview = estimate.preview
        if preview is None or preview.record is None:
            return 0.0
        metrics = preview.record.metrics
        sharpe = float(metrics.get("sharpe", 0.0))
        rank_icir = float(metrics.get("rank_icir", 0.0))
        turnover = max(0.0, float(metrics.get("turnover", 0.0)))
        drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        stability = float(metrics.get("stability_score", 0.0))
        return float(
            0.08 * np.tanh(sharpe / 2.0)
            + 0.04 * max(min(rank_icir, 0.5), -0.5)
            + 0.50 * max(min(stability, 0.05), -0.05)
            - 0.08 * turnover
            - 0.04 * drawdown
        )

    def _route_b_slow_family_bonus(self, skill_name: str, data: pd.DataFrame) -> float:
        if not is_cross_sectional_frame(data) or self.bootstrap_anchor_skill != "quality_solvency":
            return 0.0
        slow_skills = tuple(
            skill
            for skill in ("quality_solvency", "efficiency_growth", "valuation_size")
            if skill in self.agents
        )
        accepted_slow = sum(1 for skill in slow_skills if self.accepted_counts.get(skill, 0) > 0)
        if skill_name in slow_skills:
            if accepted_slow == 0 and skill_name == "quality_solvency":
                return 0.40
            if 0 < accepted_slow < 2 and self.accepted_counts.get(skill_name, 0) == 0:
                return 0.28
            return 0.0
        if skill_name == "short_horizon_flow" and 0 < accepted_slow < 2:
            return -0.30
        return 0.0

    def _decay_cooldowns(self) -> None:
        for skill_name, remaining in list(self.cooldown_remaining.items()):
            if remaining > 0:
                self.cooldown_remaining[skill_name] = remaining - 1

    def _update_routing_state(self, skill_name: str, accepted: bool, reason: str) -> None:
        if accepted:
            self.rejection_streaks[skill_name] = 0
            self.cooldown_remaining[skill_name] = 0
            return
        if reason in {
            "fast_ic_screen",
            "full_validation",
            "review_high_turnover",
            "critic_negative_gain",
            "critic_negative_walk_forward_proxy",
            "critic_bootstrap_reject",
            "critic_unstable_flow_residual",
            "validation_backed_replacement",
        }:
            self.rejection_streaks[skill_name] += 1
            if self.rejection_streaks[skill_name] >= self.rejection_cooldown_threshold:
                self.cooldown_remaining[skill_name] = self.cooldown_steps
                self.rejection_streaks[skill_name] = 0
        else:
            self.rejection_streaks[skill_name] = 0
