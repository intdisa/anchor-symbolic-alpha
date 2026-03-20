from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..memory import ExperienceMemory
from ..models.controllers import RegimeController
from ..training.curriculum import CurriculumStage
from ..training.reward_shaping import PoolRewardShaper
from ..evaluation import FactorPool, preview_candidate_on_dataset
from ..evaluation.pool_scoring import rescore_pool_on_dataset
from ..evaluation.role_profiles import normalize_role
from .base import AgentProposal, BaseRoleAgent
from .reviewer_agent import ReviewerAgent


@dataclass(frozen=True)
class CompetitiveManagerStep:
    selected_agent: str
    regime: str
    reward: float
    accepted: bool
    decision_reason: str
    review_reason: str
    pool_size: int


@dataclass(frozen=True)
class _CandidateResult:
    agent_name: str
    proposal: AgentProposal
    provisional_reward: float
    effective_reward: float
    selection_score: float
    coordination_bonus: float
    would_accept: bool
    decision_reason: str
    review_reason: str
    review_approved: bool = False
    decision_accepted: bool = False
    override_drop_roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PairCandidateResult:
    agent_names: tuple[str, ...]
    commit_order: tuple[str, ...]
    effective_reward: float
    provisional_reward: float
    selection_score: float
    would_accept: bool
    decision_reason: str
    review_reason: str
    component_tokens: dict[str, tuple[str, ...]]
    component_rewards: dict[str, float]
    component_acceptance: dict[str, bool]
    component_reasons: dict[str, str]
    proposal_valid: bool


@dataclass(frozen=True)
class _SecondaryCommitResult:
    agent_name: str
    reward: float
    accepted: bool
    reason: str


class CompetitiveManagerAgent:
    ROLE_PRIORITY = {
        "context": 0,
        "target_flow_vol": 1,
        "target_flow_gap": 1,
        "target_flow": 1,
        "target_price": 2,
    }

    def __init__(
        self,
        agents: dict[str, BaseRoleAgent],
        reviewer_agent: ReviewerAgent | None = None,
        regime_controller: RegimeController | None = None,
        reward_shaper: PoolRewardShaper | None = None,
        experience_memory: ExperienceMemory | None = None,
        fixed_agent_name: str | None = None,
        warm_start_rounds: int = 4,
        warm_start_seed_limit: int = 6,
    ) -> None:
        if not agents:
            raise ValueError("CompetitiveManagerAgent requires at least one role agent.")
        if fixed_agent_name is not None and fixed_agent_name not in agents:
            raise ValueError(f"Unknown fixed_agent_name {fixed_agent_name!r}.")
        self.experience_memory = experience_memory or ExperienceMemory()
        self.agents = dict(agents)
        self.agent_names = tuple(self.agents.keys())
        self.fixed_agent_name = fixed_agent_name
        for agent in self.agents.values():
            agent.experience_memory = self.experience_memory
        self.reviewer_agent = reviewer_agent or ReviewerAgent()
        self.regime_controller = regime_controller or RegimeController()
        self.reward_shaper = reward_shaper or PoolRewardShaper()
        self.selection_counts = {name: 0 for name in self.agent_names}
        self.accepted_counts = {name: 0 for name in self.agent_names}
        self.warm_start_rounds = warm_start_rounds
        self.warm_start_seed_limit = warm_start_seed_limit
        self._warm_started = False
        self.frozen_baseline_canonicals: set[str] = set()

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
        flow_agent_names = [name for name in self.agent_names if normalize_role(name) == "target_flow"]
        if not flow_agent_names:
            self._warm_started = True
            return
        regime = self.regime_controller.infer(data).regime
        if "target_price" in self.agents:
            target_price_agent = self.agents["target_price"]
            target_price_agent.set_context(regime)
            for tokens in target_price_agent.seed_formulas[: self.warm_start_seed_limit]:
                accepted, canonical = self._warm_start_baseline_tokens(
                    "target_price",
                    tokens,
                    data,
                    target,
                    pool,
                    validation_data=validation_data,
                    validation_target=validation_target,
                )
                if accepted and canonical is not None:
                    self.frozen_baseline_canonicals.add(canonical)
            for _ in range(max(1, self.warm_start_rounds // 2)):
                proposal = target_price_agent.propose(
                    data=data,
                    target=target,
                    pool=pool,
                    reward_shaper=self.reward_shaper,
                    validation_data=validation_data,
                    validation_target=validation_target,
                )
                accepted, canonical = self._warm_start_baseline_tokens(
                    "target_price",
                    proposal.body_tokens,
                    data,
                    target,
                    pool,
                    validation_data=validation_data,
                    validation_target=validation_target,
                )
                if accepted and canonical is not None:
                    self.frozen_baseline_canonicals.add(canonical)
        local_pool = pool.copy()
        for agent_name in flow_agent_names:
            agent = self.agents[agent_name]
            agent.set_context(regime)
            for tokens in agent.seed_formulas[: self.warm_start_seed_limit]:
                self._warm_start_tokens(
                    agent_name,
                    tokens,
                    data,
                    target,
                    local_pool,
                    validation_data=validation_data,
                    validation_target=validation_target,
                )
        for _ in range(self.warm_start_rounds):
            for agent_name in flow_agent_names:
                agent = self.agents[agent_name]
                agent.set_context(regime)
                proposal = agent.propose(
                    data=data,
                    target=target,
                    pool=local_pool,
                    reward_shaper=self.reward_shaper,
                    validation_data=validation_data,
                    validation_target=validation_target,
                )
                self._warm_start_tokens(
                    agent_name,
                    proposal.body_tokens,
                    data,
                    target,
                    local_pool,
                    validation_data=validation_data,
                    validation_target=validation_target,
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
    ) -> CompetitiveManagerStep:
        regime = self.regime_controller.infer(data).regime
        candidate_names = (self.fixed_agent_name,) if self.fixed_agent_name is not None else self.agent_names
        candidates: list[_CandidateResult] = []
        for agent_name in candidate_names:
            agent = self.agents[agent_name]
            agent.set_context(regime)
            proposal = agent.propose(
                data=data,
                target=target,
                pool=pool,
                reward_shaper=self.reward_shaper,
                validation_data=validation_data,
                validation_target=validation_target,
            )
            provisional = self.reward_shaper.shape(
                proposal.body_tokens,
                data,
                target,
                pool,
                commit=False,
                role=agent_name,
            )
            review_pool = pool
            override_drop_roles: tuple[str, ...] = ()
            review = self.reviewer_agent.review(provisional.decision, review_pool, role=agent_name)
            if normalize_role(agent_name) == "target_flow" and proposal.valid:
                flow_commit_preview = self._best_flow_commit_preview(
                    agent_name,
                    proposal.body_tokens,
                    data,
                    target,
                    pool,
                )
                if flow_commit_preview is not None:
                    preview_outcome, preview_drop_roles, preview_pool = flow_commit_preview
                    if self._prefer_flow_commit_preview(provisional, preview_outcome, override_drop_roles, preview_drop_roles):
                        provisional = preview_outcome
                        override_drop_roles = preview_drop_roles
                        review_pool = preview_pool
                        review = self.reviewer_agent.review(provisional.decision, review_pool, role=agent_name)
            coordination_bonus = self._coordination_bonus(agent_name, provisional.decision, review_pool)
            effective_reward = provisional.clipped_reward + coordination_bonus
            selection_score = effective_reward
            flow_viable = True
            if (
                validation_data is not None
                and validation_target is not None
                and normalize_role(agent_name) == "target_flow"
                and any(normalize_role(record.role) == "target_price" for record in pool.records)
                and review.approved
                and provisional.decision.accepted
                and not override_drop_roles
            ):
                flow_residual_preview = self._validation_preview(
                    agent_name,
                    proposal.body_tokens,
                    pool,
                    validation_data,
                    validation_target,
                )
                flow_standalone_preview = self._standalone_validation_preview(
                    agent_name,
                    proposal.body_tokens,
                    validation_data,
                    validation_target,
                    pool.max_size,
                )
                flow_viable = self._flow_residual_gate(flow_residual_preview, flow_standalone_preview)
            if validation_data is not None and validation_target is not None and review.approved and provisional.decision.accepted:
                selection_score += self._validation_selection_bonus(
                    agent_name,
                    proposal.body_tokens,
                    review_pool,
                    validation_data,
                    validation_target,
                )
            if not review.approved:
                effective_reward = min(-0.1, -abs(provisional.clipped_reward))
                selection_score = effective_reward
                coordination_bonus = 0.0
            elif not flow_viable:
                effective_reward = min(-0.15, -abs(provisional.clipped_reward))
                selection_score = effective_reward
                coordination_bonus = 0.0
            replaces_frozen = self._would_replace_frozen_baseline(agent_name, provisional.decision)
            if replaces_frozen:
                effective_reward = min(effective_reward, -0.2)
                selection_score = effective_reward
            decision_reason = provisional.decision.reason if flow_viable else "flow_residual_gate"
            review_reason = review.reason if flow_viable else "flow_residual_gate"
            would_accept = review.approved and provisional.decision.accepted and flow_viable and not replaces_frozen
            candidates.append(
                _CandidateResult(
                    agent_name=agent_name,
                    proposal=proposal,
                    provisional_reward=provisional.clipped_reward,
                    effective_reward=effective_reward,
                    selection_score=selection_score,
                    coordination_bonus=coordination_bonus,
                    would_accept=would_accept,
                    decision_reason=self._candidate_reason(provisional.decision, replaces_frozen) if flow_viable else decision_reason,
                    review_reason=self._review_reason(review.reason, provisional.decision, replaces_frozen) if flow_viable else review_reason,
                    review_approved=review.approved,
                    decision_accepted=provisional.decision.accepted,
                    override_drop_roles=override_drop_roles,
                )
            )
        if not candidates:
            raise RuntimeError("CompetitiveManagerAgent had no candidate agents to evaluate.")
        candidate_map = {candidate.agent_name: candidate for candidate in candidates}

        selected: _CandidateResult | _PairCandidateResult = max(candidates, key=self._candidate_key)
        pair_candidate = self._build_pair_candidate(
            candidates,
            data,
            target,
            pool,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        if pair_candidate is not None and self._candidate_key(pair_candidate) > self._candidate_key(selected):
            selected = pair_candidate
        flow_override_candidate = self._build_flow_override_candidate(
            candidates,
            data,
            target,
            pool,
            validation_data=validation_data,
            validation_target=validation_target,
        )
        if flow_override_candidate is not None and self._candidate_key(flow_override_candidate) > self._candidate_key(selected):
            selected = flow_override_candidate

        final_reward = selected.effective_reward
        accepted = False
        decision_reason = selected.decision_reason
        selected_agent_name = self._selected_name(selected)
        committed_pair_rewards: dict[str, float] = {}
        committed_pair_acceptance: dict[str, bool] = {}
        committed_pair_reasons: dict[str, str] = {}
        if isinstance(selected, _PairCandidateResult):
            if selected.would_accept and commit:
                committed_pair_rewards, committed_pair_acceptance, committed_pair_reasons = self._commit_pair(
                    selected,
                    data,
                    target,
                    pool,
                )
                final_reward = float(sum(committed_pair_rewards.values()))
                accepted = all(committed_pair_acceptance.values()) and len(committed_pair_acceptance) == len(selected.agent_names)
                decision_reason = "paired_commit" if accepted else committed_pair_reasons.get(
                    selected.commit_order[-1],
                    selected.decision_reason,
                )
        elif selected.would_accept and commit:
            original_records = None
            original_frozen = None
            if selected.override_drop_roles:
                original_records = list(pool.records)
                original_frozen = set(self.frozen_baseline_canonicals)
                self._drop_roles_in_place(pool, set(selected.override_drop_roles))
            committed = self.reward_shaper.shape(
                selected.proposal.body_tokens,
                data,
                target,
                pool,
                commit=True,
                role=selected.agent_name,
            )
            final_reward = committed.clipped_reward + selected.coordination_bonus
            accepted = committed.decision.accepted
            if not accepted and original_records is not None and original_frozen is not None:
                pool.records = original_records
                self.frozen_baseline_canonicals = original_frozen
            decision_reason = committed.decision.reason
            if accepted:
                secondary = self._maybe_commit_secondary(
                    candidate_map,
                    selected.agent_name,
                    data,
                    target,
                    pool,
                    validation_data,
                    validation_target,
                )
                if secondary is not None and secondary.accepted:
                    committed_pair_rewards = {
                        selected.agent_name: final_reward,
                        secondary.agent_name: secondary.reward,
                    }
                    committed_pair_acceptance = {
                        selected.agent_name: accepted,
                        secondary.agent_name: secondary.accepted,
                    }
                    committed_pair_reasons = {
                        selected.agent_name: decision_reason,
                        secondary.agent_name: secondary.reason,
                    }
                    final_reward += secondary.reward
                    accepted = True
                    decision_reason = f"paired_commit:{selected.agent_name}+{secondary.agent_name}"
                    selected_agent_name = f"{selected.agent_name}+{secondary.agent_name}"

        for candidate in candidates:
            agent = self.agents[candidate.agent_name]
            candidate_reward = candidate.effective_reward
            candidate_accepted = candidate.would_accept
            candidate_reason = candidate.review_reason if not candidate.would_accept else candidate.decision_reason
            if isinstance(selected, _PairCandidateResult):
                if candidate.agent_name in selected.agent_names:
                    if commit and committed_pair_rewards:
                        candidate_reward = committed_pair_rewards[candidate.agent_name]
                        candidate_accepted = committed_pair_acceptance.get(candidate.agent_name, False)
                        candidate_reason = committed_pair_reasons.get(candidate.agent_name, selected.decision_reason)
                    else:
                        candidate_reward = selected.component_rewards.get(candidate.agent_name, candidate_reward)
                        candidate_accepted = False
                        candidate_reason = selected.component_reasons.get(candidate.agent_name, selected.decision_reason)
            elif candidate.agent_name == selected.agent_name:
                candidate_reward = final_reward
                candidate_accepted = accepted
                candidate_reason = decision_reason
            agent.observe(candidate.proposal.body_tokens, candidate_reward, candidate_accepted)
            self.experience_memory.record(
                regime,
                candidate.agent_name,
                candidate.proposal.body_tokens,
                candidate_reward,
                candidate_accepted,
                candidate_reason,
            )

        for name in selected_agent_name.split("+"):
            if name in self.selection_counts:
                self.selection_counts[name] += 1
                if accepted:
                    self.accepted_counts[name] += 1

        return CompetitiveManagerStep(
            selected_agent=selected_agent_name,
            regime=regime,
            reward=float(final_reward),
            accepted=accepted,
            decision_reason=decision_reason,
            review_reason=selected.review_reason,
            pool_size=len(pool),
        )

    def _candidate_key(self, candidate: _CandidateResult | _PairCandidateResult) -> tuple[int, int, int, int, int]:
        return (
            1 if candidate.would_accept else 0,
            int(round(self._adjusted_selection_score(candidate) * 1_000_000)),
            int(round(candidate.provisional_reward * 1_000_000)),
            1 if self._proposal_valid(candidate) else 0,
            self._selection_priority(candidate),
        )

    def _selected_name(self, candidate: _CandidateResult | _PairCandidateResult) -> str:
        if isinstance(candidate, _PairCandidateResult):
            return "+".join(candidate.agent_names)
        return candidate.agent_name

    def _proposal_valid(self, candidate: _CandidateResult | _PairCandidateResult) -> bool:
        if isinstance(candidate, _PairCandidateResult):
            return candidate.proposal_valid
        return candidate.proposal.valid

    def _selection_priority(self, candidate: _CandidateResult | _PairCandidateResult) -> int:
        if isinstance(candidate, _PairCandidateResult):
            return max(self.ROLE_PRIORITY.get(name, -1) for name in candidate.agent_names)
        return self.ROLE_PRIORITY.get(candidate.agent_name, self.ROLE_PRIORITY.get(normalize_role(candidate.agent_name), -1))

    def _adjusted_selection_score(self, candidate: _CandidateResult | _PairCandidateResult) -> float:
        score = candidate.selection_score
        if isinstance(candidate, _PairCandidateResult):
            score += sum(self._quota_bonus(name) for name in candidate.agent_names)
        else:
            score += self._quota_bonus(candidate.agent_name)
        return score

    def _quota_bonus(self, agent_name: str) -> float:
        normalized = normalize_role(agent_name)
        if sum(self.selection_counts.values()) == 0:
            return 0.0
        total_target_attempts = self._family_selection_count("target_price") + self._family_selection_count("target_flow")
        if normalized == "target_flow":
            bonus = 0.0
            if self._family_accepted_count("target_flow") == 0 and total_target_attempts < 12:
                bonus += 0.12
            attempt_gap = self._family_selection_count("target_price") - self._family_selection_count("target_flow")
            if self._family_accepted_count("target_flow") == 0 and attempt_gap > 0:
                bonus += 0.02 * min(3, attempt_gap)
            return bonus
        if normalized == "context" and self._family_accepted_count("target_flow") == 0 and total_target_attempts < 12:
            return -0.05
        return 0.0

    def _build_flow_override_candidate(
        self,
        candidates: list[_CandidateResult],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> _CandidateResult | None:
        if self.fixed_agent_name is not None or validation_data is None or validation_target is None:
            return None
        if not any(normalize_role(record.role) == "target_price" for record in pool.records):
            return None
        candidate_map = {candidate.agent_name: candidate for candidate in candidates}
        if "target_price" not in candidate_map:
            return None

        baseline_validation_pool = rescore_pool_on_dataset(pool, validation_data, validation_target)
        baseline_score = baseline_validation_pool.pool_score()
        baseline_trade_proxy = baseline_validation_pool.trade_proxy_score()
        stripped_pool = self._pool_without_roles(pool, {"target_price"})
        if len(stripped_pool.records) == len(pool.records):
            return None
        stripped_validation_pool = rescore_pool_on_dataset(stripped_pool, validation_data, validation_target)
        baseline_train_score = pool.pool_score()
        baseline_train_trade_proxy = pool.trade_proxy_score()

        best_candidate: _CandidateResult | None = None
        for candidate in candidates:
            if normalize_role(candidate.agent_name) != "target_flow":
                continue
            if not candidate.proposal.valid:
                continue

            standalone_preview = self._standalone_validation_preview(
                candidate.agent_name,
                candidate.proposal.body_tokens,
                validation_data,
                validation_target,
                pool.max_size,
            )
            if standalone_preview is None or not standalone_preview.accepted or standalone_preview.record is None:
                continue

            override_preview = self._validation_preview(
                candidate.agent_name,
                candidate.proposal.body_tokens,
                stripped_validation_pool,
                validation_data,
                validation_target,
            )
            if not override_preview.accepted or override_preview.record is None:
                continue

            validation_gain = float(override_preview.new_score - baseline_score)
            validation_trade_proxy_gain = float(override_preview.new_trade_proxy - baseline_trade_proxy)
            if validation_trade_proxy_gain <= 5e-4:
                continue

            training_pool = stripped_pool.copy()
            committed = self.reward_shaper.shape(
                candidate.proposal.body_tokens,
                data,
                target,
                training_pool,
                commit=True,
                role=candidate.agent_name,
            )
            if not committed.decision.accepted:
                continue
            training_gain = float(training_pool.pool_score() - baseline_train_score)
            training_trade_proxy_gain = float(training_pool.trade_proxy_score() - baseline_train_trade_proxy)
            if training_trade_proxy_gain <= -5e-4:
                continue

            metrics = override_preview.record.metrics
            turnover = float(metrics.get("turnover", 0.0))
            drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
            override_bonus = (
                0.16
                + 5.0 * validation_gain
                + 8.0 * validation_trade_proxy_gain
                + 2.5 * float(standalone_preview.trade_proxy_gain)
                + 1.0 * float(standalone_preview.marginal_gain)
                + 2.0 * training_trade_proxy_gain
                - 0.02 * turnover
                - 0.03 * drawdown
            )
            override_candidate = _CandidateResult(
                agent_name=candidate.agent_name,
                proposal=candidate.proposal,
                provisional_reward=committed.clipped_reward + training_gain,
                effective_reward=committed.clipped_reward + training_gain + training_trade_proxy_gain,
                selection_score=candidate.selection_score + override_bonus,
                coordination_bonus=candidate.coordination_bonus,
                would_accept=True,
                decision_reason="flow_override",
                review_reason="flow_override",
                review_approved=True,
                decision_accepted=True,
                override_drop_roles=("target_price",),
            )
            if best_candidate is None or self._candidate_key(override_candidate) > self._candidate_key(best_candidate):
                best_candidate = override_candidate
        return best_candidate

    def _pool_without_roles(self, pool: FactorPool, normalized_roles: set[str]) -> FactorPool:
        trimmed = FactorPool(max_size=pool.max_size)
        trimmed.records = [
            record
            for record in pool.records
            if normalize_role(record.role) not in normalized_roles
        ]
        return trimmed

    def _best_flow_commit_preview(
        self,
        agent_name: str,
        tokens: tuple[str, ...],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
    ):
        previews = []

        direct_pool = pool.copy()
        direct_outcome = self.reward_shaper.shape(
            tokens,
            data,
            target,
            direct_pool,
            commit=True,
            role=agent_name,
        )
        if direct_outcome.decision.accepted:
            previews.append((direct_outcome, tuple(), pool))

        stripped_pool = self._pool_without_roles(pool, {"target_price"})
        if len(stripped_pool.records) != len(pool.records):
            override_pool = stripped_pool.copy()
            override_outcome = self.reward_shaper.shape(
                tokens,
                data,
                target,
                override_pool,
                commit=True,
                role=agent_name,
            )
            if override_outcome.decision.accepted:
                previews.append((override_outcome, ("target_price",), stripped_pool))

        if not previews:
            return None
        previews.sort(
            key=lambda item: (
                float(item[0].decision.trade_proxy_gain),
                float(item[0].decision.marginal_gain),
                float(item[0].clipped_reward),
                -len(item[1]),
            ),
            reverse=True,
        )
        return previews[0]

    def _prefer_flow_commit_preview(
        self,
        current_outcome,
        preview_outcome,
        current_drop_roles: tuple[str, ...],
        preview_drop_roles: tuple[str, ...],
    ) -> bool:
        if preview_outcome.decision.candidate is None:
            return False
        if current_outcome.decision.candidate is None:
            return preview_outcome.decision.accepted
        current_trade = float(current_outcome.decision.trade_proxy_gain)
        preview_trade = float(preview_outcome.decision.trade_proxy_gain)
        if preview_outcome.decision.accepted and not current_outcome.decision.accepted:
            return True
        if preview_outcome.decision.accepted and preview_trade > current_trade + 5e-4:
            return True
        if preview_drop_roles and not current_drop_roles and preview_trade > current_trade:
            return True
        return False

    def _drop_roles_in_place(self, pool: FactorPool, normalized_roles: set[str]) -> None:
        kept_records: list = []
        removed_canonicals: set[str] = set()
        for record in pool.records:
            if normalize_role(record.role) in normalized_roles:
                removed_canonicals.add(record.canonical)
            else:
                kept_records.append(record)
        pool.records = kept_records
        if removed_canonicals:
            self.frozen_baseline_canonicals.difference_update(removed_canonicals)

    def _build_pair_candidate(
        self,
        candidates: list[_CandidateResult],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> _PairCandidateResult | None:
        if self.fixed_agent_name is not None:
            return None
        candidate_map = {candidate.agent_name: candidate for candidate in candidates}
        if "target_price" not in candidate_map or not candidate_map["target_price"].would_accept:
            return None
        flow_agent_names = [
            name
            for name, candidate in candidate_map.items()
            if normalize_role(name) == "target_flow" and candidate.would_accept
        ]
        if not flow_agent_names:
            return None

        baseline = pool.pool_score()
        best_pair: _PairCandidateResult | None = None
        for flow_name in flow_agent_names:
            pair_names = ("target_price", flow_name)
            flow_residual_preview = None
            flow_standalone_preview = None
            best_single_gain = max(
                self._single_commit_gain(candidate_map[name], data, target, pool, baseline)
                for name in pair_names
            )
            best_single_validation_gain = None
            best_single_validation_trade_proxy_gain = None
            if validation_data is not None and validation_target is not None:
                flow_residual_preview = self._validation_preview(
                    flow_name,
                    candidate_map[flow_name].proposal.body_tokens,
                    pool,
                    validation_data,
                    validation_target,
                )
                flow_standalone_preview = self._standalone_validation_preview(
                    flow_name,
                    candidate_map[flow_name].proposal.body_tokens,
                    validation_data,
                    validation_target,
                    pool.max_size,
                )
                if not self._flow_residual_gate(flow_residual_preview, flow_standalone_preview):
                    continue
                best_single_validation_gain = self._best_single_validation_gain(
                    candidate_map,
                    pair_names,
                    pool,
                    validation_data,
                    validation_target,
                )
                best_single_validation_trade_proxy_gain = self._best_single_validation_trade_proxy_gain(
                    candidate_map,
                    pair_names,
                    pool,
                    validation_data,
                    validation_target,
                )
            for commit_order in (pair_names, pair_names[::-1]):
                temp_pool = pool.copy()
                total_reward = 0.0
                selection_score = 0.0
                component_rewards: dict[str, float] = {}
                component_acceptance: dict[str, bool] = {}
                component_reasons: dict[str, str] = {}
                all_accepted = True
                for name in commit_order:
                    outcome = self.reward_shaper.shape(
                        candidate_map[name].proposal.body_tokens,
                        data,
                        target,
                        temp_pool,
                        commit=True,
                        role=name,
                    )
                    component_rewards[name] = outcome.clipped_reward
                    component_acceptance[name] = outcome.decision.accepted
                    component_reasons[name] = outcome.decision.reason
                    if not outcome.decision.accepted:
                        all_accepted = False
                        break
                    total_reward += outcome.clipped_reward
                if not all_accepted:
                    continue
                combined_gain = temp_pool.pool_score() - baseline
                if combined_gain <= best_single_gain + 5e-4:
                    continue
                if validation_data is not None and validation_target is not None:
                    validation_effects = self._pair_validation_effects(
                        candidate_map,
                        commit_order,
                        pool,
                        validation_data,
                        validation_target,
                    )
                    if validation_effects is None:
                        continue
                    validation_gain, validation_trade_proxy_gain = validation_effects
                    if best_single_validation_gain is not None and validation_gain <= best_single_validation_gain + 5e-4:
                        continue
                    if (
                        best_single_validation_trade_proxy_gain is not None
                        and validation_trade_proxy_gain <= best_single_validation_trade_proxy_gain + 5e-4
                    ):
                        continue
                    selection_score = self._pair_validation_bonus(
                        validation_gain,
                        validation_trade_proxy_gain,
                        candidate_map,
                        flow_name,
                    )
                synergy_bonus = self._pair_synergy_bonus(candidate_map, temp_pool, combined_gain, best_single_gain, flow_name)
                split_bonus = synergy_bonus / float(len(pair_names))
                pair_rewards = {
                    name: component_rewards[name] + split_bonus
                    for name in pair_names
                }
                candidate = _PairCandidateResult(
                    agent_names=pair_names,
                    commit_order=commit_order,
                    effective_reward=total_reward + synergy_bonus,
                    provisional_reward=sum(candidate_map[name].provisional_reward for name in pair_names),
                    selection_score=total_reward + synergy_bonus + selection_score,
                    would_accept=True,
                    decision_reason="pair_shortlist",
                    review_reason="pair_shortlist",
                    component_tokens={name: candidate_map[name].proposal.body_tokens for name in pair_names},
                    component_rewards=pair_rewards,
                    component_acceptance=component_acceptance,
                    component_reasons=component_reasons,
                    proposal_valid=all(candidate_map[name].proposal.valid for name in pair_names),
                )
                if best_pair is None or self._candidate_key(candidate) > self._candidate_key(best_pair):
                    best_pair = candidate
        return best_pair

    def _single_commit_gain(
        self,
        candidate: _CandidateResult,
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        baseline: float,
    ) -> float:
        temp_pool = pool.copy()
        outcome = self.reward_shaper.shape(
            candidate.proposal.body_tokens,
            data,
            target,
            temp_pool,
            commit=True,
            role=candidate.agent_name,
        )
        if not outcome.decision.accepted:
            return float("-inf")
        return temp_pool.pool_score() - baseline

    def _best_single_validation_gain(
        self,
        candidate_map: dict[str, _CandidateResult],
        candidate_names: tuple[str, ...],
        pool: FactorPool,
        validation_data: pd.DataFrame,
        validation_target: pd.Series,
    ) -> float:
        scored_pool = rescore_pool_on_dataset(pool, validation_data, validation_target)
        baseline_score = scored_pool.pool_score()
        best_gain = float("-inf")
        for name in candidate_names:
            preview = self._validation_preview(
                name,
                candidate_map[name].proposal.body_tokens,
                scored_pool,
                validation_data,
                validation_target,
            )
            if preview.accepted:
                best_gain = max(best_gain, preview.new_score - baseline_score)
        return best_gain

    def _best_single_validation_trade_proxy_gain(
        self,
        candidate_map: dict[str, _CandidateResult],
        candidate_names: tuple[str, ...],
        pool: FactorPool,
        validation_data: pd.DataFrame,
        validation_target: pd.Series,
    ) -> float:
        scored_pool = rescore_pool_on_dataset(pool, validation_data, validation_target)
        best_gain = float("-inf")
        for name in candidate_names:
            preview = self._validation_preview(
                name,
                candidate_map[name].proposal.body_tokens,
                scored_pool,
                validation_data,
                validation_target,
            )
            if preview.accepted:
                best_gain = max(best_gain, float(preview.trade_proxy_gain))
        return best_gain

    def _pair_validation_effects(
        self,
        candidate_map: dict[str, _CandidateResult],
        commit_order: tuple[str, ...],
        pool: FactorPool,
        validation_data: pd.DataFrame,
        validation_target: pd.Series,
    ) -> tuple[float, float] | None:
        validation_pool = rescore_pool_on_dataset(pool, validation_data, validation_target)
        baseline_score = validation_pool.pool_score()
        baseline_trade_proxy = validation_pool.trade_proxy_score()
        for name in commit_order:
            preview = self._validation_preview(
                name,
                candidate_map[name].proposal.body_tokens,
                validation_pool,
                validation_data,
                validation_target,
            )
            if not preview.accepted or preview.record is None:
                return None
            self._apply_preview_record(validation_pool, preview)
        return (
            validation_pool.pool_score() - baseline_score,
            validation_pool.trade_proxy_score() - baseline_trade_proxy,
        )

    def _validation_selection_bonus(
        self,
        agent_name: str,
        tokens: tuple[str, ...],
        pool: FactorPool,
        validation_data: pd.DataFrame,
        validation_target: pd.Series,
    ) -> float:
        normalized_role = normalize_role(agent_name)
        preview = self._validation_preview(
            agent_name,
            tokens,
            pool,
            validation_data,
            validation_target,
        )
        if not preview.accepted or preview.record is None:
            if normalized_role == "target_flow":
                existing_roles = {
                    normalized
                    for record in pool.records
                    if (normalized := normalize_role(record.role)) is not None
                }
                if "target_flow" in existing_roles and "target_price" not in existing_roles:
                    return -0.02 - 0.005 * max(0, len(tokens) - 2)
            return -0.20
        metrics = preview.record.metrics
        valid_rank_ic = abs(float(metrics.get("rank_ic", 0.0)))
        valid_turnover = float(metrics.get("turnover", 0.0))
        valid_drawdown = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        bonus = (
            0.10
            + 4.0 * float(preview.marginal_gain)
            + 3.0 * float(preview.trade_proxy_gain)
            + 1.5 * valid_rank_ic
            - 0.02 * valid_turnover
            - 0.03 * valid_drawdown
        )
        if normalized_role == "target_flow":
            standalone_preview = self._standalone_validation_preview(
                agent_name,
                tokens,
                validation_data,
                validation_target,
                pool.max_size,
            )
            if standalone_preview is None or not standalone_preview.accepted:
                return -0.25
            bonus += (
                2.0 * float(standalone_preview.trade_proxy_gain)
                + 1.0 * float(standalone_preview.marginal_gain)
                - 0.01 * max(0, len(tokens) - 4)
            )
            if valid_turnover < 0.80:
                bonus += 0.03
            if float(preview.trade_proxy_gain) > 0.0:
                bonus += 0.05
        return bonus

    def _pair_validation_bonus(
        self,
        validation_gain: float,
        validation_trade_proxy_gain: float,
        candidate_map: dict[str, _CandidateResult],
        flow_agent_name: str,
    ) -> float:
        flow_reward = candidate_map[flow_agent_name].selection_score - candidate_map[flow_agent_name].effective_reward
        price_reward = candidate_map["target_price"].selection_score - candidate_map["target_price"].effective_reward
        flow_bonus = max(0.0, flow_reward)
        return (
            0.08
            + 6.0 * validation_gain
            + 4.0 * validation_trade_proxy_gain
            + 0.5 * max(flow_reward, price_reward)
            + 0.15 * flow_bonus
        )

    def _apply_preview_record(self, pool: FactorPool, preview) -> None:
        if preview.record is None:
            return
        if len(pool) < pool.max_size:
            pool.add(preview.record)
            return
        if preview.replaced_canonical is not None:
            for index, record in enumerate(pool.records):
                if record.canonical == preview.replaced_canonical:
                    pool.replace(index, preview.record)
                    return
        weakest_index = pool.weakest_index()
        pool.replace(weakest_index, preview.record)

    def _pair_synergy_bonus(
        self,
        candidate_map: dict[str, _CandidateResult],
        temp_pool: FactorPool,
        combined_gain: float,
        best_single_gain: float,
        flow_agent_name: str,
    ) -> float:
        candidate_roles = {normalize_role(record.role) for record in temp_pool.records}
        bonus = 0.02 + 0.5 * max(0.0, combined_gain - best_single_gain)
        if {"target_price", "target_flow"}.issubset(candidate_roles):
            bonus += 0.02
        flow_metrics = candidate_map[flow_agent_name].proposal.body_tokens
        if "CORR_5" in flow_metrics:
            bonus += 0.01
        return bonus

    def _maybe_commit_secondary(
        self,
        candidate_map: dict[str, _CandidateResult],
        primary_agent_name: str,
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        validation_data: pd.DataFrame | None,
        validation_target: pd.Series | None,
    ) -> _SecondaryCommitResult | None:
        complementary_roles = {
            "target_flow": "target_price",
        }
        primary_role = normalize_role(primary_agent_name)
        if primary_role == "target_price":
            secondary_candidates = [
                name
                for name, candidate in candidate_map.items()
                if normalize_role(name) == "target_flow" and candidate.would_accept
            ]
            secondary_name = self._best_secondary_name(
                secondary_candidates,
                candidate_map,
                pool,
                validation_data,
                validation_target,
            )
        else:
            secondary_name = complementary_roles.get(primary_role)
        if secondary_name is None or secondary_name not in candidate_map:
            return None
        candidate = candidate_map[secondary_name]
        if not candidate.would_accept:
            return None
        if validation_data is not None and validation_target is not None:
            preview = self._validation_preview(
                secondary_name,
                candidate.proposal.body_tokens,
                pool,
                validation_data,
                validation_target,
            )
            if not preview.accepted or preview.record is None or preview.marginal_gain <= 1e-3:
                return None
            preview_metrics = preview.record.metrics
            if normalize_role(secondary_name) == "target_flow":
                standalone_preview = self._standalone_validation_preview(
                    secondary_name,
                    candidate.proposal.body_tokens,
                    validation_data,
                    validation_target,
                    pool.max_size,
                )
                if not self._flow_residual_gate(preview, standalone_preview):
                    return None
                if float(preview_metrics.get("turnover", 0.0)) > 0.85:
                    return None
                if float(preview_metrics.get("max_corr", 0.0)) > 0.60:
                    return None
        committed = self.reward_shaper.shape(
            candidate.proposal.body_tokens,
            data,
            target,
            pool,
            commit=True,
            role=secondary_name,
        )
        if not committed.decision.accepted:
            return None
        return _SecondaryCommitResult(
            agent_name=secondary_name,
            reward=committed.clipped_reward,
            accepted=committed.decision.accepted,
            reason=committed.decision.reason,
        )

    def _commit_pair(
        self,
        pair_candidate: _PairCandidateResult,
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
    ) -> tuple[dict[str, float], dict[str, bool], dict[str, str]]:
        component_rewards: dict[str, float] = {}
        component_acceptance: dict[str, bool] = {}
        component_reasons: dict[str, str] = {}
        for name in pair_candidate.commit_order:
            outcome = self.reward_shaper.shape(
                pair_candidate.component_tokens[name],
                data,
                target,
                pool,
                commit=True,
                role=name,
            )
            component_rewards[name] = pair_candidate.component_rewards.get(name, outcome.clipped_reward)
            component_acceptance[name] = outcome.decision.accepted
            component_reasons[name] = outcome.decision.reason
            if not outcome.decision.accepted:
                break
        return component_rewards, component_acceptance, component_reasons

    def _coordination_bonus(self, agent_name: str, decision, pool: FactorPool) -> float:
        if not decision.accepted or decision.candidate is None:
            return 0.0
        normalized_role = normalize_role(agent_name)
        existing_roles = {
            normalized
            for record in pool.records
            if (normalized := normalize_role(record.role)) is not None
        }
        metrics = decision.candidate.metrics
        max_corr = float(metrics.get("max_corr", 0.0))
        turnover = float(metrics.get("turnover", 0.0))
        bonus = 0.0

        if normalized_role == "target_flow":
            if "target_price" in existing_roles and "target_flow" not in existing_roles:
                bonus += 0.08
                if max_corr < 0.55:
                    bonus += 0.03
                if turnover < 0.80:
                    bonus += 0.02
        elif normalized_role == "target_price":
            if "target_flow" in existing_roles and "target_price" not in existing_roles:
                bonus += 0.05
            if "target_price" in existing_roles and "target_flow" not in existing_roles:
                bonus -= 0.02
        elif normalized_role == "context":
            if existing_roles & {"target_price", "target_flow"}:
                bonus -= 0.04
            if max_corr > 0.50:
                bonus -= 0.02
        return bonus

    def _would_replace_frozen_baseline(self, agent_name: str, decision) -> bool:
        if normalize_role(agent_name) == "target_price":
            return False
        replaced_canonical = getattr(decision, "replaced_canonical", None)
        return replaced_canonical is not None and replaced_canonical in self.frozen_baseline_canonicals

    def _candidate_reason(self, decision, replaces_frozen: bool) -> str:
        if replaces_frozen:
            return "frozen_baseline"
        return decision.reason

    def _review_reason(self, review_reason: str, decision, replaces_frozen: bool) -> str:
        if replaces_frozen:
            return "review_frozen_baseline"
        if review_reason != "review_accept":
            return review_reason
        return decision.reason

    def _warm_start_baseline_tokens(
        self,
        agent_name: str,
        tokens: tuple[str, ...],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> tuple[bool, str | None]:
        preview = self.reward_shaper.shape(
            tokens,
            data,
            target,
            pool,
            commit=False,
            role=agent_name,
        )
        review = self.reviewer_agent.review(preview.decision, pool, role=agent_name)
        reward = preview.clipped_reward
        accepted = False
        reason = preview.decision.reason
        canonical: str | None = None
        validation_ok = True
        if validation_data is not None and validation_target is not None and review.approved and preview.decision.accepted:
            preview_validation = preview_candidate_on_dataset(
                tokens,
                pool,
                validation_data,
                validation_target,
                role=agent_name,
            )
            validation_ok = preview_validation.accepted
            if not validation_ok:
                reason = preview_validation.reason
        if review.approved and preview.decision.accepted and validation_ok:
            committed = self.reward_shaper.shape(
                tokens,
                data,
                target,
                pool,
                commit=True,
                role=agent_name,
            )
            reward = committed.clipped_reward
            accepted = committed.decision.accepted
            reason = committed.decision.reason
            canonical = committed.decision.candidate.canonical if committed.decision.candidate is not None else None
        else:
            reward = min(-0.1, -abs(preview.clipped_reward))
            if not review.approved:
                reason = review.reason
        agent = self.agents[agent_name]
        agent.observe(tokens, reward, accepted)
        self.experience_memory.record(
            self.regime_controller.infer(data).regime,
            agent_name,
            tokens,
            reward,
            accepted,
            reason,
        )
        return accepted, canonical

    def _warm_start_tokens(
        self,
        agent_name: str,
        tokens: tuple[str, ...],
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> None:
        preview = self.reward_shaper.shape(
            tokens,
            data,
            target,
            pool,
            commit=False,
            role=agent_name,
        )
        review = self.reviewer_agent.review(preview.decision, pool, role=agent_name)
        reward = preview.clipped_reward
        accepted = False
        reason = preview.decision.reason
        validation_ok = True
        if validation_data is not None and validation_target is not None and review.approved and preview.decision.accepted:
            preview_validation = preview_candidate_on_dataset(
                tokens,
                pool,
                validation_data,
                validation_target,
                role=agent_name,
            )
            validation_ok = preview_validation.accepted
            if not validation_ok:
                reason = preview_validation.reason
        if review.approved and preview.decision.accepted:
            if validation_ok:
                committed = self.reward_shaper.shape(
                    tokens,
                    data,
                    target,
                    pool,
                    commit=True,
                    role=agent_name,
                )
                reward = committed.clipped_reward
                accepted = committed.decision.accepted
                reason = committed.decision.reason
            else:
                reward = min(reward, -0.05)
        else:
            reward = min(-0.1, -abs(preview.clipped_reward))
            if not review.approved:
                reason = review.reason
        agent = self.agents[agent_name]
        agent.observe(tokens, reward, accepted)
        self.experience_memory.record(
            self.regime_controller.infer(data).regime,
            agent_name,
            tokens,
            reward,
            accepted,
            reason,
        )

    def _best_secondary_name(
        self,
        candidate_names: list[str],
        candidate_map: dict[str, _CandidateResult],
        pool: FactorPool,
        validation_data: pd.DataFrame | None,
        validation_target: pd.Series | None,
    ) -> str | None:
        if not candidate_names:
            return None
        if validation_data is None or validation_target is None:
            return max(candidate_names, key=lambda name: candidate_map[name].selection_score)
        previews: list[tuple[float, str]] = []
        for name in candidate_names:
            preview = self._validation_preview(
                name,
                candidate_map[name].proposal.body_tokens,
                pool,
                validation_data,
                validation_target,
            )
            if preview.accepted and preview.record is not None:
                standalone_preview = self._standalone_validation_preview(
                    name,
                    candidate_map[name].proposal.body_tokens,
                    validation_data,
                    validation_target,
                    pool.max_size,
                )
                if not self._flow_residual_gate(preview, standalone_preview):
                    continue
                previews.append(
                    (
                        preview.marginal_gain
                        + 1.5 * preview.trade_proxy_gain
                        + 1.0 * float(standalone_preview.trade_proxy_gain),
                        name,
                    )
                )
        if previews:
            previews.sort()
            return previews[-1][1]
        return None

    def _validation_preview(
        self,
        agent_name: str,
        tokens: tuple[str, ...],
        pool: FactorPool,
        validation_data: pd.DataFrame,
        validation_target: pd.Series,
    ):
        return preview_candidate_on_dataset(
            tokens,
            pool,
            validation_data,
            validation_target,
            role=agent_name,
        )

    def _standalone_validation_preview(
        self,
        agent_name: str,
        tokens: tuple[str, ...],
        validation_data: pd.DataFrame,
        validation_target: pd.Series,
        pool_max_size: int,
    ):
        return preview_candidate_on_dataset(
            tokens,
            FactorPool(max_size=pool_max_size),
            validation_data,
            validation_target,
            role=agent_name,
        )

    def _flow_residual_gate(self, residual_preview, standalone_preview) -> bool:
        if residual_preview is None or standalone_preview is None:
            return False
        if not residual_preview.accepted or residual_preview.record is None:
            return False
        if not standalone_preview.accepted or standalone_preview.record is None:
            return False
        if float(residual_preview.trade_proxy_gain) <= 5e-4:
            return False
        if float(standalone_preview.trade_proxy_gain) <= 5e-4:
            return False
        if float(standalone_preview.record.metrics.get("turnover", 0.0)) > 1.10:
            return False
        return True

    def _family_selection_count(self, normalized_role: str) -> int:
        return sum(
            count
            for name, count in self.selection_counts.items()
            if normalize_role(name) == normalized_role
        )

    def _family_accepted_count(self, normalized_role: str) -> int:
        return sum(
            count
            for name, count in self.accepted_counts.items()
            if normalize_role(name) == normalized_role
        )
