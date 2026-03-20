from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..envs import StateEncoder
from ..evaluation import FactorPool
from ..memory import ExperienceMemory
from ..models.controllers import GatingNet, RegimeController
from ..training.curriculum import CurriculumStage
from ..training.reward_shaping import PoolRewardShaper
from .base import AgentProposal
from .macro_agent import MacroAgent
from .micro_agent import MicroAgent
from .reviewer_agent import ReviewOutcome, ReviewerAgent


@dataclass(frozen=True)
class ManagerStep:
    selected_agent: str
    regime: str
    macro_weight: float
    micro_weight: float
    proposal: AgentProposal
    reward: float
    accepted: bool
    decision_reason: str
    review_reason: str
    pool_size: int


class ManagerAgent:
    def __init__(
        self,
        macro_agent: MacroAgent | None = None,
        micro_agent: MicroAgent | None = None,
        reviewer_agent: ReviewerAgent | None = None,
        gating_net: GatingNet | None = None,
        regime_controller: RegimeController | None = None,
        state_encoder: StateEncoder | None = None,
        reward_shaper: PoolRewardShaper | None = None,
        experience_memory: ExperienceMemory | None = None,
        selection_mode: str = "greedy",
        seed: int = 7,
        agent_names: tuple[str, str] = ("macro", "micro"),
    ) -> None:
        self.experience_memory = experience_memory or ExperienceMemory()
        self.agent_names = agent_names
        first_agent_name, second_agent_name = self.agent_names
        self.macro_agent = macro_agent or MacroAgent(experience_memory=self.experience_memory)
        self.micro_agent = micro_agent or MicroAgent(experience_memory=self.experience_memory)
        self.macro_agent.experience_memory = self.experience_memory
        self.micro_agent.experience_memory = self.experience_memory
        self.reviewer_agent = reviewer_agent or ReviewerAgent()
        self.gating_net = gating_net or GatingNet(agent_names=self.agent_names)
        self.regime_controller = regime_controller or RegimeController()
        self.state_encoder = state_encoder or StateEncoder(
            first_group_features=self.macro_agent.allowed_features,
            second_group_features=self.micro_agent.allowed_features,
        )
        self.reward_shaper = reward_shaper or PoolRewardShaper()
        self.selection_mode = selection_mode
        self.rng = np.random.default_rng(seed)
        self.failure_streaks = {first_agent_name: 0, second_agent_name: 0}
        self.agent_map = {
            first_agent_name: self.macro_agent,
            second_agent_name: self.micro_agent,
        }

    def apply_curriculum(self, stage: CurriculumStage) -> None:
        self.macro_agent.apply_curriculum(stage)
        self.micro_agent.apply_curriculum(stage)

    def warm_start(
        self,
        pool: FactorPool,
        data: pd.DataFrame,
        target: pd.Series,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> None:
        del pool, data, target, validation_data, validation_target

    def select_agent(self, frame: pd.DataFrame, pool: FactorPool) -> tuple[str, str, float, float]:
        regime_context = self.regime_controller.infer(frame)
        encoded = self.state_encoder.encode(frame, pool, regime_context.regime)
        weights = self.gating_net.weights(regime_context.regime, encoded)
        first_agent_name, second_agent_name = self.agent_names
        if self.selection_mode == "sample":
            agent_name = str(self.rng.choice([first_agent_name, second_agent_name], p=[weights.macro, weights.micro]))
        else:
            agent_name = first_agent_name if weights.macro >= weights.micro else second_agent_name
        other_name = second_agent_name if agent_name == first_agent_name else first_agent_name
        if self.failure_streaks[agent_name] >= 3 and self.failure_streaks[other_name] < self.failure_streaks[agent_name]:
            agent_name = other_name
        return agent_name, regime_context.regime, weights.macro, weights.micro

    def run_step(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        pool: FactorPool,
        commit: bool = True,
        validation_data: pd.DataFrame | None = None,
        validation_target: pd.Series | None = None,
    ) -> ManagerStep:
        agent_name, regime, macro_weight, micro_weight = self.select_agent(data, pool)
        agent = self.agent_map[agent_name]
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
        review = self.reviewer_agent.review(provisional.decision, pool, role=agent_name)

        final_reward = provisional.clipped_reward
        accepted = False
        decision_reason = provisional.decision.reason
        if review.approved and provisional.decision.accepted and commit:
            committed = self.reward_shaper.shape(
                proposal.body_tokens,
                data,
                target,
                pool,
                commit=True,
                role=agent_name,
            )
            final_reward = committed.clipped_reward
            accepted = committed.decision.accepted
            decision_reason = committed.decision.reason
        elif not review.approved:
            final_reward = min(-0.1, -abs(provisional.clipped_reward))

        agent.observe(proposal.body_tokens, final_reward, accepted)
        self.gating_net.update(regime, agent_name, final_reward)
        memory_reason = review.reason if not review.approved else decision_reason
        self.experience_memory.record(
            regime,
            agent_name,
            proposal.body_tokens,
            final_reward,
            accepted,
            memory_reason,
        )
        if accepted:
            self.failure_streaks[agent_name] = 0
        else:
            self.failure_streaks[agent_name] += 1
        return ManagerStep(
            selected_agent=agent_name,
            regime=regime,
            macro_weight=macro_weight,
            micro_weight=micro_weight,
            proposal=proposal,
            reward=float(final_reward),
            accepted=accepted,
            decision_reason=decision_reason,
            review_reason=review.reason,
            pool_size=len(pool),
        )
