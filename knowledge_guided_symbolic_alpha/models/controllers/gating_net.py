from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from ...envs.state_encoder import EncodedState


@dataclass(frozen=True)
class AgentWeights:
    macro: float
    micro: float


class GatingNet:
    def __init__(
        self,
        reward_history_limit: int = 32,
        base_weights: dict[str, AgentWeights] | None = None,
        agent_names: tuple[str, str] = ("macro", "micro"),
    ) -> None:
        self.reward_history_limit = reward_history_limit
        self.agent_names = agent_names
        self.base_weights = base_weights or {
            "BALANCED": AgentWeights(macro=0.40, micro=0.60),
            "HIGH_VOLATILITY": AgentWeights(macro=0.70, micro=0.30),
            "RATE_HIKING": AgentWeights(macro=0.75, micro=0.25),
            "INFLATION_SHOCK": AgentWeights(macro=0.70, micro=0.30),
            "USD_STRENGTH": AgentWeights(macro=0.65, micro=0.35),
        }
        first_name, second_name = self.agent_names
        self.reward_memory = defaultdict(
            lambda: {
                first_name: deque(maxlen=self.reward_history_limit),
                second_name: deque(maxlen=self.reward_history_limit),
            }
        )

    def weights(self, regime: str, state: EncodedState | None = None) -> AgentWeights:
        base = self.base_weights.get(regime, AgentWeights(macro=0.50, micro=0.50))
        macro = base.macro
        micro = base.micro
        if state is not None:
            imbalance = state.macro_pool_fraction - state.micro_pool_fraction
            macro -= 0.15 * imbalance
            micro += 0.15 * imbalance
        macro_bias, micro_bias = self._reward_bias(regime)
        macro += macro_bias
        micro += micro_bias
        macro = max(macro, 0.05)
        micro = max(micro, 0.05)
        total = macro + micro
        return AgentWeights(macro=macro / total, micro=micro / total)

    def update(self, regime: str, agent_name: str, reward: float) -> None:
        if agent_name not in self.reward_memory[regime]:
            return
        self.reward_memory[regime][agent_name].append(float(reward))

    def _reward_bias(self, regime: str) -> tuple[float, float]:
        memory = self.reward_memory[regime]
        first_name, second_name = self.agent_names
        first_mean = sum(memory[first_name]) / len(memory[first_name]) if memory[first_name] else 0.0
        second_mean = sum(memory[second_name]) / len(memory[second_name]) if memory[second_name] else 0.0
        relative_edge = first_mean - second_mean
        scale = 0.20
        return scale * relative_edge, -scale * relative_edge
