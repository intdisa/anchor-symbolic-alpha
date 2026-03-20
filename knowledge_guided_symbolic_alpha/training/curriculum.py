from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_length: int
    min_length: int
    temperature: float
    top_k: int
    beam_width: int
    beam_top_k: int
    mcts_simulations: int
    mcts_top_k: int
    rollout_depth: int
    exploration_constant: float
    use_mcts: bool


class FormulaCurriculum:
    def __init__(self, stages: tuple[CurriculumStage, ...] | None = None) -> None:
        self.stages = stages or (
            CurriculumStage(
                name="bootstrap",
                max_length=5,
                min_length=3,
                temperature=0.6,
                top_k=3,
                beam_width=3,
                beam_top_k=2,
                mcts_simulations=10,
                mcts_top_k=3,
                rollout_depth=4,
                exploration_constant=1.1,
                use_mcts=False,
            ),
            CurriculumStage(
                name="stabilize",
                max_length=8,
                min_length=3,
                temperature=0.7,
                top_k=4,
                beam_width=4,
                beam_top_k=3,
                mcts_simulations=16,
                mcts_top_k=4,
                rollout_depth=5,
                exploration_constant=1.2,
                use_mcts=True,
            ),
            CurriculumStage(
                name="expand",
                max_length=12,
                min_length=3,
                temperature=0.85,
                top_k=5,
                beam_width=5,
                beam_top_k=4,
                mcts_simulations=24,
                mcts_top_k=5,
                rollout_depth=6,
                exploration_constant=1.25,
                use_mcts=True,
            ),
            CurriculumStage(
                name="full",
                max_length=15,
                min_length=3,
                temperature=1.0,
                top_k=6,
                beam_width=6,
                beam_top_k=5,
                mcts_simulations=32,
                mcts_top_k=6,
                rollout_depth=8,
                exploration_constant=1.35,
                use_mcts=True,
            ),
        )

    def stage_for_episode(self, episode_index: int, total_episodes: int) -> CurriculumStage:
        if total_episodes <= 1:
            return self.stages[-1]
        progress = episode_index / max(total_episodes - 1, 1)
        boundaries = self._boundaries()
        for boundary, stage in zip(boundaries, self.stages):
            if progress <= boundary:
                return stage
        return self.stages[-1]

    def _boundaries(self) -> tuple[float, ...]:
        count = len(self.stages)
        if count == 1:
            return (1.0,)
        return tuple((index + 1) / count for index in range(count))
