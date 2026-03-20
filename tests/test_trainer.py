import numpy as np
import pandas as pd

from experiments.common import build_manager, load_yaml
from knowledge_guided_symbolic_alpha.agents import ManagerAgent
from knowledge_guided_symbolic_alpha.language import RPNGrammar
from knowledge_guided_symbolic_alpha.models.generator import TransformerGenerator
from knowledge_guided_symbolic_alpha.search import BeamSearch, FormulaSampler, SamplingConfig
from knowledge_guided_symbolic_alpha.training import (
    MultiAgentTrainer,
    PoolRewardShaper,
    SingleAgentTrainer,
    SyntheticRecoveryDatasetBuilder,
)


def make_frame() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    base = np.linspace(0.0, 1.0, len(index))
    gold_close = pd.Series(100 + np.cumsum(0.1 + base), index=index)
    gold_volume = pd.Series(1000 + 10 * np.sin(np.arange(len(index))), index=index)
    cpi = pd.Series(100 + np.floor(np.arange(len(index)) / 10), index=index)
    tnx = pd.Series(1.5 + 0.01 * np.arange(len(index)), index=index)
    vix = pd.Series(20 - 0.1 * np.arange(len(index)), index=index)
    dxy = pd.Series(100 + 0.05 * np.arange(len(index)), index=index)
    frame = pd.DataFrame(
        {
            "GOLD_CLOSE": gold_close,
            "GOLD_VOLUME": gold_volume,
            "CPI": cpi,
            "TNX": tnx,
            "VIX": vix,
            "DXY": dxy,
        }
    )
    target = frame["GOLD_CLOSE"].pct_change().shift(-1).fillna(0.0)
    return frame.iloc[:-1], target.iloc[:-1]


def test_beam_search_returns_candidates() -> None:
    grammar = RPNGrammar()
    generator = TransformerGenerator()
    search = BeamSearch(grammar, generator)
    candidates = search.search()
    assert candidates
    assert any(candidate.valid for candidate in candidates)


def test_single_agent_trainer_runs_without_nan_rewards() -> None:
    frame, target = make_frame()
    grammar = RPNGrammar()
    generator = TransformerGenerator()
    sampler = FormulaSampler(grammar, generator, SamplingConfig(seed=3))
    trainer = SingleAgentTrainer(sampler, PoolRewardShaper(), pool_max_size=4)

    summary = trainer.train(frame, target, episodes=12, validation_data=frame, validation_target=target)
    assert len(summary.history) == 12
    assert np.isfinite(summary.best_validation_pool_score)
    assert all(np.isfinite(episode.reward) for episode in summary.history)


def test_multi_agent_trainer_runs_without_nan_rewards() -> None:
    frame, target = make_frame()
    trainer = MultiAgentTrainer(ManagerAgent(selection_mode="greedy", seed=9), pool_max_size=4)
    summary = trainer.train(frame, target, episodes=8, validation_data=frame, validation_target=target)
    assert len(summary.history) == 8
    assert np.isfinite(summary.best_validation_pool_score)
    assert all(np.isfinite(episode.reward) for episode in summary.history)


def test_skill_hierarchy_manager_trainer_runs_without_nan_rewards() -> None:
    example = SyntheticRecoveryDatasetBuilder(seed=21, length=72).build(1)[0]
    manager = build_manager(
        load_yaml("configs/training.yaml"),
        dataset_name=example.dataset_name,
        partition_mode="skill_hierarchy",
    )
    trainer = MultiAgentTrainer(manager, pool_max_size=4)
    summary = trainer.train(
        example.frame,
        example.target,
        episodes=3,
        validation_data=example.validation_frame,
        validation_target=example.validation_target,
    )

    assert len(summary.history) == 3
    assert np.isfinite(summary.best_validation_pool_score)
    assert all(np.isfinite(episode.reward) for episode in summary.history)
