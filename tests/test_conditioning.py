import numpy as np
import pandas as pd

from knowledge_guided_symbolic_alpha.agents import FeatureGroupAgent
from knowledge_guided_symbolic_alpha.evaluation import FactorPool
from knowledge_guided_symbolic_alpha.evaluation.factor_pool import FactorRecord
from knowledge_guided_symbolic_alpha.language import RPNGrammar
from knowledge_guided_symbolic_alpha.models.embeddings import DatasetEmbedder
from knowledge_guided_symbolic_alpha.models.generator import GeneratorConditioningContext, TransformerGenerator


def make_frame() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2021-01-01", periods=40, freq="D")
    gap = pd.Series(np.sin(np.arange(len(index)) / 3.0), index=index)
    oc = pd.Series(np.cos(np.arange(len(index)) / 5.0), index=index)
    frame = pd.DataFrame(
        {
            "GOLD_GAP_RET": gap,
            "GOLD_OC_RET": oc,
            "GOLD_HL_SPREAD": gap.abs() * 0.5 + 0.1,
            "GOLD_REALIZED_VOL_5": gap.abs() * 0.4 + 0.05,
        }
    )
    target = gap.shift(-1).fillna(0.0)
    return frame, target


def test_dataset_embedder_biases_correlated_features() -> None:
    frame, target = make_frame()
    embedder = DatasetEmbedder()
    embedding = embedder.embed(
        frame,
        frozenset(frame.columns),
        target=target,
        regime="HIGH_VOLATILITY",
    )

    assert len(embedding.vector) == embedder.VECTOR_DIM
    assert embedding.token_biases["GOLD_GAP_RET"] > 0.0
    assert embedding.token_biases["TS_STD_5"] > 0.0


def test_generator_scores_change_with_conditioning_context() -> None:
    grammar = RPNGrammar()
    generator = TransformerGenerator(seed=13)
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)

    base_scores = generator.score_tokens(state, valid_tokens)
    generator.set_conditioning_context(
        GeneratorConditioningContext(
            summary_vector=(0.2,) * 16,
            token_biases={"GOLD_GAP_RET": 0.5},
            signature=("target_flow_gap", "HIGH_VOLATILITY"),
        )
    )
    conditioned_scores = generator.score_tokens(state, valid_tokens)

    assert conditioned_scores["GOLD_GAP_RET"] > base_scores["GOLD_GAP_RET"]


def test_feature_group_agent_sets_dataset_pool_conditioning() -> None:
    frame, target = make_frame()
    pool = FactorPool(max_size=4)
    pool.add(
        FactorRecord(
            tokens=("GOLD_CLOSE", "DELTA_1", "NEG"),
            canonical="price_baseline",
            signal=pd.Series(np.linspace(0.0, 1.0, len(frame)), index=frame.index),
            metrics={"rank_ic": 0.05, "turnover": 0.8, "max_corr": 0.1},
            role="target_price",
        )
    )
    agent = FeatureGroupAgent(
        role="target_flow_gap",
        allowed_features=frozenset({"GOLD_GAP_RET", "GOLD_OC_RET", "GOLD_HL_SPREAD"}),
        generator=TransformerGenerator(seed=19),
        max_length_cap=6,
        seed_formulas=(("GOLD_GAP_RET", "NEG"),),
    )
    agent.set_context("HIGH_VOLATILITY")
    agent.propose(data=frame, target=target, pool=pool)

    context = agent.generator.conditioning_context
    assert context is not None
    assert len(context.summary_vector) == 16
    assert context.token_biases["GOLD_GAP_RET"] > 0.0
    assert context.token_biases["NEG"] > 0.0
