import importlib.util

import pytest

from knowledge_guided_symbolic_alpha.language import RPNGrammar
from knowledge_guided_symbolic_alpha.models.generator import RNNGenerator, TransformerGenerator


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_transformer_generator_uses_torch_and_learns_from_reward() -> None:
    grammar = RPNGrammar()
    generator = TransformerGenerator(seed=11)
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)

    before = generator.score_tokens(state, valid_tokens)
    generator.observe(("DXY", "DELAY_1"), reward=1.0, accepted=True)
    after = generator.score_tokens(state, valid_tokens)

    assert generator.uses_torch
    assert after["DXY"] > before["DXY"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_rnn_generator_uses_torch_and_learns_from_reward() -> None:
    grammar = RPNGrammar()
    generator = RNNGenerator(seed=17)
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)

    before = generator.score_tokens(state, valid_tokens)
    generator.observe(("CPI", "DELAY_1"), reward=1.0, accepted=True)
    after = generator.score_tokens(state, valid_tokens)

    assert generator.uses_torch
    assert after["CPI"] > before["CPI"]
