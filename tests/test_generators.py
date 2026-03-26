import importlib.util
import os
import sys

import pytest

from knowledge_guided_symbolic_alpha.language import RPNGrammar
from knowledge_guided_symbolic_alpha.runtime import TORCH_IMPORT_ENV, probe_torch_health


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _reload_generator_module():
    for name in [
        "knowledge_guided_symbolic_alpha.models.generator",
        "knowledge_guided_symbolic_alpha.models.generator.neural_base",
        "knowledge_guided_symbolic_alpha.models.generator.rnn_generator",
        "knowledge_guided_symbolic_alpha.models.generator.transformer_generator",
        "knowledge_guided_symbolic_alpha.models.generator.tree_policy",
        "knowledge_guided_symbolic_alpha.models.generator.tree_value",
    ]:
        sys.modules.pop(name, None)


@pytest.mark.train
def test_transformer_generator_uses_torch_and_learns_from_reward() -> None:
    report = probe_torch_health()
    if not TORCH_AVAILABLE or not report.get("ok", False):
        pytest.skip(report.get("message", "torch is unavailable"))
    os.environ[TORCH_IMPORT_ENV] = "1"
    _reload_generator_module()

    from knowledge_guided_symbolic_alpha.models.generator import TransformerGenerator

    grammar = RPNGrammar()
    generator = TransformerGenerator(seed=11)
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)

    before = generator.score_tokens(state, valid_tokens)
    generator.observe(("DXY", "DELAY_1"), reward=1.0, accepted=True)
    after = generator.score_tokens(state, valid_tokens)

    assert generator.uses_torch
    assert after["DXY"] > before["DXY"]


@pytest.mark.train
def test_rnn_generator_uses_torch_and_learns_from_reward() -> None:
    report = probe_torch_health()
    if not TORCH_AVAILABLE or not report.get("ok", False):
        pytest.skip(report.get("message", "torch is unavailable"))
    os.environ[TORCH_IMPORT_ENV] = "1"
    _reload_generator_module()

    from knowledge_guided_symbolic_alpha.models.generator import RNNGenerator

    grammar = RPNGrammar()
    generator = RNNGenerator(seed=17)
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)

    before = generator.score_tokens(state, valid_tokens)
    generator.observe(("CPI", "DELAY_1"), reward=1.0, accepted=True)
    after = generator.score_tokens(state, valid_tokens)

    assert generator.uses_torch
    assert after["CPI"] > before["CPI"]
