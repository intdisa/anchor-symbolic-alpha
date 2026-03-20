from pathlib import Path

import numpy as np

from experiments.common import build_sequence_generator, load_yaml
from knowledge_guided_symbolic_alpha.language import RPNGrammar
from knowledge_guided_symbolic_alpha.models.generator import TransformerGenerator
from knowledge_guided_symbolic_alpha.training import GeneratorPretrainer, SyntheticRecoveryDatasetBuilder


def test_synthetic_recovery_builder_generates_valid_examples() -> None:
    builder = SyntheticRecoveryDatasetBuilder(seed=11, length=80)
    examples = builder.build(6)

    assert len(examples) == 6
    assert all(example.formula_tokens for example in examples)
    assert all(example.allowed_features for example in examples)
    assert all(example.frame.shape[0] > 0 for example in examples)
    assert all(example.validation_frame.shape[0] > 0 for example in examples)


def test_pretrainer_increases_conditioned_first_token_score() -> None:
    builder = SyntheticRecoveryDatasetBuilder(seed=13, length=80)
    example = builder.build(1)[0]
    generator = TransformerGenerator(seed=13)
    pretrainer = GeneratorPretrainer()
    grammar = RPNGrammar()
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)

    generator.set_conditioning_context(pretrainer._context(example))
    before = generator.score_tokens(state, valid_tokens)[example.formula_tokens[0]]
    summary = pretrainer.fit(generator, [example], epochs=2)
    generator.set_conditioning_context(pretrainer._context(example))
    after = generator.score_tokens(state, valid_tokens)[example.formula_tokens[0]]

    assert summary.first_token_accuracy >= 0.0
    assert after > before


def test_generator_checkpoint_round_trip_preserves_scores(tmp_path: Path) -> None:
    builder = SyntheticRecoveryDatasetBuilder(seed=17, length=80)
    example = builder.build(1)[0]
    generator = TransformerGenerator(seed=17)
    pretrainer = GeneratorPretrainer()
    pretrainer.fit(generator, [example], epochs=1)

    checkpoint_path = tmp_path / "prior.pt"
    generator.save_checkpoint(checkpoint_path)

    loaded = TransformerGenerator(seed=99)
    loaded.load_checkpoint(checkpoint_path)

    grammar = RPNGrammar()
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)
    context = pretrainer._context(example)
    generator.set_conditioning_context(context)
    loaded.set_conditioning_context(context)

    original_scores = generator.score_tokens(state, valid_tokens)
    loaded_scores = loaded.score_tokens(state, valid_tokens)

    for token in (example.formula_tokens[0], next(token for token in valid_tokens if token != example.formula_tokens[0])):
        assert np.isclose(original_scores[token], loaded_scores[token])


def test_build_sequence_generator_loads_pretrained_checkpoint(tmp_path: Path) -> None:
    generator = TransformerGenerator(seed=23)
    generator.observe(("GOLD_GAP_RET", "NEG"), reward=1.0, accepted=True)
    checkpoint_path = tmp_path / "synthetic_prior.pt"
    generator.save_checkpoint(checkpoint_path)

    training_config = load_yaml("configs/training.yaml")
    training_config["training"]["pretraining"]["load_checkpoint"] = True
    training_config["training"]["pretraining"]["checkpoint"] = str(checkpoint_path)
    loaded = build_sequence_generator(training_config, seed=1)

    grammar = RPNGrammar()
    state = grammar.initial_state()
    valid_tokens = grammar.valid_next_tokens(state)
    original_scores = generator.score_tokens(state, valid_tokens)
    loaded_scores = loaded.score_tokens(state, valid_tokens)

    assert np.isclose(original_scores["GOLD_GAP_RET"], loaded_scores["GOLD_GAP_RET"])
