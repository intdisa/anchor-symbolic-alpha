from __future__ import annotations

from dataclasses import dataclass
import random

from ..envs import RoleActionMask
from ..language import RPNGrammar
from ..models.embeddings import DatasetEmbedder, PoolEmbedder
from ..models.generator import GeneratorConditioningContext
from ..search import BeamSearch, BeamSearchConfig
from .synthetic_recovery import SyntheticRecoveryExample


@dataclass(frozen=True)
class PretrainingSummary:
    examples: int
    epochs: int
    exact_recovery_rate: float
    first_token_accuracy: float
    mean_supervised_loss: float


class GeneratorPretrainer:
    def __init__(
        self,
        dataset_embedder: DatasetEmbedder | None = None,
        pool_embedder: PoolEmbedder | None = None,
        seed: int = 7,
    ) -> None:
        self.dataset_embedder = dataset_embedder or DatasetEmbedder()
        self.pool_embedder = pool_embedder or PoolEmbedder()
        self.seed = seed

    def fit(
        self,
        generator,
        examples: list[SyntheticRecoveryExample],
        epochs: int = 1,
    ) -> PretrainingSummary:
        rng = random.Random(self.seed)
        supervised_losses: list[float] = []
        for _ in range(epochs):
            ordered_examples = list(examples)
            rng.shuffle(ordered_examples)
            for example in ordered_examples:
                generator.set_conditioning_context(self._context(example))
                if hasattr(generator, "supervised_update"):
                    supervised_losses.append(
                        float(generator.supervised_update(example.formula_tokens, include_eos=True, weight=1.0))
                    )
                    first_token = (example.formula_tokens[0],) if example.formula_tokens else tuple()
                    if first_token:
                        supervised_losses.append(
                            float(generator.supervised_update(first_token, include_eos=False, weight=0.35))
                        )
                generator.observe(example.formula_tokens, reward=1.0, accepted=True)

        exact_matches = 0
        first_token_matches = 0
        for example in examples:
            recovered = self.recover_formula(generator, example)
            if recovered == example.formula_tokens:
                exact_matches += 1
            if recovered and example.formula_tokens and recovered[0] == example.formula_tokens[0]:
                first_token_matches += 1
        total = max(len(examples), 1)
        return PretrainingSummary(
            examples=len(examples),
            epochs=epochs,
            exact_recovery_rate=exact_matches / total,
            first_token_accuracy=first_token_matches / total,
            mean_supervised_loss=float(sum(supervised_losses) / len(supervised_losses)) if supervised_losses else 0.0,
        )

    def recover_formula(self, generator, example: SyntheticRecoveryExample) -> tuple[str, ...]:
        generator.set_conditioning_context(self._context(example))
        grammar = RPNGrammar()
        mask = RoleActionMask(role=example.role, allowed_features=example.allowed_features)
        beam = BeamSearch(
            grammar,
            generator,
            config=BeamSearchConfig(beam_width=4, per_node_top_k=3, max_steps=14, length_penalty=0.02),
            token_filter=mask.filter_tokens,
        )
        for candidate in beam.search():
            if candidate.valid:
                return candidate.body_tokens
        return tuple()

    def _context(self, example: SyntheticRecoveryExample) -> GeneratorConditioningContext:
        dataset_embedding = self.dataset_embedder.embed(
            example.frame,
            example.allowed_features,
            target=example.target,
            regime=example.regime,
            validation_data=example.validation_frame,
            validation_target=example.validation_target,
        )
        pool_embedding = self.pool_embedder.embed(example.pool, example.allowed_features, example.role)
        token_biases = dict(dataset_embedding.token_biases)
        for token, bias in pool_embedding.token_biases.items():
            token_biases[token] = token_biases.get(token, 0.0) + bias
        signature = (
            "synthetic",
            example.dataset_name,
            example.role,
            example.regime,
            f"pool:{len(example.pool.records)}",
        )
        return GeneratorConditioningContext(
            summary_vector=dataset_embedding.vector + pool_embedding.vector,
            token_biases=token_biases,
            signature=signature,
        )
