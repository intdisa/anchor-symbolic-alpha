from __future__ import annotations

from collections import defaultdict

from ...domain.operator_registry import OPERATOR_REGISTRY
from ...language.tokens import FEATURE_TOKENS
from .neural_base import NeuralSequenceGenerator, nn


_ModuleBase = nn.Module if nn is not None else object


class _RNNLanguageModel(_ModuleBase):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        conditioning_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.context_projection = nn.Linear(conditioning_dim, embed_dim)
        self.head = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, token_ids, context_vector=None):
        embedded = self.embedding(token_ids)
        if context_vector is not None:
            embedded = embedded + self.context_projection(context_vector).unsqueeze(1)
        hidden, _ = self.gru(embedded)
        return self.head(hidden)


class RNNGenerator(NeuralSequenceGenerator):
    """
    GRU-backed generator with structural priors.

    If `torch` is unavailable, the class still works as a heuristic scorer using
    the same interface.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        grad_clip_norm: float = 1.0,
        neural_score_scale: float = 0.35,
        seed: int = 7,
        device: str | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.token_bias = defaultdict(float)
        self.transition_bias = defaultdict(lambda: defaultdict(float))
        super().__init__(
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            neural_score_scale=neural_score_scale,
            seed=seed,
            device=device,
        )

    def _build_torch_model(self):
        return _RNNLanguageModel(
            input_vocab_size=len(self.input_vocab),
            output_vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            conditioning_dim=self.conditioning_dim,
        )

    def _structural_scores(self, state, valid_tokens: tuple[str, ...]) -> dict[str, float]:
        last_token = state.body_tokens[-1] if state.body_tokens else None
        stack_depth = len(state.stack)
        needs_delay = any(expr.needs_macro_delay for expr in state.stack)
        scores: dict[str, float] = {}
        for token in valid_tokens:
            score = self.token_bias[token]
            if last_token is not None:
                score += self.transition_bias[last_token][token]
            if needs_delay and token == "DELAY_1":
                score += 2.0
            if token in FEATURE_TOKENS:
                score += 0.8 if stack_depth <= 1 else -0.8
                if token in state.body_tokens:
                    score -= 0.15
            elif token in OPERATOR_REGISTRY:
                arity = OPERATOR_REGISTRY[token].arity
                if stack_depth < arity:
                    score -= 2.0
                elif arity == 2:
                    score += 1.2
                else:
                    score += 0.2 if stack_depth <= 1 else -0.4
                if last_token == token:
                    score -= 0.5
            elif token == "<EOS>":
                score += 0.8 if stack_depth == 1 else -1.5
            scores[token] = score
        return scores

    def _observe_structural(self, tokens: tuple[str, ...], reward: float, accepted: bool) -> None:
        if not tokens:
            return
        scaled = self.learning_rate * reward / max(len(tokens), 1)
        for token in tokens:
            self.token_bias[token] += scaled
        for left, right in zip(tokens, tokens[1:]):
            self.transition_bias[left][right] += scaled
        if accepted:
            for token in tokens:
                self.token_bias[token] += self.learning_rate * 0.02

    def _serialize_structural_state(self) -> dict:
        return {
            "token_bias": dict(self.token_bias),
            "transition_bias": {
                left: dict(right_biases)
                for left, right_biases in self.transition_bias.items()
            },
        }

    def _load_structural_state(self, payload: dict) -> None:
        self.token_bias.clear()
        self.token_bias.update({token: float(value) for token, value in payload.get("token_bias", {}).items()})
        self.transition_bias.clear()
        for left, right_biases in payload.get("transition_bias", {}).items():
            self.transition_bias[left].update({right: float(value) for right, value in right_biases.items()})
