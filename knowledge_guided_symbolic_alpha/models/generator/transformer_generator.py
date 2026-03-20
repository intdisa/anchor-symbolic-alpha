from __future__ import annotations

from collections import defaultdict

from ...domain.operator_registry import OPERATOR_REGISTRY
from ...language.tokens import FEATURE_TOKENS
from .neural_base import NeuralSequenceGenerator, torch, nn


_ModuleBase = nn.Module if nn is not None else object


class _TransformerLanguageModel(_ModuleBase):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        conditioning_dim: int,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.context_projection = nn.Linear(conditioning_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(model_dim, output_vocab_size)

    def forward(self, token_ids, context_vector=None):
        sequence_length = token_ids.size(1)
        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0)
        hidden = self.embedding(token_ids) + self.position_embedding(positions)
        if context_vector is not None:
            hidden = hidden + self.context_projection(context_vector).unsqueeze(1)
        causal_mask = torch.triu(
            torch.full((sequence_length, sequence_length), float("-inf"), device=token_ids.device),
            diagonal=1,
        )
        encoded = self.encoder(hidden, mask=causal_mask)
        return self.head(encoded)


class TransformerGenerator(NeuralSequenceGenerator):
    """
    Transformer-backed generator with structural priors.

    The neural model learns sequence preferences, while the retained structural
    priors keep early exploration numerically stable.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 32,
        grad_clip_norm: float = 1.0,
        neural_score_scale: float = 0.35,
        seed: int = 7,
        device: str | None = None,
    ) -> None:
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.token_bias = defaultdict(float)
        self.position_bias = defaultdict(lambda: defaultdict(float))
        super().__init__(
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            neural_score_scale=neural_score_scale,
            seed=seed,
            device=device,
        )

    def _build_torch_model(self):
        return _TransformerLanguageModel(
            input_vocab_size=len(self.input_vocab),
            output_vocab_size=len(self.vocab),
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            conditioning_dim=self.conditioning_dim,
        )

    def _structural_scores(self, state, valid_tokens: tuple[str, ...]) -> dict[str, float]:
        stack_depth = len(state.stack)
        position = len(state.body_tokens)
        used_features = {token for token in state.body_tokens if token in FEATURE_TOKENS}
        needs_delay = any(expr.needs_macro_delay for expr in state.stack)

        scores: dict[str, float] = {}
        for token in valid_tokens:
            score = self.token_bias[token] + self.position_bias[position][token]
            if needs_delay and token == "DELAY_1":
                score += 2.0
            if token in FEATURE_TOKENS:
                score += 0.9 if stack_depth <= 1 else -0.8
                if token not in used_features:
                    score += 0.25
            elif token in OPERATOR_REGISTRY:
                arity = OPERATOR_REGISTRY[token].arity
                if stack_depth < arity:
                    score -= 2.0
                elif arity == 2:
                    score += 1.4
                else:
                    score += 0.2 if stack_depth <= 1 else -0.4
                if OPERATOR_REGISTRY[token].commutative:
                    score += 0.1
            elif token == "<EOS>":
                score += 1.0 if stack_depth == 1 else -1.5
            scores[token] = score
        return scores

    def _observe_structural(self, tokens: tuple[str, ...], reward: float, accepted: bool) -> None:
        if not tokens:
            return
        scaled = self.learning_rate * reward / max(len(tokens), 1)
        for position, token in enumerate(tokens):
            self.token_bias[token] += scaled
            self.position_bias[position][token] += scaled
        if accepted:
            for position, token in enumerate(tokens):
                self.position_bias[position][token] += self.learning_rate * 0.02

    def _serialize_structural_state(self) -> dict:
        return {
            "token_bias": dict(self.token_bias),
            "position_bias": {
                int(position): dict(token_biases)
                for position, token_biases in self.position_bias.items()
            },
        }

    def _load_structural_state(self, payload: dict) -> None:
        self.token_bias.clear()
        self.token_bias.update({token: float(value) for token, value in payload.get("token_bias", {}).items()})
        self.position_bias.clear()
        for position, token_biases in payload.get("position_bias", {}).items():
            self.position_bias[int(position)].update({token: float(value) for token, value in token_biases.items()})
