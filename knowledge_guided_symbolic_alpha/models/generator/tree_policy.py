from __future__ import annotations

from collections import defaultdict

from ...domain.operator_registry import OPERATOR_REGISTRY
from ...language.tokens import FEATURE_TOKENS
from ...runtime import load_torch_symbols

torch, nn, F = load_torch_symbols()


_ModuleBase = nn.Module if nn is not None else object


class _PolicyNet(_ModuleBase):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class TreePolicy:
    def __init__(
        self,
        learning_rate: float = 0.02,
        hidden_dim: int = 32,
        max_length: int = 15,
        device: str | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.token_bias = defaultdict(float)
        self.uses_torch = torch is not None
        self.device = None
        self.model = None
        self.optimizer = None
        if self.uses_torch:
            self.device = device or "cpu"
            self.model = _PolicyNet(hidden_dim).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_max_length(self, max_length: int) -> None:
        self.max_length = max_length

    def score_actions(self, state, valid_tokens: tuple[str, ...]) -> dict[str, float]:
        scores = {token: self.token_bias[token] for token in valid_tokens}
        if not self.uses_torch or self.model is None:
            return scores
        with torch.inference_mode():
            features = self._feature_tensor(state, valid_tokens)
            values = torch.tanh(self.model(features))
        for token, value in zip(valid_tokens, values.tolist()):
            scores[token] += 0.25 * float(value)
        return scores

    def observe(self, state, action: str, reward: float) -> None:
        self.token_bias[action] += 0.05 * reward
        if not self.uses_torch or self.model is None or self.optimizer is None:
            return
        features = self._feature_tensor(state, (action,))
        target = torch.tensor([reward], dtype=torch.float32, device=self.device)
        prediction = torch.tanh(self.model(features))
        loss = F.mse_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _feature_tensor(self, state, tokens: tuple[str, ...]):
        data = [self._features_for_token(state, token) for token in tokens]
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _features_for_token(self, state, token: str) -> list[float]:
        stack_depth = len(state.stack)
        body_length = len(state.body_tokens)
        needs_delay = 1.0 if any(expr.needs_macro_delay for expr in state.stack) else 0.0
        is_feature = 1.0 if token in FEATURE_TOKENS else 0.0
        arity = float(OPERATOR_REGISTRY[token].arity) if token in OPERATOR_REGISTRY else 0.0
        is_binary = 1.0 if arity == 2.0 else 0.0
        is_unary = 1.0 if arity == 1.0 else 0.0
        is_eos = 1.0 if token == "<EOS>" else 0.0
        seen_before = float(state.body_tokens.count(token))
        return [
            body_length / max(self.max_length, 1),
            stack_depth / max(self.max_length, 1),
            needs_delay,
            is_feature,
            is_binary,
            is_unary,
            is_eos,
            seen_before / max(self.max_length, 1),
        ]
