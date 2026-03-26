from __future__ import annotations

from ...runtime import load_torch_symbols


torch, nn, F = load_torch_symbols()


_ModuleBase = nn.Module if nn is not None else object


class _ValueNet(_ModuleBase):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class TreeValue:
    def __init__(
        self,
        learning_rate: float = 0.02,
        hidden_dim: int = 32,
        max_length: int = 15,
        device: str | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.uses_torch = torch is not None
        self.device = None
        self.model = None
        self.optimizer = None
        if self.uses_torch:
            self.device = device or "cpu"
            self.model = _ValueNet(hidden_dim).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_max_length(self, max_length: int) -> None:
        self.max_length = max_length

    def evaluate_state(self, state, valid_action_count: int) -> float:
        heuristic = self._heuristic_value(state, valid_action_count)
        if not self.uses_torch or self.model is None:
            return heuristic
        with torch.inference_mode():
            features = self._feature_tensor(state, valid_action_count)
            model_value = float(torch.tanh(self.model(features)).item())
        return 0.5 * heuristic + 0.5 * model_value

    def observe(self, state, target_value: float, valid_action_count: int) -> None:
        if not self.uses_torch or self.model is None or self.optimizer is None:
            return
        features = self._feature_tensor(state, valid_action_count)
        target = torch.tensor([target_value], dtype=torch.float32, device=self.device)
        prediction = torch.tanh(self.model(features))
        loss = F.mse_loss(prediction, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _heuristic_value(self, state, valid_action_count: int) -> float:
        stack_depth = len(state.stack)
        body_length = len(state.body_tokens)
        needs_delay = any(expr.needs_macro_delay for expr in state.stack)
        if state.finished:
            return 1.0 if state.terminal_error is None else -1.0
        score = 0.2
        score -= 0.05 * max(stack_depth - 1, 0)
        score -= 0.03 * body_length / max(self.max_length, 1)
        if needs_delay:
            score -= 0.1
        if valid_action_count <= 1:
            score -= 0.15
        return float(score)

    def _feature_tensor(self, state, valid_action_count: int):
        body_length = len(state.body_tokens)
        stack_depth = len(state.stack)
        needs_delay = 1.0 if any(expr.needs_macro_delay for expr in state.stack) else 0.0
        finished = 1.0 if state.finished else 0.0
        features = torch.tensor(
            [[
                body_length / max(self.max_length, 1),
                stack_depth / max(self.max_length, 1),
                needs_delay,
                finished,
                valid_action_count / max(self.max_length, 1),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        return features
