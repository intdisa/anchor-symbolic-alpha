from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ...language.tokens import BODY_TOKENS, EOS_TOKEN, SOS_TOKEN
from ...runtime import load_torch_symbols
from .base import GeneratorConditioningContext

torch, nn, F = load_torch_symbols()


class NeuralSequenceGenerator:
    def __init__(
        self,
        learning_rate: float = 0.05,
        grad_clip_norm: float = 1.0,
        neural_score_scale: float = 0.35,
        conditioning_scale: float = 0.40,
        conditioning_dim: int = 16,
        seed: int = 7,
        device: str | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        self.neural_score_scale = neural_score_scale
        self.conditioning_scale = conditioning_scale
        self.conditioning_dim = conditioning_dim
        self.seed = seed
        self.vocab = BODY_TOKENS + (EOS_TOKEN,)
        self.output_token_to_id = {token: index for index, token in enumerate(self.vocab)}
        self.input_vocab = (SOS_TOKEN,) + self.vocab
        self.input_token_to_id = {token: index for index, token in enumerate(self.input_vocab)}
        self.conditioning_context: GeneratorConditioningContext | None = None
        self.context_token_bias = defaultdict(lambda: defaultdict(float))
        self.uses_torch = torch is not None
        self.device = None
        self.model = None
        self.optimizer = None
        if self.uses_torch:
            torch.manual_seed(seed)
            self.device = self._resolve_device(device)
            self.model = self._build_torch_model().to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def score_tokens(self, state, valid_tokens: tuple[str, ...]) -> dict[str, float]:
        scores = self._structural_scores(state, valid_tokens)
        if not self.uses_torch or self.model is None:
            return self._apply_conditioning(scores, valid_tokens)
        with torch.inference_mode():
            logits = self._forward_logits(state.body_tokens)
        for token in valid_tokens:
            scores[token] += self.neural_score_scale * float(
                torch.tanh(logits[self.output_token_to_id[token]]).item()
            )
        return self._apply_conditioning(scores, valid_tokens)

    def observe(self, tokens: tuple[str, ...], reward: float, accepted: bool) -> None:
        self._observe_structural(tokens, reward, accepted)
        self._observe_conditioning(tokens, reward, accepted)
        if not self.uses_torch or self.model is None or self.optimizer is None or not tokens:
            return
        advantage = float(np.clip(reward + (0.05 if accepted else 0.0), -5.0, 5.0))
        if not np.isfinite(advantage) or advantage == 0.0:
            return
        losses = []
        prefix: tuple[str, ...] = ()
        for token in tokens:
            logits = self._forward_logits(prefix)
            target = torch.tensor([self.output_token_to_id[token]], dtype=torch.long, device=self.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target, reduction="mean"))
            prefix = prefix + (token,)
        loss = torch.stack(losses).mean() * advantage
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

    def supervised_update(
        self,
        tokens: tuple[str, ...],
        *,
        include_eos: bool = True,
        weight: float = 1.0,
    ) -> float:
        self._observe_structural(tokens, reward=0.25 * weight, accepted=True)
        self._observe_conditioning(tokens, reward=0.25 * weight, accepted=True)
        if not self.uses_torch or self.model is None or self.optimizer is None:
            self.observe(tokens, reward=weight, accepted=True)
            return 0.0
        sequence = list(tokens)
        if include_eos:
            sequence.append(EOS_TOKEN)
        if not sequence:
            return 0.0
        losses = []
        prefix: tuple[str, ...] = ()
        for token in sequence:
            logits = self._forward_logits(prefix)
            target = torch.tensor([self.output_token_to_id[token]], dtype=torch.long, device=self.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target, reduction="mean"))
            if token != EOS_TOKEN:
                prefix = prefix + (token,)
        loss = torch.stack(losses).mean() * float(max(weight, 1e-6))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return float(loss.detach().item())

    def _resolve_device(self, explicit_device: str | None) -> str:
        if explicit_device is not None:
            return explicit_device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _input_ids(self, prefix_tokens: Iterable[str]) -> list[int]:
        ids = [self.input_token_to_id[SOS_TOKEN]]
        ids.extend(self.input_token_to_id[token] for token in prefix_tokens)
        return ids

    def _forward_logits(self, prefix_tokens: tuple[str, ...]):
        input_ids = self._input_ids(prefix_tokens)
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        logits = self.model(tensor, self._conditioning_tensor())
        return logits[0, -1]

    def set_conditioning_context(self, context: GeneratorConditioningContext | None) -> None:
        self.conditioning_context = context

    def _conditioning_tensor(self):
        vector = np.zeros(self.conditioning_dim, dtype=np.float32)
        if self.conditioning_context is not None:
            summary = np.asarray(self.conditioning_context.summary_vector, dtype=np.float32)
            limit = min(summary.shape[0], self.conditioning_dim)
            vector[:limit] = summary[:limit]
        return torch.from_numpy(vector).unsqueeze(0).to(self.device)

    def _apply_conditioning(self, scores: dict[str, float], valid_tokens: tuple[str, ...]) -> dict[str, float]:
        if self.conditioning_context is None:
            return scores
        signature_biases = self.context_token_bias.get(self.conditioning_context.signature, {})
        for token in valid_tokens:
            scores[token] += self.conditioning_scale * float(self.conditioning_context.token_biases.get(token, 0.0))
            scores[token] += 0.5 * float(signature_biases.get(token, 0.0))
        return scores

    def _observe_conditioning(self, tokens: tuple[str, ...], reward: float, accepted: bool) -> None:
        if not tokens or self.conditioning_context is None or not self.conditioning_context.signature:
            return
        scaled = self.learning_rate * np.clip(reward + (0.05 if accepted else 0.0), -2.0, 2.0) / max(len(tokens), 1)
        learned_biases = self.context_token_bias[self.conditioning_context.signature]
        for token in tokens:
            learned_biases[token] += float(scaled)

    def save_checkpoint(self, path: str | Path) -> None:
        if not self.uses_torch or self.model is None:
            raise RuntimeError("Checkpoint saving requires a torch-backed generator.")
        payload = {
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "grad_clip_norm": self.grad_clip_norm,
            "neural_score_scale": self.neural_score_scale,
            "conditioning_scale": self.conditioning_scale,
            "conditioning_dim": self.conditioning_dim,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "structural_state": self._serialize_structural_state(),
            "context_token_bias": {
                "|".join(signature): dict(token_biases)
                for signature, token_biases in self.context_token_bias.items()
            },
        }
        torch.save(payload, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        if not self.uses_torch or self.model is None:
            raise RuntimeError("Checkpoint loading requires a torch-backed generator.")
        payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        if self.optimizer is not None and payload.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self._load_structural_state(payload.get("structural_state", {}))
        self.context_token_bias = defaultdict(lambda: defaultdict(float))
        for signature, token_biases in payload.get("context_token_bias", {}).items():
            self.context_token_bias[tuple(signature.split("|"))].update(
                {token: float(value) for token, value in token_biases.items()}
            )

    def _build_torch_model(self):
        raise NotImplementedError

    def _structural_scores(self, state, valid_tokens: tuple[str, ...]) -> dict[str, float]:
        raise NotImplementedError

    def _observe_structural(self, tokens: tuple[str, ...], reward: float, accepted: bool) -> None:
        raise NotImplementedError

    def _serialize_structural_state(self) -> dict:
        raise NotImplementedError

    def _load_structural_state(self, payload: dict) -> None:
        raise NotImplementedError
