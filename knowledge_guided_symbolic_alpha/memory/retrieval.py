from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedMemory:
    regime: str
    role: str
    token_biases: dict[str, float]
