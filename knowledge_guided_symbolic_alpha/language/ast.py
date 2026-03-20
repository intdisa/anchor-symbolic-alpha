from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExpressionNode:
    token: str
    children: tuple["ExpressionNode", ...] = field(default_factory=tuple)

    @property
    def is_leaf(self) -> bool:
        return not self.children
