from __future__ import annotations

from ..domain.feature_registry import FEATURE_REGISTRY
from ..domain.operator_registry import OPERATOR_REGISTRY

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)
FEATURE_TOKENS = tuple(FEATURE_REGISTRY.keys())
OPERATOR_TOKENS = tuple(OPERATOR_REGISTRY.keys())
BODY_TOKENS = FEATURE_TOKENS + OPERATOR_TOKENS


def is_special_token(token: str) -> bool:
    return token in SPECIAL_TOKENS
