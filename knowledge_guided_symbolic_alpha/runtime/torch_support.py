from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from typing import Any


TORCH_IMPORT_ENV = "KGSA_ENABLE_TORCH"


def torch_import_enabled() -> bool:
    raw = os.environ.get(TORCH_IMPORT_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def enable_torch_import() -> None:
    os.environ[TORCH_IMPORT_ENV] = "1"


def load_torch_symbols() -> tuple[Any, Any, Any]:
    if not torch_import_enabled():
        return None, None, None
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        functional = importlib.import_module("torch.nn.functional")
        return torch, nn, functional
    except Exception:
        return None, None, None


@lru_cache(maxsize=1)
def probe_torch_health(timeout_seconds: int = 20) -> dict[str, Any]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return {
            "ok": False,
            "status": "missing",
            "message": "torch is not installed in the active environment.",
        }

    code = """
import json
result = {"ok": True, "status": "ok", "device": "cpu", "version": None}
try:
    import torch
    from torch import nn
    result["version"] = getattr(torch, "__version__", None)
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        result["device"] = "mps"
    model = nn.Linear(2, 2).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    del optimizer
except Exception as exc:
    result = {"ok": False, "status": "error", "message": str(exc)}
print(json.dumps(result, ensure_ascii=True))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "status": "timeout",
            "message": f"torch health probe exceeded {timeout_seconds} seconds in a child process.",
        }

    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return {
            "ok": False,
            "status": "error",
            "message": completed.stderr.strip() or stdout or "torch probe failed.",
        }
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return {
            "ok": False,
            "status": "error",
            "message": stdout or "torch probe returned invalid JSON.",
        }
    return payload
