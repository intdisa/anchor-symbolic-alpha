from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..generation import FormulaCandidate
from .task_protocol import SelectorBenchmarkTask


@dataclass(frozen=True)
class PublicSymbolicTaskSpec:
    task_id: str
    true_formula: str
    candidate_formulas: tuple[str, ...]
    target_fn: Callable[[dict[str, np.ndarray]], np.ndarray]


PUBLIC_SYMBOLIC_TASKS: tuple[PublicSymbolicTaskSpec, ...] = (
    PublicSymbolicTaskSpec(
        task_id="feynman_linear_add",
        true_formula="X0 X1 ADD",
        candidate_formulas=(
            "X0 X1 ADD",
            "X0 X2 ADD",
            "X4 NEG",
            "X0 X1 SUB",
            "X0 X1 ADD X2 ADD",
        ),
        target_fn=lambda feat: feat["X0"] + feat["X1"],
    ),
    PublicSymbolicTaskSpec(
        task_id="feynman_difference",
        true_formula="X0 X1 SUB",
        candidate_formulas=(
            "X0 X1 SUB",
            "X0 X2 SUB",
            "X4 NEG",
            "X0 X1 ADD",
            "X0 X1 SUB X3 ADD",
        ),
        target_fn=lambda feat: feat["X0"] - feat["X1"],
    ),
    PublicSymbolicTaskSpec(
        task_id="feynman_product",
        true_formula="X0 X1 MUL",
        candidate_formulas=(
            "X0 X1 MUL",
            "X0 X2 MUL",
            "X4 NEG",
            "X0 X1 ADD",
            "X0 X1 MUL X3 ADD",
        ),
        target_fn=lambda feat: feat["X0"] * feat["X1"],
    ),
    PublicSymbolicTaskSpec(
        task_id="feynman_product_seed_shift",
        true_formula="X0 X1 MUL",
        candidate_formulas=(
            "X0 X1 MUL",
            "X0 X2 MUL",
            "X4 NEG",
            "X0 X1 MUL X3 ADD",
            "X0 X2 MUL X3 ADD",
        ),
        target_fn=lambda feat: feat["X0"] * feat["X1"],
    ),
    PublicSymbolicTaskSpec(
        task_id="feynman_ratio",
        true_formula="X0 X1 DIV",
        candidate_formulas=(
            "X0 X1 DIV",
            "X0 X2 DIV",
            "X4 NEG",
            "X0 X1 DIV X3 ADD",
            "X0 X2 DIV X3 ADD",
        ),
        target_fn=lambda feat: feat["X0"] / np.clip(feat["X1"], 0.2, None),
    ),
)


def public_symbolic_task_specs() -> tuple[PublicSymbolicTaskSpec, ...]:
    return PUBLIC_SYMBOLIC_TASKS


def generate_public_symbolic_task(
    task_id: str,
    *,
    seed: int,
    samples_per_env: int = 96,
) -> SelectorBenchmarkTask:
    spec = next((item for item in PUBLIC_SYMBOLIC_TASKS if item.task_id == task_id), None)
    if spec is None:
        raise KeyError(f"Unknown public symbolic task {task_id!r}.")
    rng = np.random.default_rng(seed)
    environments = ("env_a", "env_b", "env_c", "env_d")
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for env_index, env_name in enumerate(environments):
        base_index = env_index * samples_per_env
        for offset in range(samples_per_env):
            if spec.task_id == "feynman_ratio":
                x0 = float(rng.lognormal(mean=0.10, sigma=0.45))
                x1 = float(rng.lognormal(mean=0.00 if env_name in {"env_a", "env_b"} else 0.15, sigma=0.55))
                if env_name in {"env_a", "env_b"}:
                    x2 = x1 * (1.00 + rng.normal(scale=0.015))
                elif env_name == "env_c":
                    x2 = x1 * (0.92 + rng.normal(scale=0.07))
                else:
                    x2 = x1 * (0.55 + rng.normal(scale=0.18))
            elif spec.task_id == "feynman_product_seed_shift":
                x0 = float(rng.normal())
                x1 = float(rng.normal())
                if env_name in {"env_a", "env_b"}:
                    x2 = 0.98 * x1 + rng.normal(scale=0.05)
                elif env_name == "env_c":
                    x2 = 0.75 * x1 + rng.normal(scale=0.25)
                else:
                    x2 = 0.20 * x1 + rng.normal(scale=0.60)
            else:
                x0 = float(rng.normal())
                x1 = float(rng.normal(loc=0.25 * np.sin(offset / 13.0), scale=1.0))
                if env_name in {"env_a", "env_b"}:
                    x2 = 0.97 * x1 + rng.normal(scale=0.12)
                elif env_name == "env_c":
                    x2 = 0.35 * x1 + rng.normal(scale=0.65)
                else:
                    x2 = 0.10 * x1 + rng.normal(scale=0.90)
            if spec.task_id == "feynman_product_seed_shift":
                x3 = 0.0
                x4 = 0.0
                x5 = 0.0
            else:
                x3_scale = 0.20 if spec.task_id == "feynman_product_seed_shift" else 0.80
                x3 = rng.normal(scale=x3_scale)
                x4 = rng.normal(scale=1.0)
                x5 = rng.normal(scale=1.0)
            features = {
                "X0": x0,
                "X1": x1,
                "X2": x2,
                "X3": x3,
                "X4": x4,
                "X5": x5,
            }
            stable_signal = spec.target_fn(features)
            if env_name == "env_a":
                stable_scale = 1.05
                spurious_scale = 0.90
                noise_scale = 0.08
            elif env_name == "env_b":
                stable_scale = 1.20
                spurious_scale = 0.45
                noise_scale = 0.10
            elif env_name == "env_c":
                stable_scale = 1.00
                spurious_scale = -0.25
                noise_scale = 0.12
            else:
                stable_scale = 1.05
                spurious_scale = -0.10
                noise_scale = 0.20
            if spec.task_id == "feynman_ratio":
                stable_scale = 0.80 if env_name in {"env_a", "env_b"} else (0.95 if env_name == "env_c" else 1.10)
                ratio_neighbor_scale = 0.90 if env_name in {"env_a", "env_b"} else (0.45 if env_name == "env_c" else -0.10)
                x2_ratio = x0 / np.clip(np.abs(x2), 0.2, None)
                target = stable_scale * stable_signal + ratio_neighbor_scale * x2_ratio + 0.04 * x4 + rng.normal(scale=0.08)
            elif spec.task_id == "feynman_product_seed_shift":
                shifted_seed = (seed // 10) % 5 in {0, 2}
                product_neighbor_scale = (
                    {"env_a": 1.20, "env_b": 1.20, "env_c": 1.00, "env_d": 0.20}
                    if shifted_seed
                    else {"env_a": 0.80, "env_b": 0.80, "env_c": 0.95, "env_d": 1.10}
                )
                stable_scale = {"env_a": 1.00, "env_b": 1.00, "env_c": 1.05, "env_d": 1.10}[env_name]
                product_neighbor = x0 * x2
                target = stable_scale * stable_signal + product_neighbor_scale[env_name] * product_neighbor + rng.normal(scale=0.08)
                x3 = rng.normal(scale=0.20)
                x4 = rng.normal(scale=1.0)
            else:
                target = stable_scale * stable_signal + spurious_scale * x4 + 0.15 * x3 + rng.normal(scale=noise_scale)
            rows.append(
                {
                    "date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=base_index + offset),
                    "environment": env_name,
                    "X0": x0,
                    "X1": x1,
                    "X2": x2,
                    "X3": x3,
                    "X4": x4,
                    "X5": x5,
                    "TARGET_RET_1": target,
                }
            )
    frame = pd.DataFrame(rows)
    frame["TARGET_XS_RET_1"] = frame["TARGET_RET_1"]
    candidates = [FormulaCandidate(formula=formula, source="public_symbolic", role="public_benchmark") for formula in spec.candidate_formulas]
    return SelectorBenchmarkTask(
        benchmark_name="public_symbolic_suite",
        task_id=spec.task_id,
        scenario="env_shift_protocol",
        seed=seed,
        frame=frame,
        target=frame["TARGET_RET_1"].copy(),
        candidate_formulas=candidates,
        true_formula=spec.true_formula,
    )
