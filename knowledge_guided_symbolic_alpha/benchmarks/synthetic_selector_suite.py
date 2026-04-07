from __future__ import annotations

import numpy as np
import pandas as pd

from ..generation import FormulaCandidate
from .task_protocol import SelectorBenchmarkTask


SYNTHETIC_SCENARIOS = (
    "transient_spuriosity",
    "regime_sign_flip",
    "noise_variance_shift",
    "near_neighbor_ambiguity",
    "adversarial_support_lockin",
    "heavy_tail_outlier_shift",
    "support_lockin_recovery",
)


def synthetic_selector_scenarios() -> tuple[str, ...]:
    return SYNTHETIC_SCENARIOS


def generate_synthetic_selector_task(
    scenario: str,
    *,
    seed: int,
    samples_per_env: int = 96,
) -> SelectorBenchmarkTask:
    if scenario not in SYNTHETIC_SCENARIOS:
        raise KeyError(f"Unknown synthetic selector scenario {scenario!r}.")
    rng = np.random.default_rng(seed)
    environments = ("env_a", "env_b", "env_c", "env_d")
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    if scenario == "transient_spuriosity":
        true_formula = "X0 X1 ADD"
        candidate_formulas = ("X0 X1 ADD", "X0 X2 ADD", "X4 NEG", "X0 X1 SUB", "X0 X1 ADD X3 ADD")
    elif scenario == "regime_sign_flip":
        true_formula = "X0 X1 SUB"
        candidate_formulas = ("X0 X1 SUB", "X0 X1 ADD", "X0 X2 SUB", "X4 NEG", "X0 X1 SUB X3 ADD")
    elif scenario == "noise_variance_shift":
        true_formula = "X0 X1 MUL"
        candidate_formulas = ("X0 X1 MUL", "X0 X2 MUL", "X4 NEG", "X0 X1 ADD", "X0 X1 MUL X3 ADD")
    elif scenario == "adversarial_support_lockin":
        true_formula = "X0 X1 ADD"
        candidate_formulas = ("X0 X1 ADD", "X0 X2 ADD", "X4 NEG", "X0 X1 SUB", "X0 X2 ADD X3 ADD")
    elif scenario == "heavy_tail_outlier_shift":
        true_formula = "X0 X1 ADD"
        candidate_formulas = ("X0 X1 ADD", "X0 X2 ADD", "X4 NEG", "X0 X1 SUB", "X0 X1 ADD X3 ADD")
    elif scenario == "support_lockin_recovery":
        true_formula = "X0 X1 MUL"
        candidate_formulas = ("X0 X1 MUL", "X0 X2 MUL", "X4 NEG", "X0 X1 ADD", "X0 X2 MUL X3 ADD")
    else:
        true_formula = "X0 X1 DIV"
        candidate_formulas = ("X0 X1 DIV", "X0 X2 DIV", "X4 NEG", "X0 X1 DIV X3 ADD", "X0 X2 DIV X3 ADD")

    for env_index, env_name in enumerate(environments):
        base_index = env_index * samples_per_env
        for offset in range(samples_per_env):
            if scenario == "near_neighbor_ambiguity":
                x0 = float(rng.lognormal(mean=0.00, sigma=0.55))
                x1 = float(rng.lognormal(mean=0.00, sigma=0.65))
                if env_name in {"env_a", "env_b"}:
                    x2 = x1 * (1.00 + rng.normal(scale=0.012))
                elif env_name == "env_c":
                    x2 = x1 * (0.94 + rng.normal(scale=0.09))
                else:
                    x2 = x1 * (0.55 + rng.normal(scale=0.18))
            elif scenario == "adversarial_support_lockin":
                x0 = float(rng.normal())
                x1 = float(rng.normal(loc=0.10 * np.sin(offset / 11.0), scale=1.0))
                if env_name in {"env_a", "env_b", "env_c"}:
                    x2 = 1.03 * x1 + rng.normal(scale=0.05)
                else:
                    x2 = -0.20 * x1 + rng.normal(scale=1.20)
            elif scenario == "heavy_tail_outlier_shift":
                x0 = float(rng.standard_t(df=5))
                x1 = float(rng.standard_t(df=6) + 0.10 * np.cos(offset / 8.0))
                if env_name in {"env_a", "env_b"}:
                    x2 = 0.98 * x1 + rng.standard_t(df=5) * 0.08
                elif env_name == "env_c":
                    x2 = 0.70 * x1 + rng.standard_t(df=4) * 0.25
                else:
                    x2 = 0.25 * x1 + rng.standard_t(df=3) * 0.60
            elif scenario == "support_lockin_recovery":
                x0 = float(rng.normal(loc=0.05 * np.sin(offset / 7.0), scale=1.0))
                x1 = float(rng.normal(loc=0.10 * np.cos(offset / 6.0), scale=1.0))
                if env_name in {"env_a", "env_b"}:
                    x2 = 1.04 * x1 + rng.normal(scale=0.04)
                elif env_name == "env_c":
                    x2 = 0.85 * x1 + rng.normal(scale=0.22)
                else:
                    x2 = -0.10 * x1 + rng.normal(scale=0.95)
            else:
                x0 = float(rng.normal())
                x1 = float(rng.normal(loc=0.15 * np.cos(offset / 9.0), scale=1.0))
                x2 = 0.95 * x1 + rng.normal(scale=0.15 if env_name in {"env_a", "env_b"} else 0.55)
            x3 = rng.normal(scale=0.7)
            x4 = rng.normal(scale=1.0)
            x5 = rng.normal(scale=1.0)
            if scenario == "transient_spuriosity":
                stable_signal = x0 + x1
                spurious_scale = 1.10 if env_name in {"env_a", "env_b"} else (-0.45 if env_name == "env_c" else -0.10)
                noise_scale = 0.10 if env_name != "env_d" else 0.20
                target = stable_signal + spurious_scale * x4 + 0.10 * x3 + rng.normal(scale=noise_scale)
            elif scenario == "regime_sign_flip":
                stable_signal = x0 - x1
                misleading = x0 + x1
                misleading_scale = 0.95 if env_name in {"env_a", "env_b"} else (-0.40 if env_name == "env_c" else -0.15)
                target = stable_signal + misleading_scale * misleading + rng.normal(scale=0.12)
            elif scenario == "noise_variance_shift":
                stable_signal = x0 * x1
                neighbor_signal = x0 * x2
                neighbor_scale = 0.90 if env_name in {"env_a", "env_b"} else 0.20
                noise_scale = 0.05 if env_name in {"env_a", "env_b"} else (0.18 if env_name == "env_c" else 0.30)
                target = stable_signal + neighbor_scale * neighbor_signal + 0.35 * x4 + rng.normal(scale=noise_scale)
            elif scenario == "adversarial_support_lockin":
                stable_signal = x0 + x1
                misleading_signal = x0 + x2
                if env_name in {"env_a", "env_b"}:
                    target = 0.75 * stable_signal + 1.10 * misleading_signal + 0.10 * x3 + rng.normal(scale=0.08)
                elif env_name == "env_c":
                    target = 0.80 * stable_signal + 0.95 * misleading_signal + 0.10 * x3 + rng.normal(scale=0.10)
                else:
                    target = 1.15 * stable_signal - 0.35 * misleading_signal + 0.12 * x3 + rng.normal(scale=0.14)
            elif scenario == "heavy_tail_outlier_shift":
                stable_signal = x0 + x1
                neighbor_signal = x0 + x2
                outlier_scale = 0.15 if env_name in {"env_a", "env_b"} else (0.35 if env_name == "env_c" else 0.80)
                rare_outlier = (rng.random() < (0.01 if env_name != "env_d" else 0.06)) * rng.standard_t(df=2) * 3.0
                target = (
                    1.05 * stable_signal
                    + (0.85 if env_name in {"env_a", "env_b"} else (0.40 if env_name == "env_c" else -0.10)) * neighbor_signal
                    + outlier_scale * x4
                    + rare_outlier
                    + rng.normal(scale=0.10 if env_name != "env_d" else 0.22)
                )
            elif scenario == "support_lockin_recovery":
                stable_signal = x0 * x1
                misleading_signal = x0 * x2
                if env_name in {"env_a", "env_b"}:
                    target = 0.45 * stable_signal + 1.05 * misleading_signal + 0.08 * x3 + rng.normal(scale=0.08)
                elif env_name == "env_c":
                    target = 0.80 * stable_signal + 0.65 * misleading_signal + 0.10 * x3 + rng.normal(scale=0.10)
                else:
                    target = 1.25 * stable_signal - 0.10 * misleading_signal + 0.12 * x3 + rng.normal(scale=0.12)
            else:
                denominator = np.clip(np.abs(x1), 0.2, None)
                stable_signal = x0 / denominator
                neighbor_signal = x0 / np.clip(np.abs(x2), 0.2, None)
                if env_name in {"env_a", "env_b"}:
                    stable_scale = 0.80
                    neighbor_scale = 0.90
                    noise_scale = 0.08
                elif env_name == "env_c":
                    stable_scale = 0.95
                    neighbor_scale = 0.45
                    noise_scale = 0.08
                else:
                    stable_scale = 1.10
                    neighbor_scale = -0.15
                    noise_scale = 0.08
                target = stable_scale * stable_signal + neighbor_scale * neighbor_signal + 0.04 * x4 + rng.normal(scale=noise_scale)
            rows.append(
                {
                    "date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=base_index + offset),
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
    candidates = [FormulaCandidate(formula=formula, source="synthetic_suite", role="benchmark") for formula in candidate_formulas]
    return SelectorBenchmarkTask(
        benchmark_name="synthetic_selector_suite",
        task_id=scenario,
        scenario=scenario,
        seed=seed,
        frame=frame,
        target=frame["TARGET_RET_1"].copy(),
        candidate_formulas=candidates,
        true_formula=true_formula,
    )
