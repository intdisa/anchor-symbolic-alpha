from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from typing import Any

from .torch_support import probe_torch_health


CORE_MODULES = ("numpy", "pandas", "yaml")
EVAL_OPTIONAL_MODULES = ("duckdb", "pyarrow")
WRDS_MODULES = ("pg8000",)
WRDS_ENV_VARS = ("WRDS_USERNAME", "WRDS_PASSWORD")


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    message: str
    required: bool = True


@dataclass(frozen=True)
class PreflightReport:
    profile: str
    ok: bool
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "ok": self.ok,
            "checks": [asdict(item) for item in self.checks],
        }


def _module_check(name: str, *, required: bool) -> PreflightCheck:
    status = "ok" if importlib.util.find_spec(name) is not None else "missing"
    message = f"Python module `{name}` is available." if status == "ok" else f"Python module `{name}` is missing."
    return PreflightCheck(name=name, status=status, message=message, required=required)


def _env_var_check(name: str) -> PreflightCheck:
    present = bool(os.environ.get(name, ""))
    return PreflightCheck(
        name=name,
        status="ok" if present else "missing",
        message=f"Environment variable `{name}` is set." if present else f"Environment variable `{name}` is not set.",
        required=True,
    )


def run_preflight(profile: str) -> PreflightReport:
    normalized = profile.strip().lower()
    if normalized not in {"core", "eval", "train", "wrds"}:
        raise ValueError(f"Unsupported preflight profile {profile!r}.")

    checks: list[PreflightCheck] = []
    checks.extend(_module_check(name, required=True) for name in CORE_MODULES)

    if normalized == "eval":
        checks.extend(_module_check(name, required=False) for name in EVAL_OPTIONAL_MODULES)
    elif normalized == "train":
        checks.append(_module_check("torch", required=True))
        torch_report = probe_torch_health()
        torch_status = str(torch_report.get("status", "error"))
        torch_required = torch_status not in {"timeout"}
        torch_message = str(torch_report.get("message") or f"torch device={torch_report.get('device', 'unknown')}")
        if torch_status == "timeout":
            torch_message = (
                f"{torch_message} Proceeding because `torch` is installed and this check is advisory."
            )
        checks.append(
            PreflightCheck(
                name="torch_health",
                status=torch_status,
                message=torch_message,
                required=torch_required,
            )
        )
    elif normalized == "wrds":
        checks.extend(_module_check(name, required=True) for name in WRDS_MODULES)
        checks.extend(_env_var_check(name) for name in WRDS_ENV_VARS)

    ok = all(item.status == "ok" or not item.required for item in checks)
    return PreflightReport(profile=normalized, ok=ok, checks=tuple(checks))


def ensure_preflight(profile: str) -> PreflightReport:
    report = run_preflight(profile)
    if report.ok:
        return report
    lines = [f"Preflight profile `{report.profile}` failed:"]
    for check in report.checks:
        if check.status == "ok":
            continue
        prefix = "required" if check.required else "optional"
        lines.append(f"- {check.name} [{prefix}]: {check.message}")
    raise RuntimeError("\n".join(lines))
