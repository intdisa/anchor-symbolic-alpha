from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("outputs/runs")


def sanitize_run_name(run_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name.strip())
    cleaned = cleaned.strip(".-")
    return cleaned or "run"


def ensure_run_output_dirs(root: str | Path = DEFAULT_RUNS_ROOT, run_name: str = "run") -> dict[str, Path]:
    runs_root = Path(root)
    run_root = runs_root / sanitize_run_name(run_name)
    paths = {
        "runs_root": runs_root,
        "root": run_root,
        "reports": run_root / "reports",
        "factors": run_root / "factors",
        "logs": run_root / "logs",
        "checkpoints": run_root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def current_git_commit(cwd: str | Path | None = None) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def write_run_manifest(
    output_dirs: dict[str, Path],
    *,
    script_name: str,
    profile: str,
    preflight: dict[str, Any],
    config_paths: dict[str, str | None],
    dataset_name: str | None = None,
    subset: str | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "script_name": script_name,
        "profile": profile,
        "dataset_name": dataset_name,
        "subset": subset,
        "seed": seed,
        "git_commit": current_git_commit(output_dirs["root"]),
        "config_paths": config_paths,
        "preflight": preflight,
        "output_dirs": {name: str(path) for name, path in output_dirs.items()},
        "extra": extra or {},
    }
    manifest_path = output_dirs["root"] / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return manifest_path
