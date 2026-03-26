#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def merge_tree(source: Path, destination: Path) -> list[str]:
    actions: list[str] = []
    if not source.exists():
        return actions
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        target = destination / child.name
        if child.is_dir():
            actions.extend(merge_tree(child, target))
            if child.exists():
                child.rmdir()
        else:
            if target.exists():
                target.unlink()
            shutil.move(str(child), str(target))
            actions.append(f"moved {child} -> {target}")
    if source.exists():
        source.rmdir()
    return actions


def migrate_data_roots() -> list[str]:
    actions: list[str] = []
    for base in ("raw", "interim", "processed"):
        source = PROJECT_ROOT / "data" / base / "route_b"
        destination = PROJECT_ROOT / "data" / base / "us_equities"
        if not source.exists():
            continue
        actions.extend(merge_tree(source, destination))
    return actions


def archive_route_b_outputs() -> list[str]:
    outputs_root = PROJECT_ROOT / "outputs"
    legacy_root = outputs_root / "legacy" / "route_b"
    legacy_root.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []
    for child in sorted(outputs_root.glob("route_b*")):
        target = legacy_root / child.name
        if target.exists():
            continue
        shutil.move(str(child), str(target))
        actions.append(f"archived {child} -> {target}")
    return actions


def main() -> None:
    data_actions = migrate_data_roots()
    output_actions = archive_route_b_outputs()
    payload = {
        "data_actions": data_actions,
        "output_actions": output_actions,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
