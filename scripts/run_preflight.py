#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.runtime import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository preflight checks for a runtime profile.")
    parser.add_argument("--profile", choices=("core", "eval", "train", "wrds"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight(args.profile)
    print(json.dumps(report.to_dict(), ensure_ascii=True, indent=2))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
