#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.dataio.us_equities_panel import build_us_equities_panel


DEFAULT_PANEL_OUTPUT = Path("data/processed/us_equities/us_equities_panel.parquet")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/reports/us_equities_panel_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the normalized U.S. equities cross-sectional panel.")
    parser.add_argument("--raw-root", default="data/raw/us_equities")
    parser.add_argument("--config-path", default="configs/us_equities_panel.yaml")
    parser.add_argument("--panel-output", default=DEFAULT_PANEL_OUTPUT.as_posix())
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT.as_posix())
    parser.add_argument("--skip-save-panel", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_us_equities_panel(raw_root=args.raw_root, config_path=args.config_path)

    panel_output = Path(args.panel_output)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_save_panel:
        panel_output.parent.mkdir(parents=True, exist_ok=True)
        bundle.panel.to_parquet(panel_output, index=False)

    summary = {
        "panel_rows": int(len(bundle.panel)),
        "feature_count": int(len(bundle.feature_columns)),
        "feature_columns": list(bundle.feature_columns),
        "target_columns": list(bundle.target_columns),
        "date_min": str(bundle.panel["date"].min().date()),
        "date_max": str(bundle.panel["date"].max().date()),
        "permno_count": int(bundle.panel["permno"].nunique()),
        "train_rows": int(len(bundle.splits.train)),
        "valid_rows": int(len(bundle.splits.valid)),
        "test_rows": int(len(bundle.splits.test)),
        "train_dates": [
            str(bundle.splits.train["date"].min().date()),
            str(bundle.splits.train["date"].max().date()),
        ],
        "valid_dates": [
            str(bundle.splits.valid["date"].min().date()),
            str(bundle.splits.valid["date"].max().date()),
        ],
        "test_dates": [
            str(bundle.splits.test["date"].min().date()),
            str(bundle.splits.test["date"].max().date()),
        ],
        "panel_output": None if args.skip_save_panel else str(panel_output),
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
