from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_guided_symbolic_alpha.dataio import (
    DEFAULT_ROUTE_B_WRDS_ROOT,
    DEFAULT_XSC_SOURCE_ROOT,
    normalize_xsc_us_equities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize local xsc U.S. equities files into the standard raw layout.")
    parser.add_argument("--source-root", default=DEFAULT_XSC_SOURCE_ROOT.as_posix())
    parser.add_argument("--output-root", default="data/raw/us_equities/wrds")
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--datasets", default="crsp,ccm,quarterly,annual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = normalize_xsc_us_equities(
        source_root=args.source_root,
        output_root=args.output_root,
        chunksize=args.chunksize,
        datasets=tuple(part.strip() for part in args.datasets.split(",") if part.strip()),
    )
    print(summary.to_json())


if __name__ == "__main__":
    main()
