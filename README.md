# Knowledge-Guided Symbolic Alpha

This repository implements a knowledge-guided symbolic alpha discovery system for
financial data. The current Route B main line is no longer framed as "multi-agent
synergy." The working thesis is:

- one anchor agent discovers and upgrades the main cross-sectional factor family
- challenger agents try to replace or augment that baseline under strict validation
- the manager acts as a selector and gatekeeper, not as a source of independent alpha
- Route B default runs now execute the anchor agent only; challenger families are opt-in verification branches

At the moment, the empirically supported anchor family is `quality_solvency`, with
the current stable champion:

- `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`

## Current scope

- Strict RPN grammar, parser, evaluator, and admission pipeline
- Dataset/pool-conditioned generators with synthetic recovery pretraining
- Hierarchical manager with anchor-agent replacement and challenger verification
- Pool-based and trade-proxy-aware evaluation, walk-forward backtesting, and tests

## Current empirical position

- `liquid500`, 5 seeds:
  - `full == quality_solvency_only`
  - champion: `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
  - Sharpe `0.5694`, annual return `0.0439`, max drawdown `-0.2937`, turnover `0.0097`
- `liquid1000`, stricter subset:
  - same champion
  - Sharpe `0.7330`, annual return `0.0530`, max drawdown `-0.2436`, turnover `0.0098`
- current evidence supports:
  - `quality_solvency` as the anchor agent
  - `validation-backed` selection and replacement
  - strict residual gating on challengers
- current evidence does not yet support:
  - multi-agent synergy as the main claim
  - `full` outperforming the anchor baseline

## Development order

1. Protocol freezing: configs, domain registry, language layer, tests
2. Evaluation layer: evaluator, metrics, factor pool, admission
3. Pool-based reward and training loop
4. Dataset-conditioned priors and synthetic recovery pretraining
5. Hierarchical skill-family planning and library-level search
6. Distributional collection scoring, transfer, and backtesting

## Local validation

```bash
. .venv/bin/activate
python -m pytest
```

## Experiment entry points

```bash
.venv/bin/python experiments/run_pretrain.py --examples 128 --epochs 3
.venv/bin/python experiments/run_train.py --partition-mode skill_hierarchy --episodes 60
.venv/bin/python experiments/run_route_b_train.py --episodes 3
.venv/bin/python experiments/run_ablation.py --partition-mode skill_hierarchy --episodes 3
.venv/bin/python experiments/run_backtest.py
```

Synthetic prior checkpoints are written to `outputs/checkpoints/`. Training writes
the current best RPN formulas to `outputs/factors/`, and walk-forward reports to
`outputs/reports/`.

## Route B bootstrap

The route-B pivot targets a U.S. equities cross-sectional panel backed by WRDS.
The repository now includes:

- `configs/route_b_data.yaml`
- `scripts/export_wrds_route_b.py`
- `docs/route_b_data.md`

Use the export script from a Python 3.12 environment with the `wrds` client
installed. The current project environment can remain on Python 3.13 for model
development and testing.

## Paper direction

The current paper direction is documented in:

- `docs/route_b_paper_direction.md`
- `docs/route_b_experiment_log.md`
- `docs/literature_map.md`

`docs/neurips_method_pivot.md` is kept as a historical pivot memo. It is no longer
the current top-level narrative.
