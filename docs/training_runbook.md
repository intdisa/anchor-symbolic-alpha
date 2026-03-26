# Training Runbook

Use one canonical wrapper for train-side work:

```bash
.venv-train/bin/python scripts/run_standard_training.py ...
```

## Standard anchor run

```bash
.venv-train/bin/python scripts/run_standard_training.py \
  --mode anchor \
  --data-config configs/us_equities_liquid500.yaml \
  --run-name liquid500_anchor_e20 \
  --episodes 20
```

This runs:

1. train preflight
2. anchor training
3. backtest against the trained factor file

Outputs:

- workflow summary: `outputs/runs/liquid500_anchor_e20/reports/workflow_summary.json`
- train run: `outputs/runs/liquid500_anchor_e20__anchor/`
- backtest run: `outputs/runs/liquid500_anchor_e20__backtest/`
- train report and factor file now carry `selector_records` from the robust temporal selector

## Standard full run

```bash
.venv-train/bin/python scripts/run_standard_training.py \
  --mode full \
  --data-config configs/us_equities_liquid500.yaml \
  --run-name liquid500_full_e20 \
  --episodes 20
```

## Ablation run

```bash
.venv-train/bin/python scripts/run_standard_training.py \
  --mode ablation \
  --data-config configs/us_equities_liquid500.yaml \
  --run-name liquid500_ablation_e5 \
  --episodes 5 \
  --variants full,quality_solvency_only,no_validation_backed
```

## Multiseed run

```bash
.venv-train/bin/python scripts/run_standard_training.py \
  --mode multiseed \
  --data-config configs/us_equities_liquid500.yaml \
  --run-name liquid500_multiseed_e5 \
  --episodes 5 \
  --variants full,quality_solvency_only \
  --seeds 7,17,27
```

Outputs:

- workflow summary: `outputs/runs/<run_name>/reports/workflow_summary.json`
- full multiseed diagnostics: `outputs/runs/<run_name>__multiseed/reports/us_equities_multiseed.json`
- canonical consensus result: `outputs/runs/<run_name>__multiseed/reports/us_equities_multiseed_canonical.json`

Use `us_equities_multiseed_canonical.json` for main tables. The full multiseed
report remains useful for seed-dispersion diagnostics and failure analysis.

## Smoke validation

```bash
.venv-train/bin/python scripts/run_standard_training.py \
  --mode anchor \
  --smoke \
  --run-name smoke_anchor
```

`--smoke` only runs preflight + train. It skips backtest on purpose.

## Selector benchmark

```bash
.venv/bin/python scripts/run_synthetic_selector_benchmark.py \
  --run-name synthetic_selector_smoke
```

## Dry run

Use this to verify the exact commands and output names without starting a job:

```bash
.venv-train/bin/python scripts/run_standard_training.py \
  --mode anchor \
  --data-config configs/us_equities_liquid500.yaml \
  --run-name liquid500_anchor_plan \
  --dry-run
```
