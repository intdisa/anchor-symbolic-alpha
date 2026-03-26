# Knowledge-Guided Symbolic Alpha

This repository implements the current U.S. equities mainline for knowledge-guided
symbolic alpha discovery. The active research narrative is:

- one anchor generator drives candidate discovery
- a cross-seed robust selector chooses formulas under validation shift
- challenger skill families are diagnostic branches unless they show additive value

`us_equities` is now the only supported dataset name and canonical data layout.
Legacy pre-mainline artifacts are archived and not used at runtime.

## Runtime tiers

- `core`: `numpy`, `pandas`, `PyYAML`
- `train`: `torch`
- `eval`: `duckdb`, `pyarrow`
- `wrds`: `pg8000`, `pyarrow`

Recommended local env split:

- `.venv`: core / eval / docs / data plumbing
- `.venv-train`: Python 3.11 train runtime validated with torch

Install only what you need:

```bash
. .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[train]"
pip install -e ".[eval]"
```

Validated train setup:

```bash
python3.11 -m venv .venv-train
. .venv-train/bin/activate
pip install --no-build-isolation -e ".[train,dev]"
```

Run a preflight check before a long job:

```bash
.venv/bin/python scripts/run_preflight.py --profile core
.venv/bin/python scripts/run_preflight.py --profile eval
.venv-wrds/bin/python scripts/run_preflight.py --profile wrds
.venv-train/bin/python scripts/run_preflight.py --profile train
```

## Canonical layout

- raw data: `data/raw/us_equities`
- processed panel: `data/processed/us_equities`
- experiment outputs: `outputs/runs/<run_name>/`
- reusable checkpoints: `artifacts/checkpoints/`

Mainline code is now organized around:

- `knowledge_guided_symbolic_alpha/generation/`
- `knowledge_guided_symbolic_alpha/selection/`
- `knowledge_guided_symbolic_alpha/benchmarks/`

Each experiment run writes:

- `outputs/runs/<run_name>/reports/`
- `outputs/runs/<run_name>/factors/`
- `outputs/runs/<run_name>/logs/`
- `outputs/runs/<run_name>/checkpoints/`
- `outputs/runs/<run_name>/run_manifest.json`

## Main commands

```bash
. .venv/bin/activate
python experiments/run_pretrain.py --run-name pretrain_smoke
python experiments/run_train.py --episodes 60 --run-name train_smoke
python experiments/run_ablation.py --episodes 10 --run-name ablation_smoke
python experiments/run_multiseed.py --episodes 10 --run-name multiseed_smoke
python experiments/run_backtest.py --run-name backtest_smoke
python scripts/evaluate_formula.py --formula "CASH_RATIO_Q RANK" --run-name formula_eval_smoke
```

For train-side commands, use `.venv-train/bin/python` instead of `.venv/bin/python`.

Canonical train wrapper:

```bash
.venv-train/bin/python scripts/run_standard_training.py \
  --mode anchor \
  --data-config configs/us_equities_liquid500.yaml \
  --run-name liquid500_anchor_e20 \
  --episodes 20
```

Synthetic selector benchmark:

```bash
.venv/bin/python scripts/run_synthetic_selector_benchmark.py \
  --run-name synthetic_selector_smoke
```

Selector benchmark suites:

```bash
.venv/bin/python scripts/run_selector_benchmark_suite.py --suite synthetic
.venv/bin/python scripts/run_selector_benchmark_suite.py --suite public
```

Paper-facing result summary:

```bash
.venv/bin/python scripts/build_paper_results.py
```

Paper-facing drafting artifacts:

- `outputs/reports/us_equities_paper_results.md`
- `outputs/reports/us_equities_paper_results_claims.json`
- `outputs/reports/us_equities_paper_results_draft_outline.md`

For multiseed runs, treat the consensus output as canonical:

- full diagnostics: `outputs/runs/<run_name>/reports/us_equities_multiseed.json`
- paper-facing canonical result: `outputs/runs/<run_name>/reports/us_equities_multiseed_canonical.json`
- `aggregated_by_variant` is seed-dispersion diagnostics
- `canonical_by_variant` is the mainline result object

## Tests

Default `pytest` runs only the `core` suite:

```bash
python -m pytest
```

Optional suites:

```bash
python -m pytest -m eval
python -m pytest -m train
```

## Data setup

See:

- `docs/data_setup.md`
- `docs/training_runbook.md`
- `docs/repository_structure.md`
- `docs/project_evolution_history.md`
- `docs/paper_direction.md`
- `docs/selector_theory_note.md`
