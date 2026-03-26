# Repository Structure

## Mainline directories

- `configs/`: canonical `us_equities` configs only
- `experiments/`: train, ablation, multiseed, pretrain, backtest entrypoints
- `knowledge_guided_symbolic_alpha/`: package code
- `scripts/`: data export, panel build, subset build, preflight, layout migration, standard train wrapper
  - includes `build_paper_results.py` for paper-facing summaries
  - includes claim/outline generation for drafting artifacts
  - includes `run_selector_benchmark_suite.py` for synthetic/public benchmark suites
  - one-off migration scripts are kept under `scripts/archive/`
- `tests/`: `core` by default, `eval` and `train` via pytest markers

## Runtime responsibilities

- `knowledge_guided_symbolic_alpha/generation/`
  - candidate extraction and anchor-generation summaries
  - bridge from trainer outputs to selector inputs
- `knowledge_guided_symbolic_alpha/selection/`
  - robust temporal selector
  - formula ranking under validation shift
- `knowledge_guided_symbolic_alpha/benchmarks/`
  - synthetic temporal-shift benchmark generation
  - synthetic selector suite
  - public symbolic benchmark tasks
  - selector-vs-naive benchmark helpers
- `knowledge_guided_symbolic_alpha/dataio/`
  - data contracts and processed split loading
  - no legacy path fallback
- `knowledge_guided_symbolic_alpha/runtime/`
  - preflight checks
  - run output layout and manifest writing
  - gated torch import support
- `experiments/common.py`
  - canonical mainline assembly only
  - generator and selector wiring helpers
  - no legacy naming translation
  - no output path guessing

## Output contract

All experiment-style runs write to:

```text
outputs/runs/<run_name>/
├── checkpoints/
├── factors/
├── logs/
├── reports/
└── run_manifest.json
```

`run_manifest.json` records:

- script name
- config paths
- dataset and subset
- seed
- git commit
- preflight result

Reusable checkpoints that are not tied to a single run belong in:

```text
artifacts/checkpoints/
```

Multiseed runs write two report layers:

- `us_equities_multiseed.json`: raw per-seed runs, aggregated seed diagnostics,
  consensus support details
- `us_equities_multiseed_canonical.json`: cross-seed consensus result used as the
  canonical paper-facing output

## Test contract

- `python -m pytest`: `core` only
- `python -m pytest -m eval`: DuckDB/parquet-dependent tests
- `python -m pytest -m train`: torch-dependent tests

Recommended env mapping:

- `.venv`: core and eval
- `.venv-train`: train
- `.venv-wrds`: WRDS export

Selector benchmark entrypoint:

- `scripts/run_synthetic_selector_benchmark.py`
- `scripts/run_selector_benchmark_suite.py`

## Historical artifacts

- legacy pre-mainline outputs are archived under `outputs/legacy/`
- historical method changes are documented in `docs/project_evolution_history.md`
- older pivot and data-procurement notes live under `docs/archive/`
