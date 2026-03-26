# Experiment Log

This file tracks the current paper-facing outputs. Historical exploratory runs are
left in `outputs/`, but they are not the mainline evidence set.

## Canonical mainline results

### Liquid500

- workflow:
  - `outputs/runs/liquid500_multiseed_e5_r3/reports/workflow_summary.json`
- canonical consensus report:
  - `outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json`
- mainline selector:
  - `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- canonical walk-forward:
  - Sharpe `0.5694`
  - annual return `0.0439`
  - mean test rank IC `0.01315`
  - turnover `0.0097`
- interpretation:
  - raw seeds are unstable
  - support-adjusted cross-seed consensus recovers the stable cash-quality anchor

### Liquid1000

- workflow:
  - `outputs/runs/liquid1000_multiseed_e5_r4/reports/workflow_summary.json`
- canonical consensus report:
  - `outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json`
- full multiseed report:
  - `outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed.json`
- mainline selector:
  - `canonical_by_variant.full.selector_records`
  - `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- canonical walk-forward:
  - Sharpe `0.7330`
  - annual return `0.0530`
  - mean test rank IC `0.01225`
  - turnover `0.0098`
- raw seed diagnostics:
  - mean Sharpe `0.5868`
  - std Sharpe `0.2923`
  - mean test rank IC `0.01050`
  - std test rank IC `0.00349`

## Mainline method conclusions

- the anchor generator repeatedly finds the same cash-quality family
- raw single-seed outputs should not be treated as the final result object
- cross-seed support-adjusted consensus is now the canonical selector output
- `full` and `quality_solvency_only` remain behaviorally identical on the current
  mainline runs

## Supporting selector results

### Near-neighbor selector fix

- code path:
  - `knowledge_guided_symbolic_alpha/selection/robust_selector.py`
- selector reruns:
  - `outputs/runs/liquid500_selector_subset_e20_r2/reports/workflow_summary.json`
  - `outputs/runs/liquid1000_selector_subset_e20_r2/reports/workflow_summary.json`
- key effect:
  - `liquid500` no longer misranks `PROFITABILITY_A CASH_RATIO_Q RANK ADD` above
    the quarterly cash-quality anchor

### Liquid500 seed-instability exposure

- multiseed diagnostics:
  - `outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed.json`
- raw picture:
  - strong seed: `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
  - middling seeds: `CASH_RATIO_Q RANK`
  - bad seed: `PROFITABILITY_Q CASH_RATIO_Q PROFITABILITY_Q CORR_5 ADD`
- implication:
  - the remaining bottleneck is seed-robust selection, not agent routing

## Ablation status

### Liquid500 selector-synced ablation

- report:
  - `outputs/runs/liquid500_ablation_e5_r2__ablation/reports/us_equities_ablation.json`
- key findings:
  - `no_validation_backed` still collapses
  - `full` and `quality_solvency_only` still match
  - `short_horizon_flow_only` remains weaker and higher-turnover

## Synthetic selector benchmark

- smoke benchmark:
  - `outputs/runs/synthetic_selector_repo_smoke/reports/synthetic_selector_benchmark.json`
- key finding:
  - naive rank-IC selection prefers the spurious short-horizon formula
  - the robust selector recovers the stable anchor formula

## Benchmark suite path

- synthetic/public selector suites now run through:
  - `scripts/run_selector_benchmark_suite.py`
- paper-facing benchmark outputs should be read from:
  - `outputs/runs/<run_name>/reports/*_selector_benchmark_summary.json`
  - `outputs/runs/<run_name>/reports/*_selector_benchmark_leaderboard.csv`
- paper-facing summary builder now also emits:
  - `outputs/reports/us_equities_paper_results_benchmark_table.csv`
  - `outputs/reports/us_equities_paper_results_ablation_table.csv`
  - `outputs/reports/us_equities_paper_results_seed_dispersion.csv`

## Current paper rule

Use these objects in the draft:

- `liquid500`: consensus report
- `liquid1000`: `canonical_by_variant` from the multiseed report
- raw `aggregated_by_variant`: diagnostics only
- challenger-only variants: control studies only

Paper-facing summary artifact:

- script:
  - `scripts/build_paper_results.py`
- generated outputs:
  - `outputs/reports/us_equities_paper_results.json`
  - `outputs/reports/us_equities_paper_results.md`
  - `outputs/reports/us_equities_paper_results_main_table.csv`

## Benchmark suite refresh

### Synthetic suite smoke5

- report:
  - `outputs/runs/synthetic_selector_suite_smoke5/reports/synthetic_selector_benchmark_summary.json`
- leaderboard:
  - `support_adjusted_cross_seed_consensus`: selection accuracy `1.00`
  - `naive_rank_ic`: selection accuracy `0.85`
  - `best_validation_mean_rank_ic`: selection accuracy `0.70`
- key implication:
  - the tuned near-neighbor setup now makes support-adjusted consensus the clear synthetic winner

### Public suite smoke7

- report:
  - `outputs/runs/public_selector_suite_smoke7/reports/public_selector_benchmark_summary.json`
- leaderboard:
  - `support_adjusted_cross_seed_consensus`: selection accuracy `1.00`
  - `naive_rank_ic`: selection accuracy `0.92`
  - `cross_seed_mean_score_consensus`: selection accuracy `0.80`
  - `single_seed_temporal_selector`: selection accuracy `0.84`
- key implication:
  - the added `feynman_product_seed_shift` task now makes the public symbolic suite separate the main method from both naive validation ranking and mean-score consensus
