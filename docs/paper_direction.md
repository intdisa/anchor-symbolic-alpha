# Paper Direction

## Current thesis

The project should now be framed as:

- one anchor generator produces a candidate formula pool
- a cross-seed robust selector resolves formula choice under temporal shift
- auxiliary skill families are diagnostic branches until they show stable additive value

The current paper object is therefore not "multi-agent synergy." It is:

- generator-agnostic symbolic candidate generation
- robust formula selection under temporal and seed variability
- empirical analysis on U.S. equities plus selector-focused benchmarks

## Supported claims

1. The anchor generator repeatedly rediscovers a positive `cash + quality` family on
   U.S. equities.
2. Single-seed outcomes are not reliable enough to be treated as the final result.
3. Cross-seed support-adjusted consensus recovers a stronger and more stable formula
   than naive seed averaging or naive validation reselection.
4. `full` and `quality_solvency_only` remain behaviorally identical in the current
   mainline runs, so extra skill families are not yet part of the main contribution.

## Unsupported claims

1. Multi-agent cooperation is the source of the main gain.
2. Every skill family contributes marginal alpha.
3. The raw best-per-seed result is a stable discovery object.

## Canonical result objects

Mainline multiseed runs now have two layers:

- diagnostics: `outputs/runs/<run_name>__multiseed/reports/us_equities_multiseed.json`
- canonical result: `outputs/runs/<run_name>__multiseed/reports/us_equities_multiseed_canonical.json`

Use the canonical report in main tables. Treat the raw report as supporting
evidence about seed dispersion and failure modes.

## Current main results

### Liquid500

- canonical source:
  - `outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json`
- canonical selector:
  - `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- canonical walk-forward:
  - Sharpe `0.5694`
  - mean test rank IC `0.01315`
- interpretation:
  - raw seed outcomes vary materially
  - cross-seed consensus recovers the stable cash-quality anchor

### Liquid1000

- canonical source:
  - `outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json`
- canonical selector:
  - `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- canonical walk-forward:
  - Sharpe `0.7330`
  - mean test rank IC `0.01225`
- raw seed diagnostics:
  - mean Sharpe `0.5868`
  - std Sharpe `0.2923`
- interpretation:
  - the selector should be judged by the consensus result, not by raw seed averages

## Current method interpretation

### Generator

- current anchor family:
  - `quality_solvency`
- empirical behavior:
  - repeatedly surfaces `CASH_RATIO_Q`
  - repeatedly surfaces `PROFITABILITY_Q`
  - can still drift to weaker single-factor or noisy correlation formulas on some seeds

### Selector

- current mainline object:
  - support-adjusted cross-seed consensus
- current job:
  - rank candidate formulas under temporal shift
  - use cross-seed recurrence as an extra stability signal
  - break near-neighbor ties in favor of prior-aligned quarterly anchors

### Auxiliary skills

- `efficiency_growth`
- `valuation_size`
- `short_horizon_flow`

Current status:

- `short_horizon_flow` remains a useful failure-analysis branch, not a main result
- the other challenger families have not shown stable additive value
- `full` still collapses to the same result as `quality_solvency_only`

## What this means for the paper

The mainline paper should be organized around:

- `generator / selector / benchmark`
- cross-seed robust symbolic selection
- temporal distribution shift and Rashomon candidate pools
- synthetic plus public symbolic benchmark suites alongside U.S. equities

The paper should not currently be organized around:

- multi-agent coordination as the core contribution
- challenger-family synergy
- library-level diversification claims

## Immediate priorities

1. Make `canonical_by_variant` the default object used in experiment summaries and
   paper tables.
2. Expand the synthetic selector benchmark so the selector claim is supported
   outside U.S. equities.
3. Keep challenger runs as diagnostic controls, not as the headline result.

Current paper-table build script:

- `scripts/build_paper_results.py`
- `scripts/run_selector_benchmark_suite.py`
- `docs/selector_theory_note.md`

Current paper-writing artifacts:

- `outputs/reports/us_equities_paper_results.md`
- `outputs/reports/us_equities_paper_results_claims.json`
- `outputs/reports/us_equities_paper_results_draft_outline.md`
