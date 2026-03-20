# Experiment Log

This document records the experiment files that currently matter for paper writing.
It is a compact index, not a full report.

Note: many historical output directories still contain `route_b` in the path.
Those names are legacy artifacts from the exploratory phase.

## Main result files

### Liquid500, 5 seeds

- multiseed report:
  - `outputs/route_b_liquid500_multiseed_5seed/reports/route_b_multiseed.json`
- anchor-only confirmation:
  - `outputs/route_b_liquid500_quality_multiseed_5seed/reports/route_b_multiseed.json`
- current conclusion:
  - `full == quality_solvency_only`
  - `quality_solvency_only` is also identical on all 5 seeds
  - champion: `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
  - Sharpe `0.5694`
  - annual return `0.0439`
  - max drawdown `-0.2937`
  - turnover `0.0097`

### Liquid1000, stricter subset

- training report:
  - `outputs/route_b_liquid1000_full_e3/reports/route_b_train_summary.json`
- walk-forward report:
  - `outputs/route_b_liquid1000_full_e3/reports/route_b_walk_forward.json`
- anchor-only confirmation:
  - `outputs/route_b_liquid1000_quality_e3/reports/route_b_train_summary.json`
- direct formula baselines:
  - `outputs/reports/route_b_liquid1000_formula_baselines.json`
- current conclusion:
  - champion: `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
  - `quality_solvency_only` reaches the same champion in the current 3-episode run
  - Sharpe `0.7330`
  - annual return `0.0530`
  - max drawdown `-0.2436`
  - turnover `0.0098`

## Ablation files

### Liquid500 custom ablation

- report:
  - `outputs/route_b_liquid500_ablation_custom/reports/route_b_ablation.json`
- key findings:
  - removing `validation-backed` collapses to `RET_1 NEG`
  - removing `flow gate` re-admits flow and hurts the final result
  - removing `seed-priority` weakens champion selection

### Liquid1000 targeted ablations

- no validation-backed:
  - `outputs/route_b_liquid1000_ablation_no_validation/reports/route_b_ablation.json`
- no flow gate:
  - `outputs/route_b_liquid1000_ablation_no_flow_gate/reports/route_b_ablation.json`
- key findings:
  - `validation-backed` remains necessary on the stricter subset
  - `flow gate` is not binding there because flow is not admitted anyway

## Formula baseline files

### Liquid500 direct formula baselines

- summary:
  - `outputs/reports/route_b_liquid500_formula_baselines.json`
- strongest direct formula currently observed:
  - `CASH_RATIO_Q RANK SALES_TO_ASSETS_Q RANK ADD`

### Liquid1000 direct formula baselines

- summary:
  - `outputs/reports/route_b_liquid1000_formula_baselines.json`
- strongest direct formula currently observed:
  - `CASH_RATIO_Q RANK SALES_TO_ASSETS_Q RANK ADD`
- interpretation:
  - direct walk-forward on `valid + test` favors `cash + sales_to_assets`
  - current leak-free train / validation selection still favors `cash + profitability`
  - this is a selection-mismatch problem, not a reason to tune against the test split

### Liquid500 baseline + challenger combination check

- combo evaluation:
  - `outputs/route_b_liquid500_combo_eval/reports/route_b_walk_forward.json`
- key conclusion:
  - adding `SALES_TO_ASSETS_Q RANK` to the anchor baseline does not beat the baseline

## Current interpretation

- currently validated anchor:
  - `quality_solvency`
- current effective formula:
  - `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- current challengers:
  - `efficiency_growth`
  - `valuation_size`
  - `short_horizon_flow`
- current empirical status:
  - challengers have not yet shown stable marginal gain over the anchor baseline
  - default mainline runs should therefore be interpreted as anchor-only unless challengers are explicitly requested
