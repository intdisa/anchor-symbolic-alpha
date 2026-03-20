# Route B Paper Direction

## Current thesis

Route B should no longer be presented as a "multi-agent synergy" paper.

The empirically supported thesis is:

- one anchor agent discovers and upgrades the main factor family
- challenger agents are used to test replacement or additive hypotheses
- the manager's main value is validation-backed selection, replacement, and gating

In the current codebase, this means:

- anchor agent: `quality_solvency`
- challengers:
  - `efficiency_growth`
  - `valuation_size`
  - `short_horizon_flow`
- top-level controller:
  - `HierarchicalManagerAgent`
- default Route B experiment mode:
  - anchor-only (`quality_solvency`)
- challenger families are now opt-in verification branches, not part of the default main run

Note:

- older reports may still use the alias `trend_structure_only`
- in the current split Route B setup, that alias expands to:
  - `quality_solvency`
  - `efficiency_growth`
  - `valuation_size`

## What the current experiments support

### Supported claims

1. A strict symbolic search pipeline can stably discover a positive cross-sectional
   `cash + quality` factor on U.S. equities.
2. Validation-backed commit / replacement is necessary.
3. Challenger gating matters because naive re-admission of short-horizon flow can
   degrade the final result.
4. The manager is useful as a selector and gatekeeper even when only one anchor
   family dominates the final library.

### Unsupported claims

1. Multi-agent synergy is the source of the main gain.
2. `full` is better than the best single anchor family.
3. Every skill family contributes stable marginal alpha.

## Current main result

### Liquid500, 5 seeds

- champion: `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- `full == quality_solvency_only`
- `quality_solvency_only` also recovers the same champion on all 5 seeds
- Sharpe: `0.5694`
- annual return: `0.0439`
- max drawdown: `-0.2937`
- turnover: `0.0097`

### Liquid1000, stricter subset

- champion: `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- `quality_solvency_only` also recovers the same champion in the current 3-episode run
- Sharpe: `0.7330`
- annual return: `0.0530`
- max drawdown: `-0.2436`
- turnover: `0.0098`

## Current interpretation of agents

### `quality_solvency`

- role: anchor agent
- effective feature family:
  - `CASH_RATIO_Q`
  - `PROFITABILITY_Q`
  - `PROFITABILITY_A`
  - `LEVERAGE_Q`
  - `LEVERAGE_A`
  - `SIZE_LOG_MCAP`
- current champion family:
  - `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`
- direct evidence:
  - `quality_solvency_only` matches the current `full` result on liquid500 across 5 seeds

### `efficiency_growth`

- role: challenger
- representative signals such as `SALES_TO_ASSETS_Q RANK` are not pure noise
- but they do not yet improve the anchor baseline consistently under the current
  selection criterion

### `valuation_size`

- role: challenger
- currently weak; no stable accepted formula in the current smoke setting

### `short_horizon_flow`

- role: challenger
- has standalone short-horizon signal
- does not provide stable marginal value on top of the anchor baseline

## What this means for the paper

The paper should be framed as:

- anchor-agent symbolic discovery
- challenger-based verification
- validation-backed replacement and walk-forward-aware selection

The paper should not currently be framed as:

- general multi-agent cooperation
- synergistic multi-skill alpha libraries

## Immediate experimental priorities

1. Treat `quality_solvency_only` as the primary baseline to confirm the anchor-only
   thesis directly.
2. Keep challenger experiments as verification studies, not main-result drivers.
3. Strengthen baseline comparisons against direct formulas and simpler selection
   schemes.
4. Preserve current ablations around:
   - validation-backed selection
   - replacement-first admission
   - residual challenger gating

## Current selection-mismatch note

The strongest directly tested formula is currently:

- `CASH_RATIO_Q RANK SALES_TO_ASSETS_Q RANK ADD`

However, the current leak-free train / validation evidence still prefers:

- `CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`

That means the remaining mismatch is likely due to regime shift between validation
and test, not a trivial search bug. The project should therefore avoid forcing
`cash + sales_to_assets` into the main line using test-aware tuning.

## Narrative fallback rule

If future experiments still show:

- `full == anchor baseline`
- no challenger delivers stable incremental gain

then the final paper should fully drop the multi-agent claim and keep only:

- knowledge-constrained symbolic discovery
- validation-backed factor selection
- replacement-first baseline upgrades
