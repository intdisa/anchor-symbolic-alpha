# Selector Theory Note

This note records the lightweight formal layer for the current paper direction.
It is not a full proof appendix, but it defines the method object and the main
claim precisely enough for drafting.

## Objects

Let:

- `E = (e_1, ..., e_T)` be an ordered sequence of environments
- `C = {c_1, ..., c_m}` be a candidate formula pool
- `S(c, e_t)` be the temporal robust score of candidate `c` on environment `e_t`
- `R(c)` be the aggregate temporal score over the validation environments
- `U(c)` be the held-out utility on the test environment
- `A(c)` be the cross-seed support-adjusted consensus score

The current selector decomposes into two stages:

1. `TemporalRobustSelector`
   - computes per-candidate temporal diagnostics over validation slices
   - rejects candidates that fail admissibility
   - ranks survivors by robust score and redundancy-aware subset score

2. `CrossSeedConsensusSelector`
   - aggregates seed-level candidate/selector/champion support
   - reranks the consensus pool with a support-adjusted score
   - outputs the canonical cross-seed formula

## Support-adjusted consensus objective

For candidate `c`, define:

- `r(c)`: temporal robust score from reranking on the consensus pool
- `p_ch(c)`: fraction of seeds where `c` appears as the champion
- `p_sel(c)`: fraction of seeds where `c` is reselected by the temporal selector
- `p_can(c)`: fraction of seeds where `c` appears in the candidate pool
- `q(c)`: mean selector rank across seeds

The implemented consensus objective is:

`A(c) = r(c) + w_ch p_ch(c) + w_sel p_sel(c) + w_can p_can(c) - lambda max(q(c) - 1, 0)`

with positive support weights and a non-negative rank penalty.

## Informal proposition

Assume there exists a stable formula family `c*` such that:

1. `U(c*) > U(c)` for all competing high-support candidates `c`
2. `r(c*)` is within a small margin of the best temporal score
3. `p_ch(c*)` is strictly larger than that of unstable competitors
4. the estimation error of `r(c)` and `p_ch(c)` concentrates with more seeds

Then the probability that support-adjusted consensus selects an unstable formula
is lower than the probability that a naive validation-only selector selects that
formula, provided the champion-support margin dominates the temporal-score noise.

## Why this matters empirically

The finance runs already show the pattern that motivates the proposition:

- single seeds can drift to `cash-only` or noisy correlation formulas
- raw seed averages understate the stable formula's true quality
- cross-seed support-adjusted consensus recovers the recurring cash-quality
  family on both `liquid500` and `liquid1000`

The synthetic and public benchmark suites are the controlled tests for the same
claim under explicit distribution shift.
