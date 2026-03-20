# NeurIPS Method Pivot

This document is now a historical pivot memo.

The current top-level paper direction has moved to:

- anchor-agent symbolic discovery
- challenger-based verification
- validation-backed replacement and gating

Use `docs/paper_direction.md` as the current narrative source. Keep this
file as a record of the earlier multi-agent-heavy pivot reasoning.

## Why the current line is not enough

The current system is now technically stable:

- grammar / parser / evaluator are strict
- pool-based reward and admission are working
- the multi-agent manager no longer collapses to the price-only baseline
- `full` consistently finds the same two flow factors:
  - `GOLD_GAP_RET NEG`
  - `GOLD_OC_RET NEG`

This is useful progress, but it also exposes the real limit:

- the framework is no longer the main bottleneck
- the search repeatedly converges to a very small local family
- the out-of-sample result is still negative

This means the project has moved from an engineering bottleneck to a research bottleneck. A publishable NeurIPS direction needs a stronger method claim than "better local tuning of a gold timing system."


## What the eight papers suggest

### AlphaQCM

Use collection-level optimization, not single-formula reward. Alpha discovery is non-stationary and reward-sparse. A distributional critic is more suitable than a scalar reward because the value of a candidate depends on the evolving library.

### AlphaCFG

Treat alpha discovery as structured language search. The search space should be a grammar-defined tree MDP, with explicit syntax and semantic constraints, and tree-aware policy/value guidance.

### FactorMiner

Do not search from scratch every round. Use a retrieve-generate-evaluate-distill loop with experience memory that stores both successful patterns and failure regions, and use that memory to change future proposal distributions.

### AlphaFormer

Do not learn each market from scratch. Use a dataset-conditioned generator and pre-train it on synthetic or proxy tasks so the model arrives with a reusable prior over valid and useful formula families.

### KDD 2023 synergistic alpha collections

Optimize the alpha set directly against downstream collection performance. The right unit of optimization is the library, not the isolated factor.

### MACKRL

Multi-agent coordination should be built on common knowledge and a hierarchy. Agents should act on a shared state that all of them can reconstruct, while lower levels keep specialized local views.

### hhk-MARL

Human knowledge should enter as soft abstract priors, not hard-coded action scripts. The system should be able to decide when to trust or ignore human guidance.

### MA-HRL

Hierarchical decomposition matters when the state-action space is large and the reward is sparse. A high-level controller should decompose the search into semantically meaningful subtasks.


## Proposed pivot

The new target system should be:

**a hierarchical, common-knowledge, grammar-constrained library discovery framework with dataset-conditioned proposal priors, distributional collection critics, and retrieval-based experience memory**

This is a stronger research story than the current competitive price-vs-flow setup, and it matches the strongest ideas from the reference papers.


## New method: HCK-LD

Working name:

**Hierarchical Common-Knowledge Library Discovery (HCK-LD)**

The method has four levels.

### Level 1: dataset-conditioned proposal prior

Replace the current "train directly on one market" path with a proposal prior:

- input: dataset embedding, regime summary, current library embedding, task metadata
- output: a conditional distribution over formula trees or partial trees
- training:
  - synthetic formula recovery
  - formula completion under grammar constraints
  - historical library imitation from mined pools

This is the AlphaFormer-style component. It gives the search a reusable prior instead of forcing the system to rediscover short formulas from scratch for each dataset.


### Level 2: hierarchical common-knowledge planner

Replace the current flat competitive manager with a policy tree.

High-level planner actions:

- choose library slot to improve
- choose skill family
- choose search budget
- choose whether to expand, replace, or stop

Common knowledge state:

- dataset embedding
- current library embedding
- regime embedding
- validation risk summary
- current redundancy map
- slot occupancy / missing skill families

Private agent state:

- skill-specific feature whitelist
- operator whitelist
- local proposal history

This is the MACKRL / MA-HRL style component. The important change is that agents are no longer just "competing." They are coordinated by a shared hierarchical state.


### Level 3: skill-specialized grammar agents

The current `target_price`, `target_flow_vol`, `target_flow_gap`, and `context` roles should be replaced with skill families that are defined by financial structure, not by the current ad hoc split.

Suggested skill families:

- `reversal_gap`
- `intraday_imbalance`
- `volatility_liquidity`
- `trend_structure`
- `cross_asset_context`
- `regime_filter`

Each skill agent should have:

- its own feature whitelist
- its own operator whitelist
- its own seed templates
- its own memory retrieval channel
- the same strict grammar backend

The key change is that the planner chooses the skill family first, then the lower-level agent runs grammar-guided search inside that family.


### Level 4: distributional library critic

The current scalar reward shaping should be replaced with a distributional collection critic.

State:

- current library
- selected slot
- candidate formula
- regime / dataset context

Outputs:

- expected library gain
- risk-aware gain quantiles
- replacement value
- uncertainty

Training target:

- delta library score
- delta trade proxy
- delta walk-forward proxy
- replacement utility

This is the AlphaQCM-style component. The main point is that a candidate should be evaluated by its distribution of collection outcomes, not just a point estimate.


## Library evolution loop

The outer loop should follow a Ralph-style pattern:

1. retrieve relevant memory
2. plan a library improvement action
3. generate candidate(s) under grammar constraints
4. evaluate on train / validation / walk-forward proxy
5. admit / replace / reject
6. distill experience back into memory

Memory should be split into:

- successful motifs
- failure regions
- redundancy regions
- regime-specific winners
- replacement cases

This is where FactorMiner contributes the most.


## Human knowledge integration

Human knowledge should remain soft and abstract:

- operator-family priors
- unit constraints
- regime heuristics
- redundancy penalties
- skill-family routing hints

Human knowledge should not directly force formulas. It should only change the prior and critic target. This is the hhk-MARL lesson.


## Why this is more NeurIPS-worthy

This pivot gives a stronger contribution set:

1. **bi-level formulation**
   - inner grammar search over symbolic programs
   - outer library evolution over a synergistic factor set

2. **hierarchical common-knowledge MARL**
   - planner over shared state
   - skill agents over private subspaces

3. **dataset-conditioned prior**
   - pre-training and transfer, instead of one-dataset local tuning

4. **distributional collection critic**
   - directly addresses the non-stationary, sparse reward structure

5. **self-evolving memory**
   - search improves over time instead of resetting

This is substantially more defensible than a hand-tuned multi-agent search system for one asset.


## What should be kept from the current codebase

Keep:

- strict RPN grammar / parser / evaluator
- admission / replacement / reviewer pipeline
- factor pool abstraction
- walk-forward harness
- memory scaffolding
- grammar-MCTS backend

Replace or demote:

- the current flat competitive manager
- the price-vs-flow framing as the main research claim
- purely local search without pretraining
- scalar candidate scoring as the main critic


## First implementation tranche

The first implementation phase should not try to build the whole system at once.

### Tranche A: dataset-conditioned prior

Add:

- dataset embedder
- pool embedder
- skill token head
- synthetic recovery dataset generator

Goal:

- make the generator conditional on dataset and pool state

### Tranche B: planner + skill tree

Add:

- high-level planner over skill families and library slots
- new skill-family agent registry
- common-knowledge state encoder

Goal:

- replace the current flat competition with a hierarchical routing policy

### Tranche C: distributional collection critic

Add:

- quantile critic over delta library score and trade proxy
- uncertainty-aware shortlist
- replacement-aware value targets

Goal:

- score candidates by the distribution of collection outcomes

### Tranche D: memory distillation

Add:

- successful motif summarizer
- failure region summarizer
- retrieval-conditioned proposal bias

Goal:

- search behavior changes as the library grows


## Experimental target for the pivot

The paper should no longer rely only on gold timing.

The evidence chain should be:

1. **synthetic recovery**
   - formula recovery and search efficiency

2. **multi-market library discovery**
   - gold / crude oil / SP500

3. **transfer**
   - pretrain on source tasks, adapt to new target tasks

4. **ablation**
   - no grammar
   - no planner
   - no distributional critic
   - no memory
   - no dataset-conditioned prior

5. **library analysis**
   - redundancy
   - diversity
   - replacement behavior
   - regime-specific usage


## Immediate decision

Do not keep pushing the current project as a locally tuned competitive manager.

Use the current codebase as the execution substrate, but shift the research claim to:

- hierarchical library discovery
- dataset-conditioned symbolic priors
- distributional collection optimization
- retrieval-driven self-evolution

That gives the project a plausible path to a NeurIPS-level contribution. The current path alone does not.
