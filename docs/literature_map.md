# Literature Map

This file is a working reference map for later paper writing. It is not a final
bibliography. The goal is to keep one place for:

- the user's original paper set
- the later top-journal / top-conference references
- how each paper should be used in the Route B paper

Before final submission, confirm page numbers, authors, and BibTeX against the
publisher pages or PDF metadata.

## A. Original paper set

### 1. `1805_AlphaQCM_Alpha_Discovery_.pdf`

- title: `AlphaQCM: Alpha Discovery in Finance with Distributional Reinforcement Learning`
- authors: Zhoufan Zhu, Ke Zhu
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/1805_AlphaQCM_Alpha_Discovery_.pdf`
- role in our paper:
  - motivates collection-level evaluation instead of isolated single-formula reward
  - relevant when explaining why scalar train IC is not enough
- best section:
  - related work
  - method motivation for library-level or replacement-aware critic design

### 2. `2601.22119v1.pdf`

- title: `Alpha Discovery via Grammar-Guided Learning and Search`
- short name: `AlphaCFG`
- authors: Han Yang, Dong Hao, Zhuohan Wang, Qi Shi, Xingtong Li
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/2601.22119v1.pdf`
- role in our paper:
  - supports grammar-constrained symbolic search
  - relevant for strict syntax / semantics / bounded search space
- best section:
  - related work for symbolic search
  - method section on strict RPN search space

### 3. `2602.14670v1.pdf`

- title: `FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery`
- authors: Yanlong Wang, Jian Xu, Hongkang Zhang, Shao-Lun Huang, Danny Dongning Sun, Xiao-Ping Zhang
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/2602.14670v1.pdf`
- role in our paper:
  - supports retrieve-generate-evaluate-distill loops
  - supports experience memory and failure-region reuse
- best section:
  - related work
  - discussion of search memory and proposal shaping

### 4. `15375_AlphaFormer_End_to_End_S.pdf`

- title: `AlphaFormer: End-to-End Symbolic Regression of Alpha Factors with Transformers`
- venue status: under review as an ICLR 2026 submission in the current PDF
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/15375_AlphaFormer_End_to_End_S.pdf`
- role in our paper:
  - motivates dataset-conditioned proposal priors
  - relevant if we keep any pretraining / transformer generator discussion
- best section:
  - related work on learned formula proposal priors

### 5. `3580305.3599831.pdf`

- title: `Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning`
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/3580305.3599831.pdf`
- role in our paper:
  - supports optimizing alpha sets rather than isolated alphas
  - directly relevant to library-level reward and collection evaluation
- best section:
  - related work
  - discussion of why library-level selection matters

### 6. `NeurIPS-2019-multi-agent-common-knowledge-reinforcement-learning-Paper.pdf`

- title: `Multi-Agent Common Knowledge Reinforcement Learning`
- authors: Christian Schroeder de Witt, Jakob Foerster, Gregory Farquhar, Philip H. S. Torr, Wendelin Boehmer, Shimon Whiteson
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/NeurIPS-2019-multi-agent-common-knowledge-reinforcement-learning-Paper.pdf`
- role in our paper:
  - supports hierarchical coordination and common-knowledge state design
  - now more useful as architectural background than as the main claim
- best section:
  - related work on manager/controller design

### 7. `NeurIPS-2024-integrating-suboptimal-human-knowledge-with-hierarchical-reinforcement-learning-for-large-scale-multiagent-systems-Paper-Conference.pdf`

- title: `Integrating Suboptimal Human Knowledge with Hierarchical Reinforcement Learning for Large-Scale Multiagent Systems`
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/NeurIPS-2024-integrating-suboptimal-human-knowledge-with-hierarchical-reinforcement-learning-for-large-scale-multiagent-systems-Paper-Conference.pdf`
- role in our paper:
  - supports soft knowledge priors instead of hard-coded trading rules
  - still useful if we discuss knowledge-guided search
- best section:
  - related work
  - design rationale for soft priors / validation-backed guidance

### 8. `electronics-14-03001.pdf`

- title: `MA-HRL: Multi-Agent Hierarchical Reinforcement Learning for Medical Diagnostic Dialogue Systems`
- authors: Xingchuang Liao et al.
- source file:
  - `/Users/xieshangchen/Desktop/new gold/gold文献/electronics-14-03001.pdf`
- role in our paper:
  - generic hierarchical RL inspiration only
  - not a core finance citation
- best section:
  - optional method background
- note:
  - this is not a finance paper, so it should not be a central citation in the final version

## B. Added top-journal / top-conference references

### 1. Gu, Kelly, Xiu (2020)

- title: `Empirical Asset Pricing via Machine Learning`
- venue: `Review of Financial Studies`
- use:
  - canonical modern ML asset-pricing baseline
  - supports cross-sectional return prediction framing and stronger baseline design

### 2. Giglio, Kelly, Xiu (2022)

- title: `Factor Models, Machine Learning, and Asset Pricing`
- venue: `Annual Review of Financial Economics`
- use:
  - high-level survey for introduction and positioning
  - helps frame where symbolic factor discovery fits relative to ML asset pricing

### 3. Giglio, Liao, Xiu (2021)

- title: `Thousands of Alpha Tests`
- venue: `Review of Financial Studies`
- use:
  - supports the need for rigorous validation and protection against data snooping
  - important for defending validation-backed selection and conservative replacement

### 4. Chao Zhang (2025)

- title: `Alpha Go Everywhere: Machine Learning and International Stock Returns`
- venue: `Review of Asset Pricing Studies`
- use:
  - supports wider-universe and cross-market generalization arguments
  - useful if the paper later extends beyond one U.S. subset

### 5. Guanhao Feng (2020)

- title: `Firm Characteristics, Cross-Sectional Regression Estimates, and Asset Pricing Tests`
- venue: `Review of Asset Pricing Studies`
- use:
  - supports characteristic-based cross-sectional structure
  - useful for interpreting the `cash + quality` anchor formula economically

### 6. Gagliardini, Ossola, Scaillet (2020)

- title: `Testing Beta-Pricing Models Using Large Cross-Sections`
- venue: `Review of Financial Studies`
- use:
  - supports rigorous large-cross-section evaluation logic
  - useful for framing why larger universes strengthen the result

### 7. Carlos Stein Brito (ICML 2025)

- title: `Cross-regularization: Adaptive Model Complexity through Validation Gradients`
- use:
  - not a finance paper
  - useful for motivating validation-driven model selection and complexity control

### 8. Duarte (2024)

- title: `Machine Learning for Continuous-Time Finance`
- venue: `Review of Financial Studies`
- use:
  - optional broader ML-finance reference
  - useful if the paper needs a wider method-context paragraph

## C. How the literature now maps to the revised paper narrative

### Core narrative

The revised Route B paper should be positioned around:

- anchor-agent symbolic factor discovery
- validation-backed selection
- replacement-first baseline upgrades
- challenger-based verification

### Best papers for each claim

#### Symbolic / grammar-constrained discovery

- AlphaCFG
- AlphaFormer
- FactorMiner

#### Library-level or collection-level evaluation

- AlphaQCM
- Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning
- Thousands of Alpha Tests

#### Cross-sectional asset-pricing context

- Empirical Asset Pricing via Machine Learning
- Factor Models, Machine Learning, and Asset Pricing
- Firm Characteristics, Cross-Sectional Regression Estimates, and Asset Pricing Tests
- Testing Beta-Pricing Models Using Large Cross-Sections

#### Validation / replacement / verification rationale

- Thousands of Alpha Tests
- Cross-regularization
- AlphaQCM

## D. Current citation strategy

### Keep as central citations

- AlphaQCM
- AlphaCFG
- FactorMiner
- Gu, Kelly, Xiu (2020)
- Giglio, Kelly, Xiu (2022)
- Giglio, Liao, Xiu (2021)

### Keep as supporting citations

- AlphaFormer
- Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning
- MACKRL
- Integrating Suboptimal Human Knowledge with Hierarchical RL

### Keep as optional / light-touch citations

- MA-HRL in medical dialogue systems
- Duarte (2024)
- cross-market extensions such as `Alpha Go Everywhere`
