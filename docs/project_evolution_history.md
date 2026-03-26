# 项目演进全记录（从立项到当前主线）

本文档记录项目从创立以来的目标、关键问题、决策与结果。重点不是复述代码细节，而是说明“为什么要改、改了什么、结果如何”。

## 0. 立项阶段：最初目标

### 原始目标
- 自动发现可解释的白盒量化公式（RPN 符号表达）。
- 初始任务以单资产时序为主，希望通过多智能体分工提高样本外表现。
- 目标叙事是“多 agent 协同发现 alpha”。

### 最初结构
- 顶层 manager 负责任务分配与提交。
- 下游按特征族拆 agent（price / flow / context）。
- 评价链路以 pool gain、trade proxy、validation preview 为主。

---

## 1. 第一轮工程打通：框架可跑但结果退化

### 做成了什么
- 统一了语言层（grammar / parser / evaluator）。
- 打通了 pool-based reward、admission、review 机制。
- 跑通了多 seed 训练与回测流程。

### 暴露的问题
- `full` 经常退化到单一弱分支或单一简单公式。
- manager 会在某些路径上错误放行/错误拒绝，导致“训练接受但样本外更差”。
- 指标口径存在不一致：训练阶段 proxy 与最终回测目标不完全对齐。

### 结论
- 当时的主要矛盾是工程结构与选择逻辑，不是单纯算力或轮数问题。

---

## 2. 方法转向：从单资产时序 转向美股横截面

### 触发原因
- 单资产日频场景下，多 skill 协同收益长期不稳定。
- 有效信号区域过窄，框架复杂度难以转化为稳定增益。

### 关键决策
- 主任务转为美股横截面因子发现。
- 保留符号发现与层级选择框架，重建数据与评估口径。

### 落地内容
- 建立了 WRDS+公共数据的数据协议。
- 构建了 U.S. equities panel（split + subset）。
- 引入横截面评价（RankIC、long-short、turnover、stability）。

---

## 3. 迁移期主问题：训练目标与回测目标错位

### 现象
- 某些公式在训练/valid 指标看起来可接受，但 walk-forward 不理想。
- 组合公式在 valid 提升，但 test 退化。

### 修复路径
- 修正了横截面回测路径（避免误走单资产逻辑）。
- 修正了按行切分而非按交易日切分等实现问题。
- 强化了 validation-backed commit/replacement。
- 加强了 residual gate，避免有害 short-horizon flow 污染主池。

### 结果
- 核心正结果出现并稳定复现。
- 但“full 明显优于强 baseline”仍未成立。

---

## 4. 多 agent 协同假设的实证检验与否定

### 做过的验证
- 拆分 slow family（quality/efficiency/valuation）并做分阶段调度。
- 强化 second-upgrade lane、replacement-first lane。
- 多 seed、subset 扩展（liquid500/liquid1000）持续检验。

### 实证结论
- 最稳定且可复现的主结果来自 `quality_solvency`。
- challenger（`efficiency_growth`、`valuation_size`、`short_horizon_flow`）有局部信号，但未证明稳定边际增益。
- 因此“多 agent 协同是主要收益来源”不成立。

---

## 5. 主叙事重构：anchor-agent 为主体

### 现在的正式主线
- 主体：`quality_solvency` 作为 anchor agent。
- 其他 agent：作为 challenger，用于替换/增量假设验证，而非主收益来源。
- manager 的价值：validation-backed selection、replacement-first admission、risk gate。

### 当前主结果（摘要）
- 稳定冠军公式：`CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD`。
- liquid500 多 seed 为正且稳定。
- liquid1000 更严格子集仍为正。

### 当前未解决问题
- 直接公式对比中，`cash + sales_to_assets` 在部分评估窗口更强。
- 训练选择仍偏向 `cash + profitability`，存在 selection mismatch。
- 这属于“稳健选择策略”问题，不是多 agent 架构问题。

---

## 6. 当前项目结构（用于后续论文）

### 你现在可以明确主张的内容
- 知识约束的白盒符号因子发现框架。
- anchor-agent 主导的发现与升级机制。
- validation-backed + replacement-first + gate 的选择链路。

### 你现在不应主张的内容
- 多 agent 协同是主要收益来源。
- full 系统稳定显著优于 anchor baseline。

---

## 7. 决策总结（给后续写作和迭代）

### 已确认有效
- anchor 主线（quality_solvency）。
- validation-backed selection / replacement。
- 对 challenger 的严格 residual gating。

### 已确认边际收益低
- 继续堆多 agent 协同叙事。
- 仅靠增加 episode 解决选择偏差。

### 后续迭代重点
- 用更稳健的多窗口选择规则缩小 `cash+profit` 与 `cash+sales` 的选择偏差。
- 把 challenger 定位为“验证与反证工具”，而非主结果依赖。
