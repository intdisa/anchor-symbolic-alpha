# 中文报告会图表说明

本目录存放 `/Users/xieshangchen/Documents/New project/docs/chinese_report_draft.md` 使用的图形资产。

重建命令：

```bash
python3 scripts/build_chinese_report_figures.py
```

数据来源：

- `chinese_report_figure_1_pipeline.png`
  - 方法示意图，依据正文方法部分手工绘制。
- `chinese_report_figure_2_liquid500_seed_dispersion.png`
  - 来源：`/Users/xieshangchen/Documents/New project/outputs/reports/us_equities_paper_results_seed_dispersion.csv`
  - 辅助来源：`/Users/xieshangchen/Documents/New project/outputs/reports/us_equities_paper_results_main_table.csv`
- `chinese_report_figure_3_liquid1000_seed_dispersion.png`
  - 来源：`/Users/xieshangchen/Documents/New project/outputs/reports/us_equities_paper_results_seed_dispersion.csv`
  - 辅助来源：`/Users/xieshangchen/Documents/New project/outputs/reports/us_equities_paper_results_main_table.csv`
- `chinese_report_figure_4_benchmark_accuracy.png`
  - 来源：`/Users/xieshangchen/Documents/New project/outputs/reports/us_equities_paper_results_benchmark_table.csv`
- `chinese_report_figure_5_selector_case_study.png`
  - 来源：`/Users/xieshangchen/Documents/New project/outputs/reports/us_equities_paper_results_selector_case_studies.json`
- `chinese_report_figure_6_cumulative_return.png`
  - 来源：`/Users/xieshangchen/Documents/New project/outputs/runs/liquid500_selector_subset_e20_r2__backtest/reports/us_equities_walk_forward_returns.csv`
  - 来源：`/Users/xieshangchen/Documents/New project/outputs/runs/liquid1000_selector_subset_e20_r2__backtest/reports/us_equities_walk_forward_returns.csv`
- `chinese_report_figure_7_market_state.png`
  - 来源：`/Users/xieshangchen/Documents/New project/outputs/reports/chinese_report_market_state_table.csv`

这些图只服务于中文报告会稿，不改变实验主结果。
