#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import textwrap

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


REPORT_ROOT = Path("outputs/reports")
OUTPUT_DIR = Path("docs/table_pdfs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper tables as standalone PDF files for Word insertion.")
    parser.add_argument("--reports-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def wrap_text(value: object, width: int = 20) -> str:
    text = str(value)
    if text in {"nan", "None"}:
        return "—"
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def render_table_pdf(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    col_widths: list[float],
    font_size: float = 9.0,
    title_size: float = 12.5,
    scale_y: float = 1.55,
    first_col_left: bool = False,
) -> None:
    n_rows = len(df)
    fig_height = max(2.6, 1.0 + 0.48 * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    ax.axis("off")
    ax.text(0.5, 1.03, title, ha="center", va="bottom", fontsize=title_size, fontweight="bold", transform=ax.transAxes)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 0.95],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)

    n_cols = len(df.columns)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        if row == 0:
            cell.set_facecolor("#F1F5F9")
            cell.set_text_props(weight="bold")
            cell.set_linewidth(0.8)
        else:
            cell.set_linewidth(0.45)
        if first_col_left and row > 0 and col == 0:
            cell.get_text().set_ha("left")
        if row > 0 and col < n_cols and isinstance(df.iloc[row - 1, col], str) and "\n" in str(df.iloc[row - 1, col]):
            cell.get_text().set_va("center")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_table1(reports_root: Path) -> pd.DataFrame:
    src = pd.read_csv(reports_root / "chinese_report_main_results_journal.csv")
    rows = []
    for _, row in src.iterrows():
        rows.append(
            {
                "宇宙": row["universe"],
                "信号": "F*",
                "EW Gross Sharpe": f'{row["ew_gross_sharpe"]:.4f}',
                "EW Net Sharpe(15bps)": f'{row["ew_net_sharpe_15bps"]:.4f}',
                "EW NW t": f'{row["ew_nw_t"]:.4f}',
                "EW FF5 α(年化%)": f'{row["ew_ff5_alpha_ann_pct"]:.2f}',
                "EW FF5 α t": f'{row["ew_ff5_alpha_t"]:.4f}',
                "VW Gross Sharpe": f'{row["vw_gross_sharpe"]:.4f}',
                "VW Net Sharpe(15bps)": f'{row["vw_net_sharpe_15bps"]:.4f}',
                "VW FF5 α(年化%)": f'{row["vw_ff5_alpha_ann_pct"]:.2f}',
                "VW FF5 α t": f'{row["vw_ff5_alpha_t"]:.4f}',
                "EW Net Sharpe(100bps)": f'{row["ew_net_sharpe_100bps"]:.4f}',
                "覆盖率": f'{row["signal_non_null_fraction"]:.4f}',
                "换手": f'{row["turnover"]:.4f}',
            }
        )
    return pd.DataFrame(rows)


def build_table2(reports_root: Path) -> pd.DataFrame:
    src = pd.read_csv(reports_root / "finance_size_double_sort.csv")
    rows = []
    for universe in ("liquid500", "liquid1000"):
        subset = src[(src["universe"] == universe) & (src["weight_scheme"] == "value")].copy()
        subset = subset.set_index("size_bucket").loc[["Small", "Mid", "Large"]]
        rows.extend(
            [
                {
                    "宇宙": universe,
                    "指标": "VW Net Sharpe",
                    "Small": f'{subset.loc["Small", "net_sharpe_15bps"]:.4f}',
                    "Mid": f'{subset.loc["Mid", "net_sharpe_15bps"]:.4f}',
                    "Large": f'{subset.loc["Large", "net_sharpe_15bps"]:.4f}',
                },
                {
                    "宇宙": universe,
                    "指标": "VW Net FF5 α(年化%)",
                    "Small": f'{100 * subset.loc["Small", "net_ff5_alpha_ann_15bps"]:.2f}',
                    "Mid": f'{100 * subset.loc["Mid", "net_ff5_alpha_ann_15bps"]:.2f}',
                    "Large": f'{100 * subset.loc["Large", "net_ff5_alpha_ann_15bps"]:.2f}',
                },
                {
                    "宇宙": universe,
                    "指标": "VW Net FF5 α t",
                    "Small": f'{subset.loc["Small", "net_ff5_alpha_t_15bps"]:.4f}',
                    "Mid": f'{subset.loc["Mid", "net_ff5_alpha_t_15bps"]:.4f}',
                    "Large": f'{subset.loc["Large", "net_ff5_alpha_t_15bps"]:.4f}',
                },
            ]
        )
    return pd.DataFrame(rows)


def build_table3(reports_root: Path) -> pd.DataFrame:
    sparse = pd.read_csv(reports_root / "finance_sparse_signal_diagnostics.csv")
    alias = {
        ("liquid500", "shap_ranked_formula_screening"): ("F_cash", "近邻简化"),
        ("liquid500", "lasso_formula_screening"): ("PROFITABILITY_Q CASH_RATIO_Q\nRANK ADD", "近邻简化"),
        ("liquid500", "naive_rank_ic"): ("PROFITABILITY_Q ... CORR_5 ADD", "稀疏退化"),
        ("liquid1000", "shap_ranked_formula_screening"): ("F_cash", "近邻简化"),
        ("liquid1000", "pysr_pareto_selection"): ("VOLATILITY_20 RANK NEG", "目标错位"),
        ("liquid1000", "naive_rank_ic"): ("LEVERAGE_Q ... CORR_5", "熔断后不可交易"),
    }
    order = [
        ("liquid500", "shap_ranked_formula_screening", "SHAP"),
        ("liquid500", "lasso_formula_screening", "LASSO"),
        ("liquid500", "naive_rank_ic", "naive / 验证均值"),
        ("liquid1000", "shap_ranked_formula_screening", "SHAP"),
        ("liquid1000", "pysr_pareto_selection", "PySR"),
        ("liquid1000", "naive_rank_ic", "naive / 验证均值"),
    ]
    rows = []
    for universe, baseline, source_label in order:
        row = sparse[(sparse["universe"] == universe) & (sparse["baseline"] == baseline)].iloc[0]
        formula_label, explanation = alias[(universe, baseline)]
        sharpe = "—" if pd.isna(row["gross_sharpe"]) else f'{row["gross_sharpe"]:.4f}'
        rank_ic = "—" if pd.isna(row["mean_test_rank_ic"]) else f'{row["mean_test_rank_ic"]:.4f}'
        rows.append(
            {
                "宇宙": universe,
                "来源": source_label,
                "候选信号": formula_label,
                "Test Sharpe": sharpe,
                "Test Rank-IC": rank_ic,
                "覆盖率": f'{row["signal_non_null_fraction"]:.6f}',
                "可交易日占比": f'{row["tradable_date_fraction"]:.4f}',
                "中位有效股票数": f'{row["median_active_names_per_day"]:.0f}',
                "解释": explanation,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    table1 = build_table1(args.reports_root)
    table2 = build_table2(args.reports_root)
    table3 = build_table3(args.reports_root)

    render_table_pdf(
        table1,
        "表1  美国股票横截面典型信号主结果（纯测试期，执行覆盖度熔断）",
        args.output_dir / "table1_main_results.pdf",
        col_widths=[0.07, 0.07, 0.085, 0.10, 0.07, 0.09, 0.07, 0.085, 0.10, 0.09, 0.07, 0.10, 0.06, 0.055],
        font_size=8.0,
        scale_y=1.48,
    )
    render_table_pdf(
        table2,
        "表2  基于公司规模与信号强度的双重排序结果（15bps 净口径）",
        args.output_dir / "table2_double_sort.pdf",
        col_widths=[0.18, 0.28, 0.18, 0.18, 0.18],
        font_size=9.5,
        scale_y=1.58,
    )
    render_table_pdf(
        table3,
        "表3  纯测试期上的典型伪赢家与近邻退化结构",
        args.output_dir / "table3_pseudoalpha.pdf",
        col_widths=[0.09, 0.12, 0.23, 0.08, 0.09, 0.09, 0.10, 0.08, 0.12],
        font_size=8.3,
        scale_y=1.56,
    )

    combined_path = args.output_dir / "chinese_report_tables.pdf"
    with PdfPages(combined_path) as pdf:
        for df, title, widths, font_size, scale_y in [
            (table1, "表1  美国股票横截面典型信号主结果（纯测试期，执行覆盖度熔断）", [0.07, 0.07, 0.085, 0.10, 0.07, 0.09, 0.07, 0.085, 0.10, 0.09, 0.07, 0.10, 0.06, 0.055], 8.0, 1.48),
            (table2, "表2  基于公司规模与信号强度的双重排序结果（15bps 净口径）", [0.18, 0.28, 0.18, 0.18, 0.18], 9.5, 1.58),
            (table3, "表3  纯测试期上的典型伪赢家与近邻退化结构", [0.09, 0.12, 0.23, 0.08, 0.09, 0.09, 0.10, 0.08, 0.12], 8.3, 1.56),
        ]:
            n_rows = len(df)
            fig_height = max(2.6, 1.0 + 0.48 * (n_rows + 1))
            fig, ax = plt.subplots(figsize=(11.2, fig_height))
            ax.axis("off")
            ax.text(0.5, 1.03, title, ha="center", va="bottom", fontsize=12.5, fontweight="bold", transform=ax.transAxes)
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                cellLoc="center",
                colLoc="center",
                colWidths=widths,
                bbox=[0.0, 0.0, 1.0, 0.95],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)
            table.scale(1.0, scale_y)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#222222")
                if row == 0:
                    cell.set_facecolor("#F1F5F9")
                    cell.set_text_props(weight="bold")
                    cell.set_linewidth(0.8)
                else:
                    cell.set_linewidth(0.45)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
