#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path
import sys

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from matplotlib.patches import FancyArrowPatch, Rectangle

from experiments.common import dataset_columns, load_dataset_bundle, load_experiment_name
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_metrics import cross_sectional_long_short_returns


BLUE = "#355CDE"
DARK = "#1F2937"
GRAY = "#94A3B8"
LIGHT_GRID = "#D1D5DB"
RED = "#C81E1E"
GREEN = "#0F766E"
ORANGE = "#D97706"
REPORT_FILES = {
    "main": "finance_signal_main_table.csv",
    "seed": "selector_filtering_seed_rows.csv",
    "benchmark": "selector_benchmark_pareto.csv",
    "clusters": "finance_near_neighbor_clusters.csv",
    "pseudoalpha": "finance_pseudoalpha_cases.csv",
}
EXPECTED_CANONICAL_SIGNAL = "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"
EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figure assets for the Chinese report draft.")
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIX Two Text", "Times New Roman", "DejaVu Serif", "STIXGeneral"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10.5
    plt.rcParams["xtick.labelsize"] = 9.5
    plt.rcParams["ytick.labelsize"] = 9.5
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = DARK
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def shorten_formula(formula: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(str(formula), width=width, break_long_words=False, break_on_hyphens=False))


def style_axis(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.set_facecolor("white")
    if ygrid:
        ax.grid(axis="y", linestyle="--", alpha=0.35, color=LIGHT_GRID)
    else:
        ax.grid(axis="x", linestyle="--", alpha=0.35, color=LIGHT_GRID)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.2,
            color=DARK,
        )
    )


def draw_minimal_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, subtitle: str) -> None:
    patch = Rectangle((x, y), w, h, linewidth=1.2, edgecolor=DARK, facecolor="white")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=12, color=DARK, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.35, subtitle, ha="center", va="center", fontsize=10.5, color="#4B5563")


def build_pipeline_figure(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_minimal_box(ax, 0.03, 0.24, 0.18, 0.50, "Anchor Generator", "Structured candidate pool")
    draw_minimal_box(ax, 0.27, 0.24, 0.22, 0.50, "Temporal Selector", "Slice performance, dispersion, turnover")
    draw_minimal_box(ax, 0.55, 0.24, 0.22, 0.50, "Cross-Seed Consensus", "Support penalties and neighbor pruning")
    draw_minimal_box(ax, 0.83, 0.24, 0.14, 0.50, "Canonical Signal", "Cash buffer + profitability")

    add_arrow(ax, (0.21, 0.49), (0.27, 0.49))
    add_arrow(ax, (0.49, 0.49), (0.55, 0.49))
    add_arrow(ax, (0.77, 0.49), (0.83, 0.49))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _dispersion_color(selected_signal: str, canonical_signal: str, sharpe: float) -> str:
    if selected_signal == canonical_signal:
        return BLUE
    if float(sharpe) < 0:
        return RED
    return GRAY


def build_seed_dispersion_figure(seed_rows: pd.DataFrame, main_table: pd.DataFrame, universe: str, title_index: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    style_axis(ax)

    subset = seed_rows[seed_rows["universe"] == universe].copy()
    canonical = main_table.loc[main_table["universe"] == universe, "signal"].iloc[0]
    consensus_sharpe = float(main_table.loc[main_table["universe"] == universe, "gross_sharpe"].iloc[0])
    colors = [_dispersion_color(row["selector_formula"], canonical, float(row["walk_sharpe"])) for _, row in subset.iterrows()]

    ax.bar(subset["seed"].astype(str), subset["walk_sharpe"], color=colors, edgecolor=DARK, linewidth=0.8)
    ax.axhline(consensus_sharpe, color=GREEN, linestyle=(0, (4, 2)), linewidth=1.8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Out-of-sample Sharpe")
    ax.text(
        0.02,
        0.96,
        f"Canonical signal:\n{shorten_formula(canonical, width=18)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": LIGHT_GRID},
    )
    ax.text(
        0.98,
        0.04,
        f"Dashed line = canonical Sharpe ({consensus_sharpe:.4f})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=GREEN,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_benchmark_accuracy_figure(pareto_table: pd.DataFrame, output_path: Path) -> None:
    label_map = {
        "pareto_cross_seed_consensus": "Continuous Pareto",
        "pareto_discrete_legacy": "Discrete Pareto",
        "legacy_linear_selector": "Linear consensus",
        "naive_rank_ic": "naive rank-IC",
        "best_validation_mean_rank_ic": "validation mean rank-IC",
        "pareto_front_selector": "static Pareto frontier",
        "lasso_formula_screening": "LASSO",
        "single_seed_temporal_selector": "single-seed temporal",
        "pareto_discrete_temporal_selector": "single-seed discrete Pareto",
        "cross_seed_mean_score_consensus": "mean-score consensus",
        "best_validation_sharpe": "validation Sharpe",
    }
    color_map = {
        "legacy_linear_selector": BLUE,
        "pareto_cross_seed_consensus": RED,
        "naive_rank_ic": GREEN,
    }

    text_offsets = {
        "pareto_cross_seed_consensus": (0.004, 0.004),
        "pareto_discrete_legacy": (0.004, 0.009),
        "legacy_linear_selector": (0.004, -0.006),
        "naive_rank_ic": (0.004, 0.002),
        "best_validation_mean_rank_ic": (0.004, 0.008),
        "pareto_front_selector": (0.004, -0.002),
        "lasso_formula_screening": (0.004, 0.004),
        "single_seed_temporal_selector": (0.004, 0.004),
        "pareto_discrete_temporal_selector": (0.004, -0.004),
        "cross_seed_mean_score_consensus": (0.004, 0.004),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style_axis(ax, ygrid=False)
    ax.grid(axis="both", linestyle="--", alpha=0.35, color=LIGHT_GRID)

    for _, row in pareto_table.iterrows():
        baseline = row["baseline"]
        x = float(row["average_accuracy"])
        y = float(row["selected_formula_stability"])
        if baseline == "best_validation_sharpe":
            continue
        is_frontier = str(row["pareto_frontier"]).lower() == "true"
        color = color_map.get(baseline, GRAY)
        size = 72 if is_frontier else 52
        edge = DARK if is_frontier else "#64748B"
        ax.scatter(x, y, s=size, color=color, edgecolor=edge, linewidth=1.0, zorder=3)
        dx, dy = text_offsets.get(baseline, (0.004, 0.004))
        ax.text(x + dx, y + dy, label_map.get(baseline, baseline), fontsize=8.4, color=DARK)

    frontier = pareto_table[
        (pareto_table["pareto_frontier"].astype(str).str.lower() == "true")
        & (pareto_table["baseline"] != "best_validation_sharpe")
    ].copy()
    frontier = frontier.sort_values(["average_accuracy", "selected_formula_stability"])
    ax.plot(frontier["average_accuracy"], frontier["selected_formula_stability"], color="#475569", linewidth=1.2, linestyle=(0, (4, 2)))

    ax.set_xlim(0.73, 0.95)
    ax.set_ylim(0.78, 1.03)
    ax.set_xlabel("Average selection accuracy")
    ax.set_ylabel("Selected-signal stability")
    ax.text(
        0.01,
        0.02,
        "Validation-Sharpe baseline (0.30, 0.55) is omitted from the axis range for readability.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.97)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_case_study_figure(cluster_rows: pd.DataFrame, output_path: Path) -> None:
    subset = cluster_rows[cluster_rows["universe"] == "liquid500"].copy()
    subset = subset.sort_values(["is_canonical", "consensus_pareto_rank", "champion_seed_support", "selector_seed_support"], ascending=[False, True, False, False]).head(4)
    subset = subset.iloc[::-1]

    alias_map = {
        EXPECTED_CANONICAL_SIGNAL: "Canonical",
        "CASH_RATIO_Q RANK": "Cash only",
        "PROFITABILITY_Q RANK": "Profit only",
        "PROFITABILITY_A CASH_RATIO_Q RANK ADD": "Annual mix",
    }
    labels = [alias_map.get(formula, shorten_formula(formula, width=18)) for formula in subset["formula"]]
    champion = subset["champion_seed_support"].to_numpy(dtype=float)
    selector = subset["selector_seed_support"].to_numpy(dtype=float)
    candidate = subset["candidate_seed_support"].to_numpy(dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    style_axis(ax, ygrid=False)
    ax.barh(y + 0.18, candidate, height=0.22, color=GRAY, edgecolor=DARK, linewidth=0.7, label="candidate support")
    ax.barh(y, selector, height=0.22, color=GREEN, edgecolor=DARK, linewidth=0.7, label="selector support")
    ax.barh(y - 0.18, champion, height=0.22, color=BLUE, edgecolor=DARK, linewidth=0.7, label="champion support")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Support count (5 seeds)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3)

    ax.text(
        0.01,
        0.02,
        "The canonical row is the only neighbor that preserves both cash buffer and quarterly profitability.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#4B5563",
    )

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.88)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compute_market_proxy(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=["date", "RET_1"])
    panel["date"] = pd.to_datetime(panel["date"])
    market = panel.groupby("date", as_index=False)["RET_1"].mean().rename(columns={"RET_1": "ew_market_ret"})
    market = market.sort_values("date")
    market["rolling_60_cumret"] = (1.0 + market["ew_market_ret"]).rolling(60, min_periods=20).apply(np.prod, raw=True) - 1.0
    market["rolling_20_vol"] = market["ew_market_ret"].rolling(20, min_periods=20).std(ddof=0)
    vol_median = float(market["rolling_20_vol"].median(skipna=True))
    market["trend_state"] = np.where(market["rolling_60_cumret"] >= 0.0, "bull", "bear")
    market["vol_state"] = np.where(market["rolling_20_vol"] >= vol_median, "high_vol", "low_vol")
    return market


def load_formula_test_path(universe: str, formula: str) -> tuple[pd.Series, pd.Series]:
    bundle = load_dataset_bundle(Path(f"configs/us_equities_{universe}.yaml"))
    frame = bundle.splits.test.copy()
    dataset_name = load_experiment_name(EXPERIMENT_CONFIG)
    _, return_column = dataset_columns(dataset_name)
    evaluator = CrossSectionalFormulaEvaluator()
    evaluated = evaluator.evaluate(formula, frame)
    signal = evaluated.signal.astype(float)
    returns, _ = cross_sectional_long_short_returns(
        signal,
        frame.loc[signal.index, return_column],
        frame.loc[signal.index, "date"],
        frame.loc[signal.index, "permno"],
        quantile=0.2,
    )
    active = (
        pd.Series(signal.notna().astype(float), index=signal.index)
        .groupby(pd.to_datetime(frame.loc[signal.index, "date"]), sort=True)
        .sum()
        .sort_index()
    )
    return returns, active


def build_cumulative_return_figure(output_path: Path) -> None:
    main_table = pd.read_csv(Path("outputs/reports") / REPORT_FILES["main"])
    pseudoalpha = pd.read_csv(Path("outputs/reports") / REPORT_FILES["pseudoalpha"])
    universe = "liquid1000"
    canonical = str(main_table.loc[main_table["universe"] == universe, "signal"].iloc[0])
    pseudo = str(
        pseudoalpha[
            (pseudoalpha["universe"] == universe)
            & (pseudoalpha["source"] == "naive_rank_ic")
        ]["candidate_formula"].iloc[0]
    )
    canonical_returns, canonical_active = load_formula_test_path(universe, canonical)
    pseudo_returns, pseudo_active = load_formula_test_path(universe, pseudo)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.4]})
    ax_top, ax_bottom = axes
    style_axis(ax_top)
    style_axis(ax_bottom)

    canonical_cum = (1.0 + canonical_returns).cumprod() - 1.0
    pseudo_cum = (1.0 + pseudo_returns).cumprod() - 1.0
    ax_top.plot(canonical_cum.index, canonical_cum, linewidth=2.0, color=BLUE, label="canonical signal $F^*$")
    ax_top.plot(pseudo_cum.index, pseudo_cum, linewidth=1.8, color=RED, label="pseudo-winner: leverage+cash+profit corr")
    ax_top.set_ylabel("Cumulative long-short return")
    ax_top.legend(frameon=False, loc="upper left")

    canonical_active_roll = canonical_active.rolling(20, min_periods=1).mean()
    pseudo_active_roll = pseudo_active.rolling(20, min_periods=1).mean()
    ax_bottom.plot(canonical_active_roll.index, canonical_active_roll, linewidth=1.8, color=BLUE)
    ax_bottom.plot(pseudo_active_roll.index, pseudo_active_roll, linewidth=1.8, color=RED)
    ax_bottom.set_ylabel("20-day avg. active names")
    ax_bottom.set_xlabel("Date")
    ax_bottom.text(
        0.58,
        0.18,
        "The red series is driven by a tiny set of dates and names.\nIf either side has fewer than 5 names, the daily return is fused to zero.",
        transform=ax_bottom.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#4B5563",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": LIGHT_GRID, "alpha": 0.92},
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.98, hspace=0.18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_market_state_figure(output_path: Path) -> None:
    frame = pd.read_csv(Path("outputs/reports/chinese_report_market_state_table.csv"))
    label_map = {"high_vix": "High VIX state", "normal_vix": "Normal state"}
    frame["state_label"] = frame["state"].map(label_map).fillna(frame["state"])

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    style_axis(ax)
    pivot = frame.pivot(index="state_label", columns="universe", values="sharpe").loc[["High VIX state", "Normal state"]]
    x = np.arange(len(pivot.index))
    width = 0.34
    ax.bar(x - width / 2, pivot["liquid500"], width=width, color=BLUE, edgecolor=DARK, linewidth=0.7, label="liquid500")
    ax.bar(x + width / 2, pivot["liquid1000"], width=width, color=GREEN, edgecolor=DARK, linewidth=0.7, label="liquid1000")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Sharpe")
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.97)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    main_table = pd.read_csv(args.reports_root / REPORT_FILES["main"])
    try:
        seed_rows = pd.read_csv(args.reports_root / REPORT_FILES["seed"])
    except EmptyDataError:
        seed_rows = pd.DataFrame()
    pareto_table = pd.read_csv(args.reports_root / REPORT_FILES["benchmark"])
    cluster_rows = pd.read_csv(args.reports_root / REPORT_FILES["clusters"])

    mismatched = main_table.loc[main_table["signal"] != EXPECTED_CANONICAL_SIGNAL, ["universe", "signal"]]
    if not mismatched.empty:
        payload = ", ".join(f"{row.universe}={row.signal}" for row in mismatched.itertuples(index=False))
        raise ValueError(f"Historical cumulative/state return sources only match the legacy canonical signal; got {payload}.")

    build_pipeline_figure(args.output_dir / "chinese_report_figure_1_pipeline.png")
    if not seed_rows.empty:
        build_seed_dispersion_figure(seed_rows, main_table, "liquid500", 2, args.output_dir / "chinese_report_figure_2_liquid500_seed_dispersion.png")
        build_seed_dispersion_figure(seed_rows, main_table, "liquid1000", 3, args.output_dir / "chinese_report_figure_3_liquid1000_seed_dispersion.png")
    build_benchmark_accuracy_figure(pareto_table, args.output_dir / "chinese_report_figure_4_benchmark_accuracy.png")
    build_case_study_figure(cluster_rows, args.output_dir / "chinese_report_figure_5_selector_case_study.png")
    build_cumulative_return_figure(args.output_dir / "chinese_report_figure_6_cumulative_return.png")
    build_market_state_figure(args.output_dir / "chinese_report_figure_7_market_state.png")


if __name__ == "__main__":
    main()
