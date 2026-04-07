#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import (
    dataset_columns,
    load_dataset_bundle,
    load_experiment_name,
)
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_metrics import (
    cross_sectional_ic_summary,
    cross_sectional_long_short_returns,
    cross_sectional_turnover,
)
from knowledge_guided_symbolic_alpha.evaluation.finance_reporting import (
    compute_significance_metrics,
    cost_adjusted_returns,
    load_fama_french_factors,
    return_concentration_metrics,
    signal_coverage_metrics,
    summarize_returns,
)


SPLIT_WINDOWS = {
    "train_start": "2010-01-04",
    "train_end": "2014-12-31",
    "valid_start": "2015-01-02",
    "valid_end": "2018-12-31",
    "test_start": "2019-01-02",
    "test_end": "2025-12-30",
}
BASELINE_FILES = (
    "finance_walkforward_baselines_pareto.csv",
    "finance_walkforward_baselines_extended.csv",
    "finance_walkforward_baselines.csv",
)
CONSENSUS_BASELINES = (
    "legacy_linear_selector",
    "support_adjusted_cross_seed_consensus",
    "pareto_discrete_legacy",
    "pareto_cross_seed_consensus",
)
EXPECTED_CANONICAL_SIGNAL = "CASH_RATIO_Q RANK PROFITABILITY_Q RANK ADD"
EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
PANEL_PATH = Path("data/processed/us_equities/us_equities_panel.parquet")
FRED_MACRO_PATH = Path("data/raw/us_equities/public/fred_macro_daily.csv.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Word-friendly CSV tables for the Chinese signal-selection report.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports"))
    return parser.parse_args()


def load_baseline_frame(output_root: Path) -> pd.DataFrame:
    for name in BASELINE_FILES:
        path = output_root / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No finance walk-forward baseline report found.")


def consensus_row(frame: pd.DataFrame, universe: str) -> pd.Series:
    subset = frame[frame["universe"] == universe]
    for baseline in CONSENSUS_BASELINES:
        match = subset[subset["baseline"] == baseline]
        if not match.empty:
            return match.iloc[0]
    raise KeyError(f"No consensus baseline found for {universe}.")


def compute_market_proxy(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=["date", "RET_1"])
    panel["date"] = pd.to_datetime(panel["date"])
    market = panel.groupby("date", as_index=False).agg(ew_market_ret=("RET_1", "mean")).sort_values("date")
    fred = pd.read_csv(FRED_MACRO_PATH, usecols=["date", "VIXCLS"])
    fred["date"] = pd.to_datetime(fred["date"])
    fred["vix"] = pd.to_numeric(fred["VIXCLS"], errors="coerce")
    market = market.merge(fred[["date", "vix"]], on="date", how="left")
    market = market.sort_values("date")
    market["vix_threshold"] = (
        market["vix"]
        .rolling(252, min_periods=126)
        .quantile(0.8)
        .shift(1)
    )
    market["vix_state"] = np.where(market["vix"] >= market["vix_threshold"], "high_vix", "normal_vix")
    return market


def _variant_column(frame: pd.DataFrame) -> str:
    if "variant" in frame.columns:
        return "variant"
    if "label" in frame.columns:
        return "label"
    raise KeyError("No variant/label column found in hyperparameter output.")


def _safe_literal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def load_formula_test_panel(universe: str, formula: str) -> pd.DataFrame:
    bundle = load_dataset_bundle(Path(f"configs/us_equities_{universe}.yaml"))
    frame = bundle.splits.test.copy()
    dataset_name = load_experiment_name(EXPERIMENT_CONFIG)
    _, return_column = dataset_columns(dataset_name)
    evaluator = CrossSectionalFormulaEvaluator()
    evaluated = evaluator.evaluate(formula, frame)

    panel = frame.loc[evaluated.signal.index, ["date", "permno", return_column, "SIZE_LOG_MCAP"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target"] = panel[return_column].astype(float)
    panel["signal"] = evaluated.signal.astype(float)
    panel["size_proxy"] = np.exp(panel["SIZE_LOG_MCAP"].astype(float).clip(lower=-20.0, upper=20.0))
    return panel[["date", "permno", "target", "signal", "size_proxy"]].reset_index(drop=True)


def compute_portfolio_returns(panel: pd.DataFrame, *, weight_scheme: str) -> tuple[pd.Series, float]:
    kwargs: dict[str, Any] = {
        "signal": panel["signal"],
        "target": panel["target"],
        "dates": panel["date"],
        "entities": panel["permno"],
        "quantile": 0.2,
        "weight_scheme": weight_scheme,
    }
    if weight_scheme == "value":
        kwargs["size_proxy"] = panel["size_proxy"]
    daily_returns, weights = cross_sectional_long_short_returns(**kwargs)
    turnover = cross_sectional_turnover(weights, panel["date"], panel["permno"])
    return daily_returns.sort_index(), float(turnover)


def summarize_state_returns(returns: pd.Series, market_proxy: pd.DataFrame, universe: str) -> list[dict[str, object]]:
    returns_frame = pd.DataFrame({"date": pd.to_datetime(returns.index), "strategy_returns": returns.to_numpy()})
    merged = returns_frame.merge(market_proxy[["date", "vix", "vix_threshold", "vix_state"]], on="date", how="left").dropna()
    rows: list[dict[str, object]] = []
    for state, group in merged.groupby("vix_state", sort=True):
        daily = pd.Series(group["strategy_returns"].to_numpy(), index=pd.to_datetime(group["date"]))
        summary = summarize_returns(daily)
        rows.append(
            {
                "universe": universe,
                "state_group": "VIX状态",
                "state": state,
                "observation_count": int(len(group)),
                "annual_return": round(float(summary["annual_return"]), 4),
                "sharpe": round(float(summary["sharpe"]), 4),
                "mean_daily_return": round(float(daily.mean()), 6),
                "mean_vix": round(float(group["vix"].mean()), 4),
                "mean_vix_threshold": round(float(group["vix_threshold"].mean()), 4),
            }
        )
    return rows


def build_signal_main_rows(output_root: Path) -> list[dict[str, object]]:
    baseline_frame = load_baseline_frame(output_root)
    factors = load_fama_french_factors()
    rows: list[dict[str, object]] = []
    for universe in ("liquid500", "liquid1000"):
        baseline = consensus_row(baseline_frame, universe)
        formula = str(baseline["formula"])
        panel = load_formula_test_panel(universe, formula)
        ew_gross_returns, ew_turnover = compute_portfolio_returns(panel, weight_scheme="equal")
        ew_net_returns = cost_adjusted_returns(ew_gross_returns, ew_turnover, 15.0)
        vw_gross_returns, vw_turnover = compute_portfolio_returns(panel, weight_scheme="value")
        vw_net_returns = cost_adjusted_returns(vw_gross_returns, vw_turnover, 15.0)

        ew_gross_summary = summarize_returns(ew_gross_returns)
        ew_net_summary = summarize_returns(ew_net_returns)
        vw_gross_summary = summarize_returns(vw_gross_returns)
        vw_net_summary = summarize_returns(vw_net_returns)
        ew_gross_sig = compute_significance_metrics(ew_gross_returns, factors)
        ew_net_sig = compute_significance_metrics(ew_net_returns, factors)
        vw_gross_sig = compute_significance_metrics(vw_gross_returns, factors)
        vw_net_sig = compute_significance_metrics(vw_net_returns, factors)
        ic_metrics = cross_sectional_ic_summary(panel["signal"], panel["target"], panel["date"])
        coverage = signal_coverage_metrics(panel["signal"], panel["date"])
        concentration = return_concentration_metrics(ew_gross_returns)

        rows.append(
            {
                "universe": universe,
                "signal": formula,
                "evaluation_start": SPLIT_WINDOWS["test_start"],
                "evaluation_end": SPLIT_WINDOWS["test_end"],
                "gross_sharpe": round(float(ew_gross_summary["sharpe"]), 4),
                "net_sharpe_15bps": round(float(ew_net_summary["sharpe"]), 4),
                "gross_annual_return": round(float(ew_gross_summary["annual_return"]), 4),
                "net_annual_return_15bps": round(float(ew_net_summary["annual_return"]), 4),
                "gross_max_drawdown": round(float(ew_gross_summary["max_drawdown"]), 4),
                "net_max_drawdown_15bps": round(float(ew_net_summary["max_drawdown"]), 4),
                "mean_test_rank_ic": round(float(ic_metrics["rank_ic"]), 4),
                "turnover": round(float(ew_turnover), 4),
                "gross_nw_t": round(float(ew_gross_sig["nw_t"]), 4),
                "gross_nw_p": round(float(ew_gross_sig["nw_p"]), 4),
                "net_nw_t_15bps": round(float(ew_net_sig["nw_t"]), 4),
                "net_nw_p_15bps": round(float(ew_net_sig["nw_p"]), 4),
                "gross_ff5_alpha_ann": round(float(ew_gross_sig["ff5_alpha_ann"]), 4),
                "gross_ff5_alpha_t": round(float(ew_gross_sig["ff5_alpha_t"]), 4),
                "gross_ff5_alpha_p": round(float(ew_gross_sig["ff5_alpha_p"]), 4),
                "net_ff5_alpha_ann_15bps": round(float(ew_net_sig["ff5_alpha_ann"]), 4),
                "net_ff5_alpha_t_15bps": round(float(ew_net_sig["ff5_alpha_t"]), 4),
                "net_ff5_alpha_p_15bps": round(float(ew_net_sig["ff5_alpha_p"]), 4),
                "vw_gross_sharpe": round(float(vw_gross_summary["sharpe"]), 4),
                "vw_net_sharpe_15bps": round(float(vw_net_summary["sharpe"]), 4),
                "vw_gross_nw_t": round(float(vw_gross_sig["nw_t"]), 4),
                "vw_gross_nw_p": round(float(vw_gross_sig["nw_p"]), 4),
                "vw_net_nw_t_15bps": round(float(vw_net_sig["nw_t"]), 4),
                "vw_net_nw_p_15bps": round(float(vw_net_sig["nw_p"]), 4),
                "vw_gross_ff5_alpha_ann": round(float(vw_gross_sig["ff5_alpha_ann"]), 4),
                "vw_gross_ff5_alpha_t": round(float(vw_gross_sig["ff5_alpha_t"]), 4),
                "vw_gross_ff5_alpha_p": round(float(vw_gross_sig["ff5_alpha_p"]), 4),
                "vw_net_ff5_alpha_ann_15bps": round(float(vw_net_sig["ff5_alpha_ann"]), 4),
                "vw_net_ff5_alpha_t_15bps": round(float(vw_net_sig["ff5_alpha_t"]), 4),
                "vw_net_ff5_alpha_p_15bps": round(float(vw_net_sig["ff5_alpha_p"]), 4),
                "vw_turnover": round(float(vw_turnover), 4),
                "signal_non_null_fraction": round(float(coverage["signal_non_null_fraction"]), 6),
                "active_date_fraction": round(float(coverage["active_date_fraction"]), 6),
                "median_active_names_per_day": round(float(coverage["median_active_names_per_day"]), 2),
                "pnl_top_1pct_day_share": round(float(concentration["pnl_top_1pct_day_share"]), 6),
                **SPLIT_WINDOWS,
            }
        )
    return rows


def build_cost_sensitivity_rows(output_root: Path) -> list[dict[str, object]]:
    baseline_frame = load_baseline_frame(output_root)
    factors = load_fama_french_factors()
    rows: list[dict[str, object]] = []
    for universe in ("liquid500", "liquid1000"):
        formula = str(consensus_row(baseline_frame, universe)["formula"])
        panel = load_formula_test_panel(universe, formula)
        gross_returns, turnover = compute_portfolio_returns(panel, weight_scheme="equal")
        for cost_bps in (10, 30, 50, 100):
            net_returns = cost_adjusted_returns(gross_returns, turnover, float(cost_bps))
            summary = summarize_returns(net_returns)
            significance = compute_significance_metrics(net_returns, factors)
            rows.append(
                {
                    "universe": universe,
                    "evaluation_start": SPLIT_WINDOWS["test_start"],
                    "evaluation_end": SPLIT_WINDOWS["test_end"],
                    "cost_bps": cost_bps,
                    "net_sharpe": round(float(summary["sharpe"]), 4),
                    "net_annual_return": round(float(summary["annual_return"]), 4),
                    "net_max_drawdown": round(float(summary["max_drawdown"]), 4),
                    "net_nw_t": round(float(significance["nw_t"]), 4),
                    "net_nw_p": round(float(significance["nw_p"]), 4),
                    "net_ff5_alpha_ann": round(float(significance["ff5_alpha_ann"]), 4),
                    "net_ff5_alpha_t": round(float(significance["ff5_alpha_t"]), 4),
                    "net_ff5_alpha_p": round(float(significance["ff5_alpha_p"]), 4),
                }
            )
    return rows


def build_weighting_sensitivity_rows(output_root: Path) -> list[dict[str, object]]:
    signal_main = pd.DataFrame(build_signal_main_rows(output_root))
    cost_rows = pd.DataFrame(build_cost_sensitivity_rows(output_root))
    rows: list[dict[str, object]] = []
    for universe in ("liquid500", "liquid1000"):
        main_row = signal_main[signal_main["universe"] == universe].iloc[0]
        cost50 = cost_rows[(cost_rows["universe"] == universe) & (cost_rows["cost_bps"] == 50)].iloc[0]
        cost100 = cost_rows[(cost_rows["universe"] == universe) & (cost_rows["cost_bps"] == 100)].iloc[0]
        rows.append(
            {
                "universe": universe,
                "signal": main_row["signal"],
                "evaluation_start": SPLIT_WINDOWS["test_start"],
                "evaluation_end": SPLIT_WINDOWS["test_end"],
                "ew_gross_sharpe": main_row["gross_sharpe"],
                "ew_net_sharpe_15bps": main_row["net_sharpe_15bps"],
                "vw_gross_sharpe": main_row["vw_gross_sharpe"],
                "vw_net_sharpe_15bps": main_row["vw_net_sharpe_15bps"],
                "ew_net_sharpe_50bps": round(float(cost50["net_sharpe"]), 4),
                "ew_net_sharpe_100bps": round(float(cost100["net_sharpe"]), 4),
                "vw_gross_ff5_alpha_t": main_row["vw_gross_ff5_alpha_t"],
                "vw_net_ff5_alpha_t_15bps": main_row["vw_net_ff5_alpha_t_15bps"],
            }
        )
    return rows


def build_signal_main_journal_rows(output_root: Path) -> list[dict[str, object]]:
    signal_main = pd.DataFrame(build_signal_main_rows(output_root))
    cost_rows = pd.DataFrame(build_cost_sensitivity_rows(output_root))
    rows: list[dict[str, object]] = []
    for universe in ("liquid500", "liquid1000"):
        main_row = signal_main[signal_main["universe"] == universe].iloc[0]
        cost100 = cost_rows[(cost_rows["universe"] == universe) & (cost_rows["cost_bps"] == 100)].iloc[0]
        rows.append(
            {
                "universe": universe,
                "signal": main_row["signal"],
                "evaluation_start": main_row["evaluation_start"],
                "evaluation_end": main_row["evaluation_end"],
                "ew_gross_sharpe": main_row["gross_sharpe"],
                "ew_net_sharpe_15bps": main_row["net_sharpe_15bps"],
                "ew_nw_t": main_row["gross_nw_t"],
                "ew_ff5_alpha_ann_pct": round(float(main_row["gross_ff5_alpha_ann"]) * 100.0, 2),
                "ew_ff5_alpha_t": main_row["gross_ff5_alpha_t"],
                "vw_gross_sharpe": main_row["vw_gross_sharpe"],
                "vw_net_sharpe_15bps": main_row["vw_net_sharpe_15bps"],
                "vw_ff5_alpha_ann_pct": round(float(main_row["vw_gross_ff5_alpha_ann"]) * 100.0, 2),
                "vw_ff5_alpha_t": main_row["vw_gross_ff5_alpha_t"],
                "ew_net_sharpe_100bps": round(float(cost100["net_sharpe"]), 4),
                "signal_non_null_fraction": main_row["signal_non_null_fraction"],
                "turnover": main_row["turnover"],
            }
        )
    return rows


def build_parameter_basin_rows(output_root: Path) -> list[dict[str, object]]:
    meta = pd.read_csv(output_root / "finance_rolling_meta_validation_summary.csv")
    ablation = pd.read_csv(output_root / "finance_hyperparameter_ablation.csv")
    sensitivity = pd.read_csv(output_root / "finance_sensitivity_surface.csv")
    ablation_variant_col = _variant_column(ablation)
    rows: list[dict[str, object]] = []
    for universe in ("liquid500", "liquid1000"):
        meta_row = meta[meta["universe"] == universe].iloc[0]
        ablation_rows = ablation[ablation["universe"] == universe]
        sensitivity_rows = sensitivity[sensitivity["universe"] == universe]
        off_default = ablation_rows[~ablation_rows["matches_default_formula"]].copy()
        critical = off_default.sort_values("walk_forward_sharpe").iloc[0] if not off_default.empty else None
        stable_count = int(sensitivity_rows["matches_default_formula"].sum())
        total_count = int(len(sensitivity_rows))
        full_row = ablation_rows[ablation_rows[ablation_variant_col] == "full"]
        best_config = _safe_literal(meta_row["best_config"])
        best_config_label = "NA"
        if isinstance(best_config, dict):
            best_config_label = ", ".join(
                [
                    f"slices>={best_config.get('min_valid_slices', 'NA')}",
                    f"mean>={best_config.get('min_mean_rank_ic', 'NA')}",
                    f"min>={best_config.get('min_slice_rank_ic', 'NA')}",
                    f"seed>={best_config.get('min_seed_support', 'NA')}",
                ]
            )
        default_wf_sharpe = np.nan
        if not full_row.empty and pd.notna(full_row.iloc[0]["walk_forward_sharpe"]):
            default_wf_sharpe = round(float(full_row.iloc[0]["walk_forward_sharpe"]), 4)
        rows.append(
            {
                "universe": universe,
                "canonical_signal": meta_row["default_formula"],
                "rolling_meta_signal": meta_row["rolling_meta_selected_formula"],
                "window_signal_stability": round(float(meta_row["selected_formula_stability"]), 4),
                "sensitivity_match_count": stable_count,
                "sensitivity_total": total_count,
                "parameter_basin": f"{stable_count}/{total_count}",
                "critical_ablation": "none" if critical is None else critical[ablation_variant_col],
                "critical_ablation_signal": meta_row["default_formula"] if critical is None else critical["selected_formula"],
                "critical_ablation_wf_sharpe": default_wf_sharpe if critical is None else round(float(critical["walk_forward_sharpe"]), 4),
                "critical_ablation_matches_default": True if critical is None else bool(critical["matches_default_formula"]),
                "best_threshold_region": best_config_label,
                "coarse_config_count": int(meta_row["coarse_config_count"]),
                "fine_config_count": int(meta_row["fine_config_count"]),
            }
        )
    return rows


def build_pseudoalpha_summary_rows(output_root: Path) -> list[dict[str, object]]:
    path = output_root / "finance_pseudoalpha_cases.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for (universe, case_type), group in frame.groupby(["universe", "case_type"], sort=False):
        rows.append(
            {
                "universe": universe,
                "case_type": case_type,
                "case_count": int(len(group)),
                "mean_sharpe_gap": round(float(group["sharpe_gap_vs_reference"].mean()), 4),
                "worst_sharpe_gap": round(float(group["sharpe_gap_vs_reference"].min()), 4),
                "mean_rank_ic_gap": round(float(group["rank_ic_gap_vs_reference"].mean()), 4),
                "worst_source": str(group.sort_values("sharpe_gap_vs_reference").iloc[0]["source"]),
                "worst_candidate_signal": str(group.sort_values("sharpe_gap_vs_reference").iloc[0]["candidate_formula"]),
            }
        )
    return rows


def write_csv_json_md(frame: pd.DataFrame, stem: str, output_root: Path, title: str) -> None:
    csv_path = output_root / f"{stem}.csv"
    json_path = output_root / f"{stem}.json"
    md_path = output_root / f"{stem}.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    if frame.empty:
        lines = [f"# {title}", "", "(empty)"]
    else:
        header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
        divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
        body = [
            "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        lines = [f"# {title}", "", header, divider, *body]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    signal_main = pd.DataFrame(build_signal_main_rows(args.output_root))
    signal_main_journal = pd.DataFrame(build_signal_main_journal_rows(args.output_root))
    parameter_basin = pd.DataFrame(build_parameter_basin_rows(args.output_root))
    pseudoalpha_summary = pd.DataFrame(build_pseudoalpha_summary_rows(args.output_root))
    market_proxy = compute_market_proxy(PANEL_PATH)
    baseline_frame = load_baseline_frame(args.output_root)
    market_rows: list[dict[str, object]] = []
    for universe in ("liquid500", "liquid1000"):
        formula = str(consensus_row(baseline_frame, universe)["formula"])
        panel = load_formula_test_panel(universe, formula)
        ew_returns, _ = compute_portfolio_returns(panel, weight_scheme="equal")
        market_rows.extend(summarize_state_returns(ew_returns, market_proxy, universe))
    market_df = pd.DataFrame(market_rows)
    cost_df = pd.DataFrame(build_cost_sensitivity_rows(args.output_root))
    weighting_df = pd.DataFrame(build_weighting_sensitivity_rows(args.output_root))
    front_diag_path = args.output_root / "finance_pareto_front_diagnostics.csv"
    mcs_path = args.output_root / "finance_mcs_shortlist.csv"
    front_diag_df = pd.read_csv(front_diag_path) if front_diag_path.exists() else pd.DataFrame()
    mcs_df = pd.read_csv(mcs_path) if mcs_path.exists() else pd.DataFrame()

    write_csv_json_md(signal_main, "finance_signal_main_table", args.output_root, "Finance Signal Main Table")
    write_csv_json_md(signal_main_journal, "finance_signal_main_table_journal", args.output_root, "Finance Signal Main Table Journal")
    write_csv_json_md(parameter_basin, "finance_parameter_basin", args.output_root, "Finance Parameter Basin")
    write_csv_json_md(weighting_df, "finance_weighting_sensitivity_table", args.output_root, "Finance Weighting Sensitivity Table")
    if not pseudoalpha_summary.empty:
        write_csv_json_md(pseudoalpha_summary, "finance_pseudoalpha_summary", args.output_root, "Finance Pseudoalpha Summary")
    if not front_diag_df.empty:
        write_csv_json_md(front_diag_df, "finance_front_crowding_table", args.output_root, "Finance Pareto Front Diagnostics")
    if not mcs_df.empty:
        write_csv_json_md(mcs_df, "finance_mcs_table", args.output_root, "Finance MCS Shortlist")

    signal_main.to_csv(args.output_root / "chinese_report_main_results_extended.csv", index=False)
    signal_main_journal.to_csv(args.output_root / "chinese_report_main_results_journal.csv", index=False)
    parameter_basin.to_csv(args.output_root / "chinese_report_parameter_basin_table.csv", index=False)
    market_df.to_csv(args.output_root / "chinese_report_market_state_table.csv", index=False)
    cost_df.to_csv(args.output_root / "chinese_report_cost_sensitivity_table.csv", index=False)

    print(
        json.dumps(
            {
                "signal_main_table": str(args.output_root / "finance_signal_main_table.csv"),
                "signal_main_table_journal": str(args.output_root / "finance_signal_main_table_journal.csv"),
                "parameter_basin": str(args.output_root / "finance_parameter_basin.csv"),
                "market_state_table": str(args.output_root / "chinese_report_market_state_table.csv"),
                "cost_sensitivity_table": str(args.output_root / "chinese_report_cost_sensitivity_table.csv"),
                "front_crowding_table": str(args.output_root / "finance_front_crowding_table.csv") if not front_diag_df.empty else None,
                "mcs_table": str(args.output_root / "finance_mcs_table.csv") if not mcs_df.empty else None,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
