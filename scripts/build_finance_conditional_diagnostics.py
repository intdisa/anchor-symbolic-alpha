#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import dataset_columns, load_dataset_bundle, load_experiment_name
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_evaluator import CrossSectionalFormulaEvaluator
from knowledge_guided_symbolic_alpha.evaluation.cross_sectional_metrics import (
    DEFAULT_MIN_LONG_COUNT,
    DEFAULT_MIN_SHORT_COUNT,
    cross_sectional_long_short_returns,
    cross_sectional_turnover,
)
from knowledge_guided_symbolic_alpha.evaluation.finance_reporting import (
    compute_significance_metrics,
    cost_adjusted_returns,
    load_fama_french_factors,
    merge_returns_with_factors,
    summarize_returns,
)


EXPERIMENT_CONFIG = Path("configs/experiments/us_equities_anchor.yaml")
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
SIZE_BUCKETS = ("Small", "Mid", "Large")
VIX_LOOKBACK = 252
VIX_MIN_PERIODS = 126
VIX_QUANTILE = 0.8
FRED_MACRO_PATH = Path("data/raw/us_equities/public/fred_macro_daily.csv.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build size double-sort and ex-ante VIX-state diagnostics for the canonical finance signal.")
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


def load_oos_signal_panel(universe: str, formula: str) -> pd.DataFrame:
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


def load_market_proxy() -> pd.DataFrame:
    panel = pd.read_parquet("data/processed/us_equities/us_equities_panel.parquet", columns=["date", "RET_1"])
    panel["date"] = pd.to_datetime(panel["date"])
    market = panel.groupby("date", as_index=False).agg(ew_market_ret=("RET_1", "mean")).sort_values("date").reset_index(drop=True)
    fred = pd.read_csv(FRED_MACRO_PATH, usecols=["date", "VIXCLS"])
    fred["date"] = pd.to_datetime(fred["date"])
    fred["vix"] = pd.to_numeric(fred["VIXCLS"], errors="coerce")
    market = market.merge(fred[["date", "vix"]], on="date", how="left")
    market["vix_threshold"] = (
        market["vix"]
        .rolling(VIX_LOOKBACK, min_periods=VIX_MIN_PERIODS)
        .quantile(VIX_QUANTILE)
        .shift(1)
    )
    market["vix_state"] = np.where(market["vix"] >= market["vix_threshold"], "high_vix", "normal_vix")
    market = market.sort_values("date").reset_index(drop=True)
    return market


def assign_size_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    labels = list(SIZE_BUCKETS)
    bucket_chunks: list[pd.Series] = []
    for _, group in frame.groupby("date", sort=True):
        ranks = group["size_proxy"].rank(method="average", pct=True)
        bins = pd.cut(
            ranks,
            bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
            labels=labels,
            include_lowest=True,
        )
        bucket_chunks.append(pd.Series(bins.astype(str), index=group.index))
    frame["size_bucket"] = pd.concat(bucket_chunks).reindex(frame.index)
    return frame


def _bucket_weight_frame(
    frame: pd.DataFrame,
    *,
    weight_scheme: str,
    signal_quantile: float = 1.0 / 3.0,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for (_, bucket), group in frame.groupby(["date", "size_bucket"], sort=True):
        group = group.dropna(subset=["signal", "target"]).copy()
        if group.empty:
            continue
        rank = group["signal"].rank(pct=True, method="average")
        long_mask = rank >= (1.0 - signal_quantile)
        short_mask = rank <= signal_quantile
        if int(long_mask.sum()) < DEFAULT_MIN_LONG_COUNT or int(short_mask.sum()) < DEFAULT_MIN_SHORT_COUNT:
            continue
        weights = pd.Series(0.0, index=group.index)
        if weight_scheme == "value":
            long_values = group.loc[long_mask, "size_proxy"].clip(lower=0.0)
            short_values = group.loc[short_mask, "size_proxy"].clip(lower=0.0)
            long_sum = float(long_values.sum())
            short_sum = float(short_values.sum())
            weights.loc[long_mask] = 0.5 * long_values / (long_sum if long_sum > 0 else float(long_mask.sum()))
            weights.loc[short_mask] = -0.5 * short_values / (short_sum if short_sum > 0 else float(short_mask.sum()))
        else:
            weights.loc[long_mask] = 0.5 / float(long_mask.sum())
            weights.loc[short_mask] = -0.5 / float(short_mask.sum())
        group["weight"] = weights
        records.append(group)
    if not records:
        return pd.DataFrame(columns=[*frame.columns, "weight"])
    return pd.concat(records, axis=0, ignore_index=True)


def summarize_size_bucket_returns(
    frame: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    weight_scheme: str,
    cost_bps: float = 15.0,
) -> list[dict[str, Any]]:
    weighted = _bucket_weight_frame(frame, weight_scheme=weight_scheme)
    rows: list[dict[str, Any]] = []
    if weighted.empty:
        return rows
    for bucket in SIZE_BUCKETS:
        bucket_frame = weighted[weighted["size_bucket"] == bucket].copy()
        if bucket_frame.empty:
            continue
        gross_returns = (bucket_frame["weight"] * bucket_frame["target"]).groupby(bucket_frame["date"], sort=True).sum().sort_index()
        turnover = cross_sectional_turnover(bucket_frame["weight"], bucket_frame["date"], bucket_frame["permno"])
        net_returns = cost_adjusted_returns(gross_returns, turnover, cost_bps)
        gross_summary = summarize_returns(gross_returns)
        net_summary = summarize_returns(net_returns)
        gross_sig = compute_significance_metrics(gross_returns, factors)
        net_sig = compute_significance_metrics(net_returns, factors)
        rows.append(
            {
                "size_bucket": bucket,
                "weight_scheme": weight_scheme,
                "gross_sharpe": round(float(gross_summary["sharpe"]), 4),
                "net_sharpe_15bps": round(float(net_summary["sharpe"]), 4),
                "gross_annual_return": round(float(gross_summary["annual_return"]), 4),
                "net_annual_return_15bps": round(float(net_summary["annual_return"]), 4),
                "gross_nw_t": round(float(gross_sig["nw_t"]), 4),
                "gross_nw_p": round(float(gross_sig["nw_p"]), 4),
                "net_nw_t_15bps": round(float(net_sig["nw_t"]), 4),
                "net_nw_p_15bps": round(float(net_sig["nw_p"]), 4),
                "gross_ff5_alpha_ann": round(float(gross_sig["ff5_alpha_ann"]), 4),
                "gross_ff5_alpha_t": round(float(gross_sig["ff5_alpha_t"]), 4),
                "gross_ff5_alpha_p": round(float(gross_sig["ff5_alpha_p"]), 4),
                "net_ff5_alpha_ann_15bps": round(float(net_sig["ff5_alpha_ann"]), 4),
                "net_ff5_alpha_t_15bps": round(float(net_sig["ff5_alpha_t"]), 4),
                "net_ff5_alpha_p_15bps": round(float(net_sig["ff5_alpha_p"]), 4),
                "turnover": round(float(turnover), 4),
                "mean_daily_active_names": round(float(bucket_frame.groupby("date")["permno"].nunique().mean()), 2),
            }
        )
    return rows


def build_size_double_sort_rows(output_root: Path) -> list[dict[str, Any]]:
    factors = load_fama_french_factors()
    baseline_frame = load_baseline_frame(output_root)
    rows: list[dict[str, Any]] = []
    for universe in ("liquid500", "liquid1000"):
        formula = str(consensus_row(baseline_frame, universe)["formula"])
        panel = assign_size_buckets(load_oos_signal_panel(universe, formula))
        for row in summarize_size_bucket_returns(panel, factors, weight_scheme="equal"):
            row.update({"universe": universe, "signal": formula})
            rows.append(row)
        for row in summarize_size_bucket_returns(panel, factors, weight_scheme="value"):
            row.update({"universe": universe, "signal": formula})
            rows.append(row)
    return rows


def _window_metrics(returns: pd.Series, factors: pd.DataFrame) -> dict[str, float]:
    if returns.empty:
        return {
            "sharpe": float("nan"),
            "annual_return": float("nan"),
            "max_drawdown": float("nan"),
            "nw_t": float("nan"),
            "nw_p": float("nan"),
        }
    summary = summarize_returns(returns)
    sig = compute_significance_metrics(returns, factors)
    return {
        "sharpe": round(float(summary["sharpe"]), 4),
        "annual_return": round(float(summary["annual_return"]), 4),
        "max_drawdown": round(float(summary["max_drawdown"]), 4),
        "nw_t": round(float(sig["nw_t"]), 4),
        "nw_p": round(float(sig["nw_p"]), 4),
    }


def _vix_dummy_regression(returns: pd.Series, high_vix: pd.Series, factors: pd.DataFrame) -> dict[str, float]:
    merged = merge_returns_with_factors(returns, factors)
    if merged.empty:
        return {
            "alpha0_ann": float("nan"),
            "alpha0_t": float("nan"),
            "alpha0_p": float("nan"),
            "alpha1_ann": float("nan"),
            "alpha1_t": float("nan"),
            "alpha1_p": float("nan"),
            "observation_count": 0,
        }
    state = pd.Series(high_vix, copy=True)
    state.index = pd.to_datetime(state.index)
    merged = merged.merge(
        state.rename("high_vix").rename_axis("date").reset_index(),
        on="date",
        how="left",
    ).dropna(subset=["high_vix"])
    if merged.empty:
        return {
            "alpha0_ann": float("nan"),
            "alpha0_t": float("nan"),
            "alpha0_p": float("nan"),
            "alpha1_ann": float("nan"),
            "alpha1_t": float("nan"),
            "alpha1_p": float("nan"),
            "observation_count": 0,
        }
    excess = merged["strategy_returns"] - merged["RF"]
    design = sm.add_constant(merged[["high_vix"]].astype(float), has_constant="add")
    fit = sm.OLS(excess, design).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "alpha0_ann": round(float(fit.params["const"] * 252.0), 4),
        "alpha0_t": round(float(fit.tvalues["const"]), 4),
        "alpha0_p": round(float(fit.pvalues["const"]), 4),
        "alpha1_ann": round(float(fit.params["high_vix"] * 252.0), 4),
        "alpha1_t": round(float(fit.tvalues["high_vix"]), 4),
        "alpha1_p": round(float(fit.pvalues["high_vix"]), 4),
        "observation_count": int(len(merged)),
    }


def build_vix_state_rows(output_root: Path) -> list[dict[str, Any]]:
    factors = load_fama_french_factors()
    market = load_market_proxy().set_index("date")
    baseline_frame = load_baseline_frame(output_root)
    rows: list[dict[str, Any]] = []
    for universe in ("liquid500", "liquid1000"):
        formula = str(consensus_row(baseline_frame, universe)["formula"])
        panel = load_oos_signal_panel(universe, formula)
        ew_returns, _ = cross_sectional_long_short_returns(
            panel["signal"],
            panel["target"],
            panel["date"],
            panel["permno"],
            quantile=0.2,
            weight_scheme="equal",
            min_long_count=DEFAULT_MIN_LONG_COUNT,
            min_short_count=DEFAULT_MIN_SHORT_COUNT,
        )
        vw_returns, _ = cross_sectional_long_short_returns(
            panel["signal"],
            panel["target"],
            panel["date"],
            panel["permno"],
            quantile=0.2,
            weight_scheme="value",
            size_proxy=panel["size_proxy"],
            min_long_count=DEFAULT_MIN_LONG_COUNT,
            min_short_count=DEFAULT_MIN_SHORT_COUNT,
        )
        merged = pd.DataFrame({"ew_returns": ew_returns, "vw_returns": vw_returns}).join(
            market[["ew_market_ret", "vix", "vix_threshold", "vix_state"]],
            how="left",
        )
        merged = merged.dropna(subset=["vix_threshold", "vix_state"])
        for state, state_frame in merged.groupby("vix_state", sort=True):
            ew_window = state_frame["ew_returns"]
            vw_window = state_frame["vw_returns"]
            market_window = state_frame["ew_market_ret"]
            ew_metrics = _window_metrics(ew_window, factors)
            vw_metrics = _window_metrics(vw_window, factors)
            market_ann = float((1.0 + market_window).prod() ** (252.0 / max(len(market_window), 1)) - 1.0) if len(market_window) else float("nan")
            rows.append(
                {
                    "universe": universe,
                    "state": state,
                    "mean_vix": round(float(state_frame["vix"].mean()), 4),
                    "mean_vix_threshold": round(float(state_frame["vix_threshold"].mean()), 4),
                    "ew_sharpe": ew_metrics["sharpe"],
                    "ew_annual_return": ew_metrics["annual_return"],
                    "ew_nw_t": ew_metrics["nw_t"],
                    "ew_nw_p": ew_metrics["nw_p"],
                    "vw_sharpe": vw_metrics["sharpe"],
                    "vw_annual_return": vw_metrics["annual_return"],
                    "vw_nw_t": vw_metrics["nw_t"],
                    "vw_nw_p": vw_metrics["nw_p"],
                    "market_annual_return": round(market_ann, 4) if np.isfinite(market_ann) else float("nan"),
                    "observation_count": int(len(state_frame)),
                }
            )
    return rows


def build_vix_regression_rows(output_root: Path) -> list[dict[str, Any]]:
    factors = load_fama_french_factors()
    market = load_market_proxy().set_index("date")
    baseline_frame = load_baseline_frame(output_root)
    rows: list[dict[str, Any]] = []
    for universe in ("liquid500", "liquid1000"):
        formula = str(consensus_row(baseline_frame, universe)["formula"])
        panel = load_oos_signal_panel(universe, formula)
        ew_returns, _ = cross_sectional_long_short_returns(
            panel["signal"],
            panel["target"],
            panel["date"],
            panel["permno"],
            quantile=0.2,
            weight_scheme="equal",
            min_long_count=DEFAULT_MIN_LONG_COUNT,
            min_short_count=DEFAULT_MIN_SHORT_COUNT,
        )
        vw_returns, _ = cross_sectional_long_short_returns(
            panel["signal"],
            panel["target"],
            panel["date"],
            panel["permno"],
            quantile=0.2,
            weight_scheme="value",
            size_proxy=panel["size_proxy"],
            min_long_count=DEFAULT_MIN_LONG_COUNT,
            min_short_count=DEFAULT_MIN_SHORT_COUNT,
        )
        state = market.loc[market.index.intersection(ew_returns.index), "vix_state"].eq("high_vix").astype(float)
        ew_reg = _vix_dummy_regression(ew_returns, state, factors)
        vw_reg = _vix_dummy_regression(vw_returns, state, factors)
        rows.append(
            {
                "universe": universe,
                "ew_alpha0_ann": ew_reg["alpha0_ann"],
                "ew_alpha0_t": ew_reg["alpha0_t"],
                "ew_alpha0_p": ew_reg["alpha0_p"],
                "ew_high_vix_alpha_ann": ew_reg["alpha1_ann"],
                "ew_high_vix_t": ew_reg["alpha1_t"],
                "ew_high_vix_p": ew_reg["alpha1_p"],
                "vw_alpha0_ann": vw_reg["alpha0_ann"],
                "vw_alpha0_t": vw_reg["alpha0_t"],
                "vw_alpha0_p": vw_reg["alpha0_p"],
                "vw_high_vix_alpha_ann": vw_reg["alpha1_ann"],
                "vw_high_vix_t": vw_reg["alpha1_t"],
                "vw_high_vix_p": vw_reg["alpha1_p"],
                "observation_count": int(vw_reg["observation_count"]),
            }
        )
    return rows


def _write_outputs(output_root: Path, stem: str, rows: list[dict[str, Any]], markdown_builder) -> None:
    frame = pd.DataFrame(rows)
    (output_root / f"{stem}.csv").write_text(frame.to_csv(index=False), encoding="utf-8")
    (output_root / f"{stem}.json").write_text(frame.to_json(orient="records", force_ascii=True, indent=2) + "\n", encoding="utf-8")
    (output_root / f"{stem}.md").write_text(markdown_builder(frame) + "\n", encoding="utf-8")


def _build_size_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Finance Size-by-Signal Double Sort", ""]
    for universe in ("liquid500", "liquid1000"):
        lines.extend([f"## {universe}", ""])
        lines.append("| Weight | Size | Net Sharpe(15bps) | Net FF5 alpha_ann | Net FF5 t | Active names/day |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        subset = frame[frame["universe"] == universe]
        for weight_scheme in ("equal", "value"):
            block = subset[subset["weight_scheme"] == weight_scheme]
            for bucket in SIZE_BUCKETS:
                row = block[block["size_bucket"] == bucket].iloc[0]
                lines.append(
                    f"| {weight_scheme} | {bucket} | {row['net_sharpe_15bps']:.4f} | "
                    f"{row['net_ff5_alpha_ann_15bps']:.4f} | {row['net_ff5_alpha_t_15bps']:.4f} | {row['mean_daily_active_names']:.2f} |"
                )
        lines.append("")
    return "\n".join(lines)


def _build_vix_state_markdown(frame: pd.DataFrame) -> str:
    lines = ["# Finance Ex-Ante VIX State Diagnostics", ""]
    for universe in ("liquid500", "liquid1000"):
        lines.extend([f"## {universe}", ""])
        lines.append("| State | EW Sharpe | VW Sharpe | EW NW t | VW NW t | Mean VIX | Mean Threshold | Market Ann. Return | Obs |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        subset = frame[frame["universe"] == universe]
        order = {"high_vix": 0, "normal_vix": 1}
        subset = subset.assign(_order=subset["state"].map(order)).sort_values("_order")
        for _, row in subset.iterrows():
            lines.append(
                f"| {row['state']} | {row['ew_sharpe']:.4f} | {row['vw_sharpe']:.4f} | {row['ew_nw_t']:.4f} | "
                f"{row['vw_nw_t']:.4f} | {row['mean_vix']:.4f} | {row['mean_vix_threshold']:.4f} | "
                f"{row['market_annual_return']:.4f} | {int(row['observation_count'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _build_vix_regression_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Finance Ex-Ante VIX Dummy Regression",
        "",
        "| Universe | EW alpha1_ann | EW t | EW p | VW alpha1_ann | VW t | VW p | Obs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['universe']} | {row['ew_high_vix_alpha_ann']:.4f} | {row['ew_high_vix_t']:.4f} | {row['ew_high_vix_p']:.4f} | "
            f"{row['vw_high_vix_alpha_ann']:.4f} | {row['vw_high_vix_t']:.4f} | {row['vw_high_vix_p']:.4f} | {int(row['observation_count'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    size_rows = build_size_double_sort_rows(args.output_root)
    vix_rows = build_vix_state_rows(args.output_root)
    vix_reg_rows = build_vix_regression_rows(args.output_root)
    _write_outputs(args.output_root, "finance_size_double_sort", size_rows, _build_size_markdown)
    _write_outputs(args.output_root, "finance_vix_state_table", vix_rows, _build_vix_state_markdown)
    _write_outputs(args.output_root, "finance_vix_regression_table", vix_reg_rows, _build_vix_regression_markdown)
    print(
        json.dumps(
            {
                "size_double_sort": str(args.output_root / "finance_size_double_sort.csv"),
                "vix_state_table": str(args.output_root / "finance_vix_state_table.csv"),
                "vix_regression_table": str(args.output_root / "finance_vix_regression_table.csv"),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
