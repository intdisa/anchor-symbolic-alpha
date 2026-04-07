#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.common import (
    build_portfolio_config,
    build_signal_fusion_config,
    build_walk_forward_config,
    dataset_columns,
    load_dataset_bundle,
    load_experiment_name,
    load_yaml,
)
from knowledge_guided_symbolic_alpha.backtest import WalkForwardBacktester
from knowledge_guided_symbolic_alpha.benchmarks.task_protocol import select_best_formula_by_mean_slice_rank_ic
from knowledge_guided_symbolic_alpha.evaluation.panel_dispatch import evaluate_formula_metrics
from knowledge_guided_symbolic_alpha.generation import FormulaCandidate
from knowledge_guided_symbolic_alpha.selection import (
    CrossSeedConsensusConfig,
    CrossSeedConsensusSelector,
    CrossSeedSelectionRun,
    RobustSelectorConfig,
    TemporalRobustSelector,
)


UNIVERSE_SOURCES = {
    'liquid500': {
        'data_config': Path('configs/us_equities_liquid500.yaml'),
        'canonical': Path('outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json'),
        'multiseed': Path('outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed.json'),
    },
    'liquid1000': {
        'data_config': Path('configs/us_equities_liquid1000.yaml'),
        'canonical': Path('outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json'),
        'multiseed': Path('outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed.json'),
    },
}

BACKTEST_CONFIG = Path('configs/backtest.yaml')
EXPERIMENT_CONFIG = Path('configs/experiments/us_equities_anchor.yaml')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run finance meta-validation for selector hyperparameters.')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/reports'))
    parser.add_argument('--universes', type=str, default='liquid500,liquid1000')
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def parse_universes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def build_candidate_pool(raw_runs: list[dict[str, Any]]) -> list[FormulaCandidate]:
    formulas: list[str] = []
    seen: set[str] = set()
    for run in raw_runs:
        for formula in run.get('candidate_records', []):
            if formula and formula not in seen:
                seen.add(formula)
                formulas.append(formula)
    return [FormulaCandidate(formula=formula, source='finance_meta_validation', role='finance') for formula in formulas]


def split_valid_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(pd.to_datetime(frame['date']).sort_values().unique())
    midpoint = len(dates) // 2
    left_dates = set(dates[:midpoint])
    right_dates = set(dates[midpoint:])
    left = frame[pd.to_datetime(frame['date']).isin(left_dates)].copy()
    right = frame[pd.to_datetime(frame['date']).isin(right_dates)].copy()
    return left, right


def selection_score(metrics: dict[str, float]) -> float:
    rank_ic = float(metrics.get('rank_ic') or 0.0)
    sharpe = float(metrics.get('sharpe') or 0.0)
    turnover = float(metrics.get('turnover') or 0.0)
    return rank_ic + 0.03 * np.tanh(sharpe / 2.0) - 0.015 * turnover


def derive_seed_runs(
    raw_runs: list[dict[str, Any]],
    frame: pd.DataFrame,
    target: pd.Series,
    temporal_selector: TemporalRobustSelector,
) -> list[CrossSeedSelectionRun]:
    derived: list[CrossSeedSelectionRun] = []
    for raw in raw_runs:
        candidates = [
            FormulaCandidate(formula=formula, source='finance_meta_validation', role='finance')
            for formula in raw.get('candidate_records', [])
            if formula
        ]
        outcome = temporal_selector.select(candidates, frame, target)
        champion_formula = select_best_formula_by_mean_slice_rank_ic(candidates, frame, target)
        derived.append(
            CrossSeedSelectionRun(
                seed=int(raw['seed']),
                candidate_records=tuple(candidate.formula for candidate in candidates),
                selector_records=tuple(outcome.selected_formulas),
                champion_records=(champion_formula,) if champion_formula else tuple(),
                selector_ranked_records=tuple(outcome.records),
            )
        )
    return derived


def evaluate_formula(formula: str, frame: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    metrics = evaluate_formula_metrics(formula, frame, target).metrics
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float)) and np.isfinite(value)}


def walk_forward_metrics(universe: str, formula: str) -> dict[str, float]:
    data_config = UNIVERSE_SOURCES[universe]['data_config']
    bundle = load_dataset_bundle(data_config)
    dataset_name = load_experiment_name(EXPERIMENT_CONFIG)
    target_column, return_column = dataset_columns(dataset_name)
    backtest_frame = pd.concat([bundle.splits.valid, bundle.splits.test], axis=0)
    backtest_config = load_yaml(BACKTEST_CONFIG)
    backtester = WalkForwardBacktester(
        signal_fusion_config=build_signal_fusion_config(backtest_config),
        portfolio_config=build_portfolio_config(backtest_config),
    )
    report = backtester.run(
        formulas=[formula],
        frame=backtest_frame,
        feature_columns=bundle.feature_columns,
        target_column=target_column,
        return_column=return_column,
        config=build_walk_forward_config(backtest_config),
    )
    return {key: float(value) for key, value in report.aggregate_metrics.items() if isinstance(value, (int, float))}


def default_formula(universe: str) -> str:
    payload = load_json(UNIVERSE_SOURCES[universe]['canonical'])
    return str(payload['canonical_by_variant']['full']['selector_records'][0])


def config_grid() -> list[dict[str, float]]:
    values = {
        'rank_ic_std_penalty': (0.25, 0.35, 0.45),
        'min_rank_ic_bonus': (0.45, 0.55, 0.65),
        'turnover_weight': (0.010, 0.015, 0.020),
        'champion_support_weight': (0.02, 0.03, 0.04),
        'selector_support_weight': (0.005, 0.01, 0.015),
    }
    keys = list(values.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(values[key] for key in keys))]


def run_universe(universe: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    multiseed = load_json(UNIVERSE_SOURCES[universe]['multiseed'])
    raw_runs = multiseed['runs_by_variant']['full']
    candidates = build_candidate_pool(raw_runs)
    bundle = load_dataset_bundle(UNIVERSE_SOURCES[universe]['data_config'])
    valid_frame = bundle.splits.valid.copy()
    test_frame = bundle.splits.test.copy()
    valid_target = valid_frame['TARGET_XS_RET_1'].copy()
    test_target = test_frame['TARGET_XS_RET_1'].copy()
    calib_frame, meta_frame = split_valid_frame(valid_frame)
    calib_target = calib_frame['TARGET_XS_RET_1'].copy()
    meta_target = meta_frame['TARGET_XS_RET_1'].copy()

    records: list[dict[str, object]] = []
    best_record: dict[str, object] | None = None
    for config_id, cfg in enumerate(config_grid(), start=1):
        temporal_selector = TemporalRobustSelector(
            RobustSelectorConfig(
                rank_ic_std_penalty=float(cfg['rank_ic_std_penalty']),
                min_rank_ic_bonus=float(cfg['min_rank_ic_bonus']),
                turnover_weight=float(cfg['turnover_weight']),
            )
        )
        seed_runs = derive_seed_runs(raw_runs, calib_frame, calib_target, temporal_selector)
        consensus_selector = CrossSeedConsensusSelector(
            temporal_selector=temporal_selector,
            config=CrossSeedConsensusConfig(
                champion_support_weight=float(cfg['champion_support_weight']),
                selector_support_weight=float(cfg['selector_support_weight']),
                candidate_support_weight=0.005,
                selector_rank_penalty=0.002,
                rerank_mode='shared_frame',
            ),
        )
        outcome = consensus_selector.select(seed_runs, calib_frame, calib_target, base_candidates=candidates)
        formula = outcome.selected_formulas[0] if outcome.selected_formulas else ''
        meta_metrics = evaluate_formula(formula, meta_frame, meta_target) if formula else {}
        meta_score = selection_score(meta_metrics) if meta_metrics else float('-inf')
        row = {
            'universe': universe,
            'config_id': config_id,
            'formula': formula,
            'meta_score': round(float(meta_score), 6),
            'meta_rank_ic': round(float(meta_metrics.get('rank_ic') or 0.0), 6) if meta_metrics else None,
            'meta_sharpe': round(float(meta_metrics.get('sharpe') or 0.0), 6) if meta_metrics else None,
            'meta_turnover': round(float(meta_metrics.get('turnover') or 0.0), 6) if meta_metrics else None,
            **cfg,
        }
        records.append(row)
        if best_record is None or meta_score > float(best_record['meta_score']):
            best_record = row
    assert best_record is not None

    best_temporal = TemporalRobustSelector(
        RobustSelectorConfig(
            rank_ic_std_penalty=float(best_record['rank_ic_std_penalty']),
            min_rank_ic_bonus=float(best_record['min_rank_ic_bonus']),
            turnover_weight=float(best_record['turnover_weight']),
        )
    )
    best_seed_runs = derive_seed_runs(raw_runs, valid_frame, valid_target, best_temporal)
    best_consensus = CrossSeedConsensusSelector(
        temporal_selector=best_temporal,
        config=CrossSeedConsensusConfig(
            champion_support_weight=float(best_record['champion_support_weight']),
            selector_support_weight=float(best_record['selector_support_weight']),
            candidate_support_weight=0.005,
            selector_rank_penalty=0.002,
            rerank_mode='shared_frame',
        ),
    )
    final_outcome = best_consensus.select(best_seed_runs, valid_frame, valid_target, base_candidates=candidates)
    selected_formula = final_outcome.selected_formulas[0] if final_outcome.selected_formulas else ''
    test_metrics = evaluate_formula(selected_formula, test_frame, test_target) if selected_formula else {}
    wf_metrics = walk_forward_metrics(universe, selected_formula) if selected_formula else {}
    default = default_formula(universe)
    summary = {
        'universe': universe,
        'default_formula': default,
        'meta_selected_formula': selected_formula,
        'matches_default_formula': selected_formula == default,
        'meta_validation_best_config': {
            key: best_record[key]
            for key in ['config_id', 'rank_ic_std_penalty', 'min_rank_ic_bonus', 'turnover_weight', 'champion_support_weight', 'selector_support_weight']
        },
        'meta_validation_metrics': {
            'meta_score': best_record['meta_score'],
            'meta_rank_ic': best_record['meta_rank_ic'],
            'meta_sharpe': best_record['meta_sharpe'],
            'meta_turnover': best_record['meta_turnover'],
        },
        'test_metrics': {key: round(float(value), 6) for key, value in test_metrics.items()},
        'walk_forward_metrics': {key: round(float(value), 6) for key, value in wf_metrics.items()},
        'formula_vote_count': int(sum(1 for row in records if row['formula'] == selected_formula)),
        'config_count': len(records),
    }
    return records, summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    universes = parse_universes(args.universes)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for universe in universes:
        rows, summary = run_universe(universe)
        all_rows.extend(rows)
        summaries.append(summary)
    detail = pd.DataFrame(all_rows)
    summary_frame = pd.DataFrame(summaries)
    detail_csv = args.output_root / 'finance_meta_validation_detail.csv'
    summary_csv = args.output_root / 'finance_meta_validation_summary.csv'
    detail.to_csv(detail_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    json_path = args.output_root / 'finance_meta_validation_report.json'
    md_path = args.output_root / 'finance_meta_validation_report.md'
    payload = {'summaries': summaries, 'detail_csv': str(detail_csv), 'summary_csv': str(summary_csv)}
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')
    lines = ['# Finance Meta-Validation Report', '', '| Universe | Default Formula | Meta-Selected Formula | Matches Default | Formula Vote Count | Walk-Forward Sharpe | Test Rank-IC |', '| --- | --- | --- | --- | ---: | ---: | ---: |']
    for item in summaries:
        wf_sharpe = item['walk_forward_metrics'].get('sharpe') if item['walk_forward_metrics'] else None
        test_rank_ic = item['test_metrics'].get('rank_ic') if item['test_metrics'] else None
        lines.append(
            f"| {item['universe']} | {item['default_formula']} | {item['meta_selected_formula']} | {item['matches_default_formula']} | {item['formula_vote_count']} / {item['config_count']} | {wf_sharpe:.4f} | {test_rank_ic:.4f} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'detail_csv': str(detail_csv), 'summary_csv': str(summary_csv), 'json': str(json_path), 'markdown': str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
