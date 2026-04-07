#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from knowledge_guided_symbolic_alpha.benchmarks.task_protocol import (
    formula_complexity,
    select_best_formula_by_mean_slice_rank_ic,
    select_best_formula_by_metric,
    select_formula_by_lasso_screening,
    select_formula_by_pareto_front,
)
from knowledge_guided_symbolic_alpha.evaluation.panel_dispatch import evaluate_formula_metrics
from knowledge_guided_symbolic_alpha.generation import FormulaCandidate


SOURCES = {
    'liquid500': Path('outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json'),
    'liquid1000': Path('outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate stronger selection baselines on the finance candidate pools.')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/reports'))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_candidates(universe: str) -> tuple[list[FormulaCandidate], str]:
    payload = load_json(SOURCES[universe])
    full = payload['canonical_by_variant']['full']
    candidates = [
        FormulaCandidate(formula=item['formula'], source='finance_consensus_pool', role='finance_baseline')
        for item in full['support_adjusted_ranked_records']
    ]
    consensus_formula = full['selector_records'][0]
    return candidates, consensus_formula


def load_split(universe: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_parquet(f'data/processed/us_equities/subsets/{universe}_2010_2025/{split}.parquet')
    target = frame['TARGET_XS_RET_1'].copy()
    return frame, target


def metric_summary(formula: str, frame: pd.DataFrame, target: pd.Series) -> dict[str, float | None]:
    metrics = evaluate_formula_metrics(formula, frame, target).metrics
    return {
        'rank_ic': float(metrics['rank_ic']) if 'rank_ic' in metrics else None,
        'sharpe': float(metrics['sharpe']) if 'sharpe' in metrics else None,
        'turnover': float(metrics['turnover']) if 'turnover' in metrics else None,
        'annual_return': float(metrics['annual_return']) if 'annual_return' in metrics else None,
        'max_drawdown': float(metrics['max_drawdown']) if 'max_drawdown' in metrics else None,
    }


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for universe in SOURCES:
        candidates, consensus_formula = load_candidates(universe)
        valid_frame, valid_target = load_split(universe, 'valid')
        test_frame, test_target = load_split(universe, 'test')
        selections = {
            'consensus_formula': consensus_formula,
            'naive_rank_ic': select_best_formula_by_metric(candidates, valid_frame, valid_target, 'rank_ic'),
            'best_validation_sharpe': select_best_formula_by_metric(candidates, valid_frame, valid_target, 'sharpe'),
            'best_validation_mean_rank_ic': select_best_formula_by_mean_slice_rank_ic(candidates, valid_frame, valid_target),
            'pareto_front_selector': select_formula_by_pareto_front(candidates, valid_frame, valid_target),
            'lasso_formula_screening': select_formula_by_lasso_screening(candidates, valid_frame, valid_target),
        }
        for baseline, formula in selections.items():
            valid_metrics = metric_summary(formula, valid_frame, valid_target)
            test_metrics = metric_summary(formula, test_frame, test_target)
            rows.append(
                {
                    'universe': universe,
                    'baseline': baseline,
                    'formula': formula,
                    'complexity': formula_complexity(formula),
                    'valid_rank_ic': round(valid_metrics['rank_ic'], 4) if valid_metrics['rank_ic'] is not None else None,
                    'valid_sharpe': round(valid_metrics['sharpe'], 4) if valid_metrics['sharpe'] is not None else None,
                    'test_rank_ic': round(test_metrics['rank_ic'], 4) if test_metrics['rank_ic'] is not None else None,
                    'test_sharpe': round(test_metrics['sharpe'], 4) if test_metrics['sharpe'] is not None else None,
                    'test_turnover': round(test_metrics['turnover'], 4) if test_metrics['turnover'] is not None else None,
                    'test_annual_return': round(test_metrics['annual_return'], 4) if test_metrics['annual_return'] is not None else None,
                    'test_max_drawdown': round(test_metrics['max_drawdown'], 4) if test_metrics['max_drawdown'] is not None else None,
                    'matches_consensus_formula': formula == consensus_formula,
                }
            )
    return rows


def build_markdown(rows: list[dict[str, object]]) -> str:
    lines = ['# Finance Selection Baselines', '']
    for universe in sorted({row['universe'] for row in rows}):
        lines.extend([f'## {universe}', ''])
        lines.append('| Baseline | Formula | Valid Rank-IC | Test Sharpe | Test Rank-IC | Matches Consensus |')
        lines.append('| --- | --- | ---: | ---: | ---: | --- |')
        for row in [item for item in rows if item['universe'] == universe]:
            lines.append(
                f"| {row['baseline']} | {row['formula']} | {row['valid_rank_ic']:.4f} | {row['test_sharpe']:.4f} | {row['test_rank_ic']:.4f} | {row['matches_consensus_formula']} |"
            )
        lines.append('')
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    frame = pd.DataFrame(rows)
    csv_path = args.output_root / 'finance_selection_baselines.csv'
    json_path = args.output_root / 'finance_selection_baselines.json'
    md_path = args.output_root / 'finance_selection_baselines.md'
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient='records', force_ascii=True, indent=2) + '\n', encoding='utf-8')
    md_path.write_text(build_markdown(rows) + '\n', encoding='utf-8')
    print(json.dumps({'csv': str(csv_path), 'json': str(json_path), 'markdown': str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
