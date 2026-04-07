#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

BASE_CONFIG = {
    'champion_support_weight': 0.03,
    'selector_support_weight': 0.01,
    'candidate_support_weight': 0.005,
    'selector_rank_penalty': 0.002,
}
MULTIPLIERS = (0.8, 1.0, 1.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a hyperparameter-ensemble report from consensus ranked records.')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/reports'))
    parser.add_argument(
        '--benchmark-summary',
        type=Path,
        default=Path('outputs/runs/selector_suite_strong_baselines_r3/reports/all_selector_benchmark_summary.json'),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def score_record(record: dict[str, Any], seed_count: int, cfg: dict[str, float]) -> float:
    candidate_frac = float(record.get('candidate_seed_support', 0)) / max(seed_count, 1)
    selector_frac = float(record.get('selector_seed_support', 0)) / max(seed_count, 1)
    champion_frac = float(record.get('champion_seed_support', 0)) / max(seed_count, 1)
    mean_rank = record.get('mean_selector_rank')
    rank_penalty = 0.0 if mean_rank is None else cfg['selector_rank_penalty'] * max(float(mean_rank) - 1.0, 0.0)
    return (
        float(record.get('robust_score') or 0.0)
        + cfg['champion_support_weight'] * champion_frac
        + cfg['selector_support_weight'] * selector_frac
        + cfg['candidate_support_weight'] * candidate_frac
        - rank_penalty
    )


def ensemble_winner(records: list[dict[str, Any]], seed_count: int) -> tuple[str, dict[str, int]]:
    vote_counts: dict[str, int] = {}
    for m1 in MULTIPLIERS:
        for m2 in MULTIPLIERS:
            for m3 in MULTIPLIERS:
                for m4 in MULTIPLIERS:
                    cfg = {
                        'champion_support_weight': BASE_CONFIG['champion_support_weight'] * m1,
                        'selector_support_weight': BASE_CONFIG['selector_support_weight'] * m2,
                        'candidate_support_weight': BASE_CONFIG['candidate_support_weight'] * m3,
                        'selector_rank_penalty': BASE_CONFIG['selector_rank_penalty'] * m4,
                    }
                    ranked = sorted(
                        records,
                        key=lambda item: (
                            score_record(item, seed_count, cfg),
                            int(item.get('champion_seed_support', 0)),
                            int(item.get('selector_seed_support', 0)),
                            int(item.get('candidate_seed_support', 0)),
                            -(float(item.get('mean_selector_rank') or 1e9)),
                        ),
                        reverse=True,
                    )
                    if ranked:
                        formula = str(ranked[0]['formula'])
                        vote_counts[formula] = vote_counts.get(formula, 0) + 1
    if not vote_counts:
        return '', {}
    winner = max(vote_counts, key=lambda formula: (vote_counts[formula], formula))
    return winner, vote_counts


def finance_payload() -> list[dict[str, Any]]:
    sources = {
        'liquid500': Path('outputs/runs/liquid500_multiseed_e5_r3__multiseed/reports/us_equities_multiseed_canonical.json'),
        'liquid1000': Path('outputs/runs/liquid1000_multiseed_e5_r4__multiseed/reports/us_equities_multiseed_canonical.json'),
    }
    rows = []
    for universe, path in sources.items():
        payload = load_json(path)
        full = payload['canonical_by_variant']['full']
        records = full['support_adjusted_ranked_records']
        seed_count = int(full['seed_support']['seed_count'])
        ensemble_formula, votes = ensemble_winner(records, seed_count)
        rows.append(
            {
                'scope': 'finance',
                'id': universe,
                'true_formula': None,
                'canonical_formula': full['selector_records'][0],
                'ensemble_formula': ensemble_formula,
                'ensemble_matches_canonical': ensemble_formula == full['selector_records'][0],
                'winner_vote_share': round(votes.get(ensemble_formula, 0) / 81.0, 4) if ensemble_formula else None,
                'vote_counts_json': json.dumps(votes, ensure_ascii=True),
            }
        )
    return rows


def benchmark_payload(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    payload = load_json(summary_path)
    rows = []
    for task in payload.get('task_results', []):
        baseline = task['baselines'].get('support_adjusted_cross_seed_consensus', {})
        diagnostics = baseline.get('diagnostics', {})
        records = diagnostics.get('support_adjusted_ranked_records', [])
        seed_count = int(task.get('seed_count') or 0)
        ensemble_formula, votes = ensemble_winner(records, seed_count)
        rows.append(
            {
                'scope': 'benchmark',
                'id': task['task_id'],
                'true_formula': task['true_formula'],
                'canonical_formula': baseline.get('selected_formula'),
                'ensemble_formula': ensemble_formula,
                'ensemble_matches_canonical': ensemble_formula == baseline.get('selected_formula'),
                'ensemble_matches_true': ensemble_formula == task['true_formula'],
                'winner_vote_share': round(votes.get(ensemble_formula, 0) / 81.0, 4) if ensemble_formula else None,
                'vote_counts_json': json.dumps(votes, ensure_ascii=True),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = finance_payload() + benchmark_payload(args.benchmark_summary)
    frame = pd.DataFrame(rows)
    csv_path = args.output_root / 'hyperparameter_ensemble_report.csv'
    json_path = args.output_root / 'hyperparameter_ensemble_report.json'
    md_path = args.output_root / 'hyperparameter_ensemble_report.md'
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient='records', force_ascii=True, indent=2) + '\n', encoding='utf-8')
    lines = ['# Hyperparameter Ensemble Report', '', '| Scope | ID | Canonical | Ensemble | Matches Canonical | Matches True | Vote Share |', '| --- | --- | --- | --- | --- | --- | ---: |']
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['id']} | {row['canonical_formula']} | {row['ensemble_formula']} | {row['ensemble_matches_canonical']} | {row.get('ensemble_matches_true', 'NA')} | {row['winner_vote_share']:.4f} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'csv': str(csv_path), 'json': str(json_path), 'markdown': str(md_path)}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
