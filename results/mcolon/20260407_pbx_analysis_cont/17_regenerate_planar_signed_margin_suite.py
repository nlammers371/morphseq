from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRWISE_ROOT = SCRIPT_DIR / 'results' / 'positioning' / 'pairwise' / 'combined_pairwise_5class_bin2_perm500'
MODULE_PATH = SCRIPT_DIR / '13_pairwise_pbx4_vs_double_by_experiment.py'

spec = importlib.util.spec_from_file_location('pbx_pairwise_module', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PAIRS = [
    ('inj_ctrl', 'pbx1b_crispant'),
    ('inj_ctrl', 'pbx1b_pbx4_crispant'),
    ('inj_ctrl', 'pbx4_crispant'),
    ('inj_ctrl', 'wik_ab'),
    ('wik_ab', 'pbx1b_pbx4_crispant'),
    ('pbx4_crispant', 'pbx1b_pbx4_crispant'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Regenerate plain signed-margin PBX figures directly from the saved all-pairs bundle.')
    parser.add_argument('--results-subdir', default='plain_signed_margin_all_embryos_bin2')
    parser.add_argument('--pairwise-root', type=Path, default=PAIRWISE_ROOT)
    return parser.parse_args()


def _comparison_record(metadata: dict[str, object], group1: str, group2: str) -> tuple[str, dict[str, object]]:
    comparisons = metadata['comparisons']
    direct = f'{group1}__vs__{group2}'
    reverse = f'{group2}__vs__{group1}'
    if direct in comparisons:
        return direct, comparisons[direct]
    if reverse in comparisons:
        return reverse, comparisons[reverse]
    raise KeyError(f'No comparison found for {group1} vs {group2}')


def _build_embryo_df(predictions: pd.DataFrame, comparison_id: str, group1: str, group2: str) -> pd.DataFrame:
    df = predictions[predictions['comparison_id'].astype(str) == comparison_id].copy()
    if df.empty:
        raise ValueError(f'No predictions found for {comparison_id}')
    df = df[df['true_label'].astype(str).isin([group1, group2])].copy()
    return df[[
        'embryo_id',
        'time_bin',
        'time_bin_center',
        'true_label',
        'predicted_label',
        'support_true',
        'confidence',
        'signed_margin',
        'is_correct',
    ]].sort_values(['true_label', 'embryo_id', 'time_bin_center']).reset_index(drop=True)


def _build_auc_df(scores: pd.DataFrame, comparison_id: str) -> pd.DataFrame:
    df = scores[scores['comparison_id'].astype(str) == comparison_id].copy()
    if df.empty:
        raise ValueError(f'No score rows found for {comparison_id}')
    return df[[
        'time_bin',
        'time_bin_center',
        'auroc_obs',
        'pval',
        'n_positive',
        'n_negative',
        'auroc_null_mean',
        'auroc_null_std',
        'n_permutations',
    ]].sort_values('time_bin_center').reset_index(drop=True)


def main() -> None:
    args = parse_args()
    results_dir = mod.RESULTS_ROOT / 'results' / 'classification' / args.results_subdir
    figures_dir = mod.RESULTS_ROOT / 'figures' / 'classification' / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(args.pairwise_root / 'predictions.parquet')
    scores = pd.read_parquet(args.pairwise_root / 'scores.parquet')
    metadata = json.loads((args.pairwise_root / 'metadata.json').read_text())

    manifest_pairs = []
    for group1, group2 in PAIRS:
        mod.GROUP1 = group1
        mod.GROUP2 = group2
        comparison_id, rec = _comparison_record(metadata, group1, group2)
        embryo_df = _build_embryo_df(predictions, comparison_id, group1, group2)
        pen_df = mod._compute_penetrance(embryo_df)
        auc_df = _build_auc_df(scores, comparison_id)
        stem = f'{group1}_vs_{group2}'

        pred_path = results_dir / f'embryo_predictions_{stem}.csv'
        pen_path = results_dir / f'embryo_penetrance_{stem}.csv'
        auc_path = results_dir / f'classification_auroc_{stem}.csv'
        fig_path = figures_dir / f'embryo_trajectories_signed_margin_{stem}.png'

        embryo_df.to_csv(pred_path, index=False)
        pen_df.to_csv(pen_path, index=False)
        auc_df.to_csv(auc_path, index=False)
        mod._plot_multiline(
            embryo_df,
            pen_df,
            max_embryos=-1,
            title=f'Embryo signed-margin trajectories: {mod._pretty(group1)} vs {mod._pretty(group2)}',
            output_path=fig_path,
        )
        print(fig_path)
        manifest_pairs.append({
            'group1': group1,
            'group2': group2,
            'comparison_id': comparison_id,
            'positive_label_in_bundle': rec['positive_label'],
            'negative_label_in_bundle': rec['negative_label'],
            'predictions_csv': str(pred_path),
            'penetrance_csv': str(pen_path),
            'auroc_csv': str(auc_path),
            'signed_margin_png': str(fig_path),
        })

    manifest = {
        'analysis': 'plain_signed_margin_suite_from_all_pairs',
        'pairwise_root': str(args.pairwise_root),
        'source_files': [
            str(args.pairwise_root / 'predictions.parquet'),
            str(args.pairwise_root / 'scores.parquet'),
            str(args.pairwise_root / 'metadata.json'),
        ],
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'pairs': manifest_pairs,
    }
    manifest_path = results_dir / 'plain_signed_margin_suite_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(manifest_path)


if __name__ == '__main__':
    main()
