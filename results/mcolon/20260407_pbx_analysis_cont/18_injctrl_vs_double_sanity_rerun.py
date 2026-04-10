from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.classification import run_classification
from common import ALL_EXPERIMENT_IDS, RESULTS_ROOT, load_combined_pbx_dataframe

GROUP1 = 'inj_ctrl'
GROUP2 = 'pbx1b_pbx4_crispant'
BIN_WIDTH = 4.0
N_PERM = 500
PAIRWISE_ROOT = RESULTS_ROOT / 'results' / 'positioning' / 'pairwise' / 'combined_pairwise_5class_bin4_perm500'
OUT_ROOT = RESULTS_ROOT / 'results' / 'classification' / 'injctrl_vs_double_sanity_rerun'


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_combined_pbx_dataframe(experiment_ids=ALL_EXPERIMENT_IDS, genotypes=[GROUP1, GROUP2])
    result = run_classification(
        df,
        class_col='genotype',
        id_col='embryo_id',
        time_col='stage_hpf',
        positive=GROUP2,
        negative=GROUP1,
        features={'vae': 'z_mu_b'},
        bin_width=BIN_WIDTH,
        n_permutations=N_PERM,
        n_jobs=8,
        min_samples_per_group=3,
        min_samples_per_member=2,
        random_state=42,
        class_weight='balanced',
        verbose=False,
        save_predictions=True,
        save_contrast_coordinates=True,
        save_dir=OUT_ROOT / 'binary_run',
    )

    preds = result.layers['predictions'].copy()
    raw_long = result.layers['raw_contrast_scores_long'].copy()
    scores = result.scores.copy()

    preds['time_bin'] = (preds['time_bin_center'].astype(float) - BIN_WIDTH / 2.0).round().astype(int)
    preds['comparison_id'] = f'{GROUP1}__vs__{GROUP2}'
    preds['signed_margin_from_p'] = preds['signed_margin'].astype(float)

    saved_raw = pd.read_parquet(PAIRWISE_ROOT / 'raw_contrast_scores_long.parquet')
    saved_raw = saved_raw[saved_raw['comparison_id'].astype(str) == f'{GROUP1}__vs__{GROUP2}'].copy()
    saved_scores = pd.read_parquet(PAIRWISE_ROOT / 'scores.parquet')
    saved_scores = saved_scores[saved_scores['comparison_id'].astype(str) == f'{GROUP1}__vs__{GROUP2}'].copy()

    raw_compare = raw_long.merge(
        saved_raw[[
            'embryo_id', 'time_bin', 'time_bin_center', 'comparison_id', 'm_raw', 'signed_margin'
        ]].rename(columns={
            'm_raw': 'm_raw_saved',
            'signed_margin': 'signed_margin_saved',
        }),
        on=['embryo_id', 'time_bin', 'time_bin_center', 'comparison_id'],
        how='outer',
        indicator=True,
        validate='one_to_one',
    )
    raw_compare['m_raw_diff'] = raw_compare['m_raw'].astype(float) - raw_compare['m_raw_saved'].astype(float)
    raw_compare['signed_margin_diff'] = raw_compare['signed_margin'].astype(float) - raw_compare['signed_margin_saved'].astype(float)

    pred_compare = preds.merge(
        saved_raw[[
            'embryo_id', 'time_bin', 'time_bin_center', 'comparison_id', 'm_raw', 'signed_margin'
        ]].rename(columns={
            'm_raw': 'm_raw_saved',
            'signed_margin': 'signed_margin_saved',
        }),
        on=['embryo_id', 'time_bin', 'time_bin_center', 'comparison_id'],
        how='outer',
        indicator=True,
        validate='one_to_one',
    )
    pred_compare['m_raw_from_pred'] = 2.0 * pred_compare['signed_margin_from_p'].astype(float)
    pred_compare['m_raw_vs_saved_diff'] = pred_compare['m_raw_from_pred'] - pred_compare['m_raw_saved'].astype(float)
    pred_compare['signed_margin_vs_saved_diff'] = pred_compare['signed_margin_from_p'].astype(float) - pred_compare['signed_margin_saved'].astype(float)

    score_compare = scores.merge(
        saved_scores[[
            'time_bin', 'time_bin_center', 'auroc_obs', 'pval', 'n_positive', 'n_negative', 'auroc_null_mean', 'auroc_null_std'
        ]].rename(columns={
            'auroc_obs': 'auroc_obs_saved',
            'pval': 'pval_saved',
            'n_positive': 'n_positive_saved',
            'n_negative': 'n_negative_saved',
            'auroc_null_mean': 'auroc_null_mean_saved',
            'auroc_null_std': 'auroc_null_std_saved',
        }),
        on=['time_bin', 'time_bin_center'],
        how='outer',
        indicator=True,
        validate='one_to_one',
    )
    if 'auroc_obs' in score_compare.columns:
        score_compare['auroc_diff'] = score_compare['auroc_obs'].astype(float) - score_compare['auroc_obs_saved'].astype(float)
        score_compare['pval_diff'] = score_compare['pval'].astype(float) - score_compare['pval_saved'].astype(float)

    raw_compare.to_csv(OUT_ROOT / 'raw_long_comparison.csv', index=False)
    pred_compare.to_csv(OUT_ROOT / 'prediction_vs_saved_comparison.csv', index=False)
    score_compare.to_csv(OUT_ROOT / 'score_vs_saved_comparison.csv', index=False)

    summary = {
        'pair': f'{GROUP1} vs {GROUP2}',
        'settings': {
            'bin_width': BIN_WIDTH,
            'n_permutations': N_PERM,
            'random_state': 42,
            'class_weight': 'balanced',
            'experiments': list(ALL_EXPERIMENT_IDS),
        },
        'raw_long_rows_binary': int(len(raw_long)),
        'raw_long_rows_saved': int(len(saved_raw)),
        'raw_long_outer_status': raw_compare['_merge'].value_counts(dropna=False).to_dict(),
        'raw_long_max_abs_m_raw_diff': float(raw_compare['m_raw_diff'].abs().max()),
        'raw_long_max_abs_signed_margin_diff': float(raw_compare['signed_margin_diff'].abs().max()),
        'pred_outer_status': pred_compare['_merge'].value_counts(dropna=False).to_dict(),
        'pred_max_abs_m_raw_vs_saved_diff': float(pred_compare['m_raw_vs_saved_diff'].abs().max()),
        'pred_max_abs_signed_margin_vs_saved_diff': float(pred_compare['signed_margin_vs_saved_diff'].abs().max()),
        'score_outer_status': score_compare['_merge'].value_counts(dropna=False).to_dict(),
        'score_max_abs_auroc_diff': float(score_compare['auroc_diff'].abs().max()),
        'score_max_abs_pval_diff': float(score_compare['pval_diff'].abs().max()),
    }
    (OUT_ROOT / 'sanity_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(OUT_ROOT)
    print(OUT_ROOT / 'sanity_summary.json')


if __name__ == '__main__':
    main()
