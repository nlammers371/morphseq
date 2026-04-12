from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

OUT_ROOT = SCRIPT_DIR / 'results' / 'classification' / 'injctrl_vs_double_sanity_rerun'
BINARY_ROOT = OUT_ROOT / 'binary_run'
PAIRWISE_ROOT = SCRIPT_DIR / 'results' / 'positioning' / 'pairwise' / 'combined_pairwise_5class_bin4_perm500'
FIG_ROOT = SCRIPT_DIR / 'figures' / 'classification' / 'injctrl_vs_double_sanity_rerun'
FIG_ROOT.mkdir(parents=True, exist_ok=True)

PAIR_SAVED = 'inj_ctrl__vs__pbx1b_pbx4_crispant'
PAIR_BINARY = 'pbx1b_pbx4_crispant__vs__inj_ctrl'


def main() -> None:
    preds = pd.read_parquet(BINARY_ROOT / 'predictions.parquet').copy()
    raw_long = pd.read_parquet(BINARY_ROOT / 'raw_contrast_scores_long.parquet').copy()
    scores = pd.read_parquet(BINARY_ROOT / 'scores.parquet').copy()
    saved_raw = pd.read_parquet(PAIRWISE_ROOT / 'raw_contrast_scores_long.parquet').copy()
    saved_scores = pd.read_parquet(PAIRWISE_ROOT / 'scores.parquet').copy()

    preds = preds[preds['comparison_id'].astype(str) == PAIR_BINARY].copy()
    raw_long = raw_long[raw_long['comparison_id'].astype(str) == PAIR_BINARY].copy()
    scores = scores[scores['comparison_id'].astype(str) == PAIR_BINARY].copy()
    saved_raw = saved_raw[saved_raw['comparison_id'].astype(str) == PAIR_SAVED].copy()
    saved_scores = saved_scores[saved_scores['comparison_id'].astype(str) == PAIR_SAVED].copy()

    # Normalize fresh binary orientation to match saved all-pairs orientation.
    raw_long['comparison_id_norm'] = PAIR_SAVED
    raw_long['signed_margin_norm'] = -raw_long['signed_margin'].astype(float)
    raw_long['m_raw_norm'] = -raw_long['m_raw'].astype(float)

    preds['time_bin'] = (preds['time_bin_center'].astype(float) - 2.0).round().astype(int)
    preds['comparison_id_norm'] = PAIR_SAVED
    preds['signed_margin_norm'] = -preds['signed_margin'].astype(float)
    preds['m_raw_norm'] = -2.0 * preds['signed_margin'].astype(float)

    raw_compare = raw_long.merge(
        saved_raw.rename(columns={'m_raw': 'm_raw_saved'}),
        left_on=['embryo_id', 'time_bin', 'time_bin_center', 'comparison_id_norm'],
        right_on=['embryo_id', 'time_bin', 'time_bin_center', 'comparison_id'],
        how='outer',
        indicator=True,
        validate='one_to_one',
    )
    raw_compare['signed_margin_saved'] = 0.5 * raw_compare['m_raw_saved'].astype(float)
    raw_compare['m_raw_diff'] = raw_compare['m_raw_norm'].astype(float) - raw_compare['m_raw_saved'].astype(float)
    raw_compare['signed_margin_diff'] = raw_compare['signed_margin_norm'].astype(float) - raw_compare['signed_margin_saved'].astype(float)

    pred_compare = preds.merge(
        saved_raw.rename(columns={'m_raw': 'm_raw_saved'}),
        left_on=['embryo_id', 'time_bin', 'time_bin_center', 'comparison_id_norm'],
        right_on=['embryo_id', 'time_bin', 'time_bin_center', 'comparison_id'],
        how='outer',
        indicator=True,
        validate='one_to_one',
    )
    pred_compare['signed_margin_saved'] = 0.5 * pred_compare['m_raw_saved'].astype(float)
    pred_compare['m_raw_diff'] = pred_compare['m_raw_norm'].astype(float) - pred_compare['m_raw_saved'].astype(float)
    pred_compare['signed_margin_diff'] = pred_compare['signed_margin_norm'].astype(float) - pred_compare['signed_margin_saved'].astype(float)

    score_compare = scores.merge(
        saved_scores.rename(columns={
            'auroc_obs': 'auroc_obs_saved',
            'pval': 'pval_saved',
            'n_positive': 'n_positive_saved',
            'n_negative': 'n_negative_saved',
            'auroc_null_mean': 'auroc_null_mean_saved',
            'auroc_null_std': 'auroc_null_std_saved',
        }),
        on=['time_bin', 'time_bin_center', 'bin_width', 'feature_set'],
        how='outer',
        indicator=True,
        validate='one_to_one',
    )
    score_compare['auroc_diff'] = score_compare['auroc_obs'].astype(float) - score_compare['auroc_obs_saved'].astype(float)
    score_compare['pval_diff'] = score_compare['pval'].astype(float) - score_compare['pval_saved'].astype(float)

    raw_compare.to_csv(OUT_ROOT / 'raw_long_comparison.csv', index=False)
    pred_compare.to_csv(OUT_ROOT / 'prediction_vs_saved_comparison.csv', index=False)
    score_compare.to_csv(OUT_ROOT / 'score_vs_saved_comparison.csv', index=False)

    summary = {
        'pair_saved': PAIR_SAVED,
        'pair_binary': PAIR_BINARY,
        'orientation_fix_applied': 'binary_sign_flipped_to_match_saved_all_pairs',
        'raw_long_outer_status': raw_compare['_merge'].value_counts(dropna=False).to_dict(),
        'raw_long_max_abs_m_raw_diff': float(np.nanmax(np.abs(raw_compare['m_raw_diff'].to_numpy(dtype=float)))),
        'raw_long_max_abs_signed_margin_diff': float(np.nanmax(np.abs(raw_compare['signed_margin_diff'].to_numpy(dtype=float)))),
        'pred_outer_status': pred_compare['_merge'].value_counts(dropna=False).to_dict(),
        'pred_max_abs_m_raw_diff': float(np.nanmax(np.abs(pred_compare['m_raw_diff'].to_numpy(dtype=float)))),
        'pred_max_abs_signed_margin_diff': float(np.nanmax(np.abs(pred_compare['signed_margin_diff'].to_numpy(dtype=float)))),
        'score_outer_status': score_compare['_merge'].value_counts(dropna=False).to_dict(),
        'score_max_abs_auroc_diff': float(np.nanmax(np.abs(score_compare['auroc_diff'].to_numpy(dtype=float)))),
        'score_max_abs_pval_diff': float(np.nanmax(np.abs(score_compare['pval_diff'].to_numpy(dtype=float)))),
    }
    (OUT_ROOT / 'sanity_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    # Plot 1: AUROC overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(score_compare['time_bin_center'], score_compare['auroc_obs_saved'], label='saved all-pairs', color='#2166ac', linewidth=2.5)
    ax.plot(score_compare['time_bin_center'], score_compare['auroc_obs'], label='fresh binary rerun', color='#b2182b', linewidth=2.0, linestyle='--')
    ax.axhline(0.5, color='0.5', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Hours Post Fertilization (hpf)')
    ax.set_ylabel('AUROC')
    ax.set_title('inj_ctrl vs pbx1b_pbx4_crispant: saved all-pairs vs fresh binary rerun')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / 'auroc_saved_vs_rerun.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: signed-margin scatter
    matched = pred_compare[pred_compare['_merge'] == 'both'].copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(matched['signed_margin_saved'], matched['signed_margin_norm'], s=10, alpha=0.35, color='#7b3294')
    lim = 0.55
    ax.plot([-lim, lim], [-lim, lim], color='0.4', linestyle='--', linewidth=1.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('saved all-pairs signed margin')
    ax.set_ylabel('fresh binary rerun signed margin')
    ax.set_title('Per-embryo signed margin agreement')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / 'signed_margin_saved_vs_rerun_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot 3: signed-margin difference histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    diffs = matched['signed_margin_diff'].astype(float)
    ax.hist(diffs[np.isfinite(diffs)], bins=50, color='#4d9221', alpha=0.8)
    ax.axvline(0.0, color='0.3', linestyle='--', linewidth=1.5)
    ax.set_xlabel('fresh - saved signed margin')
    ax.set_ylabel('count')
    ax.set_title('Signed-margin difference after orientation fix')
    fig.tight_layout()
    fig.savefig(FIG_ROOT / 'signed_margin_difference_hist.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(OUT_ROOT / 'sanity_summary.json')
    print(FIG_ROOT / 'auroc_saved_vs_rerun.png')
    print(FIG_ROOT / 'signed_margin_saved_vs_rerun_scatter.png')
    print(FIG_ROOT / 'signed_margin_difference_hist.png')


if __name__ == '__main__':
    main()
