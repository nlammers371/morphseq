from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from trajectory_cosmology import animation as tc_animation, plotting, schema
from trajectory_cosmology.condensation.api import run_condensation
from trajectory_cosmology.condensation.engine.stopping import StoppingConfig
from trajectory_cosmology.condensation.geometry_refs import estimate_local_spacing_ref
from trajectory_cosmology.condensation.state import CondensationConfig
from trajectory_cosmology.iteration_ranking import (
    plot_iteration_scores,
    render_selected_iteration_bundle,
    save_ranking_outputs,
    score_saved_iterations,
)

GENOTYPE_COLORS: dict[str, str] = {
    'inj_ctrl': '#2166AC',
    'wik_ab': '#808080',
    'pbx1b_crispant': '#9467bd',
    'pbx4_crispant': '#F7B267',
    'pbx1b_pbx4_crispant': '#B2182B',
}


def _gamma_from_half_life_iters(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Force-separation ablation sweep for PBX condensation.')
    p.add_argument(
        '--input',
        default='results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_positioning_pairwise_bin4_perm500/pairwise_raw_vectors.csv',
        help='Path to pairwise raw vector CSV.',
    )
    p.add_argument('--input-type', choices=['auto', 'multiclass', 'pairwise'], default='pairwise')
    p.add_argument(
        '--x0-path',
        default='results/mcolon/20260329_pbx_crispant_analysis_cont/results/pairwise_raw_condensation_iter1000_ranked_umap/x0_init.npz',
        help='Path to saved x0_init.npz used by every ablation run.',
    )
    p.add_argument(
        '--output-dir',
        default='results/mcolon/20260329_pbx_crispant_analysis_cont/results/force_ablation_coherence_iter500',
    )
    p.add_argument('--n-iter', type=int, default=500)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--sigma', type=float, default=0.5, help='Fixed attraction bandwidth for the ablation family.')
    p.add_argument('--sigma-coh-mults', default='0.1,0.3,0.4,0.5,0.6')
    p.add_argument('--render-animations', action='store_true', help='Render GIF animations for each ablation run.')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_iter = 100
        args.save_every = 10
        args.top_k = min(args.top_k, 2)
        args.log_every = 10
        sigma_mults = [0.1, 0.5]
    else:
        sigma_mults = [float(x) for x in args.sigma_coh_mults.split(',') if x.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_cosmology_data(args.input, args.input_type)
    schema.validate(data, allow_feature_nans=bool(np.isnan(data.features[data.mask]).any()))

    x0_payload = np.load(args.x0_path)
    x0 = np.asarray(x0_payload['x0'], dtype=float)
    if x0.shape != (data.features.shape[0], data.features.shape[1], 2):
        raise ValueError(
            f'x0 shape mismatch: expected {(data.features.shape[0], data.features.shape[1], 2)}, found {tuple(x0.shape)}'
        )

    s_local = float(estimate_local_spacing_ref(x0, data.mask, k=5))
    color_map = {g: GENOTYPE_COLORS.get(g, '#555555') for g in np.unique(data.labels)}
    stopping = StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )

    ablations = _build_ablations(args.n_iter, args.sigma, s_local, sigma_mults)
    if args.smoke:
        keep = {'repel_fid_only', 'attract_uniform_plus_repel_fid', 'attract_computed_plus_repel_fid_sigma0p5'}
        ablations = [spec for spec in ablations if spec['name'] in keep]

    manifest = {
        'input': str(Path(args.input)),
        'x0_path': str(Path(args.x0_path)),
        'n_iter': int(args.n_iter),
        'save_every': int(args.save_every),
        'top_k': int(args.top_k),
        'fixed_sigma_attract': float(args.sigma),
        's_local': s_local,
        'sigma_coh_multipliers': sigma_mults,
        'smoke': bool(args.smoke),
        'runs': [
            {
                'name': spec['name'],
                'description': spec['description'],
                'coherence_mode': spec['config'].coherence_mode,
                'sigma_coh': spec['config'].sigma_coh,
                'k_attract': spec['config'].k_attract,
                'weights': _config_weights(spec['config']),
            }
            for spec in ablations
        ],
    }
    (output_dir / 'ablation_manifest.json').write_text(json.dumps(manifest, indent=2))

    summary_rows: list[dict[str, Any]] = []
    for spec in ablations:
        run_dir = output_dir / spec['name']
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {spec['name']} ===")
        print(spec['description'])
        result = run_condensation(
            x0=x0,
            mask=data.mask,
            config=spec['config'],
            stopping=stopping,
            log_every=args.log_every,
            save_every=args.save_every,
            verbose=True,
        )
        summary_rows.append(
            _save_run_bundle(
                run_dir=run_dir,
                name=spec['name'],
                description=spec['description'],
                data=data,
                x0=x0,
                result=result,
                color_map=color_map,
                top_k=args.top_k,
                objective='balanced',
                render_animations=bool(args.render_animations),
            )
            | {
                'coherence_mode': spec['config'].coherence_mode,
                'sigma': float(spec['config'].sigma),
                'sigma_coh': _float_or_none(spec['config'].sigma_coh),
                'k_attract': spec['config'].k_attract,
            }
            | _config_weights(spec['config'])
        )

    summary = pd.DataFrame(summary_rows).sort_values('name').reset_index(drop=True)
    summary.to_csv(output_dir / 'ablation_summary.csv', index=False)
    _plot_ablation_summary(summary, output_dir / 'plot_ablation_summary.png')
    print(f"\nSaved sweep summary to: {output_dir}")


def _build_ablations(
    n_iter: int,
    sigma: float,
    s_local: float,
    sigma_mults: list[float],
) -> list[dict[str, Any]]:
    base_kwargs = dict(
        sigma=sigma,
        delta=3,
        epsilon_r=0.005,
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        lr=1e-4,
        alpha=0.9,
        max_iter=n_iter,
    )
    ablations: list[dict[str, Any]] = [
        {
            'name': 'repel_fid_only',
            'description': 'Repulsion + fidelity only. Attraction fully disabled.',
            'config': CondensationConfig(
                **base_kwargs,
                coherence_mode='uniform',
                k_attract=None,
                lambda_stretch=0.0,
                lambda_bend=0.0,
                epsilon_void=0.0,
                lambda_scale=0.0,
                w_attract=0.0,
                w_repel=1.0,
                w_fidelity=1.0,
                w_elastic=0.0,
                w_void=0.0,
                w_scale=0.0,
            ),
        },
        {
            'name': 'attract_uniform_plus_repel_fid',
            'description': 'Attraction + repulsion + fidelity, but coherence replaced with a uniform observed-pair mask.',
            'config': CondensationConfig(
                **base_kwargs,
                coherence_mode='uniform',
                k_attract=None,
                lambda_stretch=0.0,
                lambda_bend=0.0,
                epsilon_void=0.0,
                lambda_scale=0.0,
                w_attract=1.0,
                w_repel=1.0,
                w_fidelity=1.0,
                w_elastic=0.0,
                w_void=0.0,
                w_scale=0.0,
            ),
        },
    ]

    for mult in sigma_mults:
        tag = str(mult).replace('.', 'p')
        ablations.append(
            {
                'name': f'attract_computed_plus_repel_fid_sigma{tag}',
                'description': f'Attraction + repulsion + fidelity with computed coherence at sigma_coh={mult:.2f} * s_local.',
                'config': CondensationConfig(
                    **base_kwargs,
                    coherence_mode='computed',
                    sigma_coh=float(mult) * s_local,
                    k_attract=None,
                    lambda_stretch=0.0,
                    lambda_bend=0.0,
                    epsilon_void=0.0,
                    lambda_scale=0.0,
                    w_attract=1.0,
                    w_repel=1.0,
                    w_fidelity=1.0,
                    w_elastic=0.0,
                    w_void=0.0,
                    w_scale=0.0,
                ),
            }
        )

    ablations.append(
        {
            'name': 'baseline_current',
            'description': 'Current PBX baseline used for direct comparison against the force-isolated ablations.',
            'config': CondensationConfig(
                sigma=sigma,
                delta=3,
                epsilon_r=0.005,
                lambda_stretch=0.04,
                lambda_bend=0.04,
                fidelity_init_strength=0.25,
                fidelity_half_life=_gamma_from_half_life_iters(70.0),
                epsilon_void=0.014,
                k_attract=20,
                lr=1e-4,
                alpha=0.9,
                max_iter=n_iter,
            ),
        }
    )
    return ablations


def _save_run_bundle(
    *,
    run_dir: Path,
    name: str,
    description: str,
    data: schema.CosmologyData,
    x0: np.ndarray,
    result: Any,
    color_map: dict[str, str],
    top_k: int,
    objective: str,
    render_animations: bool,
) -> dict[str, Any]:
    payload = {
        'positions': result.positions,
        'x0': x0,
        'mask': data.mask,
        'time_values': data.time_values,
        'embryo_ids': data.embryo_ids,
        'labels': data.labels,
    }
    if result.position_history is not None:
        payload['position_history'] = result.position_history
        payload['snapshot_iters'] = np.asarray(result.snapshot_iters, dtype=int)
    np.savez(run_dir / 'condensed_positions.npz', **payload)

    metrics_df = pd.DataFrame(result.metrics_history)
    metrics_df.to_csv(run_dir / 'metrics.csv', index=False)

    fig, _ = plotting.plot_trajectories(
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        title=f'{name} | final',
    )
    fig.savefig(run_dir / 'plot_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, _ = plotting.plot_trajectories(
        x0,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        title=f'{name} | init',
    )
    fig.savefig(run_dir / 'plot_trajectories_init.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    snap_indices = np.linspace(0, len(data.time_values) - 1, min(6, len(data.time_values)), dtype=int)
    snapshot_times = [float(data.time_values[i]) for i in snap_indices]
    fig, _ = plotting.plot_panels(
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        snapshot_times=snapshot_times,
        title=f'{name} | final panels',
    )
    fig.savefig(run_dir / 'plot_panels.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        result.positions,
        data.mask,
        data.time_values,
        labels=data.labels,
        color_map=color_map,
        title=f'{name} | stacked 3D',
    )
    fig.savefig(run_dir / 'plot_stacked_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    _plot_metrics(metrics_df, run_dir / 'plot_metrics.png', title=name)

    if result.position_history is not None:
        if render_animations:
            tc_animation.animate_rotation(
                result.positions,
                data.mask,
                data.time_values,
                labels=data.labels,
                color_map=color_map,
                output_path=run_dir / 'rotation.gif',
                title=f'{name} | rotation',
            )
            tc_animation.animate_iterations(
                result.position_history,
                data.mask,
                data.time_values,
                iter_labels=result.snapshot_iters,
                labels=data.labels,
                color_map=color_map,
                output_path=run_dir / 'iterations.gif',
                fps=4,
                title=f'{name} | iterations',
            )

        ranking = _rank_and_render_candidates(
            output_dir=run_dir,
            metrics_df=metrics_df,
            position_history=result.position_history,
            snapshot_iters=list(result.snapshot_iters),
            final_positions=result.positions,
            final_iter=max(0, result.n_iter - 1),
            mask=data.mask,
            labels=data.labels,
            time_values=data.time_values,
            color_map=color_map,
            objective=objective,
            top_k=top_k,
            title_prefix=name,
        )
    else:
        ranking = pd.DataFrame()

    summary = {
        'name': name,
        'description': description,
        'n_iter': int(result.n_iter),
        'converged': bool(result.converged),
        'final_energy_total': _last_metric(metrics_df, 'energy_total'),
        'final_disp_rms_rel': _last_metric(metrics_df, 'disp_rms_rel'),
        'final_disp_max_rel': _last_metric(metrics_df, 'disp_max_rel'),
        'final_energy_change_rel': _last_metric(metrics_df, 'energy_change_rel'),
        'final_coherence_change_rel': _last_metric(metrics_df, 'coherence_change_rel'),
        'top_geometry_score': _float_or_none(ranking.iloc[0]['geometry_score']) if not ranking.empty else None,
        'top_iter': int(ranking.iloc[0]['iter']) if not ranking.empty else None,
    }
    (run_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def _rank_and_render_candidates(
    *,
    output_dir: Path,
    metrics_df: pd.DataFrame,
    position_history: np.ndarray,
    snapshot_iters: list[int],
    final_positions: np.ndarray,
    final_iter: int,
    mask: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    color_map: dict[str, str],
    objective: str,
    top_k: int,
    title_prefix: str,
) -> pd.DataFrame:
    history = position_history
    iters = list(snapshot_iters)
    if not iters or iters[-1] != final_iter:
        history = np.concatenate([history, final_positions[None, ...]], axis=0)
        iters = iters + [int(final_iter)]

    ranking_dir = output_dir / 'iteration_ranking'
    scores = score_saved_iterations(
        history,
        iters,
        mask,
        labels,
        time_values,
        metrics_df,
        objective=objective,
    )
    save_ranking_outputs(
        scores,
        output_dir=ranking_dir,
        config_payload={
            'geometry_objective': objective,
            'top_k': top_k,
            'n_saved_iterations': len(iters),
            'density_metrics': ['density_knn_mean', 'density_knn_cv'],
        },
    )
    plot_iteration_scores(scores, ranking_dir / 'plot_iteration_scores.png', title=f'{title_prefix} | geometry ranking')

    selected_root = output_dir / 'selected_iterations'
    selected_root.mkdir(parents=True, exist_ok=True)
    for _, row in scores.head(top_k).iterrows():
        iter_idx = int(row['iter'])
        snapshot_index = int(row['snapshot_index'])
        candidate_dir = selected_root / f'iter_{iter_idx:04d}_rank_{int(row["rank"]):02d}'
        metadata = {
            'iter': iter_idx,
            'rank': int(row['rank']),
            'geometry_score': float(row['geometry_score']),
            'selection_score': float(row['geometry_score']),
            'geometry_objective': objective,
            'density_knn_mean': _float_or_none(row.get('density_knn_mean')),
            'density_knn_cv': _float_or_none(row.get('density_knn_cv')),
            'compactness_mean': _float_or_none(row.get('compactness_mean')),
            'centroid_separation_median': _float_or_none(row.get('centroid_separation_median')),
            'crowding_p10': _float_or_none(row.get('crowding_p10')),
            'centroid_shift_mean': _float_or_none(row.get('centroid_shift_mean')),
        }
        render_selected_iteration_bundle(
            positions=history[snapshot_index],
            mask=mask,
            time_values=time_values,
            labels=labels,
            color_map=color_map,
            output_dir=candidate_dir,
            title_prefix=title_prefix,
            snapshot_iter=iter_idx,
            metadata=metadata,
        )
    scores.head(top_k).to_csv(selected_root / 'top_k_candidates.csv', index=False)
    return scores


def _plot_metrics(metrics_df: pd.DataFrame, output_path: Path, title: str = '') -> None:
    cols_labels = [
        ('energy_total', 'Total energy'),
        ('disp_rms_rel', 'RMS displacement (rel)'),
        ('energy_change_rel', 'Energy change (rel)'),
        ('coherence_change_rel', 'Coherence change (rel)'),
    ]
    available = [(c, l) for c, l in cols_labels if c in metrics_df.columns]
    if not available:
        return
    ncols = min(2, len(available))
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, available):
        ax.plot(metrics_df['iter'], metrics_df[col], lw=1.4)
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(available):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _plot_ablation_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    metrics = [
        ('top_geometry_score', 'Top geometry score'),
        ('final_energy_total', 'Final total energy'),
        ('final_disp_rms_rel', 'Final RMS displacement'),
        ('final_coherence_change_rel', 'Final coherence change'),
    ]
    x = np.arange(len(summary))
    labels = summary['name'].tolist()
    for ax, (col, title) in zip(axes, metrics):
        vals = summary[col].astype(float)
        ax.bar(x, vals, color='#4C78A8')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def _config_weights(config: CondensationConfig) -> dict[str, float]:
    return {
        'w_attract': float(config.w_attract),
        'w_repel': float(config.w_repel),
        'w_fidelity': float(config.w_fidelity),
        'w_elastic': float(config.w_elastic),
        'w_void': float(config.w_void),
        'w_scale': float(config.w_scale),
    }


def _last_metric(metrics_df: pd.DataFrame, col: str) -> float | None:
    if col not in metrics_df.columns or metrics_df.empty:
        return None
    return _float_or_none(metrics_df.iloc[-1][col])


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _load_cosmology_data(input_path: str, input_type: str) -> schema.CosmologyData:
    if input_type == 'multiclass':
        return schema.from_multiclass_csv(input_path, label_col='genotype')
    if input_type == 'pairwise':
        return schema.from_pairwise_margin_csv(input_path, label_col='genotype')

    df = pd.read_csv(input_path, nrows=5)
    has_prob = any(c.startswith('p_') or c.startswith('pred_proba_') for c in df.columns)
    has_pairwise = any('_vs_' in c for c in df.columns)
    if has_pairwise and not has_prob:
        return schema.from_pairwise_margin_csv(input_path, label_col='genotype')
    if has_prob:
        return schema.from_multiclass_csv(input_path, label_col='genotype')
    raise ValueError(
        f'Could not infer input type for {input_path}. '
        "Expected probability columns ('p_*') or pairwise columns ('*_vs_*')."
    )


if __name__ == '__main__':
    main()
