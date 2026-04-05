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

from trajectory_cosmology import plotting, schema
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
    p = argparse.ArgumentParser(description='Attraction-weight sweep with fixed computed coherence.')
    p.add_argument(
        '--input',
        default='results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_positioning_pairwise_bin4_perm500/pairwise_raw_vectors.csv',
    )
    p.add_argument('--input-type', choices=['auto', 'multiclass', 'pairwise'], default='pairwise')
    p.add_argument(
        '--x0-path',
        default='results/mcolon/20260329_pbx_crispant_analysis_cont/results/pairwise_raw_condensation_iter1000_ranked_umap/x0_init.npz',
    )
    p.add_argument(
        '--output-dir',
        default='results/mcolon/20260329_pbx_crispant_analysis_cont/results/attraction_weight_sweep_iter500',
    )
    p.add_argument('--n-iter', type=int, default=500)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--top-k', type=int, default=1)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--sigma', type=float, default=0.5)
    p.add_argument('--sigma-coh-mult', type=float, default=0.5)
    p.add_argument('--w-attract-values', default='0,0.25,0.5,1,2,4')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_iter = 100
        args.save_every = 10
        args.top_k = 1
        args.log_every = 10
        w_values = [0.0, 1.0, 4.0]
    else:
        w_values = [float(x) for x in args.w_attract_values.split(',') if x.strip()]

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
    sigma_coh = float(args.sigma_coh_mult) * s_local
    color_map = {g: GENOTYPE_COLORS.get(g, '#555555') for g in np.unique(data.labels)}
    stopping = StoppingConfig(
        disp_max_rel_threshold=None,
        disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None,
        coherence_change_rel_threshold=None,
    )

    sweep_specs = _build_sweep_specs(args.n_iter, args.sigma, sigma_coh, w_values)
    manifest = {
        'input': str(Path(args.input)),
        'x0_path': str(Path(args.x0_path)),
        'n_iter': int(args.n_iter),
        'save_every': int(args.save_every),
        'top_k': int(args.top_k),
        'sigma': float(args.sigma),
        'sigma_coh_mult': float(args.sigma_coh_mult),
        'sigma_coh': sigma_coh,
        's_local': s_local,
        'w_attract_values': w_values,
        'smoke': bool(args.smoke),
    }
    (output_dir / 'sweep_manifest.json').write_text(json.dumps(manifest, indent=2))

    summary_rows: list[dict[str, Any]] = []
    reference_positions: np.ndarray | None = None
    reference_name = 'w_attract_0p00'

    for spec in sweep_specs:
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
        if spec['name'] == reference_name:
            reference_positions = result.positions.copy()
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
            )
            | {
                'w_attract': float(spec['config'].w_attract),
                'sigma': float(spec['config'].sigma),
                'sigma_coh': float(spec['config'].sigma_coh),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values('w_attract').reset_index(drop=True)
    if reference_positions is None:
        raise RuntimeError('Reference w_attract=0 run did not complete.')
    summary = _append_reference_distortion(summary, output_dir, reference_positions, data.mask)
    summary.to_csv(output_dir / 'attraction_weight_summary.csv', index=False)
    _plot_weight_summary(summary, output_dir / 'plot_attraction_weight_summary.png')
    print(f"\nSaved sweep summary to: {output_dir}")

def _build_sweep_specs(
    n_iter: int,
    sigma: float,
    sigma_coh: float,
    w_values: list[float],
) -> list[dict[str, Any]]:
    base_kwargs = dict(
        sigma=sigma,
        sigma_coh=sigma_coh,
        temporal_cohere_mode='computed',
        temporal_cohere_window=3,
        epsilon_r=0.005,
        fidelity_init_strength=0.25,
        fidelity_half_life=_gamma_from_half_life_iters(70.0),
        solver_lr=1e-4,
        solver_momentum=0.9,
        solver_max_iter=n_iter,
        attract_k=None,
        lambda_stretch=0.0,
        lambda_bend=0.0,
        epsilon_void=0.0,
        lambda_scale=0.0,
        w_repel=1.0,
        w_fidelity=1.0,
        w_elastic=0.0,
        w_void=0.0,
        w_scale=0.0,
    )
    specs: list[dict[str, Any]] = []
    for w in w_values:
        tag = f'{w:.2f}'.replace('.', 'p')
        specs.append(
            {
                'name': f'w_attract_{tag}',
                'description': f'Computed coherence fixed; attraction weight set to {w:.2f}.',
                'config': CondensationConfig(**base_kwargs, w_attract=float(w)),
            }
        )
    return specs


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
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f'{name} | final',
    )
    fig.savefig(run_dir / 'plot_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, _ = plotting.plot_trajectories(
        x0, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f'{name} | init',
    )
    fig.savefig(run_dir / 'plot_trajectories_init.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    snap_indices = np.linspace(0, len(data.time_values) - 1, min(6, len(data.time_values)), dtype=int)
    snapshot_times = [float(data.time_values[i]) for i in snap_indices]
    fig, _ = plotting.plot_panels(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        snapshot_times=snapshot_times,
        title=f'{name} | final panels',
    )
    fig.savefig(run_dir / 'plot_panels.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        result.positions, data.mask, data.time_values,
        labels=data.labels, color_map=color_map,
        title=f'{name} | stacked 3D',
    )
    fig.savefig(run_dir / 'plot_stacked_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    _plot_metrics(metrics_df, run_dir / 'plot_metrics.png', title=name)

    ranking = pd.DataFrame()
    if result.position_history is not None:
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
    scores = score_saved_iterations(history, iters, mask, labels, time_values, metrics_df, objective=objective)
    save_ranking_outputs(
        scores,
        output_dir=ranking_dir,
        config_payload={
            'geometry_objective': objective,
            'top_k': top_k,
            'n_saved_iterations': len(iters),
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
            'w_attract_sweep': True,
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


def _append_reference_distortion(
    summary: pd.DataFrame,
    output_dir: Path,
    reference_positions: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        run_dir = output_dir / str(row['name'])
        payload = np.load(run_dir / 'condensed_positions.npz')
        positions = np.asarray(payload['positions'], dtype=float)
        delta = positions - reference_positions
        delta_norm = np.linalg.norm(delta, axis=-1)
        delta_norm = delta_norm[np.asarray(mask, dtype=bool)]
        rows.append({
            'name': row['name'],
            'mean_shift_from_w0': float(np.nanmean(delta_norm)),
            'max_shift_from_w0': float(np.nanmax(delta_norm)),
        })
    shifts = pd.DataFrame(rows)
    return summary.merge(shifts, on='name', how='left')


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
        ax.grid(True, solver_momentum=0.25)
    for ax in axes[len(available):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _plot_weight_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    x = summary['w_attract'].astype(float)
    metrics = [
        ('top_geometry_score', 'Top geometry score'),
        ('final_disp_rms_rel', 'Final RMS displacement'),
        ('mean_shift_from_w0', 'Mean shift from w=0'),
        ('max_shift_from_w0', 'Max shift from w=0'),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(x, summary[col].astype(float), marker='o', lw=1.8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, solver_momentum=0.25)
        ax.set_xlabel('w_attract', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


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
