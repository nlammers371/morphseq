"""
run_condensation_experiment.py
------------------------------
Thin driver for a single condensation experiment run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from trajectory_cosmology.schema import from_multiclass_csv, from_pairwise_margin_csv
from trajectory_cosmology.init_embedding import pca_init, aligned_umap_init
from trajectory_cosmology.condensation.api import run_condensation
from trajectory_cosmology.condensation.state import CondensationConfig
from trajectory_cosmology.condensation.stopping import StoppingConfig
from trajectory_cosmology import plotting


def _parse_optional_int(value: str) -> int | None:
    if value.lower() == 'none':
        return None
    return int(value)


def parse_args():
    p = argparse.ArgumentParser(description="Run one condensation experiment.")
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--input-type", choices=["multiclass", "pairwise"], default="multiclass")
    p.add_argument("--prob-cols", default=None,
                   help="Comma-separated feature column names. Auto-detected if omitted.")
    p.add_argument("--label-col", default="true_class")
    p.add_argument("--init", choices=["pca", "umap"], default="pca")
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--delta", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda-stretch", type=float, default=0.1)
    p.add_argument("--lambda-bend", type=float, default=0.05)
    p.add_argument("--mu0", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--epsilon-r", type=float, default=0.01)
    p.add_argument("--eta", type=float, default=1e-4)
    p.add_argument("--k-attract", type=_parse_optional_int, default=15,
                   help="kNN attraction degree; pass 'none' for all-pairs attraction")
    p.add_argument("--subtract-mean-attraction", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prob_cols = args.prob_cols.split(",") if args.prob_cols else None
    if args.input_type == "multiclass":
        data = from_multiclass_csv(args.input, prob_cols=prob_cols, label_col=args.label_col)
    else:
        data = from_pairwise_margin_csv(args.input, margin_cols=prob_cols, label_col=args.label_col)

    print(f"Loaded: {data.features.shape[0]} embryos × {data.features.shape[1]} time bins × {data.features.shape[2]} features")
    print(f"Observed entries: {data.mask.sum()} / {data.mask.size}")
    print(f"Conditions: {sorted(set(data.labels.tolist()))}")

    print(f"Initializing with {args.init} ...")
    if args.init == "pca":
        x0 = pca_init(data.features, data.mask)
    else:
        x0 = aligned_umap_init(data.features, data.mask)

    config = CondensationConfig(
        sigma=args.sigma,
        delta=args.delta,
        lr=args.lr,
        lambda_stretch=args.lambda_stretch,
        lambda_bend=args.lambda_bend,
        mu0=args.mu0,
        gamma=args.gamma,
        epsilon_r=args.epsilon_r,
        eta=args.eta,
        k_attract=args.k_attract,
        subtract_mean_attraction=args.subtract_mean_attraction,
        max_iter=args.n_iter,
    )
    stopping = StoppingConfig(patience=args.n_iter + 1)

    print(f"Running {args.n_iter} iterations ...")
    result = run_condensation(
        x0=x0,
        mask=data.mask,
        config=config,
        stopping=stopping,
        log_every=max(1, args.n_iter // 20),
        save_every=args.save_every,
        verbose=True,
    )
    print(f"Done. Converged: {result.converged} | Iterations: {result.n_iter}")

    metrics_df = pd.DataFrame(result.metrics_history)
    metrics_path = output_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics: {metrics_path}")

    pos_path = output_dir / 'condensed_positions.npz'
    np.savez(
        pos_path,
        positions=result.positions,
        x0=result.x0,
        mask=result.mask,
        time_values=data.time_values,
    )
    print(f"Positions: {pos_path}")

    fig, _ = plotting.plot_trajectories(
        result.positions, result.mask, data.time_values, labels=data.labels,
        title=f"Condensed trajectories ({args.init} init, {result.n_iter} iters)",
    )
    fig.savefig(output_dir / 'plot_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    T = len(data.time_values)
    snap_idx = np.linspace(0, T - 1, min(6, T), dtype=int)
    snapshot_times = [float(data.time_values[i]) for i in snap_idx]
    fig, _ = plotting.plot_panels(
        result.positions, result.mask, data.time_values,
        labels=data.labels, snapshot_times=snapshot_times,
        title=f"Panels ({args.init} init)",
    )
    fig.savefig(output_dir / 'plot_panels.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, col, ylabel in zip(
        axes,
        ['energy_total', 'disp_rms_rel', 'energy_change_rel', 'coherence_change_rel'],
        ['Total energy', 'RMS displacement (rel)', 'Energy change (rel)', 'Coherence change (rel)'],
    ):
        if col in metrics_df.columns:
            ax.plot(metrics_df['iter'], metrics_df[col])
            ax.set_xlabel('Iteration')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
    fig.suptitle(
        f"Convergence metrics — {args.init} init, sigma={args.sigma}, delta={args.delta}, k={args.k_attract}")
    fig.tight_layout()
    fig.savefig(output_dir / 'plot_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figures saved to {output_dir}")

    last = metrics_df.iloc[-1]
    print("\n--- Final iteration summary ---")
    for col in ['energy_total', 'disp_rms_rel', 'disp_max_rel', 'energy_change_rel', 'coherence_change_rel']:
        if col in last:
            print(f"  {col}: {last[col]:.6f}")


if __name__ == '__main__':
    main()
