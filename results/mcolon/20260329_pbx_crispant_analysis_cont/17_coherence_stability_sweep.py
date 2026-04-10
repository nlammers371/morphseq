"""
17_coherence_stability_sweep.py
-------------------------------
Sweep temporal coherence stability across saved condensation iterations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from trajectory_cosmology.condensation.coherence.compute import compute_coherence
from trajectory_cosmology.condensation.geometry_refs import estimate_local_spacing_ref


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coherence stability sweep across saved condensation iterations.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--sigma-mults", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    p.add_argument("--delta", type=int, default=3)
    p.add_argument("--viz-iters", type=int, nargs="+", default=None)
    return p.parse_args()


def load_history(run_dir: Path):
    npz = np.load(run_dir / "condensed_positions.npz", allow_pickle=True)
    x0 = npz["x0"]
    mask = npz["mask"]
    history = npz["position_history"]
    snapshot_iters = np.asarray(npz["snapshot_iters"], dtype=int)
    final_positions = npz["positions"]
    time_values = npz["time_values"]
    metrics_df = pd.read_csv(run_dir / "metrics.csv")
    final_iter = int(metrics_df["iter"].iloc[-1]) if "iter" in metrics_df.columns else int(snapshot_iters[-1])
    if snapshot_iters[-1] != final_iter:
        history = np.concatenate([history, final_positions[None, ...]], axis=0)
        snapshot_iters = np.concatenate([snapshot_iters, [final_iter]])
    return x0, history, mask, snapshot_iters, time_values


def coherence_similarity(Ca: np.ndarray, Cb: np.ndarray) -> float:
    vals = []
    n = Ca.shape[0]
    tri = np.triu_indices(n, k=1)
    for t in range(Ca.shape[2]):
        a = Ca[:, :, t][tri]
        b = Cb[:, :, t][tri]
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() < 3:
            continue
        aa = a[keep]
        bb = b[keep]
        if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            vals.append(float(np.mean(np.isclose(aa, bb, atol=1e-8))))
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def similarity_matrix(history: np.ndarray, mask: np.ndarray, sigma: float, delta: int) -> tuple[np.ndarray, list[np.ndarray]]:
    coherence_stack = [compute_coherence(history[i], mask, sigma=sigma, delta=delta) for i in range(history.shape[0])]
    n = len(coherence_stack)
    mat = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            sim = coherence_similarity(coherence_stack[i], coherence_stack[j])
            mat[i, j] = sim
            mat[j, i] = sim
    return mat, coherence_stack


def previous_similarity_df(mat: np.ndarray, snapshot_iters: np.ndarray, sigma_mult: float) -> pd.DataFrame:
    rows = []
    for i in range(1, len(snapshot_iters)):
        rows.append({
            "iter": int(snapshot_iters[i]),
            "prev_iter": int(snapshot_iters[i - 1]),
            "similarity": float(mat[i - 1, i]),
            "sigma_mult": float(sigma_mult),
        })
    return pd.DataFrame(rows)


def plot_heatmaps(sim_by_mult: dict[float, np.ndarray], snapshot_iters: np.ndarray, output_path: Path) -> None:
    mults = list(sim_by_mult.keys())
    ncols = min(2, len(mults))
    nrows = int(np.ceil(len(mults) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, mult in zip(axes, mults):
        mat = sim_by_mult[mult]
        im = ax.imshow(mat, origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
        tick_idx = np.linspace(0, len(snapshot_iters) - 1, min(8, len(snapshot_iters)), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(snapshot_iters[tick_idx], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(snapshot_iters[tick_idx], fontsize=8)
        ax.set_title(f"sigma_mult={mult:.2f}", fontsize=10)
        ax.set_xlabel("iteration", fontsize=9)
        ax.set_ylabel("iteration", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(mults):]:
        ax.set_visible(False)
    fig.suptitle("Coherence stability across saved iterations", fontsize=11, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_previous(prev_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for mult, sub in prev_df.groupby("sigma_mult"):
        ax.plot(sub["iter"], sub["similarity"], marker="o", ms=3, lw=1.5, label=f"sigma_mult={mult:.2f}")
    ax.set_xlabel("iteration", fontsize=9)
    ax.set_ylabel("coherence similarity to previous saved iteration", fontsize=9)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def resolve_viz_indices(snapshot_iters: np.ndarray, requested: list[int] | None) -> list[int]:
    if requested:
        idxs = [int(np.argmin(np.abs(snapshot_iters - int(v)))) for v in requested]
        return sorted(set(idxs))
    return sorted(set([0, len(snapshot_iters) // 2, len(snapshot_iters) - 1]))


def plot_coherence_panels(coherence_stack: list[np.ndarray], snapshot_iters: np.ndarray, time_values: np.ndarray, output_dir: Path, sigma_mult: float, viz_indices: list[int]) -> None:
    time_idx = np.linspace(0, len(time_values) - 1, min(3, len(time_values)), dtype=int)
    for idx in viz_indices:
        C = coherence_stack[idx]
        fig, axes = plt.subplots(1, len(time_idx), figsize=(4.8 * len(time_idx), 4.2), squeeze=False)
        axes = axes.ravel()
        for ax, t in zip(axes, time_idx):
            im = ax.imshow(C[:, :, int(t)], vmin=0.0, vmax=1.0, cmap="magma")
            ax.set_title(f"iter {int(snapshot_iters[idx])} | {time_values[int(t)]:.0f} hpf", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Coherence matrices | sigma_mult={sigma_mult:.2f}", fontsize=10, y=0.98)
        fig.tight_layout()
        fig.savefig(output_dir / f"coherence_iter_{int(snapshot_iters[idx]):04d}_sigma_{str(sigma_mult).replace('.', 'p')}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "coherence_stability_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    x0, history, mask, snapshot_iters, time_values = load_history(run_dir)
    s_local = estimate_local_spacing_ref(x0, mask, k=5)
    sigma_mults = [float(x) for x in args.sigma_mults]
    sigma_values = [mult * s_local for mult in sigma_mults]

    sim_by_mult: dict[float, np.ndarray] = {}
    prev_frames = []
    summary_rows = []
    viz_indices = resolve_viz_indices(snapshot_iters, args.viz_iters)
    viz_dir = output_dir / "coherence_matrices"
    viz_dir.mkdir(exist_ok=True)

    for mult, sigma in zip(sigma_mults, sigma_values):
        mat, coherence_stack = similarity_matrix(history, mask, sigma=sigma, delta=args.delta)
        sim_by_mult[mult] = mat
        pd.DataFrame(mat, index=snapshot_iters, columns=snapshot_iters).to_csv(output_dir / f"coherence_similarity_sigma_{str(mult).replace('.', 'p')}.csv")
        prev_df = previous_similarity_df(mat, snapshot_iters, mult)
        prev_frames.append(prev_df)
        tri = mat[np.triu_indices_from(mat, k=1)]
        tri = tri[np.isfinite(tri)]
        summary_rows.append({
            "sigma_mult": mult,
            "sigma": sigma,
            "s_local": s_local,
            "mean_pairwise_similarity": float(np.mean(tri)) if len(tri) else np.nan,
            "median_pairwise_similarity": float(np.median(tri)) if len(tri) else np.nan,
            "mean_prev_similarity": float(prev_df["similarity"].mean()) if not prev_df.empty else np.nan,
            "min_prev_similarity": float(prev_df["similarity"].min()) if not prev_df.empty else np.nan,
        })
        plot_coherence_panels(coherence_stack, snapshot_iters, time_values, viz_dir, mult, viz_indices)

    prev_all = pd.concat(prev_frames, ignore_index=True)
    prev_all.to_csv(output_dir / "coherence_previous_iteration_similarity.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "coherence_similarity_summary.csv", index=False)

    plot_heatmaps(sim_by_mult, snapshot_iters, output_dir / "coherence_similarity_heatmaps.png")
    plot_previous(prev_all, output_dir / "coherence_previous_iteration_similarity.png")

    manifest = {
        "run_dir": str(run_dir),
        "delta": int(args.delta),
        "s_local": float(s_local),
        "sigma_mults": sigma_mults,
        "sigma_values": sigma_values,
        "kNN_gating": None,
        "viz_iterations": [int(snapshot_iters[idx]) for idx in viz_indices],
    }
    (output_dir / "coherence_stability_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved coherence stability sweep to: {output_dir}")


if __name__ == "__main__":
    main()
