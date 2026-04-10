"""
16_knn_stability_sweep.py
-------------------------
Sweep k-nearest-neighbor stability across saved condensation iterations.
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

from trajectory_cosmology.knn_stability_metrics import (
    finite_points_at_time,
    jaccard_similarity_matrix,
    knn_edge_index,
    previous_iteration_similarity,
)

GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
    "pbx1b_crispant": "#9467bd",
    "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kNN stability sweep across saved condensation iterations.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--viz-iters", type=int, nargs="+", default=None,
                   help="Specific saved iterations to visualize as kNN graphs. Defaults to first, mid, final.")
    return p.parse_args()


def load_history(run_dir: Path):
    npz = np.load(run_dir / "condensed_positions.npz", allow_pickle=True)
    mask = npz["mask"]
    history = npz["position_history"]
    snapshot_iters = np.asarray(npz["snapshot_iters"], dtype=int)
    final_positions = npz["positions"]
    labels = npz["labels"]
    time_values = npz["time_values"]
    metrics_df = pd.read_csv(run_dir / "metrics.csv")
    final_iter = int(metrics_df["iter"].iloc[-1]) if "iter" in metrics_df.columns else int(snapshot_iters[-1])
    if snapshot_iters[-1] != final_iter:
        history = np.concatenate([history, final_positions[None, ...]], axis=0)
        snapshot_iters = np.concatenate([snapshot_iters, [final_iter]])
    return history, mask, snapshot_iters, labels, time_values


def plot_heatmaps(sim_by_k: dict[int, np.ndarray], snapshot_iters: np.ndarray, output_path: Path) -> None:
    ks = sorted(sim_by_k)
    ncols = min(2, len(ks))
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, k in zip(axes, ks):
        mat = sim_by_k[k]
        im = ax.imshow(mat, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        tick_idx = np.linspace(0, len(snapshot_iters) - 1, min(8, len(snapshot_iters)), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(snapshot_iters[tick_idx], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels(snapshot_iters[tick_idx], fontsize=8)
        ax.set_title(f"k={k}", fontsize=10)
        ax.set_xlabel("iteration", fontsize=9)
        ax.set_ylabel("iteration", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(ks):]:
        ax.set_visible(False)
    fig.suptitle("kNN neighborhood stability across saved iterations", fontsize=11, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_previous_similarity(prev_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, sub in prev_df.groupby("k"):
        ax.plot(sub["iter"], sub["similarity"], marker="o", ms=3, lw=1.5, label=f"k={k}")
    ax.set_xlabel("iteration", fontsize=9)
    ax.set_ylabel("similarity to previous saved iteration", fontsize=9)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_knn_graph_panels(
    positions: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    ks: list[int],
    output_path: Path,
    *,
    iteration_label: int,
) -> None:
    time_indices = np.linspace(0, len(time_values) - 1, min(3, len(time_values)), dtype=int)
    nrows = len(time_indices)
    ncols = len(ks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), squeeze=False)

    for r, t_idx in enumerate(time_indices):
        obs, pts = finite_points_at_time(positions, mask, int(t_idx))
        label_slice = labels[obs] if len(obs) else np.asarray([])
        for c, k in enumerate(ks):
            ax = axes[r, c]
            ax.set_title(f"iter {iteration_label} | {time_values[t_idx]:.0f} hpf | k={k}", fontsize=9)
            if len(obs) <= 1:
                ax.set_axis_off()
                continue
            edges = knn_edge_index(pts, k)
            for a, b in edges:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="#999999", alpha=0.25, lw=0.5)
            for genotype in np.unique(label_slice):
                idx = np.where(label_slice == genotype)[0]
                ax.scatter(
                    pts[idx, 0],
                    pts[idx, 1],
                    s=10,
                    color=GENOTYPE_COLORS.get(str(genotype), "#555555"),
                    alpha=0.8,
                    label=str(genotype) if (r == 0 and c == 0) else None,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
    handles, labels_unique = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_unique, loc="upper center", ncol=min(5, len(handles)), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def resolve_viz_indices(snapshot_iters: np.ndarray, requested: list[int] | None) -> list[int]:
    if requested:
        picked = []
        for iter_value in requested:
            idx = int(np.argmin(np.abs(snapshot_iters - int(iter_value))))
            picked.append(idx)
        return sorted(set(picked))
    return sorted(set([0, len(snapshot_iters) // 2, len(snapshot_iters) - 1]))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "knn_stability_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    history, mask, snapshot_iters, labels, time_values = load_history(run_dir)
    ks = sorted(set(int(k) for k in args.ks))

    sim_by_k = jaccard_similarity_matrix(history, mask, ks)
    prev_rows = []
    summary_rows = []
    for k in ks:
        mat = sim_by_k[k]
        pd.DataFrame(mat, index=snapshot_iters, columns=snapshot_iters).to_csv(output_dir / f"knn_similarity_k{k:02d}.csv")
        prev = previous_iteration_similarity(mat, snapshot_iters)
        prev_df = pd.DataFrame(prev, columns=["iter", "prev_iter", "similarity"])
        prev_df["k"] = k
        prev_rows.append(prev_df)
        tri = mat[np.triu_indices_from(mat, k=1)]
        tri = tri[np.isfinite(tri)]
        summary_rows.append({
            "k": k,
            "mean_pairwise_similarity": float(np.mean(tri)) if len(tri) else np.nan,
            "median_pairwise_similarity": float(np.median(tri)) if len(tri) else np.nan,
            "mean_prev_similarity": float(prev_df["similarity"].mean()) if not prev_df.empty else np.nan,
            "min_prev_similarity": float(prev_df["similarity"].min()) if not prev_df.empty else np.nan,
        })

    prev_all = pd.concat(prev_rows, ignore_index=True)
    prev_all.to_csv(output_dir / "knn_previous_iteration_similarity.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "knn_similarity_summary.csv", index=False)

    plot_heatmaps(sim_by_k, snapshot_iters, output_dir / "knn_similarity_heatmaps.png")
    plot_previous_similarity(prev_all, output_dir / "knn_previous_iteration_similarity.png")

    viz_dir = output_dir / "knn_graphs"
    viz_dir.mkdir(exist_ok=True)
    for idx in resolve_viz_indices(snapshot_iters, args.viz_iters):
        iter_label = int(snapshot_iters[idx])
        plot_knn_graph_panels(
            history[idx],
            mask,
            labels,
            time_values,
            ks,
            viz_dir / f"knn_graphs_iter_{iter_label:04d}.png",
            iteration_label=iter_label,
        )

    manifest = {
        "run_dir": str(run_dir),
        "ks": ks,
        "n_saved_iterations": int(len(snapshot_iters)),
        "viz_iterations": [int(snapshot_iters[idx]) for idx in resolve_viz_indices(snapshot_iters, args.viz_iters)],
    }
    (output_dir / "knn_stability_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved kNN stability sweep to: {output_dir}")


if __name__ == "__main__":
    main()
