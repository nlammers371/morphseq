"""
animate_slice_sandbox.py
------------------------
Generate animated GIFs showing iteration-by-iteration evolution of force law dynamics.

Takes the metrics_history.csv from slice_sandbox.py runs and creates side-by-side
comparison GIFs: good vs bad parameter combinations.

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/animate_slice_sandbox.py \\
      --good-run results/slice_sandbox_all_variants_v1/separated/run_0008 \\
      --bad-run results/slice_sandbox_all_variants_v1/separated/run_0037 \\
      --output-dir /tmp/animations
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from PIL import Image
except ImportError:
    Image = None

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from trajectory_cosmology.condensation.forces import attraction, repulsion


# ===========================================================================
# Helpers (copied from slice_sandbox.py for self-containment)
# ===========================================================================

def _pack(pos2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = pos2d.shape[0]
    return pos2d[:, None, :].copy(), np.ones((N, 1), dtype=bool)


def _unpack(positions_3d: np.ndarray) -> np.ndarray:
    return positions_3d[:, 0, :]


def _make_coherence(labels: np.ndarray, mode: str) -> np.ndarray:
    N = len(labels)
    if mode == "oracle":
        C2d = (labels[:, None] == labels[None, :]).astype(float)
    else:
        C2d = np.ones((N, N), dtype=float)
    np.fill_diagonal(C2d, 0.0)
    return C2d[:, :, None]


def radial_spread(pos: np.ndarray) -> float:
    center = pos.mean(axis=0)
    return float(np.sqrt(((pos - center) ** 2).sum(axis=1).mean()))


def _cluster_metrics(pos: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    groups = sorted(set(labels.tolist()))
    centroids = np.array([pos[labels == g].mean(axis=0) for g in groups])
    cdiffs = centroids[:, None, :] - centroids[None, :, :]
    cdists = np.sqrt((cdiffs ** 2).sum(axis=-1))
    iu = np.triu_indices(len(groups), k=1)
    centroid_distance = float(cdists[iu].mean()) if len(iu[0]) > 0 else 0.0
    within_rms_list = []
    for g, c in zip(groups, centroids):
        pts = pos[labels == g]
        rms = float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean()))
        within_rms_list.append(rms)
    within_cluster_rms = float(np.mean(within_rms_list))
    separation_ratio = centroid_distance / (within_cluster_rms + 1e-8)
    global_spread = radial_spread(pos)
    return {
        "centroid_distance": centroid_distance,
        "within_cluster_rms": within_cluster_rms,
        "separation_ratio": separation_ratio,
        "global_spread": global_spread,
    }


_CLUSTER_COLORS = ["#2166AC", "#B2182B", "#4DAC26", "#F1A340"]


def _resolve_color_map(labels: np.ndarray) -> dict[int, str]:
    groups = sorted(set(labels.tolist()))
    return {g: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, g in enumerate(groups)}


# ===========================================================================
# Animation generation
# ===========================================================================

def _load_run_config(run_dir: Path) -> dict:
    """Read this run's parameters from the parent summary.csv."""
    summary_path = run_dir.parent / "summary.csv"
    run_id = run_dir.name.replace("run_", "")
    df = pd.read_csv(summary_path)
    row = df[df["run_id"] == run_id]
    if len(row) == 0:
        # Try as int
        row = df[df["run_id"] == int(run_id)]
    if len(row) == 0:
        raise ValueError(f"Run {run_id} not found in {summary_path}")
    row = row.iloc[0]
    k = row["k_attract"]
    return {
        "sigma": float(row["sigma"]),
        "epsilon_r": float(row["epsilon_r"]),
        "k_attract": None if (pd.isna(k) or str(k).lower() == "none") else int(k),
        "subtract_mean": bool(row["subtract_mean"]),
        "coherence_mode": str(row["coherence_mode"]),
        "lr": float(row.get("lr", 5e-4)),
    }


def _replay_run(
    dataset_pos: np.ndarray,
    dataset_labels: np.ndarray,
    config_dict: dict,
    metrics_df: pd.DataFrame,
) -> list[np.ndarray]:
    """Replay the run to get positions at each iteration.

    config_dict must contain: sigma, epsilon_r, k_attract, subtract_mean,
    coherence_mode, lr.
    """
    sigma = config_dict["sigma"]
    epsilon_r = config_dict["epsilon_r"]
    k_attract = config_dict["k_attract"]
    subtract_mean = config_dict["subtract_mean"]
    coherence_mode = config_dict["coherence_mode"]
    lr = config_dict.get("lr", 5e-4)
    alpha = 0.9

    positions_3d, mask = _pack(dataset_pos)
    coherence = _make_coherence(dataset_labels, coherence_mode)
    velocities = np.zeros_like(positions_3d)

    positions_history = [_unpack(positions_3d).copy()]

    n_iter = len(metrics_df)
    for i in range(n_iter):
        if k_attract == 0:
            e_att, g_att = 0.0, np.zeros_like(positions_3d)
        else:
            e_att, g_att = attraction(
                positions_3d, mask, coherence,
                sigma=sigma,
                k_attract=k_attract,
                subtract_mean=subtract_mean,
            )
        e_rep, g_rep = repulsion(positions_3d, mask, epsilon_r=epsilon_r, eta=1e-4)
        grad = g_att + g_rep
        velocities = alpha * velocities - lr * grad
        positions_3d = positions_3d + velocities
        positions_history.append(_unpack(positions_3d).copy())

    return positions_history


def _make_frame(
    pos: np.ndarray,
    labels: np.ndarray,
    color_map: dict[int, str],
    title: str,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
) -> np.ndarray:
    """Create a matplotlib figure as a numpy array (RGB)."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    for g, c in color_map.items():
        mask = labels == g
        ax.scatter(pos[mask, 0], pos[mask, 1], s=25, alpha=0.7, color=c, label=f"cluster {g}")
        centroid = pos[mask].mean(axis=0)
        ax.scatter(*centroid, s=250, marker="+", color="black", linewidths=2.5, zorder=5)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def generate_comparison_gif(
    good_run_dir: Path,
    bad_run_dir: Path,
    dataset: dict,  # must have 'pos' and 'labels'
    output_dir: Path,
    max_frames: int = 100,
    frame_interval: int = 3,
) -> None:
    """Generate side-by-side GIF comparing good vs bad runs with actual cluster positions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset_pos = dataset["pos"]
    dataset_labels = dataset["labels"]
    color_map = _resolve_color_map(dataset_labels)

    # Load metrics for both runs
    good_metrics = pd.read_csv(good_run_dir / "metrics_history.csv")
    bad_metrics = pd.read_csv(bad_run_dir / "metrics_history.csv")

    print(f"Good run: {good_run_dir.name}")
    print(f"  Metrics shape: {good_metrics.shape}")
    print(f"  Final sep_ratio: {good_metrics['separation_ratio'].iloc[-1]:.3f}")
    print(f"  collapse_score: {good_metrics['global_spread'].iloc[-1] / good_metrics['global_spread'].iloc[0]:.3f}")

    print(f"\nBad run: {bad_run_dir.name}")
    print(f"  Metrics shape: {bad_metrics.shape}")
    print(f"  Final sep_ratio: {bad_metrics['separation_ratio'].iloc[-1]:.3f}")
    print(f"  collapse_score: {bad_metrics['global_spread'].iloc[-1] / bad_metrics['global_spread'].iloc[0]:.3f}")

    good_config = _load_run_config(good_run_dir)
    bad_config = _load_run_config(bad_run_dir)
    print(f"\nGood config: k={good_config['k_attract']} sigma={good_config['sigma']:.3f} eps_r={good_config['epsilon_r']:.5f} lr={good_config['lr']}")
    print(f"Bad config:  k={bad_config['k_attract']} sigma={bad_config['sigma']:.3f} eps_r={bad_config['epsilon_r']:.5f} lr={bad_config['lr']}")

    print("\nReplaying good run to get position history...")
    good_positions = _replay_run(dataset_pos, dataset_labels, good_config, good_metrics)
    print(f"  Got {len(good_positions)} position snapshots")

    print("Replaying bad run to get position history...")
    bad_positions = _replay_run(dataset_pos, dataset_labels, bad_config, bad_metrics)
    print(f"  Got {len(bad_positions)} position snapshots")

    print("Generating animation from positions...")

    n_frames_good = min(len(good_metrics), max_frames)
    n_frames_bad = min(len(bad_metrics), max_frames)
    n_frames = max(n_frames_good, n_frames_bad)

    frames = []

    # Compute per-panel axis limits (locked to initial cloud so axes stay constant per panel)
    pad = 2.0
    init_pos = good_positions[0]
    good_xlim = (init_pos[:, 0].min() - pad, init_pos[:, 0].max() + pad)
    good_ylim = (init_pos[:, 1].min() - pad, init_pos[:, 1].max() + pad)
    # Bad panel: lock to initial too, so blow-up is visible as points leaving the frame
    bad_xlim = good_xlim
    bad_ylim = good_ylim

    good_label = f"GOOD  k={good_config['k_attract']}  ε_r={good_config['epsilon_r']:.4f}  lr={good_config['lr']:.0e}"
    bad_label  = f"BAD   k={bad_config['k_attract']}  ε_r={bad_config['epsilon_r']:.4f}  lr={bad_config['lr']:.0e}"

    for frame_idx in range(0, n_frames, frame_interval):
        # Create side-by-side scatter plots
        fig, (ax_good, ax_bad) = plt.subplots(1, 2, figsize=(13, 6), dpi=100)

        # Good run
        if frame_idx < len(good_positions):
            pos = good_positions[frame_idx]
            row = good_metrics.iloc[frame_idx]
            for g, c in color_map.items():
                mask = dataset_labels == g
                ax_good.scatter(pos[mask, 0], pos[mask, 1], s=30, alpha=0.75, color=c)
                centroid = pos[mask].mean(axis=0)
                ax_good.scatter(*centroid, s=300, marker="+", color="black", linewidths=2.5, zorder=5)
            ax_good.set_title(
                f"{good_label}\niter {frame_idx}",
                fontsize=10, fontweight="bold", color="#2166AC"
            )
            ax_good.set_xlim(good_xlim)
            ax_good.set_ylim(good_ylim)
            ax_good.set_aspect("equal")
            ax_good.grid(True, alpha=0.3)
            # Metrics annotation top-right
            ax_good.text(
                0.98, 0.98,
                f"sep_ratio = {row['separation_ratio']:.3f}\nglobal_spread = {row['global_spread']:.3f}\ne_att = {row['e_att']:.1f}\ne_rep = {row['e_rep']:.1f}",
                transform=ax_good.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # Bad run
        if frame_idx < len(bad_positions):
            pos = bad_positions[frame_idx]
            row = bad_metrics.iloc[frame_idx]
            for g, c in color_map.items():
                mask = dataset_labels == g
                ax_bad.scatter(pos[mask, 0], pos[mask, 1], s=30, alpha=0.75, color=c)
                centroid = pos[mask].mean(axis=0)
                ax_bad.scatter(*centroid, s=300, marker="+", color="black", linewidths=2.5, zorder=5)
            ax_bad.set_title(
                f"{bad_label}\niter {frame_idx}",
                fontsize=10, fontweight="bold", color="#B2182B"
            )
            ax_bad.set_xlim(bad_xlim)
            ax_bad.set_ylim(bad_ylim)
            ax_bad.set_aspect("equal")
            ax_bad.grid(True, alpha=0.3)
            # Metrics annotation top-right
            ax_bad.text(
                0.98, 0.98,
                f"sep_ratio = {row['separation_ratio']:.3f}\nglobal_spread = {row['global_spread']:.3f}\ne_att = {row['e_att']:.1f}\ne_rep = {row['e_rep']:.1f}",
                transform=ax_bad.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        fig.tight_layout()
        fig.canvas.draw()
        # Use buffer_rgba for newer matplotlib versions
        try:
            img_buf = fig.canvas.buffer_rgba()
        except AttributeError:
            img_buf = fig.canvas.tostring_argb()
        img = np.frombuffer(img_buf, dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        # buffer_rgba returns RGBA, tostring_argb returns ARGB
        if len(img) == w * h * 4:
            img = img.reshape((h, w, 4))[:, :, :3]  # Drop alpha or reorder
        else:
            img = img.reshape((h, w, 4))
            img = np.roll(img, -1, axis=2)[:, :, :3]  # ARGB -> RGB
        frames.append(Image.fromarray(img))
        plt.close(fig)

    # Save GIF
    good_id = good_run_dir.name
    bad_id = bad_run_dir.name
    if frames and Image is not None:
        gif_path = output_dir / f"{good_id}_vs_{bad_id}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # 100 ms per frame for faster viewing
            loop=0,
        )
        print(f"\nSaved GIF: {gif_path}")
    else:
        print("PIL not available or no frames generated")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate animated GIFs comparing good vs bad slice sandbox runs."
    )
    p.add_argument("--good-run", required=True, help="Path to good run directory (with metrics_history.csv)")
    p.add_argument("--bad-run", required=True, help="Path to bad run directory (with metrics_history.csv)")
    p.add_argument("--output-dir", default="/tmp/slice_animations")
    args = p.parse_args()

    good_run_dir = Path(args.good_run)
    bad_run_dir = Path(args.bad_run)
    output_dir = Path(args.output_dir)

    # Load synthetic dataset (need to regenerate it with same seed)
    from slice_sandbox import make_two_cluster_dataset
    dataset_obj = make_two_cluster_dataset("separated", n_per_cluster=60, random_seed=42)
    dataset = {"pos": dataset_obj.pos, "labels": dataset_obj.labels}

    generate_comparison_gif(
        good_run_dir,
        bad_run_dir,
        dataset,
        output_dir,
        max_frames=300,
        frame_interval=3,
    )

    # Also save to slice_sandbox_examples/ next to this script
    examples_dir = _HERE / "slice_sandbox_examples"
    examples_dir.mkdir(exist_ok=True)
    import shutil
    for gif in output_dir.glob("*.gif"):
        dest = examples_dir / gif.name
        if gif.resolve() != dest.resolve():
            shutil.copy(gif, dest)
            print(f"Copied to: {dest}")


if __name__ == "__main__":
    main()
