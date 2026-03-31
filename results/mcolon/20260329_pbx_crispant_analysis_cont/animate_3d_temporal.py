"""
animate_3d_temporal.py
----------------------
Generate a 3D trajectory GIF where axes are (x, y, time).

Each embryo is a curve through (x_t, y_t, t) space.
The animation rotates the azimuth so you can see the 3D structure from all angles.
Optionally also shows iteration-by-iteration evolution (optimization snapshots).

Usage — view final positions as rotating 3D:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/animate_3d_temporal.py \\
      --variant crossing_bundles \\
      --output-dir /tmp/3d_anim \\
      --mode rotate

Usage — animate optimization progress (requires save_every snapshots):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/animate_3d_temporal.py \\
      --variant crossing_bundles \\
      --output-dir /tmp/3d_anim \\
      --mode optimize
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from temporal_sandbox import (
    TemporalRunConfig,
    make_crossing_bundles,
    make_stable_bundles,
    run_temporal,
    _resolve_color_map,
)

_CLUSTER_COLORS = ["#2166AC", "#B2182B"]


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return img.reshape((h, w, 4))[:, :, :3]


def _draw_3d_trajectories(
    ax,
    positions: np.ndarray,   # (N_e, T, 2)
    labels: np.ndarray,      # (N_e,)
    time_values: np.ndarray, # (T,)
    alpha_points: float = 0.6,
    alpha_lines: float = 0.25,
    title: str = "",
) -> None:
    """Draw each embryo as a 3D trajectory line + scatter."""
    ax.clear()
    groups = sorted(set(labels.tolist()))
    color_map = {g: _CLUSTER_COLORS[i] for i, g in enumerate(groups)}

    for g in groups:
        idx = np.flatnonzero(labels == g)
        c = color_map[g]

        # Draw trajectory lines (thin, faded)
        for i in idx:
            ax.plot(
                positions[i, :, 0],   # x
                time_values,           # time axis
                positions[i, :, 1],   # y
                color=c, alpha=alpha_lines, lw=0.8,
            )

        # Draw scatter points at each time slice
        xs = positions[idx, :, 0].ravel()
        ts = np.tile(time_values, len(idx))
        ys = positions[idx, :, 1].ravel()
        ax.scatter(xs, ts, ys, color=c, s=12, alpha=alpha_points,
                   depthshade=True, label=f"cluster {g}")

        # Draw cluster centroid trajectory (thick)
        centroids = positions[idx].mean(axis=0)  # (T, 2)
        ax.plot(
            centroids[:, 0],  # x
            time_values,      # time
            centroids[:, 1],  # y
            color=c, lw=3, alpha=0.95, zorder=10,
        )
        ax.scatter(
            centroids[:, 0], time_values, centroids[:, 1],
            color=c, s=80, marker="o", zorder=11, edgecolors="black", linewidths=0.8,
        )

    ax.set_xlabel("x", fontsize=9, labelpad=6)
    ax.set_ylabel("time", fontsize=9, labelpad=6)
    ax.set_zlabel("y", fontsize=9, labelpad=6)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=7)


def make_rotate_gif(
    positions: np.ndarray,     # (N_e, T, 2) — final positions
    labels: np.ndarray,
    time_values: np.ndarray,
    output_path: Path,
    n_frames: int = 72,        # one full rotation = 360 / 5 degrees per frame
    elev: float = 25,
    title: str = "",
    fps_ms: int = 80,
) -> None:
    """Rotate azimuth through 360°, save as GIF."""
    if not _PIL_AVAILABLE:
        print("PIL not available — cannot save GIF")
        return

    frames = []
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    for az in azimuths:
        fig = plt.figure(figsize=(8, 6), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        _draw_3d_trajectories(ax, positions, labels, time_values, title=title)
        ax.view_init(elev=elev, azim=az)
        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=fps_ms,
        loop=0,
    )
    print(f"Saved: {output_path}")


def make_optimize_gif(
    position_history: np.ndarray,  # (N_snapshots, N_e, T, 2)
    snapshot_iters: list[int],
    labels: np.ndarray,
    time_values: np.ndarray,
    output_path: Path,
    azim: float = 45,
    elev: float = 25,
    fps_ms: int = 120,
    frame_interval: int = 1,
) -> None:
    """Show how positions evolve over optimization iterations."""
    if not _PIL_AVAILABLE:
        print("PIL not available — cannot save GIF")
        return

    # Fix axis limits from full position history
    all_pos = position_history.reshape(-1, 2)
    pad = 0.5
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    frames = []
    indices = range(0, len(snapshot_iters), frame_interval)

    for snap_idx in indices:
        iter_n = snapshot_iters[snap_idx]
        pos = position_history[snap_idx]  # (N_e, T, 2)

        fig = plt.figure(figsize=(8, 6), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        _draw_3d_trajectories(
            ax, pos, labels, time_values,
            title=f"iter {iter_n}",
        )
        ax.set_xlim(xlim)
        ax.set_zlim(ylim)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=fps_ms,
        loop=0,
    )
    print(f"Saved: {output_path}")


def make_side_by_side_gif(
    pos_initial: np.ndarray,       # (N_e, T, 2)
    pos_final: np.ndarray,         # (N_e, T, 2)
    labels: np.ndarray,
    time_values: np.ndarray,
    output_path: Path,
    n_frames: int = 72,
    elev: float = 25,
    title_left: str = "initial",
    title_right: str = "final",
    fps_ms: int = 80,
) -> None:
    """Rotate two 3D views side by side: initial vs final positions."""
    if not _PIL_AVAILABLE:
        print("PIL not available — cannot save GIF")
        return

    # Fixed limits from both
    all_pos = np.vstack([pos_initial.reshape(-1, 2), pos_final.reshape(-1, 2)])
    pad = 0.5
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    frames = []
    azimuths = np.linspace(0, 360, n_frames, endpoint=False)

    for az in azimuths:
        fig = plt.figure(figsize=(14, 6), dpi=100)

        ax_l = fig.add_subplot(121, projection="3d")
        _draw_3d_trajectories(ax_l, pos_initial, labels, time_values, title=title_left)
        ax_l.set_xlim(xlim)
        ax_l.set_zlim(ylim)
        ax_l.view_init(elev=elev, azim=az)

        ax_r = fig.add_subplot(122, projection="3d")
        _draw_3d_trajectories(ax_r, pos_final, labels, time_values, title=title_right)
        ax_r.set_xlim(xlim)
        ax_r.set_zlim(ylim)
        ax_r.view_init(elev=elev, azim=az)

        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=fps_ms,
        loop=0,
    )
    print(f"Saved: {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="3D trajectory GIF generator for temporal sandbox.")
    p.add_argument("--variant", choices=["crossing_bundles", "stable_bundles"],
                   default="crossing_bundles")
    p.add_argument("--output-dir", default="/tmp/3d_anim")
    p.add_argument("--mode", choices=["rotate", "optimize", "both", "side_by_side"],
                   default="side_by_side",
                   help="rotate: final positions rotating | optimize: iter evolution | "
                        "side_by_side: initial vs final rotating | both: rotate + optimize")
    p.add_argument("--n-per-cluster", type=int, default=40)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--n-frames", type=int, default=72, help="Frames per full rotation")
    p.add_argument("--delta", type=int, default=3)
    p.add_argument("--mu0", type=float, default=0.0)
    p.add_argument("--eps-mult-scale", type=float, default=1.0,
                   help="Multiplicative scale on eps_mult. Use 0.1 for corrected dynamics.")
    p.add_argument("--r-cut-frac", type=float, default=0.0,
                   help="Truncated repulsion cutoff as fraction of median 5-NN. 0=soft-core.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-interval", type=int, default=1,
                   help="Use every Nth optimization snapshot for optimize mode")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print(f"Generating {args.variant}...")
    if args.variant == "crossing_bundles":
        dataset = make_crossing_bundles(n_per_cluster=args.n_per_cluster, random_seed=args.seed)
    else:
        dataset = make_stable_bundles(n_per_cluster=args.n_per_cluster, random_seed=args.seed)

    # Run dynamics
    cfg = TemporalRunConfig(
        delta=args.delta,
        mu0=args.mu0,
        n_iter=args.n_iter,
        k_attract=20,
        lr=1e-4,
        eps_mult_scale=args.eps_mult_scale,
        r_cut_frac=args.r_cut_frac,
    )
    needs_snapshots = args.mode in ("optimize", "both")
    print(f"Running dynamics (n_iter={args.n_iter}, save_snapshots={needs_snapshots})...")
    result = run_temporal(dataset, cfg, save_snapshots=needs_snapshots, verbose=True)

    pos_init = dataset.positions   # (N_e, T, 2)
    pos_final = result.cond_result.positions
    labels = dataset.labels
    time_values = dataset.time_values

    # --- Generate requested GIFs ---
    if args.mode in ("rotate", "both"):
        print("\nGenerating rotating 3D GIF (final positions)...")
        make_rotate_gif(
            pos_final, labels, time_values,
            output_dir / f"{args.variant}_3d_rotate.gif",
            n_frames=args.n_frames,
            title=f"{args.variant} | final positions | k=20 δ={args.delta} mu0={args.mu0}",
        )

    if args.mode in ("optimize", "both"):
        if result.cond_result.position_history is not None:
            print("\nGenerating optimization evolution GIF...")
            make_optimize_gif(
                result.cond_result.position_history,
                result.cond_result.snapshot_iters,
                labels, time_values,
                output_dir / f"{args.variant}_3d_optimize.gif",
                frame_interval=args.frame_interval,
            )
        else:
            print("No position history available — run without --no-snapshots")

    if args.mode == "side_by_side":
        print("\nGenerating side-by-side initial vs final GIF...")
        cond_tag = f"eps{args.eps_mult_scale}" + (f"_rcut{args.r_cut_frac}" if args.r_cut_frac > 0 else "")
        make_side_by_side_gif(
            pos_init, pos_final, labels, time_values,
            output_dir / f"{args.variant}_3d_before_after_{cond_tag}.gif",
            n_frames=args.n_frames,
            title_left=f"initial",
            title_right=f"final (iter {args.n_iter}, eps_scale={args.eps_mult_scale})",
        )


if __name__ == "__main__":
    main()
