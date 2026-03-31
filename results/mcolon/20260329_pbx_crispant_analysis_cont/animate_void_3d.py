"""
animate_void_3d.py
------------------
Generate 3D GIFs showing void sandbox optimization evolution.

Axes: x (spatial), y (spatial), iteration (depth axis).
Each bundle is a tube through (x_t, iter_t, y_t) space as optimization proceeds.
The azimuth rotates so you can see the 3D structure from all angles.

Generates clearly-named GIFs for a GOOD and a BAD condition:
  GOOD — lambda_void=0.005, sigma_void_frac=5.0
         Bundles spread out; local density preserved.
  BAD  — lambda_void=0.05,  sigma_void_frac=2.0
         sigma_void too small (~bundle size) -> acts like extra repulsion -> bundles explode.

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/animate_void_3d.py \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/void_3d_gifs
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

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from void_sandbox import (
    VoidRunConfig,
    make_crowded_one_side,
    run_void,
    _color_map,
)

_BUNDLE_COLORS = ["#2166AC", "#B2182B", "#4DAC26", "#F1A340"]


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]


def _draw_3d_void(
    ax,
    positions_history: np.ndarray,   # (S, N, 2) — S snapshots
    snapshot_iters: list[int],
    labels: np.ndarray,               # (N,)
    title: str = "",
    alpha_points: float = 0.5,
    alpha_lines: float = 0.15,
    frame_stride: int = 1,
) -> None:
    """Draw each bundle as a 3D tube through (x, iter, y) space."""
    ax.clear()
    groups = sorted(set(labels.tolist()))
    color_map = {g: _BUNDLE_COLORS[i % len(_BUNDLE_COLORS)] for i, g in enumerate(groups)}

    iters = np.array(snapshot_iters[::frame_stride], dtype=float)
    pos = positions_history[::frame_stride]   # (S', N, 2)

    for g in groups:
        idx = np.flatnonzero(labels == g)
        c = color_map[g]

        # Draw trajectory line for each point in bundle (thin, faded)
        for i in idx:
            ax.plot(
                pos[:, i, 0],   # x over iters
                iters,           # iteration axis
                pos[:, i, 1],   # y over iters
                color=c, alpha=alpha_lines, lw=0.6,
            )

        # Scatter at each snapshot
        xs = pos[:, idx, 0].ravel()
        its = np.repeat(iters, len(idx))
        ys = pos[:, idx, 1].ravel()
        ax.scatter(xs, its, ys, color=c, s=8, alpha=alpha_points,
                   depthshade=True, label=f"bundle {g}")

        # Centroid trajectory (thick)
        centroids = pos[:, idx, :].mean(axis=1)   # (S', 2)
        ax.plot(
            centroids[:, 0], iters, centroids[:, 1],
            color=c, lw=3, alpha=0.95, zorder=10,
        )
        ax.scatter(
            centroids[:, 0], iters, centroids[:, 1],
            color=c, s=60, marker="o", zorder=11,
            edgecolors="black", linewidths=0.6,
        )

    ax.set_xlabel("x", fontsize=9, labelpad=5)
    ax.set_ylabel("iteration", fontsize=9, labelpad=5)
    ax.set_zlabel("y", fontsize=9, labelpad=5)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)


def make_void_3d_gif(
    positions_history: np.ndarray,    # (S, N, 2)
    snapshot_iters: list[int],
    labels: np.ndarray,
    output_path: Path,
    title: str = "",
    n_frames: int = 72,
    elev: float = 25,
    fps_ms: int = 80,
    frame_stride: int = 1,
) -> None:
    """Rotating 3D GIF: bundles traced through (x, iteration, y) space."""
    if not _PIL_AVAILABLE:
        print("PIL not available — cannot save GIF")
        return

    # Fix spatial axis limits from full history
    all_pos = positions_history.reshape(-1, 2)
    pad = 0.5
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    frames = []

    for az in azimuths:
        fig = plt.figure(figsize=(9, 7), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        _draw_3d_void(ax, positions_history, snapshot_iters, labels,
                      title=title, frame_stride=frame_stride)
        ax.set_xlim(xlim)
        ax.set_zlim(ylim)
        ax.view_init(elev=elev, azim=az)
        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=fps_ms,
        loop=0,
    )
    print(f"Saved: {output_path}")


def make_void_3d_side_by_side_gif(
    result_good,
    result_bad,
    output_path: Path,
    label_good: str,
    label_bad: str,
    n_frames: int = 72,
    elev: float = 25,
    fps_ms: int = 80,
    frame_stride: int = 1,
) -> None:
    """Side-by-side rotating 3D GIF: GOOD vs BAD void condition."""
    if not _PIL_AVAILABLE:
        print("PIL not available — cannot save GIF")
        return

    def _limits(hist):
        p = hist.reshape(-1, 2)
        pad = 0.5
        return (p[:, 0].min() - pad, p[:, 0].max() + pad), \
               (p[:, 1].min() - pad, p[:, 1].max() + pad)

    xlim_g, ylim_g = _limits(result_good.positions_history)
    xlim_b, ylim_b = _limits(result_bad.positions_history)
    xlim = (min(xlim_g[0], xlim_b[0]), max(xlim_g[1], xlim_b[1]))
    ylim = (min(ylim_g[0], ylim_b[0]), max(ylim_g[1], ylim_b[1]))

    labels = result_good.dataset.labels

    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    frames = []

    for az in azimuths:
        fig = plt.figure(figsize=(16, 7), dpi=100)

        ax_l = fig.add_subplot(121, projection="3d")
        _draw_3d_void(ax_l, result_good.positions_history, result_good.snapshot_iters,
                      labels, title=label_good, frame_stride=frame_stride)
        ax_l.set_xlim(xlim)
        ax_l.set_zlim(ylim)
        ax_l.view_init(elev=elev, azim=az)

        ax_r = fig.add_subplot(122, projection="3d")
        _draw_3d_void(ax_r, result_bad.positions_history, result_bad.snapshot_iters,
                      labels, title=label_bad, frame_stride=frame_stride)
        ax_r.set_xlim(xlim)
        ax_r.set_zlim(ylim)
        ax_r.view_init(elev=elev, azim=az)

        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=fps_ms,
        loop=0,
    )
    print(f"Saved: {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="3D void sandbox GIF generator.")
    p.add_argument("--output-dir",
                   default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/void_3d_gifs")
    p.add_argument("--n-per-bundle", type=int, default=30)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--n-frames", type=int, default=72, help="Rotation frames per GIF")
    p.add_argument("--frame-stride", type=int, default=3,
                   help="Use every Nth optimization snapshot for drawing (reduces clutter)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_crowded_one_side(n_per_bundle=args.n_per_bundle, random_seed=args.seed)

    # ---- GOOD condition: lambda=0.005, sigma_void_frac=5.0 ----
    print("Running GOOD condition (lambda_void=0.005, sigma_void_frac=5.0)...")
    cfg_good = VoidRunConfig(
        lambda_void=0.005,
        sigma_void_frac=5.0,
        n_iter=args.n_iter,
        save_every=5,
    )
    result_good = run_void(dataset, cfg_good, verbose=True)

    label_good = (
        "GOOD: λ_void=0.005, σ_void_frac=5.0\n"
        f"spread_ratio={result_good.final_metrics['within_bundle_spread_ratio']:.2f}  "
        f"escape={result_good.final_metrics['domain_escape_frac']:.2f}"
    )

    print("\nGenerating GOOD 3D rotating GIF...")
    make_void_3d_gif(
        result_good.positions_history,
        result_good.snapshot_iters,
        dataset.labels,
        output_dir / "void_3d_GOOD_lv0.005_sv5.0.gif",
        title=label_good,
        n_frames=args.n_frames,
        frame_stride=args.frame_stride,
    )

    # ---- BAD condition: lambda=0.05, sigma_void_frac=2.0 ----
    print("\nRunning BAD condition (lambda_void=0.05, sigma_void_frac=2.0)...")
    cfg_bad = VoidRunConfig(
        lambda_void=0.05,
        sigma_void_frac=2.0,
        n_iter=args.n_iter,
        save_every=5,
    )
    result_bad = run_void(dataset, cfg_bad, verbose=True)

    label_bad = (
        "BAD: λ_void=0.05, σ_void_frac=2.0\n"
        f"spread_ratio={result_bad.final_metrics['within_bundle_spread_ratio']:.2f}  "
        f"escape={result_bad.final_metrics['domain_escape_frac']:.2f}"
    )

    print("\nGenerating BAD 3D rotating GIF...")
    make_void_3d_gif(
        result_bad.positions_history,
        result_bad.snapshot_iters,
        dataset.labels,
        output_dir / "void_3d_BAD_lv0.05_sv2.0.gif",
        title=label_bad,
        n_frames=args.n_frames,
        frame_stride=args.frame_stride,
    )

    # ---- Side-by-side comparison ----
    print("\nGenerating side-by-side GOOD vs BAD GIF...")
    make_void_3d_side_by_side_gif(
        result_good, result_bad,
        output_dir / "void_3d_GOOD_vs_BAD_side_by_side.gif",
        label_good=label_good,
        label_bad=label_bad,
        n_frames=args.n_frames,
        frame_stride=args.frame_stride,
    )

    print(f"\nAll GIFs saved to: {output_dir}")


if __name__ == "__main__":
    main()
