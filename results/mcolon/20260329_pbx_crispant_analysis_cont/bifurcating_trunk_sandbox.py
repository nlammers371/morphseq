"""
bifurcating_trunk_sandbox.py
----------------------------
Synthetic Y-shaped branching dataset for testing whether the model can:
  1. Preserve a single compact trunk (early time)
  2. Allow a trunk-to-branch transition (middle–late time)
  3. Maintain branch identity once the split has begun

Ground truth structure
----------------------
t=0–3   : one compact dense trunk along the y-axis.
           All N embryos live near x=0, y spread in [-3, 3].
t=4–7   : trunk elongates. Upper region (y > 0) begins splitting.
           Points start to favour one of two directions: up-left or up-right.
           Lower region (y < 0) remains a single trunk.
t=8–12  : clear Y-shape. Two distinct branches emerge from a common base.
           Branch 0 (left):  upper points drift toward x=-2.5, y=+4
           Branch 1 (right): upper points drift toward x=+2.5, y=+4
           Base (bottom half of each branch): still aligned near x=0

Labels
------
  label=0 : branch-0 embryos (will go left)
  label=1 : branch-1 embryos (will go right)

Both labels share the common trunk at early time, so initial coherence is high
across labels. The test is whether the model eventually separates them.

Branch-geometry metrics
-----------------------
  linearity_score(t)
    Per time-slice: ratio λ_1 / (λ_1 + λ_2) of 2D covariance eigenvalues.
    1.0 = perfect line, 0.5 = circular blob.
    Trunk metric: should be high at early t, maintained at middle t.

  branch_separation(t)
    At late t: centroid distance between branch-0 and branch-1,
    normalised by their pooled within-branch std.
    > 2.0 = well-separated branches.

  within_branch_spread_ratio
    Final / initial within-branch std (label-aware).
    Should stay near 1.0 — branches should not bloat.

Four conditions run automatically (all from the same saved initialization)
---------------------------------------------------------------------------
  A. isotropic              — coherence + repulsion only (baseline)
  B. fidelity               — + fidelity anchor (fidelity_init_strength=1.0,
                               fidelity_half_life=0.999 — slow/persistent decay)
  C. pairwise_void_proxy    — + broad pairwise Gaussian void (epsilon_void=0.005,
                               sigma_void_frac=5.0). NOTE: this is NOT the grid-based
                               occupancy void. It is a broad pairwise density-field
                               repulsion — essentially another broad repulsion force.
                               Labelled as "proxy" to distinguish from the grid void.
  D. elasticity             — + stretch + bending penalty (lambda_stretch=0.05,
                               lambda_bend=0.02)
  E. all_on                 — all forces on: fidelity (fidelity_init_strength=1.0,
                               fidelity_half_life=0.1 — fast decay, anchor dies quickly)
                               + void proxy (epsilon_void=0.005, sigma_void_frac=5.0)
                               + elasticity defaults (stretch_strength_mult=0.001,
                               bend_strength_mult=0.001)

Anisotropy: not yet implemented. Excluded from all plots until it exists.

Shared initialization
---------------------
The initialization is saved once to initialization.npz and reloaded for all
conditions, so that x0, coherence inputs, s_local, and sigma are identical.
This makes condition comparisons valid — any difference in results is due to
the force term, not to a different starting point.

Summary outputs
---------------
  summary_comparison.png : tall figure — initialization slices at top, then
                           final slices for each condition below, all on the
                           same axis limits. Shows "what changed" vs reference.
  metrics_comparison.png : bar chart of 4 scalar metrics across conditions A–D.
  summary.csv            : numeric summary table.

Run (smoke test, no animation):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/bifurcating_trunk_sandbox.py \\
      --output-dir /tmp/bifurcating_trunk_test \\
      --n-per-branch 20 --n-iter 100 --no-animation

Run (full):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/bifurcating_trunk_sandbox.py \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/bifurcating_trunk_v2
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
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from temporal_sandbox import (
    TemporalDataset,
    TemporalRunConfig,
    TemporalRunResult,
    run_temporal,
    _resolve_color_map,
    _within_bundle_spread,
    _fig_to_rgb,
)
_BRANCH_COLORS = ["#2166AC", "#B2182B"]


# ===========================================================================
# Section 0: Shared initialization — save / load
# ===========================================================================

def save_initialization(dataset: TemporalDataset, path: Path) -> None:
    """Save the dataset positions/labels/mask to .npz so all conditions share
    the exact same starting point, coherence inputs, and s_local."""
    np.savez(
        path,
        positions=dataset.positions,
        labels=dataset.labels,
        mask=dataset.mask.astype(np.uint8),
        time_values=dataset.time_values,
    )
    print(f"  Saved initialization: {path}")


def load_initialization(path: Path, variant: str, n_per_cluster: int) -> TemporalDataset:
    """Reload a saved initialization as a TemporalDataset."""
    d = np.load(path)
    n_time = int(d["time_values"].shape[0])
    return TemporalDataset(
        positions=d["positions"],
        labels=d["labels"],
        mask=d["mask"].astype(bool),
        time_values=d["time_values"],
        variant=variant,
        n_per_cluster=n_per_cluster,
        n_time=n_time,
    )


# ===========================================================================
# Section 1: Synthetic dataset
# ===========================================================================

def make_bifurcating_trunk(
    n_per_branch: int = 40,
    n_time: int = 13,
    trunk_noise: float = 0.25,
    branch_noise: float = 0.20,
    trunk_length: float = 3.0,
    branch_spread: float = 2.5,
    branch_height: float = 4.0,
    split_start: int = 4,     # time index where upper trunk begins splitting
    split_full: int = 8,      # time index where branches are clearly distinct
    random_seed: int = 42,
) -> TemporalDataset:
    """Y-shaped temporal dataset: single trunk at early t, two branches at late t.

    Parameters
    ----------
    n_per_branch : embryos per branch (total N = 2 * n_per_branch)
    n_time       : total time steps (default 13: t=0..12)
    trunk_noise  : within-trunk Gaussian noise (spatial units)
    branch_noise : within-branch noise at late time
    trunk_length : half-length of trunk along y (points span [-trunk_length, trunk_length])
    branch_spread: final x-offset of branch centroids (± from y-axis)
    branch_height: final y-position of branch tip
    split_start  : first t where upper region starts to bifurcate (default 4)
    split_full   : t where the Y is fully formed (default 8)
    random_seed  : RNG seed

    Ground truth labels:
        label=0 → will go to upper-left branch
        label=1 → will go to upper-right branch
    Both start near x=0 at t=0.
    """
    rng = np.random.default_rng(random_seed)
    N = 2 * n_per_branch
    labels = np.array([0] * n_per_branch + [1] * n_per_branch, dtype=int)
    positions = np.zeros((N, n_time, 2), dtype=float)

    # Fixed per-embryo "rank" along the trunk: from -trunk_length to +trunk_length
    # Lower-rank embryos stay in the base; higher-rank embryos end up in branch tips.
    # Branch-0 gets the first n_per_branch, branch-1 the second.
    trunk_y_0 = np.linspace(-trunk_length, trunk_length, n_per_branch)
    trunk_y_1 = np.linspace(-trunk_length, trunk_length, n_per_branch)

    # Per-embryo fixed offsets (their "identity noise" — constant across time)
    offset_0 = rng.normal(0, trunk_noise * 0.5, size=(n_per_branch, 2))
    offset_1 = rng.normal(0, trunk_noise * 0.5, size=(n_per_branch, 2))

    for t in range(n_time):
        # Fraction through the split [0, 1]: 0 before split_start, 1 at split_full
        split_frac = float(np.clip((t - split_start) / max(split_full - split_start, 1), 0.0, 1.0))
        # Smooth step (ease in/out)
        split_frac = split_frac * split_frac * (3.0 - 2.0 * split_frac)

        # Upper fraction: portion of trunk that has bifurcated at this time.
        # Only embryos with high trunk_y are affected early.
        # For embryo i with normalized rank u_i in [0,1]:
        #   branch influence = max(0, (u_i - (1 - split_frac))) / split_frac
        #   = how far into the "splitting front" this embryo is

        for branch_idx, (trunk_y, offsets) in enumerate([
            (trunk_y_0, offset_0),
            (trunk_y_1, offset_1),
        ]):
            n = n_per_branch
            start_idx = branch_idx * n

            # Normalized rank in [0, 1] for each embryo
            u = (trunk_y - trunk_y.min()) / (trunk_y.max() - trunk_y.min() + 1e-12)

            # Branch influence per embryo: how much this embryo has left the trunk
            # Embryos at top (u≈1) split first; base (u≈0) stays trunk
            if split_frac > 0:
                branch_influence = np.clip((u - (1.0 - split_frac)) / (split_frac + 1e-6), 0.0, 1.0)
            else:
                branch_influence = np.zeros(n)

            # Lateral drift: branch 0 goes left (x → -branch_spread), branch 1 right
            sign = -1.0 if branch_idx == 0 else +1.0
            x_branch = sign * branch_spread * branch_influence

            # Vertical lift: upper part of trunk lifts toward branch_height as it splits
            # Base of trunk stays near its original y; tips lift to branch_height
            y_trunk = trunk_y  # original y positions along trunk
            y_lift = (branch_height - trunk_y) * branch_influence  # how much upper part rises
            y_branch = y_trunk + y_lift

            # Small per-step noise (shrinks toward branch_noise at split completion)
            noise_scale = trunk_noise * (1.0 - 0.3 * split_frac) + branch_noise * 0.3 * split_frac
            noise = rng.normal(0, noise_scale, size=(n, 2))

            positions[start_idx:start_idx + n, t, 0] = x_branch + offsets[:, 0] + noise[:, 0]
            positions[start_idx:start_idx + n, t, 1] = y_branch + offsets[:, 1] + noise[:, 1]

    mask = np.ones((N, n_time), dtype=bool)
    time_values = np.arange(n_time, dtype=float)

    return TemporalDataset(
        positions=positions,
        mask=mask,
        labels=labels,
        time_values=time_values,
        variant="bifurcating_trunk",
        n_per_cluster=n_per_branch,
        n_time=n_time,
    )


# ===========================================================================
# Section 2: Branch-geometry metrics
# ===========================================================================

def linearity_score(positions_t: np.ndarray) -> float:
    """λ_1 / (λ_1 + λ_2) from 2D covariance of all points at one time slice.

    1.0 = perfect line, 0.5 = isotropic blob.
    """
    if positions_t.shape[0] < 3:
        return float("nan")
    centered = positions_t - positions_t.mean(axis=0)
    cov = centered.T @ centered / (len(centered) - 1)
    eigvals = np.linalg.eigvalsh(cov)   # ascending order
    lam1, lam2 = eigvals[1], eigvals[0]  # lam1 ≥ lam2
    return float(lam1 / (lam1 + lam2 + 1e-12))


def branch_separation(positions_t: np.ndarray, labels: np.ndarray) -> float:
    """Centroid distance between branch-0 and branch-1 at one time slice,
    normalised by their pooled within-branch std.

    > 2.0 = well-separated branches.
    """
    g0 = positions_t[labels == 0]
    g1 = positions_t[labels == 1]
    if len(g0) < 2 or len(g1) < 2:
        return float("nan")
    c0 = g0.mean(axis=0)
    c1 = g1.mean(axis=0)
    dist = float(np.linalg.norm(c0 - c1))
    within_std = float(np.std(np.vstack([g0 - c0, g1 - c1])))
    return dist / (within_std + 1e-12)


def trunk_linearity_profile(
    positions: np.ndarray,   # (N_e, T, 2)
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-time-slice linearity and branch-separation scores."""
    T = positions.shape[1]
    lin_all = np.array([linearity_score(positions[:, t, :]) for t in range(T)])
    lin_0   = np.array([linearity_score(positions[labels == 0, t, :]) for t in range(T)])
    lin_1   = np.array([linearity_score(positions[labels == 1, t, :]) for t in range(T)])
    sep     = np.array([branch_separation(positions[:, t, :], labels) for t in range(T)])
    return {
        "linearity_all": lin_all,
        "linearity_branch0": lin_0,
        "linearity_branch1": lin_1,
        "branch_separation": sep,
    }


def trunk_summary_metrics(
    positions_final: np.ndarray,   # (N_e, T, 2)
    positions_init: np.ndarray,
    labels: np.ndarray,
    split_full: int = 8,
) -> dict:
    """Scalar summary metrics for the bifurcating trunk test.

    trunk_linearity_early  : mean linearity score over t < split_full
    branch_sep_late        : mean branch_separation over t >= split_full
    within_branch_spread_ratio : final / initial within-branch std
    """
    T = positions_final.shape[1]
    prof = trunk_linearity_profile(positions_final, labels)

    early_mask = np.arange(T) < split_full
    late_mask  = np.arange(T) >= split_full

    trunk_lin_early = float(np.nanmean(prof["linearity_all"][early_mask])) if early_mask.any() else float("nan")
    branch_sep_late = float(np.nanmean(prof["branch_separation"][late_mask])) if late_mask.any() else float("nan")

    init_spread  = _within_bundle_spread(positions_init, labels)
    final_spread = _within_bundle_spread(positions_final, labels)
    spread_ratio = final_spread / (init_spread + 1e-12)

    return {
        "trunk_linearity_early": trunk_lin_early,
        "branch_sep_late": branch_sep_late,
        "within_branch_spread_ratio": spread_ratio,
    }


# ===========================================================================
# Section 3: Plots
# ===========================================================================

def plot_branch_geometry(
    result: TemporalRunResult,
    output_dir: Path,
    split_full: int = 8,
    condition_label: str = "",
) -> None:
    """Save: slices grid, branch-geometry metric profiles, coherence heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = result.dataset
    pos_init  = ds.positions
    pos_final = result.cond_result.positions
    labels    = ds.labels
    T         = ds.n_time
    color_map = {0: _BRANCH_COLORS[0], 1: _BRANCH_COLORS[1]}

    all_pos = np.vstack([pos_init.reshape(-1, 2), pos_final.reshape(-1, 2)])
    pad = 1.0
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    # --- slices grid (initial vs final) ---
    ncols = min(T, 7)
    nrows = (T + ncols - 1) // ncols

    for pos, tag in [(pos_init, "initial"), (pos_final, "final")]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
        axes_flat = np.array(axes).ravel().tolist()
        for t in range(T):
            ax = axes_flat[t]
            pt = pos[:, t, :]
            for g, c in color_map.items():
                m = labels == g
                ax.scatter(pt[m, 0], pt[m, 1], s=12, alpha=0.7, color=c)
                ax.scatter(*pt[m].mean(axis=0), s=120, marker="+", color="k", lw=2, zorder=5)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            sep = branch_separation(pt, labels)
            lin = linearity_score(pt)
            ax.set_title(f"t={t}\nsep={sep:.1f} lin={lin:.2f}", fontsize=7)
            ax.tick_params(labelsize=5)
            # Mark split boundary
            if t == split_full:
                ax.set_facecolor("#fff8e1")
        for t in range(T, len(axes_flat)):
            axes_flat[t].set_visible(False)
        fig.suptitle(
            f"bifurcating_trunk | {tag} | {condition_label}",
            fontsize=9, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"slices_{tag}.png", dpi=120)
        plt.close(fig)

    # --- branch geometry metric profiles ---
    prof_init  = trunk_linearity_profile(pos_init,  labels)
    prof_final = trunk_linearity_profile(pos_final, labels)
    t_vals = ds.time_values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(t_vals, prof_init["linearity_all"],  color="gray",         lw=1.5, ls="--", label="initial (all)")
    ax.plot(t_vals, prof_final["linearity_all"], color="black",        lw=2,            label="final (all)")
    ax.plot(t_vals, prof_final["linearity_branch0"], color=_BRANCH_COLORS[0], lw=1.5, ls="-.", label="final (branch-0)")
    ax.plot(t_vals, prof_final["linearity_branch1"], color=_BRANCH_COLORS[1], lw=1.5, ls="-.", label="final (branch-1)")
    ax.axvline(split_full, color="orange", lw=1, ls=":", label=f"split_full (t={split_full})")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel("linearity score (0.5=blob, 1.0=line)", fontsize=8)
    ax.set_title("Linearity over time", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(t_vals, prof_init["branch_separation"],  color="gray",  lw=1.5, ls="--", label="initial")
    ax.plot(t_vals, prof_final["branch_separation"], color="#B2182B", lw=2,          label="final")
    ax.axvline(split_full, color="orange", lw=1, ls=":", label=f"split_full (t={split_full})")
    ax.axhline(2.0, color="green", lw=0.8, ls=":", label="sep=2 (well-separated)")
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel("branch separation (centroid dist / within-std)", fontsize=8)
    ax.set_title("Branch separation over time", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Branch geometry metrics | {condition_label}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "branch_geometry.png", dpi=120)
    plt.close(fig)


def _draw_3d_trunk(
    ax,
    positions: np.ndarray,   # (N_e, T, 2)
    labels: np.ndarray,
    time_values: np.ndarray,
    title: str = "",
    alpha_pts: float = 0.5,
    alpha_lines: float = 0.15,
) -> None:
    """3D plot: x (spatial), time (depth axis), y (spatial)."""
    ax.clear()
    color_map = {0: _BRANCH_COLORS[0], 1: _BRANCH_COLORS[1]}
    for g, c in color_map.items():
        idx = np.flatnonzero(labels == g)
        # Thin lines per embryo
        for i in idx:
            ax.plot(
                positions[i, :, 0], time_values, positions[i, :, 1],
                color=c, alpha=alpha_lines, lw=0.7,
            )
        # Scatter
        xs = positions[idx, :, 0].ravel()
        ts = np.tile(time_values, len(idx))
        ys = positions[idx, :, 1].ravel()
        ax.scatter(xs, ts, ys, color=c, s=10, alpha=alpha_pts, depthshade=True,
                   label=f"branch {g}")
        # Centroid trajectory (thick)
        cents = positions[idx].mean(axis=0)   # (T, 2)
        ax.plot(cents[:, 0], time_values, cents[:, 1], color=c, lw=3, alpha=0.95, zorder=10)
        ax.scatter(cents[:, 0], time_values, cents[:, 1],
                   color=c, s=60, marker="o", zorder=11, edgecolors="black", linewidths=0.6)

    ax.set_xlabel("x", fontsize=8, labelpad=4)
    ax.set_ylabel("time", fontsize=8, labelpad=4)
    ax.set_zlabel("y", fontsize=8, labelpad=4)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=6)


def make_trunk_3d_gif(
    positions: np.ndarray,    # (N_e, T, 2)
    labels: np.ndarray,
    time_values: np.ndarray,
    output_path: Path,
    title: str = "",
    n_frames: int = 72,
    elev: float = 25,
    fps_ms: int = 80,
) -> None:
    if not _PIL_AVAILABLE:
        print("PIL not available — skipping GIF")
        return
    all_pos = positions.reshape(-1, 2)
    pad = 0.5
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    frames = []
    for az in np.linspace(0, 360, n_frames, endpoint=False):
        fig = plt.figure(figsize=(9, 7), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        _draw_3d_trunk(ax, positions, labels, time_values, title=title)
        ax.set_xlim(xlim)
        ax.set_zlim(ylim)
        ax.view_init(elev=elev, azim=az)
        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=fps_ms, loop=0)
    print(f"  Saved: {output_path}")


def make_trunk_side_by_side_gif(
    pos_initial: np.ndarray,
    pos_final: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    output_path: Path,
    title_left: str = "initial",
    title_right: str = "final",
    n_frames: int = 72,
    elev: float = 25,
    fps_ms: int = 80,
) -> None:
    if not _PIL_AVAILABLE:
        print("PIL not available — skipping GIF")
        return
    all_pos = np.vstack([pos_initial.reshape(-1, 2), pos_final.reshape(-1, 2)])
    pad = 0.5
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    frames = []
    for az in np.linspace(0, 360, n_frames, endpoint=False):
        fig = plt.figure(figsize=(16, 7), dpi=100)

        ax_l = fig.add_subplot(121, projection="3d")
        _draw_3d_trunk(ax_l, pos_initial, labels, time_values, title=title_left)
        ax_l.set_xlim(xlim); ax_l.set_zlim(ylim)
        ax_l.view_init(elev=elev, azim=az)

        ax_r = fig.add_subplot(122, projection="3d")
        _draw_3d_trunk(ax_r, pos_final, labels, time_values, title=title_right)
        ax_r.set_xlim(xlim); ax_r.set_zlim(ylim)
        ax_r.view_init(elev=elev, azim=az)

        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=fps_ms, loop=0)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Section 4: Run conditions
# ===========================================================================

def _run_and_plot(
    dataset: TemporalDataset,
    config: TemporalRunConfig,
    output_dir: Path,
    condition_label: str,
    split_full: int,
    no_animation: bool,
    n_frames: int,
    pos_init_fixed: np.ndarray,   # shared init positions for GIF left panel + metrics
    verbose: bool = True,
) -> dict:
    """Run one condition, generate all plots and GIFs, return scalar summary.

    pos_init_fixed is the shared initialization used for the GIF left panel and
    spread_ratio computation — always the same array regardless of which condition.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_temporal(dataset, config, save_snapshots=not no_animation, verbose=verbose)
    result.metrics_df.to_csv(output_dir / "metrics_history.csv", index=False)

    plot_branch_geometry(result, output_dir, split_full=split_full,
                         condition_label=condition_label)

    pos_final = result.cond_result.positions
    labels    = dataset.labels
    time_vals = dataset.time_values

    if not no_animation:
        print(f"  Generating 3D side-by-side GIF ({condition_label})...")
        make_trunk_side_by_side_gif(
            pos_init_fixed, pos_final, labels, time_vals,
            output_dir / "3d_before_after.gif",
            title_left="initialization (shared)",
            title_right=f"final | {condition_label}",
            n_frames=n_frames,
        )

    summary = trunk_summary_metrics(pos_final, pos_init_fixed, labels, split_full=split_full)
    summary["condition"] = condition_label
    summary["collapse_score"] = result.collapse_score
    summary["coherence_selectivity"] = result.final_metrics.get("coherence_selectivity", float("nan"))
    summary["sep_ratio_mean"] = result.final_metrics.get("sep_ratio_mean", float("nan"))

    return summary, result.cond_result.positions


# ===========================================================================
# Section 5: Summary comparison figures
# ===========================================================================

def plot_summary_comparison(
    pos_init: np.ndarray,                         # (N_e, T, 2) — shared initialization
    condition_finals: list[tuple[str, np.ndarray]],  # [(label, (N_e, T, 2)), ...]
    labels: np.ndarray,
    time_values: np.ndarray,
    output_path: Path,
    split_full: int = 8,
) -> None:
    """Tall figure: initialization slice row at top, one row per condition below.

    All rows use the same axis limits so geometry changes are directly visible.
    Each cell shows sep and lin scores. Yellow background = split_full time step.
    """
    T = pos_init.shape[1]
    ncols = min(T, 7)
    n_conditions = len(condition_finals)
    n_rows = 1 + n_conditions          # init row + one per condition
    color_map = {0: _BRANCH_COLORS[0], 1: _BRANCH_COLORS[1]}

    # Shared axis limits across init + all finals
    all_arrays = [pos_init] + [pf for _, pf in condition_finals]
    all_pos = np.vstack([a.reshape(-1, 2) for a in all_arrays])
    pad = 0.8
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    cell_size = 2.3
    fig, axes = plt.subplots(
        n_rows, ncols,
        figsize=(cell_size * ncols, cell_size * n_rows),
        squeeze=False,
    )

    def _fill_row(row_idx: int, pos: np.ndarray, row_label: str) -> None:
        for t in range(T):
            if t >= ncols:
                break
            ax = axes[row_idx, t]
            pt = pos[:, t, :]
            for g, c in color_map.items():
                m = labels == g
                ax.scatter(pt[m, 0], pt[m, 1], s=8, alpha=0.65, color=c, rasterized=True)
                ax.scatter(*pt[m].mean(axis=0), s=80, marker="+",
                           color="k", lw=1.5, zorder=5)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=4, length=2)
            sep = branch_separation(pt, labels)
            lin = linearity_score(pt)
            ax.set_title(f"t={t}  sep={sep:.1f}\nlin={lin:.2f}", fontsize=6)
            if t == split_full:
                ax.set_facecolor("#fff8e1")
            # Row label on leftmost cell
            if t == 0:
                ax.set_ylabel(row_label, fontsize=7, fontweight="bold", labelpad=4)
        # Hide unused cells in row
        for t in range(T, ncols):
            axes[row_idx, t].set_visible(False)

    _fill_row(0, pos_init, "initialization\n(ground truth)")
    for i, (cond_label, pos_final) in enumerate(condition_finals):
        _fill_row(i + 1, pos_final, cond_label)

    fig.suptitle(
        "Bifurcating trunk — initialization vs condition finals\n"
        "(yellow = split_full; all panels share same axis limits)",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_metrics_comparison(
    summaries: list[dict],
    output_path: Path,
) -> None:
    """Bar chart of 4 scalar metrics across all conditions."""
    df = pd.DataFrame(summaries)
    metrics = [
        ("trunk_linearity_early",      "Trunk linearity (early t)\n>0.7 = line-like",   0.5, 1.0),
        ("branch_sep_late",            "Branch separation (late t)\n>2.0 = split",       0.0, None),
        ("within_branch_spread_ratio", "Within-branch spread ratio\n~1.0 = preserved",   0.0, None),
        ("coherence_selectivity",      "Coherence selectivity\n>1.0 = distinguishes",    0.0, None),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    conditions = df["condition"].tolist()
    x = np.arange(len(conditions))
    bar_colors = ["#4D4D4D", "#2166AC", "#B2182B", "#4DAC26"]

    for ax, (col, ylabel, ref_lo, ref_hi) in zip(axes, metrics):
        vals = df[col].tolist()
        bars = ax.bar(x, vals, color=bar_colors[:len(conditions)], width=0.6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.split(":")[0] for c in conditions], fontsize=8, rotation=15)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(col, fontsize=8, fontweight="bold")
        if ref_lo is not None:
            ax.axhline(ref_lo, color="gray", lw=0.8, ls=":")
        # Annotate values
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Force comparison: bifurcating trunk\n(all conditions from same initialization)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_force_sweeps(
    init_path: Path,
    base_kwargs: dict,
    output_path: Path,
    n_per_branch: int = 40,
    split_full: int = 8,
    verbose: bool = False,
) -> None:
    """2×3 grid of force-family sweep plots.

    Each facet = one force family. Within each facet: two stacked subplots sharing
    the x-axis — top: branch_sep_late, bottom: trunk_linearity_early.
    X-axis: parameter value (log scale). Vertical dashed line: default value.
    Horizontal dotted line: isotropic baseline for each metric.

    Families (row-major order):
      [0,0] repulsion_strength_mult   [0,1] fidelity_init_strength  [0,2] fidelity_half_life
      [1,0] epsilon_void              [1,1] stretch_strength_mult   [1,2] bend_strength_mult
    """
    # --- Define sweep specs ---
    # Each entry: (param_name, values, default, fixed_overrides, xlabel_suffix)
    # fixed_overrides: kwargs to hold constant while sweeping this param
    # (isotropic baseline always has repulsion_strength_mult=default, everything else off)
    sweep_specs = [
        (
            "repulsion_strength_mult",
            np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]),
            0.005,
            dict(fidelity_init_strength=0.0, epsilon_void=0.0),
            "",
        ),
        (
            "fidelity_init_strength",
            np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]),
            1.0,
            dict(fidelity_half_life=0.99, epsilon_void=0.0),
            "(fidelity_half_life=0.99 fixed)",
        ),
        (
            "fidelity_half_life",
            np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999]),
            0.99,
            dict(fidelity_init_strength=1.0, epsilon_void=0.0),
            "(fidelity_init_strength=1.0 fixed)",
        ),
        (
            "epsilon_void",
            np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]),
            0.005,
            dict(fidelity_init_strength=0.0, sigma_void_frac=5.0),
            "(sigma_void_frac=5.0 fixed)",
        ),
        (
            "stretch_strength_mult",
            np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]),
            0.001,
            dict(fidelity_init_strength=0.0, epsilon_void=0.0),
            "",
        ),
        (
            "bend_strength_mult",
            np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]),
            0.001,
            dict(fidelity_init_strength=0.0, epsilon_void=0.0),
            "",
        ),
    ]

    # --- Run isotropic baseline once ---
    ds_base = load_initialization(init_path, variant="bifurcating_trunk",
                                  n_per_cluster=n_per_branch)
    pos_init_fixed = ds_base.positions.copy()
    cfg_base = TemporalRunConfig(**base_kwargs, fidelity_init_strength=0.0, epsilon_void=0.0)
    print("  Running isotropic baseline...")
    res_base = run_temporal(ds_base, cfg_base, save_snapshots=False, verbose=False)
    base_summary = trunk_summary_metrics(res_base.cond_result.positions, pos_init_fixed,
                                         ds_base.labels, split_full=split_full)
    baseline_sep = base_summary["branch_sep_late"]
    baseline_lin = base_summary["trunk_linearity_early"]
    print(f"  baseline: branch_sep={baseline_sep:.4f}  trunk_lin={baseline_lin:.4f}")

    color_sep = "#2166AC"   # blue
    color_lin = "#D6604D"   # red-orange

    # --- Build figure: 2 rows × 3 cols of facets, each facet = 2 stacked subplots ---
    # Use gridspec: outer 2×3, inner each cell split into 2 rows
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(15, 10))
    outer = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    for facet_idx, (param, values, default, fixed_kw, xlabel_suffix) in enumerate(sweep_specs):
        row, col = divmod(facet_idx, 3)
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col],
                                        hspace=0.05, height_ratios=[1, 1])
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

        branch_seps = []
        trunk_lins = []
        print(f"\n  Sweeping {param}...")
        for v in values:
            ds = load_initialization(init_path, variant="bifurcating_trunk",
                                     n_per_cluster=n_per_branch)
            kw = {**fixed_kw, param: float(v)}
            cfg = TemporalRunConfig(**base_kwargs, **kw)
            res = run_temporal(ds, cfg, save_snapshots=False, verbose=False)
            s = trunk_summary_metrics(res.cond_result.positions, pos_init_fixed,
                                      ds.labels, split_full=split_full)
            branch_seps.append(s["branch_sep_late"])
            trunk_lins.append(s["trunk_linearity_early"])
            print(f"    {param}={v:.5g}  sep={branch_seps[-1]:.4f}  lin={trunk_lins[-1]:.4f}")

        # Top subplot: branch_sep_late
        ax_top.plot(values, branch_seps, "o-", color=color_sep, lw=1.6, ms=4)
        ax_top.axhline(baseline_sep, color=color_sep, lw=2.0, ls=":", alpha=0.85)
        ax_top.axvline(default, color="black", lw=1.2, ls="--", alpha=0.7)
        ax_top.set_ylabel("branch_sep_late", fontsize=7, color=color_sep, fontweight="bold")
        ax_top.tick_params(axis="y", labelsize=6, colors=color_sep)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.grid(True, which="both", alpha=0.2, lw=0.4)
        ax_top.set_xscale("log")
        ax_top.set_title(
            f"{param}\n{xlabel_suffix}" if xlabel_suffix else param,
            fontsize=8, fontweight="bold", pad=3,
        )

        # Bottom subplot: trunk_linearity_early
        ax_bot.plot(values, trunk_lins, "s--", color=color_lin, lw=1.6, ms=4)
        ax_bot.axhline(baseline_lin, color=color_lin, lw=2.0, ls=":", alpha=0.85)
        ax_bot.axvline(default, color="black", lw=1.2, ls="--", alpha=0.7)
        ax_bot.set_ylabel("trunk_lin_early", fontsize=7, color=color_lin, fontweight="bold")
        ax_bot.tick_params(axis="y", labelsize=6, colors=color_lin)
        ax_bot.tick_params(axis="x", labelsize=6, rotation=30)
        ax_bot.grid(True, which="both", alpha=0.2, lw=0.4)
        ax_bot.set_xlabel(f"value (log)\ndefault={default}", fontsize=7)

    # Legend as figure-level annotation
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=color_sep, marker="o", lw=1.6, ms=4, label="branch_sep_late"),
        Line2D([0], [0], color=color_lin, marker="s", lw=1.6, ms=4, ls="--", label="trunk_linearity_early"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="default value"),
        Line2D([0], [0], color="gray", lw=2.0, ls=":", label="isotropic baseline"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.01), frameon=True)

    fig.suptitle("Force family sweeps — bifurcating trunk benchmark\n"
                 "Each panel: branch_sep_late (top) and trunk_linearity_early (bottom) vs parameter value",
                 fontsize=10, fontweight="bold", y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {output_path}")


# ===========================================================================
# Section 6: Main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bifurcating trunk force comparison sandbox."
    )
    p.add_argument("--output-dir",
                   default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/bifurcating_trunk_v2")
    p.add_argument("--n-per-branch", type=int, default=40)
    p.add_argument("--n-time", type=int, default=13)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--k-attract", type=int, default=20)
    p.add_argument("--delta", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--split-start", type=int, default=4)
    p.add_argument("--split-full", type=int, default=8)
    p.add_argument("--n-frames", type=int, default=72, help="Frames per 3D rotating GIF")
    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fidelity-half-life-sweep", action="store_true",
                   help="Run all force family sweeps and generate 2x3 sweep figure")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate and save shared initialization ---
    print("Generating bifurcating_trunk dataset...")
    dataset = make_bifurcating_trunk(
        n_per_branch=args.n_per_branch,
        n_time=args.n_time,
        split_start=args.split_start,
        split_full=args.split_full,
        random_seed=args.seed,
    )
    init_path = output_dir / "initialization.npz"
    save_initialization(dataset, init_path)

    # All conditions reload from this file — guarantees identical x0 / coherence inputs
    pos_init_fixed = dataset.positions.copy()

    # 3D GIF of ground truth
    if not args.no_animation:
        print("Generating ground truth 3D GIF...")
        make_trunk_3d_gif(
            pos_init_fixed, dataset.labels, dataset.time_values,
            output_dir / "ground_truth_3d.gif",
            title="Ground truth: Y-trunk bifurcation",
            n_frames=args.n_frames,
        )

    # Base config shared across all conditions
    base_kwargs = dict(
        k_attract=args.k_attract,
        delta=args.delta,
        lr=args.lr,
        n_iter=args.n_iter,
    )

    conditions = [
        (
            "A_isotropic",
            "A: isotropic",
            TemporalRunConfig(**base_kwargs, fidelity_init_strength=0.0, epsilon_void=0.0),
        ),
        (
            "B_fidelity",
            "B: + fidelity\n(init_str=1.0, half_life=0.999)",
            TemporalRunConfig(**base_kwargs, fidelity_init_strength=1.0, fidelity_half_life=0.999, epsilon_void=0.0),
        ),
        (
            "C_pairwise_void_proxy",
            "C: + pairwise void proxy\n(ε_void=0.005, σ_void_frac=5.0)\n[NOT grid void]",
            TemporalRunConfig(**base_kwargs, fidelity_init_strength=0.0, epsilon_void=0.005, sigma_void_frac=5.0),
        ),
        (
            "D_elasticity",
            "D: + elasticity\n(λ_stretch=0.05, λ_bend=0.02)",
            TemporalRunConfig(**base_kwargs, fidelity_init_strength=0.0, epsilon_void=0.0,
                              lambda_stretch=0.05, lambda_bend=0.02),
        ),
        (
            "E_all_on",
            "E: all ON\n(fid half_life=0.1 fast + void + ela)",
            TemporalRunConfig(**base_kwargs, fidelity_init_strength=1.0, fidelity_half_life=0.1,
                              epsilon_void=0.005, sigma_void_frac=5.0),
        ),
    ]

    summaries = []
    condition_finals = []   # [(label, pos_final), ...] for summary comparison plot

    for dir_name, cond_label, cfg in conditions:
        print(f"\n=== Condition {cond_label.split(chr(10))[0]} ===")
        # Reload from saved init to guarantee identical dataset object
        ds = load_initialization(init_path, variant="bifurcating_trunk",
                                 n_per_cluster=args.n_per_branch)
        summary, pos_final = _run_and_plot(
            ds, cfg,
            output_dir / dir_name,
            condition_label=cond_label,
            split_full=args.split_full,
            no_animation=args.no_animation,
            n_frames=args.n_frames,
            pos_init_fixed=pos_init_fixed,
        )
        summaries.append(summary)
        condition_finals.append((cond_label, pos_final))

    # --- Summary comparison figure ---
    print("\nGenerating summary_comparison.png...")
    plot_summary_comparison(
        pos_init_fixed, condition_finals,
        dataset.labels, dataset.time_values,
        output_dir / "summary_comparison.png",
        split_full=args.split_full,
    )

    # --- Metrics bar chart ---
    print("Generating metrics_comparison.png...")
    plot_metrics_comparison(summaries, output_dir / "metrics_comparison.png")

    # --- CSV ---
    df = pd.DataFrame(summaries)
    df.to_csv(output_dir / "summary.csv", index=False)

    print("\n=== Summary ===")
    print(df[["condition", "trunk_linearity_early", "branch_sep_late",
              "within_branch_spread_ratio", "coherence_selectivity"]].to_string(index=False))

    print("\n--- What to look for ---")
    print("  trunk_linearity_early        : >0.7 = line-like trunk  (~0.5 = blob)")
    print("  branch_sep_late              : >2.0 = clear split      (<1.0 = still fused)")
    print("  within_branch_spread_ratio   : ~1.0 = local structure preserved  (>2 = inflated)")
    print("  coherence_selectivity        : >1.0 = model distinguishes branches")

    # --- Optional force family sweeps ---
    if args.fidelity_half_life_sweep:
        print("\n=== Force family sweeps ===")
        plot_force_sweeps(
            init_path=init_path,
            base_kwargs=base_kwargs,
            output_path=output_dir / "force_sweeps.png",
            n_per_branch=args.n_per_branch,
            split_full=args.split_full,
            verbose=False,
        )

    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
