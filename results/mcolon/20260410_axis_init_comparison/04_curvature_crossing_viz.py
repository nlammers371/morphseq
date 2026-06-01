"""Curvature crossing visualization — 3D spacetime.

X/Y = condensed 2D embedding space.
Z   = developmental time (hpf).

Produces:
  figures/crossing/
    3d_spacetime_static.png   — static 3D view: faint individual traces + bold mean tube + variance band
    3d_spacetime_anim.gif     — animated rotation of the same 3D structure

Groups shown: High_to_Low, Low_to_High, cep290_wildtype.

Run:
  conda run -n segmentation_grounded_sam --no-capture-output \
      python results/mcolon/20260410_axis_init_comparison/04_curvature_crossing_viz.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="using precomputed metric", category=UserWarning)
warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

_SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_DIR  = _SCRIPT_DIR / "results" / "reference_genotype_condensation" / "curvature" / "run"
_WIDE_PARQUET = _SCRIPT_DIR / "results" / "reference_genotype_condensation" / "curvature" / "wide_scores.parquet"
_OUT_DIR  = _SCRIPT_DIR / "figures" / "crossing"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
COLORS = {
    "High_to_Low":     "#E76FA2",
    "Low_to_High":     "#2FB7B0",
    "cep290_wildtype": "#2166AC",
}
DISPLAY = {
    "High_to_Low":     "High→Low",
    "Low_to_High":     "Low→High",
    "cep290_wildtype": "Wildtype",
}
GROUPS = list(COLORS.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    d = np.load(_RUN_DIR / "condensed_positions.npz", allow_pickle=True)
    positions   = d["positions"]    # (N_e, T, 2)
    mask        = d["mask"]         # (N_e, T)
    time_values = d["time_values"]  # (T,)
    embryo_ids  = d["embryo_ids"]   # (N_e,)

    wide = pd.read_parquet(_WIDE_PARQUET)
    embryo_to_viz = (
        wide[["embryo_id", "viz_label"]]
        .drop_duplicates("embryo_id")
        .set_index("embryo_id")["viz_label"]
        .to_dict()
    )
    viz_labels = np.array(
        [embryo_to_viz.get(str(eid), "other") for eid in embryo_ids],
        dtype=object,
    )
    group_idx = {g: np.where(viz_labels == g)[0] for g in GROUPS}
    return positions, mask, time_values, group_idx


def group_stats(positions, mask, idx):
    """Per-time mean and std for a group. Returns (T,2) arrays."""
    T = positions.shape[1]
    mean_xy = np.full((T, 2), np.nan)
    std_xy  = np.full((T, 2), np.nan)
    for t in range(T):
        obs = idx[mask[idx, t]]
        if len(obs) >= 2:
            pts = positions[obs, t, :]
            mean_xy[t] = pts.mean(axis=0)
            std_xy[t]  = pts.std(axis=0)
    return mean_xy, std_xy


# ---------------------------------------------------------------------------
# Core drawing helper — draws one 3D scene onto ax
# ---------------------------------------------------------------------------

def draw_3d_scene(ax, positions, mask, time_values, group_idx,
                  individual_alpha=0.08, mean_lw=3.0, band_alpha=0.18,
                  n_band_samples=40, elev=20, azim=-60):
    """
    Draw individual traces (faint), mean tube, and variance band on a 3D axes.

    Z = time_values (hpf), X/Y = condensed embedding dims.
    """
    T = len(time_values)

    for group, idx in group_idx.items():
        color = COLORS[group]
        mean_xy, std_xy = group_stats(positions, mask, idx)

        # ---- faint individual traces ----------------------------------------
        for ei in idx:
            obs_t = np.where(mask[ei])[0]
            if len(obs_t) < 2:
                continue
            x = positions[ei, obs_t, 0]
            y = positions[ei, obs_t, 1]
            z = time_values[obs_t]
            ax.plot(x, y, z, color=color, linewidth=0.5,
                    alpha=individual_alpha, zorder=1)

        # ---- mean tube -------------------------------------------------------
        valid = ~np.isnan(mean_xy[:, 0])
        t_idx = np.where(valid)[0]
        if len(t_idx) < 2:
            continue

        mx = mean_xy[t_idx, 0]
        my = mean_xy[t_idx, 1]
        mz = time_values[t_idx]

        # Draw mean as colored segments fading from lighter to full saturation
        # by breaking into individual segments with increasing alpha
        n_seg = len(t_idx) - 1
        for i in range(n_seg):
            seg_alpha = 0.4 + 0.6 * (i / max(n_seg - 1, 1))
            ax.plot(mx[i:i+2], my[i:i+2], mz[i:i+2],
                    color=color, linewidth=mean_lw, alpha=seg_alpha, zorder=4,
                    solid_capstyle="round")

        # ---- variance band (±1 SD, sampled as a ribbon) ----------------------
        # Build a ribbon by sweeping ±std in the direction perpendicular to the
        # mean trajectory projected onto the XY plane, then stacking into a surface.
        sx = std_xy[t_idx, 0]
        sy = std_xy[t_idx, 1]
        std_mag = np.sqrt(sx**2 + sy**2)  # scalar SD magnitude per time bin

        # Parametric ribbon: sample angles around the mean in XY
        # For simplicity use an ellipse of ±std_x, ±std_y
        theta = np.linspace(0, 2 * np.pi, n_band_samples)
        # Surface: (n_band_samples, len(t_idx))
        band_x = mx[np.newaxis, :] + sx[np.newaxis, :] * np.cos(theta)[:, np.newaxis]
        band_y = my[np.newaxis, :] + sy[np.newaxis, :] * np.sin(theta)[:, np.newaxis]
        band_z = np.tile(mz[np.newaxis, :], (n_band_samples, 1))

        ax.plot_surface(
            band_x, band_y, band_z,
            color=color, alpha=band_alpha,
            linewidth=0, antialiased=True, zorder=2,
        )

        # ---- endpoint dot ----------------------------------------------------
        ax.scatter([mx[-1]], [my[-1]], [mz[-1]],
                   color=color, s=60, zorder=6,
                   edgecolors="white", linewidths=1.2)

    # ---- axes formatting -----------------------------------------------------
    ax.set_xlabel("dim 1", fontsize=9, labelpad=4)
    ax.set_ylabel("dim 2", fontsize=9, labelpad=4)
    ax.set_zlabel("hpf", fontsize=9, labelpad=4)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[g], linewidth=2.5, label=DISPLAY[g])
               for g in GROUPS]
    ax.legend(handles=handles, fontsize=8, frameon=False,
              loc="upper left", bbox_to_anchor=(0.0, 1.0))


# ---------------------------------------------------------------------------
# Output A: static PNG
# ---------------------------------------------------------------------------

def plot_3d_static(positions, mask, time_values, group_idx, out_path,
                   individual_alpha=0.08):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    draw_3d_scene(ax, positions, mask, time_values, group_idx,
                  individual_alpha=individual_alpha, elev=22, azim=-55)
    ax.set_title("Curvature trajectories in developmental spacetime\n"
                 "(X/Y = embedding, Z = hpf | faint = individuals, bold = group mean ± 1 SD)",
                 fontsize=9, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Output B: rotating animation
# ---------------------------------------------------------------------------

def make_rotation_anim(positions, mask, time_values, group_idx, out_path,
                       fps=8, n_frames=72):
    """Slow rotation (360° / n_frames), one full revolution."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw static scene once
    draw_3d_scene(ax, positions, mask, time_values, group_idx,
                  individual_alpha=0.08, elev=22, azim=-55)
    ax.set_title("Curvature trajectories — developmental spacetime", fontsize=9, pad=8)

    azim_start = -55

    def _update(frame):
        ax.view_init(elev=22, azim=azim_start + frame * (360 / n_frames))
        return []

    anim = animation.FuncAnimation(fig, _update, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Output C: time-reveal animation — traces emerge from bottom (early) upward
# ---------------------------------------------------------------------------

def make_time_reveal_anim(positions, mask, time_values, group_idx, out_path,
                          fps=5):
    """Each frame reveals one more time bin, growing upward in Z."""
    T = len(time_values)

    # Precompute global axis bounds from all 3 groups
    all_x, all_y = [], []
    for idx in group_idx.values():
        for ei in idx:
            obs_t = np.where(mask[ei])[0]
            if len(obs_t):
                all_x.extend(positions[ei, obs_t, 0].tolist())
                all_y.extend(positions[ei, obs_t, 1].tolist())
    pad = 0.4
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)
    zlim = (float(time_values[0]) - 2, float(time_values[-1]) + 2)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_xlabel("dim 1", fontsize=9, labelpad=4)
    ax.set_ylabel("dim 2", fontsize=9, labelpad=4)
    ax.set_zlabel("hpf", fontsize=9, labelpad=4)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-55)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[g], linewidth=2.5, label=DISPLAY[g])
               for g in GROUPS]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                          fontsize=11, fontweight="bold")

    # Precompute means
    means = {g: group_stats(positions, mask, idx)[0] for g, idx in group_idx.items()}

    def _update(frame):
        # frame = current max time bin revealed
        t_max = frame
        ax.cla()
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel("dim 1", fontsize=9, labelpad=4)
        ax.set_ylabel("dim 2", fontsize=9, labelpad=4)
        ax.set_zlabel("hpf", fontsize=9, labelpad=4)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=22, azim=-55)
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
        ax.text2D(0.02, 0.95, f"{time_values[t_max]:.0f} hpf",
                  transform=ax.transAxes, fontsize=11, fontweight="bold")

        for group, idx in group_idx.items():
            color = COLORS[group]
            m = means[group]

            # Faint individual traces up to t_max
            for ei in idx:
                obs_t = np.where(mask[ei] & (np.arange(T) <= t_max))[0]
                if len(obs_t) < 2:
                    continue
                ax.plot(positions[ei, obs_t, 0],
                        positions[ei, obs_t, 1],
                        time_values[obs_t],
                        color=color, linewidth=0.5, alpha=0.10, zorder=1)

            # Mean line up to t_max
            valid_t = [t for t in range(t_max + 1) if not np.isnan(m[t, 0])]
            if len(valid_t) < 2:
                continue
            vt = np.array(valid_t)
            ax.plot(m[vt, 0], m[vt, 1], time_values[vt],
                    color=color, linewidth=3.0, alpha=0.95, zorder=4)

            # Current mean dot
            t_last = valid_t[-1]
            ax.scatter([m[t_last, 0]], [m[t_last, 1]], [time_values[t_last]],
                       color=color, s=70, zorder=6,
                       edgecolors="white", linewidths=1.2)

        return []

    anim = animation.FuncAnimation(fig, _update, frames=T,
                                   interval=1000 // fps, blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    positions, mask, time_values, group_idx = load_data()
    for g, idx in group_idx.items():
        print(f"  {DISPLAY[g]}: {len(idx)} embryos")

    print("Static 3D spacetime — alpha sweep...")
    for alpha in [0.03, 0.06, 0.10, 0.18, 0.30]:
        plot_3d_static(positions, mask, time_values, group_idx,
                       _OUT_DIR / f"3d_spacetime_alpha{int(alpha*100):02d}.png",
                       individual_alpha=alpha)

    print("Rotation animation...")
    make_rotation_anim(positions, mask, time_values, group_idx,
                       _OUT_DIR / "3d_rotation.gif", fps=8, n_frames=72)

    print("Time-reveal animation (traces emerge over time)...")
    make_time_reveal_anim(positions, mask, time_values, group_idx,
                          _OUT_DIR / "3d_time_reveal.gif", fps=5)

    print(f"\nDone. Outputs in: {_OUT_DIR}")


if __name__ == "__main__":
    main()
