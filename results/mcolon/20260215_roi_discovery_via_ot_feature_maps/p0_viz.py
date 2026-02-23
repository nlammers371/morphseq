"""
Phase 0 Visualization (Section 3 of PHASE0_SPEC).

Provides the main debugging/reporting figures for Phase 0:
  3.1 Cost density maps (mean WT, mean cep290, difference, smoothed contours)
  3.2 Displacement dynamics maps (quiver + magnitude)
  AUROC vs S-bin curves
  Logistic coefficient profiles
  Interval search results
  Null distribution histograms
  Bootstrap confidence bands

All visual smoothing is for display only — stats use unsmoothed data.

Preferred clean style:
  - embryo outline (mask_ref boundary)
  - filled contours (smoothed scalar field)
  - thin contour lines on top (same levels)
  - consistent color scale across WT/mutant/diff
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embryo_outline(mask_ref: np.ndarray, ax: plt.Axes, color: str = "white", lw: float = 1.0):
    """Draw embryo outline (mask_ref boundary) on axes."""
    ax.contour(mask_ref.astype(float), levels=[0.5], colors=[color],
               linewidths=lw, linestyles="-", origin="upper")


def _apply_mask_nan(field: np.ndarray, mask_ref: np.ndarray) -> np.ndarray:
    """Mask field outside embryo with NaN."""
    return np.where(mask_ref.astype(bool), field, np.nan)


# ---------------------------------------------------------------------------
# 3.1 Cost density maps
# ---------------------------------------------------------------------------

def plot_cost_density_suite(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    outlier_flag: np.ndarray,
    sigma_grid: Tuple[float, ...] = (1.0, 2.0, 4.0),
    label_names: Dict[int, str] = None,
    save_dir: Optional[str | Path] = None,
) -> Dict[str, plt.Figure]:
    """
    Figs A1–A6: Mean cost density maps (raw + smoothed contour versions).
    """
    if label_names is None:
        label_names = {0: "WT", 1: "cep290"}
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    valid = ~outlier_flag
    X_valid = X[valid]
    y_valid = y[valid]
    cost_ch = X_valid[:, :, :, 0]
    mask_bool = mask_ref.astype(bool)

    # Compute means
    means = {}
    for label_int, label_str in label_names.items():
        sel = y_valid == label_int
        if sel.sum() > 0:
            means[label_str] = np.mean(cost_ch[sel], axis=0)
        else:
            means[label_str] = np.zeros_like(mask_ref, dtype=np.float32)

    diff = means.get(label_names[1], np.zeros_like(mask_ref)) - means.get(label_names[0], np.zeros_like(mask_ref))

    figs = {}

    # A1-A3: Raw mean maps
    fig_raw, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax_raw = max(np.nanmax(_apply_mask_nan(m, mask_ref)) for m in means.values() if np.any(np.isfinite(_apply_mask_nan(m, mask_ref))))

    for idx, (label_str, mean_map) in enumerate(means.items()):
        ax = axes[idx]
        display = _apply_mask_nan(mean_map, mask_ref)
        im = ax.imshow(display, cmap="hot", vmin=0, vmax=vmax_raw, interpolation="bilinear", origin="upper")
        _embryo_outline(mask_ref, ax)
        ax.set_title(f"Fig A{idx+1}: Mean Cost — {label_str}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # A3: difference
    diff_display = _apply_mask_nan(diff, mask_ref)
    vabs = np.nanmax(np.abs(diff_display)) if np.any(np.isfinite(diff_display)) else 1.0
    im = axes[2].imshow(diff_display, cmap="RdBu_r", vmin=-vabs, vmax=vabs, interpolation="bilinear", origin="upper")
    _embryo_outline(mask_ref, axes[2], color="black")
    axes[2].set_title(f"Fig A3: Difference ({label_names[1]} − {label_names[0]})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Δ cost")

    fig_raw.tight_layout()
    figs["cost_raw"] = fig_raw
    if save_dir:
        fig_raw.savefig(save_dir / "fig_A1_A3_cost_density_raw.png", dpi=150, bbox_inches="tight")

    # A4-A6: Smoothed contour versions
    for sigma in sigma_grid:
        fig_smooth, axes_s = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (label_str, mean_map) in enumerate(means.items()):
            ax = axes_s[idx]
            smoothed = gaussian_filter(mean_map * mask_bool, sigma=sigma)
            smoothed = _apply_mask_nan(smoothed, mask_ref)
            levels = np.linspace(0, vmax_raw, 12)

            ax.contourf(smoothed, levels=levels, cmap="hot", origin="upper")
            ax.contour(smoothed, levels=levels, colors="k", linewidths=0.3, origin="upper")
            _embryo_outline(mask_ref, ax, color="white", lw=1.5)
            ax.set_title(f"{label_str} (σ={sigma})")
            ax.axis("off")

        # Difference contour
        diff_smoothed = gaussian_filter(diff * mask_bool, sigma=sigma)
        diff_smoothed = _apply_mask_nan(diff_smoothed, mask_ref)
        levels_diff = np.linspace(-vabs, vabs, 15)

        axes_s[2].contourf(diff_smoothed, levels=levels_diff, cmap="RdBu_r", origin="upper")
        axes_s[2].contour(diff_smoothed, levels=levels_diff, colors="k", linewidths=0.3, origin="upper")
        _embryo_outline(mask_ref, axes_s[2], color="black", lw=1.5)
        axes_s[2].set_title(f"Diff ({label_names[1]}−{label_names[0]}) σ={sigma}")
        axes_s[2].axis("off")

        fig_smooth.suptitle(f"Fig A4+: Smoothed Contours (σ={sigma})", fontsize=12)
        fig_smooth.tight_layout(rect=[0, 0, 1, 0.95])
        figs[f"cost_contour_sigma{sigma}"] = fig_smooth
        if save_dir:
            fig_smooth.savefig(save_dir / f"fig_A_cost_contour_sigma{sigma:.0f}.png",
                               dpi=150, bbox_inches="tight")

    return figs


# ---------------------------------------------------------------------------
# 3.2 Displacement dynamics maps
# ---------------------------------------------------------------------------

def plot_displacement_suite(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    outlier_flag: np.ndarray,
    stride: int = 8,
    label_names: Dict[int, str] = None,
    save_dir: Optional[str | Path] = None,
) -> Dict[str, plt.Figure]:
    """
    Figs B1–B3: Mean displacement vector fields (quiver).
    Requires V1_DYNAMICS channel set (ch1=disp_u, ch2=disp_v).
    """
    if label_names is None:
        label_names = {0: "WT", 1: "cep290"}
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    C = X.shape[-1]
    if C < 3:
        logger.warning("Displacement suite requires V1_DYNAMICS (C>=5), skipping")
        return {}

    valid = ~outlier_flag
    X_valid = X[valid]
    y_valid = y[valid]

    # Compute mean displacement per class
    mean_disp = {}
    for label_int, label_str in label_names.items():
        sel = y_valid == label_int
        if sel.sum() > 0:
            mean_u = np.mean(X_valid[sel, :, :, 1], axis=0)
            mean_v = np.mean(X_valid[sel, :, :, 2], axis=0)
        else:
            mean_u = np.zeros_like(mask_ref, dtype=np.float32)
            mean_v = np.zeros_like(mask_ref, dtype=np.float32)
        mean_disp[label_str] = (mean_u, mean_v)

    # Difference displacement
    wt_name, mut_name = label_names[0], label_names[1]
    diff_u = mean_disp[mut_name][0] - mean_disp[wt_name][0]
    diff_v = mean_disp[mut_name][1] - mean_disp[wt_name][1]
    mean_disp["Difference"] = (diff_u, diff_v)

    figs = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mask_bool = mask_ref.astype(bool)

    H, W = mask_ref.shape
    yy, xx = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")

    for idx, (label, (u_map, v_map)) in enumerate(mean_disp.items()):
        ax = axes[idx]

        # Background: embryo mask
        ax.imshow(mask_ref.astype(float), cmap="gray", alpha=0.3,
                  extent=[0, W, H, 0], origin="upper")

        # Quiver
        u_sub = u_map[::stride, ::stride]
        v_sub = v_map[::stride, ::stride]
        mag_sub = np.sqrt(u_sub**2 + v_sub**2)

        # Only plot where there's signal
        threshold = np.percentile(mag_sub[mag_sub > 0], 10) if np.any(mag_sub > 0) else 0
        show = mag_sub > threshold

        if np.any(show):
            ax.quiver(xx[show], yy[show], u_sub[show], v_sub[show],
                      mag_sub[show], cmap="hot", scale=150, scale_units="xy",
                      angles="xy", width=0.002)

        _embryo_outline(mask_ref, ax, color="white")
        ax.set_title(f"Fig B{idx+1}: {label}")
        ax.axis("off")

    fig.suptitle("Displacement Vector Fields (Mean)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figs["displacement_quiver"] = fig
    if save_dir:
        fig.savefig(save_dir / "fig_B1_B3_displacement_quiver.png", dpi=150, bbox_inches="tight")

    # Optional: displacement magnitude maps
    fig_mag, axes_m = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (label, (u_map, v_map)) in enumerate(mean_disp.items()):
        ax = axes_m[idx]
        mag = np.sqrt(u_map**2 + v_map**2)
        display = _apply_mask_nan(mag, mask_ref)
        im = ax.imshow(display, cmap="viridis", interpolation="bilinear", origin="upper")
        _embryo_outline(mask_ref, ax)
        ax.set_title(f"|d| — {label}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig_mag.suptitle("Displacement Magnitude", fontsize=12)
    fig_mag.tight_layout(rect=[0, 0, 1, 0.95])
    figs["displacement_magnitude"] = fig_mag
    if save_dir:
        fig_mag.savefig(save_dir / "fig_B_displacement_magnitude.png", dpi=150, bbox_inches="tight")

    return figs


# ---------------------------------------------------------------------------
# AUROC per S-bin plots
# ---------------------------------------------------------------------------

def plot_auroc_vs_sbin(
    auroc_dfs: Dict[str, pd.DataFrame],
    K: int = 10,
    title: str = "AUROC vs S-bin",
    save_path: Optional[str | Path] = None,
    bootstrap_result: Optional[Dict] = None,
) -> plt.Figure:
    """
    Plot AUROC vs S-bin for one or more features.

    auroc_dfs: dict mapping feature_name → DataFrame with k_bin, auroc columns
    bootstrap_result: if provided, add confidence bands
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(auroc_dfs), 1)))
    s_centers = [(k + 0.5) / K for k in range(K)]

    for i, (feat_name, df) in enumerate(auroc_dfs.items()):
        df_sorted = df.sort_values("k_bin")
        aurocs = df_sorted["auroc"].values
        ax.plot(s_centers[:len(aurocs)], aurocs, "o-", color=colors[i],
                label=feat_name, linewidth=2, markersize=6)

    # Bootstrap confidence bands
    if bootstrap_result is not None:
        ci_lo = bootstrap_result.get("auroc_ci_lo")
        ci_hi = bootstrap_result.get("auroc_ci_hi")
        if ci_lo is not None and ci_hi is not None:
            ax.fill_between(s_centers[:len(ci_lo)], ci_lo, ci_hi,
                            alpha=0.2, color="gray", label="95% CI (bootstrap)")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("S (rostral → caudal)")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Logistic coefficient profile
# ---------------------------------------------------------------------------

def plot_coefficient_profile(
    coef_mean: np.ndarray,
    K: int = 10,
    feature_cols: List[str] = ["cost_mean"],
    title: str = "Logistic Regression Coefficients",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot coefficient magnitude profile over S-bins."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    n_features = len(feature_cols)
    s_centers = [(k + 0.5) / K for k in range(K)]

    for j, feat_name in enumerate(feature_cols):
        coefs = coef_mean[j * K:(j + 1) * K]
        ax.bar([s + j * 0.03 for s in s_centers[:len(coefs)]], np.abs(coefs),
               width=0.08, alpha=0.7, label=feat_name)

    ax.set_xlabel("S (rostral → caudal)")
    ax.set_ylabel("|coefficient|")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Interval search visualization
# ---------------------------------------------------------------------------

def plot_interval_results(
    interval_df: pd.DataFrame,
    selected: Dict,
    K: int = 10,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot interval search results: AUROC vs interval, selected interval highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: best AUROC vs interval length
    ax = axes[0]
    best_by_len = interval_df.groupby("n_bins")["auroc"].max().reset_index()
    ax.plot(best_by_len["n_bins"], best_by_len["auroc"], "o-", color="steelblue",
            linewidth=2, markersize=8)
    ax.axhline(selected["auroc"], color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Interval Length (bins)")
    ax.set_ylabel("Best AUROC")
    ax.set_title("Best AUROC vs Interval Length")
    ax.grid(True, alpha=0.3)

    # Right: selected interval on S axis
    ax = axes[1]
    s_centers = [(k + 0.5) / K for k in range(K)]
    ax.bar(s_centers, [0.1] * K, width=1/K, color="lightgray", edgecolor="gray")

    # Highlight selected interval
    for k in range(selected["bin_start"], selected["bin_end"]):
        ax.bar((k + 0.5) / K, 0.1, width=1/K, color="red", alpha=0.6)

    ax.set_xlabel("S (rostral → caudal)")
    ax.set_title(f"Selected Interval: S=[{selected['S_lo']:.2f}, {selected['S_hi']:.2f})\n"
                 f"AUROC={selected['auroc']:.4f}, {selected['n_bins']} bins")
    ax.set_xlim(0, 1)
    ax.set_yticks([])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Null distribution plots
# ---------------------------------------------------------------------------

def plot_permutation_null(
    null_result: Dict,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Histogram of permutation null distribution with observed value marked."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    null_dist = null_result["null_distribution"]
    observed = null_result["observed"]
    pvalue = null_result["pvalue"]

    ax.hist(null_dist, bins=30, alpha=0.7, color="steelblue", edgecolor="k",
            label="Null distribution")
    ax.axvline(observed, color="red", linewidth=2, linestyle="--",
               label=f"Observed={observed:.4f}\np={pvalue:.4f}")

    ax.set_xlabel(f"AUROC ({null_result['statistic']})")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Null ({null_result['test']})\n"
                 f"n_perm={null_result['n_permute']}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_bootstrap_interval_stability(
    bootstrap_result: Dict,
    K: int = 10,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot bootstrap distributions of interval start/end points."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    starts = bootstrap_result["interval_starts"]
    ends = bootstrap_result["interval_ends"]

    if len(starts) > 0:
        axes[0].hist(starts / K, bins=K, alpha=0.7, color="steelblue", edgecolor="k")
        axes[0].axvline(bootstrap_result["interval_start_mean"] / K, color="red", linewidth=2)
    axes[0].set_xlabel("S start")
    axes[0].set_title("Bootstrap: Interval Start Distribution")

    if len(ends) > 0:
        axes[1].hist(ends / K, bins=K, alpha=0.7, color="darkorange", edgecolor="k")
        axes[1].axvline(bootstrap_result["interval_end_mean"] / K, color="red", linewidth=2)
    axes[1].set_xlabel("S end")
    axes[1].set_title("Bootstrap: Interval End Distribution")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# S coordinate visualization
# ---------------------------------------------------------------------------

def plot_s_map(
    S_map_ref: np.ndarray,
    mask_ref: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot the S coordinate map on the reference mask."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    display = _apply_mask_nan(S_map_ref, mask_ref)
    im = ax.imshow(display, cmap="viridis", vmin=0, vmax=1, interpolation="bilinear", origin="upper")
    _embryo_outline(mask_ref, ax, color="white")
    plt.colorbar(im, ax=ax, label="S (0=head, 1=tail)")
    ax.set_title("S Coordinate Map (Rostral → Caudal)")
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


__all__ = [
    "plot_cost_density_suite",
    "plot_displacement_suite",
    "plot_auroc_vs_sbin",
    "plot_coefficient_profile",
    "plot_interval_results",
    "plot_permutation_null",
    "plot_bootstrap_interval_stability",
    "plot_s_map",
]
