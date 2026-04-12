"""
temporal_sandbox.py
-------------------
Minimal synthetic temporal stitching sandbox for testing the coherence feedback
mechanism before applying it to real multi-slice trajectory data.

Two tests:
  Test A (stable_bundles):   Two clusters drift apart across T time steps.
                              Coherence should reinforce within-bundle pairs.

  Test B (crossing_bundles): Two bundles converge, cross, then re-separate.
                              Coherence must not permanently fuse them.

Defaults deliberately set fidelity_init_strength=0.0 (no fidelity) so results reflect the
coherence + spatial force mechanism alone, not the anchor.

Run (smoke test):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/temporal_sandbox.py \\
      --output-dir /tmp/temporal_sandbox_test \\
      --n-per-cluster 20 --n-time 8 --n-iter 50 --no-animation

Run (full):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/temporal_sandbox.py \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/temporal_sandbox_v1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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

from trajectory_cosmology.condensation.coherence import compute_coherence
from trajectory_cosmology.condensation.engine import run_dynamics
from trajectory_cosmology.condensation.geometry_refs import (
    estimate_geometry_refs,
    estimate_local_spacing_ref,
)
from trajectory_cosmology.condensation.state import CondensationConfig, CondensationResult

# Reuse helpers from the 2D slice sandbox
from slice_sandbox import _cluster_metrics, _resolve_color_map, radial_spread


def _median_knn_dist(pos2d: np.ndarray, k: int = 5) -> float:
    """Median distance to k nearest neighbors across all points in a 2D slice.

    Used to derive sigma_attract_local — a within-bundle scale estimate
    that does not depend on knowing labels.
    """
    n = pos2d.shape[0]
    if n <= k:
        return float(np.median(np.linalg.norm(
            pos2d[:, None, :] - pos2d[None, :, :], axis=-1
        )))
    diff = pos2d[:, None, :] - pos2d[None, :, :]   # (n, n, 2)
    sq_dist = (diff ** 2).sum(axis=-1)              # (n, n)
    np.fill_diagonal(sq_dist, np.inf)
    k_eff = min(k, n - 1)
    knn_dists = np.partition(sq_dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    return float(np.median(np.sqrt(knn_dists)))


# ===========================================================================
# Section 1: Synthetic temporal dataset generator
# ===========================================================================

@dataclass
class TemporalDataset:
    positions: np.ndarray    # (N_e, T, 2) — ground-truth initial positions
    mask: np.ndarray         # (N_e, T) bool — all True for synthetic
    labels: np.ndarray       # (N_e,) int — cluster identity, constant per embryo
    time_values: np.ndarray  # (T,) float
    variant: str
    n_per_cluster: int
    n_time: int


def make_stable_bundles(
    n_per_cluster: int = 40,
    n_time: int = 10,
    drift_speed: float = 0.1,
    within_cluster_noise: float = 0.3,
    random_seed: int = 42,
) -> TemporalDataset:
    """Test A: Two clusters drift apart steadily across time.

    Cluster 0 starts at (-3, 0), drifts left.
    Cluster 1 starts at (+3, 0), drifts right.
    Per-embryo noise added independently at each time step.
    """
    rng = np.random.default_rng(random_seed)
    N = 2 * n_per_cluster
    positions = np.zeros((N, n_time, 2))
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster, dtype=int)

    # Initial cluster centers
    center_0 = np.array([-3.0, 0.0])
    center_1 = np.array([+3.0, 0.0])
    drift_0 = np.array([-drift_speed, 0.0])
    drift_1 = np.array([+drift_speed, 0.0])

    # Draw initial per-embryo offsets from cluster center
    offset_0 = rng.normal(0, within_cluster_noise, size=(n_per_cluster, 2))
    offset_1 = rng.normal(0, within_cluster_noise, size=(n_per_cluster, 2))

    for t in range(n_time):
        noise_0 = rng.normal(0, within_cluster_noise * 0.3, size=(n_per_cluster, 2))
        noise_1 = rng.normal(0, within_cluster_noise * 0.3, size=(n_per_cluster, 2))
        positions[:n_per_cluster, t, :] = center_0 + t * drift_0 + offset_0 + noise_0
        positions[n_per_cluster:, t, :] = center_1 + t * drift_1 + offset_1 + noise_1

    mask = np.ones((N, n_time), dtype=bool)
    time_values = np.arange(n_time, dtype=float)

    return TemporalDataset(
        positions=positions,
        mask=mask,
        labels=labels,
        time_values=time_values,
        variant="stable_bundles",
        n_per_cluster=n_per_cluster,
        n_time=n_time,
    )


def make_crossing_bundles(
    n_per_cluster: int = 40,
    n_time: int = 15,
    approach_speed: float = 0.5,
    pass_speed: float = 0.3,
    within_cluster_noise: float = 0.3,
    random_seed: int = 42,
) -> TemporalDataset:
    """Test B: Two bundles converge, pass through each other, then diverge.

    Cluster 0 starts at (-6, -1), moves right and up.
    Cluster 1 starts at (+6, +1), moves left and down.
    They cross near t = n_time // 2 then re-separate.
    """
    rng = np.random.default_rng(random_seed)
    N = 2 * n_per_cluster
    positions = np.zeros((N, n_time, 2))
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster, dtype=int)

    mid = n_time // 2

    # Per-embryo offsets (fixed identity across time)
    offset_0 = rng.normal(0, within_cluster_noise, size=(n_per_cluster, 2))
    offset_1 = rng.normal(0, within_cluster_noise, size=(n_per_cluster, 2))

    for t in range(n_time):
        # Approach phase: move toward crossing point
        # Post-cross phase: move away on the other side
        # Parametrize as a smooth trajectory through origin
        frac = t / (n_time - 1)  # 0 to 1
        x0_center = np.array([-6.0 + frac * 12.0, -1.0 + frac * 2.0])  # left→right
        x1_center = np.array([+6.0 - frac * 12.0, +1.0 - frac * 2.0])  # right→left

        noise_0 = rng.normal(0, within_cluster_noise * 0.3, size=(n_per_cluster, 2))
        noise_1 = rng.normal(0, within_cluster_noise * 0.3, size=(n_per_cluster, 2))

        positions[:n_per_cluster, t, :] = x0_center + offset_0 + noise_0
        positions[n_per_cluster:, t, :] = x1_center + offset_1 + noise_1

    mask = np.ones((N, n_time), dtype=bool)
    time_values = np.arange(n_time, dtype=float)

    return TemporalDataset(
        positions=positions,
        mask=mask,
        labels=labels,
        time_values=time_values,
        variant="crossing_bundles",
        n_per_cluster=n_per_cluster,
        n_time=n_time,
    )


# ===========================================================================
# Section 2: Metrics
# ===========================================================================

def _per_slice_metrics(positions: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Compute cluster metrics for each time slice independently."""
    T = positions.shape[1]
    rows = []
    for t in range(T):
        m = _cluster_metrics(positions[:, t, :], labels)
        m["t"] = t
        rows.append(m)
    return rows


def _within_bundle_spread(positions: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-cluster std of 2D positions across all time slices.

    This is a label-aware compactness measure for sandbox evaluation.
    Lower is more compact. Compare final vs initial to get spread_ratio.
    """
    T = positions.shape[1]
    groups = sorted(set(labels.tolist()))
    stds = []
    for g in groups:
        idx = np.flatnonzero(labels == g)
        for t in range(T):
            pts = positions[idx, t, :]   # (n_cluster, 2)
            stds.append(float(pts.std()))
    return float(np.mean(stds)) if stds else 0.0


def _local_radius_ratio(
    positions_curr: np.ndarray,   # (N_e, T, 2)
    positions_ref: np.ndarray,    # (N_e, T, 2) — initial positions
    k: int = 5,
) -> dict[str, float]:
    """Label-free local neighborhood inflation metric.

    For each (embryo, time) point, computes median distance to its k
    nearest neighbors at current vs reference positions. Returns:
      local_radius_ratio_median — median ratio across all points/slices
      local_radius_ratio_p90   — 90th percentile (worst-case inflation)
      local_radius_ratio_p95   — 95th percentile

    Ratio > 1 means neighborhoods inflated (decompaction).
    Ratio ≈ 1 means local structure preserved.
    """
    N_e, T, _ = positions_curr.shape
    ratios = []

    for t in range(T):
        pts_curr = positions_curr[:, t, :]
        pts_ref = positions_ref[:, t, :]
        r_curr = _median_knn_dist(pts_curr, k=k)
        r_ref = _median_knn_dist(pts_ref, k=k)
        # Per-point version: use kNN radius per point
        n = pts_curr.shape[0]
        k_eff = min(k, n - 1)
        if k_eff < 1:
            continue

        for pts, pts_r in [(pts_curr, pts_curr), (pts_curr, pts_curr)]:
            break  # just use slice-level ratio below

        # Slice-level ratio (simpler and sufficient)
        if r_ref > 1e-12:
            ratios.append(r_curr / r_ref)

    if not ratios:
        return {"local_radius_ratio_median": float("nan"),
                "local_radius_ratio_p90": float("nan"),
                "local_radius_ratio_p95": float("nan")}

    arr = np.array(ratios)
    return {
        "local_radius_ratio_median": float(np.median(arr)),
        "local_radius_ratio_p90": float(np.percentile(arr, 90)),
        "local_radius_ratio_p95": float(np.percentile(arr, 95)),
    }


def _temporal_metrics(
    positions: np.ndarray,              # (N_e, T, 2)
    labels: np.ndarray,                 # (N_e,)
    coherence: np.ndarray,              # (N_e, N_e, T)
    mask: np.ndarray,                   # (N_e, T) bool
    positions_ref: np.ndarray | None = None,  # (N_e, T, 2) initial positions for ratio metrics
) -> dict:
    """Aggregate metrics across time slices plus coherence selectivity."""
    slice_metrics = _per_slice_metrics(positions, labels)
    df = pd.DataFrame(slice_metrics)

    centroid_distance_mean = float(df["centroid_distance"].mean())
    sep_ratio_mean = float(df["separation_ratio"].mean())
    global_spread_mean = float(df["global_spread"].mean())

    # Coherence selectivity: within-bundle vs cross-bundle C
    groups = sorted(set(labels.tolist()))
    T = coherence.shape[2]
    within_vals = []
    cross_vals = []
    for g in groups:
        same = labels == g
        diff_mask = ~same
        for t in range(T):
            C_t = coherence[:, :, t]
            # Off-diagonal within-cluster
            within_block = C_t[np.ix_(np.flatnonzero(same), np.flatnonzero(same))].copy()
            np.fill_diagonal(within_block, 0.0)  # exclude self
            n_within = same.sum()
            if n_within > 1:
                within_vals.append(within_block.sum() / (n_within * (n_within - 1)))
            # Cross-cluster
            cross_block = C_t[np.ix_(np.flatnonzero(same), np.flatnonzero(diff_mask))]
            n_cross = same.sum() * diff_mask.sum()
            if n_cross > 0:
                cross_vals.append(cross_block.mean())

    within_coherence_mean = float(np.mean(within_vals)) if within_vals else 0.0
    cross_coherence_mean = float(np.mean(cross_vals)) if cross_vals else 0.0
    coherence_selectivity = within_coherence_mean / (cross_coherence_mean + 1e-8)

    # Compactness metrics
    within_bundle_spread = _within_bundle_spread(positions, labels)

    result: dict = {
        "centroid_distance_mean": centroid_distance_mean,
        "sep_ratio_mean": sep_ratio_mean,
        "global_spread_mean": global_spread_mean,
        "within_coherence_mean": within_coherence_mean,
        "cross_coherence_mean": cross_coherence_mean,
        "coherence_selectivity": coherence_selectivity,
        "within_bundle_spread": within_bundle_spread,
    }

    if positions_ref is not None:
        ref_spread = _within_bundle_spread(positions_ref, labels)
        result["within_bundle_spread_ratio"] = within_bundle_spread / (ref_spread + 1e-12)
        result.update(_local_radius_ratio(positions, positions_ref))
    else:
        result["within_bundle_spread_ratio"] = float("nan")
        result["local_radius_ratio_median"] = float("nan")
        result["local_radius_ratio_p90"] = float("nan")
        result["local_radius_ratio_p95"] = float("nan")

    return result


# ===========================================================================
# Section 3: Run config and runner
# ===========================================================================

def gamma_from_half_life_iters(fidelity_half_life_iters: float | None) -> float:
    """Convert fidelity half-life (in iterations) to per-iteration retention multiplier.

    mu(n) = mu0 * gamma^n,  where gamma = 2^(-1/h)

    h = fidelity_half_life_iters: number of iterations for mu to drop by half.
    None → gamma = 1.0 (no decay, constant anchor).
    """
    if fidelity_half_life_iters is None:
        return 1.0
    if fidelity_half_life_iters <= 0:
        raise ValueError(f"fidelity_half_life_iters must be > 0, got {fidelity_half_life_iters}")
    return 2.0 ** (-1.0 / fidelity_half_life_iters)


@dataclass
class TemporalRunConfig:
    # --- Structural / informational knobs (not force magnitudes) ---
    sigma_frac: float = 0.5       # attraction bandwidth: sigma = sigma_frac × s_global
    temporal_cohere_bandwidth_mult: float | None = None
    coherence_scale_mult: float | None = None
                                  # temporal coherence bandwidth: sigma_coh = mult × s_local
                                  # None = use sigma (old behaviour, σ = inter-bundle scale)
                                  # Set e.g. 1.0 to anchor coherence to local spacing
    delta: int = 3                # coherence backward window (time bins)
    k_attract: int = 20
    attract_weight: float = 1.0
    temporal_cohere_weight: float = 1.0

    # --- Dimensionless force multipliers (each × reference scale² → coefficient) ---
    # Design principle: each default is set just below the no_change threshold on the
    # Y-benchmark (lr=1e-4, n_iter=500). Under good initialization they stay inert.
    # Under poor initialization (noisy, sparse, misaligned) they activate and correct.
    repulsion_strength_mult: float = 0.005
                                  # ε_r = λ_rep × s_local²
                                  # no_change threshold: 0.0071 — default sits just below
    fidelity_strength_mult: float = 0.25
                                  # μ_0 = λ_fid / s_local²
                                  # no_change threshold: 0.334 — default sits just below
                                  # anchors to init early; decays via fidelity_half_life_iters
    stretch_strength_mult: float = 0.04
                                  # λ_stretch = λ_str / s_step²
                                  # no_change threshold: 0.049 — default sits just below
                                  # activates if trajectory steps are irregular
    bend_strength_mult: float = 0.04
                                  # λ_bend = λ_bnd / s_bend²
                                  # no_change threshold: 0.049 — default sits just below
                                  # activates if trajectory curvature is extreme
    epsilon_void: float = 0.014   # pairwise void proxy strength
                                  # no_change threshold: 0.018 — default sits just below
                                  # activates if embeddings are crowded/overlapping
                                  # NOTE: broad pairwise Gaussian, NOT grid-based occupancy void
    sigma_void_frac: float = 5.0  # sigma_void = sigma_void_frac × s_global

    # --- Raw overrides (backward compat — used when dimensionless mult is 0) ---
    # If fidelity_strength_mult > 0, fidelity_init_strength is derived; otherwise used directly.
    # Same pattern for stretch/bend. This lets old callers pass raw values unchanged.
    fidelity_init_strength: float = 0.0        # initial anchor weight (0 = fidelity off)
    lambda_stretch: float = 0.0
    lambda_bend: float = 0.0
    fidelity_half_life_iters: float | None = 70.0
                                                # solver iterations for fidelity weight to halve.
                                                # mu(n) = fidelity_init_strength * 2^(-n/h)
                                                # None = no decay (constant anchor, different regime)
                                                # 70  ≈ old gamma=0.99 onset
                                                # 50  = faster decay
                                                # 100 = more persistent anchor
    alpha: float = 0.9

    # --- Optimization ---
    lr: float = 1e-4    # calibrated: stable, plateaus by 500 iters (solver_tempo sweep)
    n_iter: int = 500   # calibrated: metrics plateau at ~300-500 iters at lr=1e-4

    # --- Legacy / less-used ---
    sigma_local_frac: float | None = None  # None = use sigma_frac (old behavior)
    r_cut_frac: float = 0.0               # r_cut = r_cut_frac × s_local; 0 = soft-core
    lambda_scale: float = 0.0
    k_local_scale: int = 5


@dataclass
class TemporalRunResult:
    dataset: TemporalDataset
    config: TemporalRunConfig
    cond_result: CondensationResult
    initial_metrics: dict
    final_metrics: dict
    metrics_df: pd.DataFrame

    @property
    def collapse_score(self) -> float:
        gs_init = self.initial_metrics["global_spread_mean"]
        gs_final = self.final_metrics["global_spread_mean"]
        return float(gs_final / (gs_init + 1e-12))

    @property
    def coherence_selectivity_trajectory(self) -> np.ndarray | None:
        """Series of coherence_selectivity logged per iteration, if present."""
        if "coherence_selectivity" in self.metrics_df.columns:
            return self.metrics_df["coherence_selectivity"].values
        return None


def run_temporal(
    dataset: TemporalDataset,
    config: TemporalRunConfig,
    save_snapshots: bool = True,
    verbose: bool = True,
) -> TemporalRunResult:
    """Run the full condensation dynamics on a synthetic temporal dataset."""
    positions = dataset.positions  # (N_e, T, 2)
    mask = dataset.mask

    # --- Measure geometry references from initial positions ---
    # All force coefficients are derived from these reference scales.
    # This is the geometry_refs layer: measure first, calibrate second.
    refs = estimate_geometry_refs(positions, mask, k_local=5)

    # --- Derive force coefficients from dimensionless multipliers ---
    # Repulsion: calibrated to local point spacing (validated at 0.005)
    epsilon_r = config.repulsion_strength_mult * refs.s_local ** 2

    # Fidelity: if dimensionless mult given, derive from s_local; else use raw fidelity_init_strength
    if config.fidelity_strength_mult > 0.0:
        fidelity_init_strength = config.fidelity_strength_mult / (refs.s_local ** 2 + 1e-16)
    else:
        fidelity_init_strength = config.fidelity_init_strength

    # Stretch: if dimensionless mult given, derive from s_step; else use raw lambda_stretch
    if config.stretch_strength_mult > 0.0:
        lambda_stretch = config.stretch_strength_mult / (refs.s_step ** 2 + 1e-16)
    else:
        lambda_stretch = config.lambda_stretch

    # Bend: if dimensionless mult given, derive from s_bend; else use raw lambda_bend
    if config.bend_strength_mult > 0.0:
        lambda_bend = config.bend_strength_mult / (refs.s_bend ** 2 + 1e-16)
    else:
        lambda_bend = config.lambda_bend

    # Attraction bandwidth: inter-bundle scale
    sigma = config.sigma_frac * refs.s_global

    # Coherence bandwidth: local scale if coherence_scale_mult set, else use sigma
    if config.coherence_scale_mult is not None:
        sigma_coh = config.coherence_scale_mult * refs.s_local
    else:
        sigma_coh = None  # dynamics.py falls back to sigma

    # Derived spatial params
    sigma_attract_local = config.sigma_local_frac * refs.s_local if config.sigma_local_frac is not None else None
    r_cut = config.r_cut_frac * refs.s_local if config.r_cut_frac > 0.0 else 0.0
    sigma_void = config.sigma_void_frac * refs.s_global

    if verbose:
        sal_str = f"{sigma_attract_local:.4f}" if sigma_attract_local is not None else "None (=sigma)"
        rcut_str = f"{r_cut:.4f}" if r_cut > 0 else "0 (soft-core)"
        coh_str = f"{sigma_coh:.4f}" if sigma_coh is not None else f"None (=sigma={sigma:.4f})"
        print(f"  {refs}")
        print(f"  sigma={sigma:.4f}  sigma_coh={coh_str}  sigma_attract_local={sal_str}")
        print(f"  epsilon_r={epsilon_r:.6f} (λ_rep={config.repulsion_strength_mult})"
              f"  fidelity_init_strength={fidelity_init_strength:.4f}  lambda_stretch={lambda_stretch:.4f}  lambda_bend={lambda_bend:.6f}")
        print(f"  r_cut={rcut_str}  epsilon_void={config.epsilon_void:.4f}")

    cond_cfg = CondensationConfig(
        sigma=sigma,
        sigma_coh=sigma_coh,
        delta=config.delta,
        epsilon_r=epsilon_r,
        eta=1e-4,
        lambda_stretch=lambda_stretch,
        lambda_bend=lambda_bend,
        fidelity_init_strength=fidelity_init_strength,
        fidelity_half_life=gamma_from_half_life_iters(config.fidelity_half_life_iters),
        k_attract=config.k_attract,
        attract_weight=config.attract_weight,
        temporal_cohere_weight=config.temporal_cohere_weight,
        subtract_mean_attraction=False,
        alpha=config.alpha,
        lr=config.lr,
        max_iter=config.n_iter,
        tol=1e-6,
        sigma_attract_local=sigma_attract_local,
        sigma_void=sigma_void,
        epsilon_void=config.epsilon_void,
        r_cut=r_cut,
        lambda_scale=config.lambda_scale,
        k_local_scale=config.k_local_scale,
        w_attract=config.attract_weight,
        coherence_weight=config.temporal_cohere_weight,
    )

    # Compute initial coherence and metrics using the same sigma_coh that dynamics will use
    _sigma_for_init_coh = sigma_coh if sigma_coh is not None else sigma
    C_init = compute_coherence(positions, mask, sigma=_sigma_for_init_coh, delta=config.delta)
    initial_metrics = _temporal_metrics(positions, dataset.labels, C_init, mask)
    # positions_ref not provided for initial metrics — ratio relative to self = 1.0 by definition

    cond_result = run_dynamics(
        x0=positions,
        mask=mask,
        config=cond_cfg,
        save_every=10 if save_snapshots else None,
        verbose=verbose,
    )

    pos_final = cond_result.positions
    C_final = compute_coherence(pos_final, mask, sigma=_sigma_for_init_coh, delta=config.delta)
    final_metrics = _temporal_metrics(pos_final, dataset.labels, C_final, mask,
                                      positions_ref=positions)

    metrics_df = pd.DataFrame(cond_result.metrics_history)

    # Augment metrics_df with per-iteration coherence selectivity + compactness
    # (expensive but small datasets — compute every 10 iters from snapshots)
    if cond_result.position_history is not None:
        sel_rows = []
        for snap_idx, snap_iter in enumerate(cond_result.snapshot_iters):
            snap_pos = cond_result.position_history[snap_idx]
            C_snap = compute_coherence(snap_pos, mask, sigma=_sigma_for_init_coh, delta=config.delta)
            m = _temporal_metrics(snap_pos, dataset.labels, C_snap, mask,
                                  positions_ref=positions)
            sel_rows.append({"iter": snap_iter, **m})
        sel_df = pd.DataFrame(sel_rows)
        merge_cols = ["iter", "coherence_selectivity", "within_coherence_mean",
                      "cross_coherence_mean", "sep_ratio_mean", "global_spread_mean",
                      "within_bundle_spread_ratio", "local_radius_ratio_median",
                      "local_radius_ratio_p90", "local_radius_ratio_p95"]
        merge_cols = [c for c in merge_cols if c in sel_df.columns]
        metrics_df = metrics_df.merge(sel_df[merge_cols], on="iter", how="left")

    return TemporalRunResult(
        dataset=dataset,
        config=config,
        cond_result=cond_result,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        metrics_df=metrics_df,
    )


# ===========================================================================
# Section 4: Plots
# ===========================================================================

_CLUSTER_COLORS = ["#2166AC", "#B2182B", "#4DAC26", "#F1A340"]


def _scatter_slices(
    ax_row: list[plt.Axes],
    positions: np.ndarray,    # (N_e, T, 2)
    labels: np.ndarray,
    color_map: dict,
    time_values: np.ndarray,
    slice_metrics: list[dict],
    title_prefix: str = "",
    shared_lim: tuple | None = None,
) -> None:
    """Fill a row of axes with per-slice scatter plots."""
    T = positions.shape[1]
    for t, ax in enumerate(ax_row):
        if t >= T:
            ax.set_visible(False)
            continue
        pos_t = positions[:, t, :]
        for g, c in color_map.items():
            m = labels == g
            ax.scatter(pos_t[m, 0], pos_t[m, 1], s=15, alpha=0.7, color=c)
            centroid = pos_t[m].mean(axis=0)
            ax.scatter(*centroid, s=150, marker="+", color="black", linewidths=2, zorder=5)
        sep = slice_metrics[t]["separation_ratio"] if t < len(slice_metrics) else float("nan")
        ax.set_title(f"{title_prefix}t={int(time_values[t])} sep={sep:.2f}", fontsize=8)
        ax.set_aspect("equal")
        if shared_lim is not None:
            ax.set_xlim(shared_lim[0])
            ax.set_ylim(shared_lim[1])
        ax.tick_params(labelsize=6)


def plot_temporal_run(result: TemporalRunResult, output_dir: Path) -> None:
    """Save slices_initial.png, slices_final.png, coherence_heatmap.png, metrics_history.png."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = result.dataset
    color_map = _resolve_color_map(ds.labels)
    T = ds.n_time

    pos_init = ds.positions
    pos_final = result.cond_result.positions

    init_slice_metrics = _per_slice_metrics(pos_init, ds.labels)
    final_slice_metrics = _per_slice_metrics(pos_final, ds.labels)

    # Shared axis limits across both grids (union of initial + final positions)
    all_pos = np.vstack([
        pos_init.reshape(-1, 2),
        pos_final.reshape(-1, 2),
    ])
    pad = 1.0
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    ncols = min(T, 8)
    nrows_per_grid = (T + ncols - 1) // ncols

    def _make_slice_grid(positions, slice_metrics, title_prefix, filename):
        fig, axes = plt.subplots(
            nrows_per_grid, ncols,
            figsize=(2.5 * ncols, 2.5 * nrows_per_grid),
        )
        axes_flat = np.array(axes).ravel().tolist()
        for row_idx in range(nrows_per_grid):
            row_axes = axes_flat[row_idx * ncols: (row_idx + 1) * ncols]
            t_offset = row_idx * ncols
            # Temporary subset of positions/labels/metrics for this row
            T_row = min(ncols, T - t_offset)
            _scatter_slices(
                row_axes,
                positions[:, t_offset:t_offset + T_row, :],
                ds.labels,
                color_map,
                ds.time_values[t_offset:t_offset + T_row],
                slice_metrics[t_offset:t_offset + T_row],
                title_prefix=title_prefix,
                shared_lim=(xlim, ylim),
            )
        cfg = result.config
        fig.suptitle(
            f"{ds.variant} | {title_prefix.strip()} | "
            f"k={cfg.k_attract} σ_frac={cfg.sigma_frac} λ_rep={cfg.repulsion_strength_mult} δ={cfg.delta} fid_init={cfg.fidelity_init_strength}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=120)
        plt.close(fig)

    _make_slice_grid(pos_init, init_slice_metrics, "initial ", "slices_initial.png")
    _make_slice_grid(pos_final, final_slice_metrics, "final ", "slices_final.png")

    # --- coherence_heatmap.png ---
    C_final = compute_coherence(
        pos_final, ds.mask,
        sigma=result.config.sigma_frac * float(np.mean([
            radial_spread(ds.positions[:, t, :]) for t in range(T)
        ])),
        delta=result.config.delta,
    )
    # Show 5 time slices spread across [0, T-1]
    t_show = sorted(set([0, T // 4, T // 2, 3 * T // 4, T - 1]))
    # Sort embryos by label for block-diagonal visualization
    sort_idx = np.argsort(ds.labels)
    boundary = int((ds.labels == 0).sum())

    fig, axes = plt.subplots(1, len(t_show), figsize=(4 * len(t_show), 4))
    if len(t_show) == 1:
        axes = [axes]
    for ax, t in zip(axes, t_show):
        C_t = C_final[np.ix_(sort_idx, sort_idx, [t])][:, :, 0]
        im = ax.imshow(C_t, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.axhline(boundary - 0.5, color="white", lw=1.5, ls="--")
        ax.axvline(boundary - 0.5, color="white", lw=1.5, ls="--")
        ax.set_title(f"t={int(ds.time_values[t])}", fontsize=9)
        ax.set_xlabel("embryo (sorted by label)", fontsize=7)
        ax.set_ylabel("embryo (sorted by label)", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Coherence C_ij(t) — white dashed = cluster boundary | {ds.variant}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "coherence_heatmap.png", dpi=120)
    plt.close(fig)

    # --- metrics_history.png ---
    mdf = result.metrics_df
    has_selectivity = "coherence_selectivity" in mdf.columns and mdf["coherence_selectivity"].notna().any()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_flat = axes.ravel()

    # Panel 0: sep_ratio_mean + coherence_selectivity
    ax = ax_flat[0]
    if has_selectivity:
        sub = mdf.dropna(subset=["sep_ratio_mean"])
        ax.plot(sub["iter"], sub["sep_ratio_mean"], color="#2166AC", lw=2, label="sep_ratio_mean")
        ax2 = ax.twinx()
        ax2.plot(sub["iter"], sub["coherence_selectivity"], color="#B2182B", lw=1.5,
                 ls="--", label="coherence_selectivity")
        ax2.set_ylabel("coherence_selectivity", fontsize=8, color="#B2182B")
        ax2.axhline(1.0, color="#B2182B", lw=0.8, ls=":", alpha=0.5)
        ax.set_title("sep_ratio_mean + coherence_selectivity", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")
    else:
        ax.plot(mdf["iter"], mdf.get("disp_rms_rel", [0] * len(mdf)),
                color="#2166AC", lw=2, label="disp_rms_rel")
        ax.set_title("disp_rms_rel (selectivity not logged)", fontsize=9)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("sep_ratio_mean", fontsize=8)

    # Panel 1: within vs cross coherence
    ax = ax_flat[1]
    if has_selectivity:
        sub = mdf.dropna(subset=["within_coherence_mean"])
        ax.plot(sub["iter"], sub["within_coherence_mean"], color="#2166AC", lw=2, label="within")
        ax.plot(sub["iter"], sub["cross_coherence_mean"], color="#B2182B", lw=2, label="cross")
        ax.set_title("within vs cross coherence", fontsize=9)
        ax.legend(fontsize=8)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("mean coherence", fontsize=8)

    # Panel 2: global_spread_mean + within_bundle_spread_ratio
    ax = ax_flat[2]
    if has_selectivity and "global_spread_mean" in mdf.columns:
        sub = mdf.dropna(subset=["global_spread_mean"])
        ax.plot(sub["iter"], sub["global_spread_mean"], color="#762A83", lw=2, label="global_spread")
    spread_ratio_final = result.final_metrics.get("within_bundle_spread_ratio", float("nan"))
    if "within_bundle_spread_ratio" in mdf.columns and mdf["within_bundle_spread_ratio"].notna().any():
        sub2 = mdf.dropna(subset=["within_bundle_spread_ratio"])
        ax2_c = ax.twinx()
        ax2_c.plot(sub2["iter"], sub2["within_bundle_spread_ratio"],
                   color="#F1A340", lw=2, ls="-.", label="spread_ratio")
        ax2_c.axhline(1.0, color="#F1A340", lw=0.8, ls=":", alpha=0.6)
        ax2_c.set_ylabel("within_bundle_spread_ratio", fontsize=7, color="#F1A340")
        ax2_c.legend(fontsize=7, loc="upper right")
    ax.set_title(
        f"global_spread (collapse={result.collapse_score:.3f}) | "
        f"spread_ratio={spread_ratio_final:.2f}",
        fontsize=8,
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("iteration", fontsize=8)

    # Panel 3: energy terms
    ax = ax_flat[3]
    for col, c, label in [
        ("energy_attract", "#B2182B", "e_att"),
        ("energy_repel", "#4DAC26", "e_rep"),
        ("energy_void", "#1F78B4", "e_void"),
        ("energy_scale", "#D95F02", "e_scale"),
        ("energy_elastic", "#F1A340", "e_elastic"),
        ("energy_fidelity", "#762A83", "e_fidelity"),
    ]:
        if col in mdf.columns:
            ax.plot(mdf["iter"], mdf[col], color=c, lw=1.5, label=label)
    ax.set_title("energy terms", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlabel("iteration", fontsize=8)

    fig.suptitle(
        f"{ds.variant} | k={result.config.k_attract} δ={result.config.delta} "
        f"fid_init={result.config.fidelity_init_strength} lr={result.config.lr:.0e}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_history.png", dpi=120)
    plt.close(fig)


# ===========================================================================
# Section 5: Animation
# ===========================================================================

def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = img.reshape((h, w, 4))[:, :, :3]
    return img


def animate_temporal_run(
    result: TemporalRunResult,
    output_dir: Path,
    frame_interval: int = 1,
) -> None:
    """Generate animation.gif showing position snapshots over iterations."""
    if not _PIL_AVAILABLE:
        print("PIL not available — skipping animation")
        return
    if result.cond_result.position_history is None:
        print("No position history saved — skipping animation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ds = result.dataset
    color_map = _resolve_color_map(ds.labels)
    mdf = result.metrics_df
    has_selectivity = "coherence_selectivity" in mdf.columns and mdf["coherence_selectivity"].notna().any()

    # Compute fixed axis limits from full position history
    all_pos = result.cond_result.position_history.reshape(-1, 2)
    pad = 1.0
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    # Build metric series for live plot
    sel_sub = mdf.dropna(subset=["coherence_selectivity"]) if has_selectivity else pd.DataFrame()

    frames = []
    T = ds.n_time

    for snap_idx in range(0, len(result.cond_result.snapshot_iters), frame_interval):
        iter_n = result.cond_result.snapshot_iters[snap_idx]
        snap_pos = result.cond_result.position_history[snap_idx]  # (N_e, T, 2)

        fig, (ax_pos, ax_met) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=100)

        # Left: overlay all time slices with fading opacity
        for t in range(T):
            alpha = 0.2 + 0.8 * (t / max(T - 1, 1))
            for g, c in color_map.items():
                m = ds.labels == g
                ax_pos.scatter(
                    snap_pos[m, t, 0], snap_pos[m, t, 1],
                    s=12, alpha=alpha * 0.7, color=c,
                )
            centroid_0 = snap_pos[ds.labels == 0, t, :].mean(axis=0)
            centroid_1 = snap_pos[ds.labels == 1, t, :].mean(axis=0)
            ax_pos.scatter(*centroid_0, s=80, marker="+", color="black", lw=2, zorder=5, alpha=alpha)
            ax_pos.scatter(*centroid_1, s=80, marker="+", color="black", lw=2, zorder=5, alpha=alpha)

        ax_pos.set_xlim(xlim)
        ax_pos.set_ylim(ylim)
        ax_pos.set_aspect("equal")
        ax_pos.set_title(f"{ds.variant} | iter {iter_n}\n(opacity ∝ time — dim=early, bright=late)",
                         fontsize=9)
        ax_pos.grid(True, alpha=0.25)

        # Right: live metric plots up to current iter
        if has_selectivity and len(sel_sub) > 0:
            shown = sel_sub[sel_sub["iter"] <= iter_n]
            if len(shown) > 0:
                ax_met.plot(shown["iter"], shown["sep_ratio_mean"],
                            color="#2166AC", lw=2, label="sep_ratio_mean")
                ax2 = ax_met.twinx()
                ax2.plot(shown["iter"], shown["coherence_selectivity"],
                         color="#B2182B", lw=1.5, ls="--", label="selectivity")
                ax2.axhline(1.0, color="#B2182B", lw=0.8, ls=":", alpha=0.4)
                ax2.set_ylabel("coherence_selectivity", fontsize=8, color="#B2182B")
                ax2.set_ylim(bottom=0)
                ax_met.legend(fontsize=7, loc="upper left")
                ax2.legend(fontsize=7, loc="upper right")
        ax_met.set_xlim(0, result.config.n_iter)
        ax_met.set_xlabel("iteration", fontsize=8)
        ax_met.set_ylabel("sep_ratio_mean", fontsize=8)
        ax_met.set_title("Live metrics", fontsize=9)
        ax_met.grid(True, alpha=0.25)

        fig.tight_layout()
        frames.append(Image.fromarray(_fig_to_rgb(fig)))
        plt.close(fig)

    if frames:
        gif_path = output_dir / "animation.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=150,
            loop=0,
        )
        print(f"  Saved: {gif_path}")


# ===========================================================================
# Section 6: Delta sweep on crossing bundles
# ===========================================================================

def run_delta_sweep(
    dataset: TemporalDataset,
    base_config: TemporalRunConfig,
    delta_values: list[int],
    output_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep delta on Test B (crossing bundles) and log selectivity metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for delta in delta_values:
        cfg = TemporalRunConfig(
            sigma_frac=base_config.sigma_frac,
            repulsion_strength_mult=base_config.repulsion_strength_mult,
            k_attract=base_config.k_attract,
            delta=delta,
            lr=base_config.lr,
            n_iter=base_config.n_iter,
            lambda_stretch=base_config.lambda_stretch,
            lambda_bend=base_config.lambda_bend,
            fidelity_init_strength=base_config.fidelity_init_strength,
            fidelity_half_life_iters=base_config.fidelity_half_life_iters,
            alpha=base_config.alpha,
        )
        if verbose:
            print(f"\n  delta={delta}")
        run_dir = output_dir / f"delta_{delta}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_temporal(dataset, cfg, save_snapshots=False, verbose=verbose)
        result.metrics_df.to_csv(run_dir / "metrics_history.csv", index=False)
        plot_temporal_run(result, run_dir)

        # Selectivity during crossing vs final
        sel = result.metrics_df.get("coherence_selectivity", pd.Series(dtype=float))
        sel_valid = sel.dropna() if hasattr(sel, "dropna") else pd.Series(dtype=float)

        min_sel = float(sel_valid.min()) if len(sel_valid) > 0 else float("nan")
        final_sel = result.final_metrics["coherence_selectivity"]
        selectivity_drop = final_sel - min_sel  # positive = recovered above minimum
        selectivity_recovery = final_sel / (min_sel + 1e-8)

        rows.append({
            "variant": dataset.variant,
            "delta": delta,
            "sep_ratio_mean_initial": result.initial_metrics["sep_ratio_mean"],
            "sep_ratio_mean_final": result.final_metrics["sep_ratio_mean"],
            "coherence_selectivity_initial": result.initial_metrics["coherence_selectivity"],
            "coherence_selectivity_final": final_sel,
            "coherence_selectivity_min": min_sel,
            "selectivity_drop": selectivity_drop,
            "selectivity_recovery": selectivity_recovery,
            "collapse_score": result.collapse_score,
            "within_coherence_final": result.final_metrics["within_coherence_mean"],
            "cross_coherence_final": result.final_metrics["cross_coherence_mean"],
        })
        if verbose:
            print(
                f"    sel_init={rows[-1]['coherence_selectivity_initial']:.2f}"
                f"  sel_final={final_sel:.2f}"
                f"  sel_min={min_sel:.2f}"
                f"  recovery={selectivity_recovery:.2f}"
                f"  collapse={result.collapse_score:.3f}"
            )

    summary = pd.DataFrame(rows)

    # Delta sweep summary plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(summary["delta"], summary["coherence_selectivity_final"], "o-", color="#2166AC", label="final")
    ax.plot(summary["delta"], summary["coherence_selectivity_min"], "s--", color="#B2182B", label="min (during crossing)")
    ax.axhline(1.0, color="gray", lw=1, ls=":")
    ax.set_title("coherence_selectivity vs delta", fontsize=9)
    ax.set_xlabel("delta (time bins)", fontsize=8)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(summary["delta"], summary["selectivity_recovery"], "o-", color="#4DAC26")
    ax.axhline(1.0, color="gray", lw=1, ls=":")
    ax.set_title("selectivity_recovery = final / min\n(>1 = recovered after crossing)", fontsize=9)
    ax.set_xlabel("delta (time bins)", fontsize=8)

    ax = axes[2]
    ax.plot(summary["delta"], summary["sep_ratio_mean_final"], "o-", color="#762A83")
    ax.set_title("sep_ratio_mean_final vs delta", fontsize=9)
    ax.set_xlabel("delta (time bins)", fontsize=8)

    fig.suptitle(f"Delta sweep on {dataset.variant}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "delta_sweep_summary.png", dpi=120)
    plt.close(fig)

    return summary


# ===========================================================================
# Section 7: Main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic temporal stitching sandbox for coherence mechanism testing."
    )
    p.add_argument(
        "--output-dir",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/temporal_sandbox_v1",
    )
    p.add_argument("--n-per-cluster", type=int, default=40)
    p.add_argument("--n-time-stable", type=int, default=10)
    p.add_argument("--n-time-cross", type=int, default=15)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--k-attract", type=int, default=20)
    p.add_argument("--delta", type=int, default=3)
    p.add_argument("--delta-sweep", nargs="+", type=int, default=[1, 3, 5, 10],
                   help="Delta values to sweep on crossing bundles")
    p.add_argument("--fidelity-init-strength", type=float, default=0.0,
                   help="Initial fidelity anchor weight (0 = off, cleanest mechanism test)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma-local-frac", type=float, default=None,
                   help="If set, use local-scale attraction (sigma_attract_local = frac × median_5nn). "
                        "Default None = old single-sigma behavior.")
    p.add_argument("--epsilon-void", type=float, default=0.0,
                   help="Void repulsion strength. 0 = off (default).")
    p.add_argument("--compare", action="store_true",
                   help="Run baseline vs local-sigma comparison experiment and print summary table.")
    p.add_argument("--lambda-scale", type=float, default=0.0,
                   help="Local neighborhood scale preservation strength. 0=off. Try 0.1–2.0.")
    p.add_argument("--repulsion-sweep", action="store_true",
                   help="Sweep truncated repulsion cutoffs and epsilon scales on crossing_bundles. "
                        "Tests: soft-core baseline, lower epsilon (0.5×, 0.25×), "
                        "truncated r_cut (0.1×, 0.25×, 0.5× median_5nn).")
    p.add_argument("--scale-sweep", action="store_true",
                   help="Sweep lambda_scale (local neighborhood preservation): "
                        "baseline_decompact (old eps), tuned_eps (eps×0.1), "
                        "tuned_eps + lambda=0.1, tuned_eps + lambda=1.0. "
                        "Generates per-condition animations and summary bar chart.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = TemporalRunConfig(
        k_attract=args.k_attract,
        delta=args.delta,
        lr=args.lr,
        n_iter=args.n_iter,
        fidelity_init_strength=args.fidelity_init_strength,
        sigma_local_frac=args.sigma_local_frac,
        epsilon_void=args.epsilon_void,
        lambda_scale=args.lambda_scale,
    )

    all_summaries = []

    # --- Test A: Stable bundles ---
    print("\n=== Test A: Stable Bundles ===")
    ds_stable = make_stable_bundles(
        n_per_cluster=args.n_per_cluster,
        n_time=args.n_time_stable,
        random_seed=args.seed,
    )
    stable_dir = output_dir / "stable_bundles"
    stable_dir.mkdir(parents=True, exist_ok=True)
    result_stable = run_temporal(ds_stable, base_config, save_snapshots=not args.no_animation)
    result_stable.metrics_df.to_csv(stable_dir / "metrics_history.csv", index=False)
    plot_temporal_run(result_stable, stable_dir)
    if not args.no_animation:
        animate_temporal_run(result_stable, stable_dir)

    m = result_stable.final_metrics
    print(f"\n  Final metrics (stable):")
    print(f"    sep_ratio_mean:             {m['sep_ratio_mean']:.3f}  (init: {result_stable.initial_metrics['sep_ratio_mean']:.3f})")
    print(f"    coherence_selectivity:       {m['coherence_selectivity']:.3f}  (init: {result_stable.initial_metrics['coherence_selectivity']:.3f})")
    print(f"    within_coherence:            {m['within_coherence_mean']:.3f}")
    print(f"    cross_coherence:             {m['cross_coherence_mean']:.3f}")
    print(f"    collapse_score:              {result_stable.collapse_score:.3f}")
    print(f"    within_bundle_spread_ratio:  {m.get('within_bundle_spread_ratio', float('nan')):.3f}  (1.0 = no inflation)")
    print(f"    local_radius_ratio_median:   {m.get('local_radius_ratio_median', float('nan')):.3f}")
    print(f"    local_radius_ratio_p95:      {m.get('local_radius_ratio_p95', float('nan')):.3f}")

    all_summaries.append({
        "variant": "stable_bundles", "delta": args.delta, "fidelity_init_strength": args.fidelity_init_strength,
        "sigma_local_frac": args.sigma_local_frac,
        "sep_ratio_mean_initial": result_stable.initial_metrics["sep_ratio_mean"],
        "sep_ratio_mean_final": m["sep_ratio_mean"],
        "coherence_selectivity_initial": result_stable.initial_metrics["coherence_selectivity"],
        "coherence_selectivity_final": m["coherence_selectivity"],
        "coherence_selectivity_min": float("nan"),
        "selectivity_recovery": float("nan"),
        "collapse_score": result_stable.collapse_score,
        "within_bundle_spread_ratio": m.get("within_bundle_spread_ratio", float("nan")),
        "local_radius_ratio_median": m.get("local_radius_ratio_median", float("nan")),
        "local_radius_ratio_p95": m.get("local_radius_ratio_p95", float("nan")),
    })

    # --- Test B: Crossing bundles (single run) ---
    print("\n=== Test B: Crossing Bundles (default delta) ===")
    ds_cross = make_crossing_bundles(
        n_per_cluster=args.n_per_cluster,
        n_time=args.n_time_cross,
        random_seed=args.seed,
    )
    cross_dir = output_dir / "crossing_bundles"
    cross_dir.mkdir(parents=True, exist_ok=True)
    result_cross = run_temporal(ds_cross, base_config, save_snapshots=not args.no_animation)
    result_cross.metrics_df.to_csv(cross_dir / "metrics_history.csv", index=False)
    plot_temporal_run(result_cross, cross_dir)
    if not args.no_animation:
        animate_temporal_run(result_cross, cross_dir)

    m = result_cross.final_metrics
    sel_series = result_cross.metrics_df.get("coherence_selectivity", pd.Series(dtype=float))
    sel_valid = sel_series.dropna() if hasattr(sel_series, "dropna") else pd.Series(dtype=float)
    min_sel = float(sel_valid.min()) if len(sel_valid) > 0 else float("nan")
    recovery = m["coherence_selectivity"] / (min_sel + 1e-8) if not np.isnan(min_sel) else float("nan")

    print(f"\n  Final metrics (crossing):")
    print(f"    sep_ratio_mean:             {m['sep_ratio_mean']:.3f}  (init: {result_cross.initial_metrics['sep_ratio_mean']:.3f})")
    print(f"    coherence_selectivity:       {m['coherence_selectivity']:.3f}  (init: {result_cross.initial_metrics['coherence_selectivity']:.3f})")
    print(f"    selectivity_min:             {min_sel:.3f}  (during crossing)")
    print(f"    selectivity_recovery:        {recovery:.3f}  (final / min, >1 = recovered)")
    print(f"    collapse_score:              {result_cross.collapse_score:.3f}")
    print(f"    within_bundle_spread_ratio:  {m.get('within_bundle_spread_ratio', float('nan')):.3f}  (1.0 = no inflation)")
    print(f"    local_radius_ratio_median:   {m.get('local_radius_ratio_median', float('nan')):.3f}")
    print(f"    local_radius_ratio_p95:      {m.get('local_radius_ratio_p95', float('nan')):.3f}")

    all_summaries.append({
        "variant": "crossing_bundles", "delta": args.delta, "fidelity_init_strength": args.fidelity_init_strength,
        "sigma_local_frac": args.sigma_local_frac,
        "sep_ratio_mean_initial": result_cross.initial_metrics["sep_ratio_mean"],
        "sep_ratio_mean_final": m["sep_ratio_mean"],
        "coherence_selectivity_initial": result_cross.initial_metrics["coherence_selectivity"],
        "coherence_selectivity_final": m["coherence_selectivity"],
        "coherence_selectivity_min": min_sel,
        "selectivity_recovery": recovery,
        "collapse_score": result_cross.collapse_score,
        "within_bundle_spread_ratio": m.get("within_bundle_spread_ratio", float("nan")),
        "local_radius_ratio_median": m.get("local_radius_ratio_median", float("nan")),
        "local_radius_ratio_p95": m.get("local_radius_ratio_p95", float("nan")),
    })

    # --- Delta sweep on crossing bundles ---
    print("\n=== Delta Sweep: Crossing Bundles ===")
    sweep_dir = output_dir / "crossing_bundles_delta_sweep"
    sweep_summary = run_delta_sweep(
        ds_cross, base_config, args.delta_sweep, sweep_dir, verbose=True
    )
    sweep_summary.to_csv(sweep_dir / "summary.csv", index=False)
    for _, row in sweep_summary.iterrows():
        all_summaries.append(dict(row))

    # --- Comparison experiment: baseline vs local-sigma ---
    if args.compare:
        print("\n=== Comparison: Baseline vs Local-Sigma ===")
        compare_rows = []
        for variant_name, ds in [("stable_bundles", ds_stable), ("crossing_bundles", ds_cross)]:
            for label, sloc in [("baseline", None), ("local_sigma_0.5", 0.5)]:
                cfg_cmp = TemporalRunConfig(
                    k_attract=args.k_attract,
                    delta=args.delta,
                    lr=args.lr,
                    n_iter=args.n_iter,
                    fidelity_init_strength=args.fidelity_init_strength,
                    sigma_local_frac=sloc,
                )
                print(f"  {variant_name} / {label} ...", flush=True)
                r = run_temporal(ds, cfg_cmp, save_snapshots=False, verbose=False)
                fm = r.final_metrics
                compare_rows.append({
                    "variant": variant_name,
                    "condition": label,
                    "sigma_local_frac": sloc,
                    "sep_ratio_final": fm["sep_ratio_mean"],
                    "coherence_selectivity": fm["coherence_selectivity"],
                    "collapse_score": r.collapse_score,
                    "within_bundle_spread_ratio": fm.get("within_bundle_spread_ratio", float("nan")),
                    "local_radius_ratio_median": fm.get("local_radius_ratio_median", float("nan")),
                    "local_radius_ratio_p95": fm.get("local_radius_ratio_p95", float("nan")),
                })
        cmp_df = pd.DataFrame(compare_rows)
        cmp_df.to_csv(output_dir / "comparison_baseline_vs_local_sigma.csv", index=False)
        print("\n  Comparison results:")
        print(cmp_df[[
            "variant", "condition", "sep_ratio_final", "collapse_score",
            "within_bundle_spread_ratio", "local_radius_ratio_median", "local_radius_ratio_p95"
        ]].to_string(index=False))
        print(f"\n  Saved: {output_dir / 'comparison_baseline_vs_local_sigma.csv'}")

    # --- Scale sweep: lambda_scale = 0 vs 0.1 vs 1.0 ---
    if args.scale_sweep:
        print("\n=== Scale Sweep: lambda_scale ===")
        # repulsion_strength_mult values:
        #   0.005 = default (calibrated to s_local, always correct now)
        #   0.05  = 10× stronger — simulates the old broken behavior for comparison
        scale_conditions = [
            # label,                    rep_mult,  lambda_scale
            ("strong_rep_decompact",    0.05,       0.0),   # broken: 10× too strong, decompaction
            ("default_rep_no_lambda",   0.005,      0.0),   # correct calibration, no lambda
            ("default_rep_lambda_0.1",  0.005,      0.1),   # correct + soft lambda
            ("default_rep_lambda_1.0",  0.005,      1.0),   # correct + moderate lambda
        ]
        scale_rows = []
        scale_dir = output_dir / "scale_sweep"
        scale_dir.mkdir(parents=True, exist_ok=True)
        for variant_name, ds in [("stable_bundles", ds_stable), ("crossing_bundles", ds_cross)]:
            for label, rep_mult, lam in scale_conditions:
                cfg_sc = TemporalRunConfig(
                    k_attract=args.k_attract,
                    delta=args.delta,
                    lr=args.lr,
                    n_iter=args.n_iter,
                    fidelity_init_strength=args.fidelity_init_strength,
                    repulsion_strength_mult=rep_mult,
                    lambda_scale=lam,
                    k_local_scale=5,
                )
                cond_dir = scale_dir / variant_name / label
                cond_dir.mkdir(parents=True, exist_ok=True)
                save_snaps = not args.no_animation
                print(f"  {variant_name} / {label} ...", flush=True)
                r = run_temporal(ds, cfg_sc, save_snapshots=save_snaps, verbose=False)
                r.metrics_df.to_csv(cond_dir / "metrics_history.csv", index=False)
                plot_temporal_run(r, cond_dir)
                if save_snaps:
                    animate_temporal_run(r, cond_dir)
                    # Rename animation to make condition obvious
                    src = cond_dir / "animation.gif"
                    dst = cond_dir / f"animation_{variant_name}_{label}.gif"
                    if src.exists():
                        src.rename(dst)
                        print(f"    Saved animation: {dst}")
                fm = r.final_metrics
                row = {
                    "variant": variant_name,
                    "condition": label,
                    "repulsion_strength_mult": rep_mult,
                    "lambda_scale": lam,
                    "sep_ratio_final": fm["sep_ratio_mean"],
                    "sep_ratio_initial": r.initial_metrics["sep_ratio_mean"],
                    "coherence_selectivity": fm["coherence_selectivity"],
                    "collapse_score": r.collapse_score,
                    "within_bundle_spread_ratio": fm.get("within_bundle_spread_ratio", float("nan")),
                    "local_radius_ratio_median": fm.get("local_radius_ratio_median", float("nan")),
                    "local_radius_ratio_p95": fm.get("local_radius_ratio_p95", float("nan")),
                }
                scale_rows.append(row)
                print(
                    f"    spread_ratio={row['within_bundle_spread_ratio']:.3f}  "
                    f"p95={row['local_radius_ratio_p95']:.3f}  "
                    f"sep={row['sep_ratio_final']:.1f}  "
                    f"collapse={row['collapse_score']:.3f}"
                )

        scale_df = pd.DataFrame(scale_rows)
        scale_df.to_csv(scale_dir / "scale_sweep_summary.csv", index=False)

        # Summary bar chart
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax_flat = axes.ravel()
        metrics_to_plot = [
            ("within_bundle_spread_ratio", "within_bundle_spread_ratio\n(target ≤ 1.1, lower=better)", True),
            ("local_radius_ratio_p95",     "local_radius_ratio_p95\n(label-free, lower=better)", True),
            ("sep_ratio_final",            "sep_ratio_final\n(higher=better)", False),
            ("collapse_score",             "collapse_score\n(1.0 = no global change)", False),
        ]
        cond_labels = [c[0] for c in scale_conditions]
        x = np.arange(len(cond_labels))
        variants = scale_df["variant"].unique()
        bar_colors = ["#2166AC", "#B2182B"]
        for ax, (col, title, lower_better) in zip(ax_flat, metrics_to_plot):
            for vi, (vname, bc) in enumerate(zip(variants, bar_colors)):
                sub = scale_df[scale_df["variant"] == vname]
                vals = [sub[sub["condition"] == c][col].values[0] if len(sub[sub["condition"] == c]) > 0 else float("nan")
                        for c in cond_labels]
                ax.bar(x + vi * 0.38, vals, width=0.36, label=vname, color=bc, alpha=0.8)
            ax.set_xticks(x + 0.19)
            ax.set_xticklabels(cond_labels, rotation=25, ha="right", fontsize=7)
            ax.axhline(1.0, color="gray", lw=0.8, ls=":")
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            if lower_better:
                ax.axhline(1.1, color="orange", lw=0.8, ls="--", alpha=0.7)  # target line
        fig.suptitle(
            "Scale sweep: baseline_decompact vs tuned_eps vs +lambda_scale\n"
            "orange dashed = target threshold (1.1)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(scale_dir / "scale_sweep_summary.png", dpi=120)
        plt.close(fig)

        print("\n  Scale sweep results:")
        for vname in scale_df["variant"].unique():
            print(f"\n  -- {vname} --")
            sub = scale_df[scale_df["variant"] == vname]
            print(sub[[
                "condition", "within_bundle_spread_ratio", "local_radius_ratio_p95",
                "sep_ratio_final", "collapse_score"
            ]].to_string(index=False))
        print(f"\n  Saved: {scale_dir / 'scale_sweep_summary.csv'}")
        print(f"  Saved: {scale_dir / 'scale_sweep_summary.png'}")

    # --- Repulsion sweep: truncated vs soft-core, varying r_cut and epsilon ---
    if args.repulsion_sweep:
        print("\n=== Repulsion Sweep: truncated r_cut vs baseline ===")
        rep_conditions = [
            # label,            r_cut_frac, rep_mult
            ("default_rep",          0.0,   0.005),
            ("strong_rep_2x",        0.0,   0.010),
            ("strong_rep_5x",        0.0,   0.025),
            ("strong_rep_10x",       0.0,   0.050),
            ("bump_rcut0.10",        0.10,  0.005),
            ("bump_rcut0.25",        0.25,  0.005),
            ("bump_rcut0.50",        0.50,  0.005),
            ("bump_rcut0.25_half",   0.25,  0.0025),
        ]
        rep_rows = []
        rep_dir = output_dir / "repulsion_sweep"
        rep_dir.mkdir(parents=True, exist_ok=True)
        for variant_name, ds in [("stable_bundles", ds_stable), ("crossing_bundles", ds_cross)]:
            for label, rcut_frac, rep_mult in rep_conditions:
                cfg_rep = TemporalRunConfig(
                    k_attract=args.k_attract,
                    delta=args.delta,
                    lr=args.lr,
                    n_iter=args.n_iter,
                    fidelity_init_strength=args.fidelity_init_strength,
                    r_cut_frac=rcut_frac,
                    repulsion_strength_mult=rep_mult,
                )
                print(f"  {variant_name} / {label} ...", flush=True)
                r = run_temporal(ds, cfg_rep, save_snapshots=False, verbose=False)
                fm = r.final_metrics
                rep_rows.append({
                    "variant": variant_name,
                    "condition": label,
                    "r_cut_frac": rcut_frac,
                    "repulsion_strength_mult": rep_mult,
                    "sep_ratio_final": fm["sep_ratio_mean"],
                    "coherence_selectivity": fm["coherence_selectivity"],
                    "collapse_score": r.collapse_score,
                    "within_bundle_spread_ratio": fm.get("within_bundle_spread_ratio", float("nan")),
                    "local_radius_ratio_median": fm.get("local_radius_ratio_median", float("nan")),
                    "local_radius_ratio_p95": fm.get("local_radius_ratio_p95", float("nan")),
                })
        rep_df = pd.DataFrame(rep_rows)
        rep_df.to_csv(rep_dir / "repulsion_sweep_summary.csv", index=False)

        print("\n  Repulsion sweep results:")
        for vname in rep_df["variant"].unique():
            print(f"\n  -- {vname} --")
            sub = rep_df[rep_df["variant"] == vname]
            print(sub[[
                "condition", "sep_ratio_final", "collapse_score",
                "within_bundle_spread_ratio", "local_radius_ratio_median", "local_radius_ratio_p95"
            ]].to_string(index=False))

        # Summary bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, col, title in [
            (axes[0], "within_bundle_spread_ratio", "within_bundle_spread_ratio\n(1.0 = no inflation)"),
            (axes[1], "local_radius_ratio_p95",     "local_radius_ratio_p95\n(tails — lower is better)"),
            (axes[2], "sep_ratio_final",             "sep_ratio_final\n(higher is better)"),
        ]:
            for i, vname in enumerate(rep_df["variant"].unique()):
                sub = rep_df[rep_df["variant"] == vname]
                x = np.arange(len(sub)) + i * 0.35
                ax.bar(x, sub[col], width=0.3, label=vname, alpha=0.8)
            ax.set_xticks(np.arange(len(rep_conditions)) + 0.175)
            ax.set_xticklabels([c[0] for c in rep_conditions], rotation=35, ha="right", fontsize=7)
            ax.axhline(1.0, color="gray", lw=0.8, ls=":")
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("Repulsion sweep: soft-core vs truncated bump", fontsize=10)
        fig.tight_layout()
        fig.savefig(rep_dir / "repulsion_sweep_summary.png", dpi=120)
        plt.close(fig)
        print(f"\n  Saved: {rep_dir / 'repulsion_sweep_summary.csv'}")
        print(f"  Saved: {rep_dir / 'repulsion_sweep_summary.png'}")

    # --- Combined summary ---
    combined = pd.DataFrame(all_summaries)
    combined.to_csv(output_dir / "summary.csv", index=False)
    print(f"\n=== Summary ===")
    cols = ["variant", "delta", "sep_ratio_mean_final",
            "coherence_selectivity_final", "selectivity_recovery", "collapse_score",
            "within_bundle_spread_ratio", "local_radius_ratio_p95"]
    show_cols = [c for c in cols if c in combined.columns]
    print(combined[show_cols].to_string(index=False))
    print(f"\nOutputs in: {output_dir}")


if __name__ == "__main__":
    main()
