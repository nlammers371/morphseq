"""
void_sandbox.py
---------------
Diagnostic sandbox for the grid-based void term.

Job description of the void term
---------------------------------
Input:  compact bundles crowded unevenly within a bounded domain.
Output: bundle centers redistributed more evenly through that domain.
Constraint: local bundle density unchanged.

The void term acts at bundle-centroid scale (sigma_void >> s_local),
not at point-to-point spacing.

Grid-based occupancy formulation
---------------------------------
Lay down a coarse G_x x G_y grid of cell centers c_g in the fixed domain.
For each cell g, define smooth occupancy:

    m_g = sum_i exp(-||x_i - c_g||^2 / 2 sigma_void^2)

Energy = penalize deviation from uniform occupancy:

    E_void = lambda_void * sum_g (m_g - m_bar)^2
    where m_bar = (1/G) sum_g m_g

Gradient:

    grad_{x_i} E_void = 2 * lambda_void * sum_g (m_g - m_bar) * d(m_g)/d(x_i)
    d(m_g)/d(x_i) = -(x_i - c_g) / sigma_void^2 * exp(-||x_i - c_g||^2 / 2 sigma_void^2)

Why this is the right formulation:
  - sigma_void broad -> sees bundle-scale density, not point spacing
  - If sigma_void >> within-bundle spacing, all points in a bundle see ~same grid
    cell occupancy -> void force is nearly uniform across the bundle -> mostly
    translates bundle centers, not internal structure
  - Quadratic penalty on occupancy variance is convex in m_g, clean gradient
  - No need for pairwise point iteration; O(N * G) per iteration

Confinement is sandbox scaffolding only — a soft elastic penalty for leaving
the fixed [-5,5]x[-5,5] test domain. This makes "spread out" meaningful by
preventing outward drift. It is NOT proposed as a permanent force in the
trajectory model.

Four synthetic tests
--------------------
  crowded_one_side   — 4 bundles packed into left half; should spread across domain
  well_spaced        — 4 bundles already uniform; void should do almost nothing
  mixed_sizes        — 3 bundles N=10/30/60; should spread centers, preserve sizes
  line_crowded       — 4 bundles along a line, too close; should spread, preserve topology

Metrics
-------
  Local  (should stay near 1.0):
    within_bundle_spread_ratio    — final/initial within-cluster std
    local_radius_ratio_p95        — 95th pctile of kNN radius ratio (label-free)
  Global (should improve for crowded cases):
    mean_centroid_dist            — mean pairwise distance between bundle centroids
    centroid_spacing_cv           — coefficient of variation of pairwise centroid dists
                                    (lower = more uniform spacing)
  Stability:
    collapse_score                — final/initial global cloud spread
    domain_escape_frac            — fraction of points outside padded test domain

Run (smoke test):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/void_sandbox.py \\
      --output-dir /tmp/void_sandbox_test --n-iter 100 --no-animation

Run (full sweep):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/void_sandbox.py \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/void_sandbox_v1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from itertools import combinations

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

from trajectory_cosmology.condensation.forces import (
    attraction,
    repulsion,
    estimate_local_spacing_ref,
)

# Fixed sandbox domain
DOMAIN = (-5.0, 5.0)   # same for x and y
DOMAIN_PAD = 0.5       # confinement kicks in outside DOMAIN +/- PAD

_BUNDLE_COLORS = ["#2166AC", "#B2182B", "#4DAC26", "#F1A340",
                  "#762A83", "#D95F02", "#1F78B4", "#E78AC3"]


# ===========================================================================
# Grid-based void term (sandbox-only implementation)
# ===========================================================================

def make_grid_centers(
    domain: tuple[float, float],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Build (G_x * G_y, 2) array of grid cell centers in the fixed domain.

    Parameters
    ----------
    domain : (lo, hi) — same for x and y
    grid_shape : (G_x, G_y)
    """
    lo, hi = domain
    xs = np.linspace(lo, hi, grid_shape[0])
    ys = np.linspace(lo, hi, grid_shape[1])
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)   # (G, 2)


def grid_void_energy_and_grad(
    positions: np.ndarray,         # (N, 2)
    grid_centers: np.ndarray,      # (G, 2) — fixed, computed once
    sigma_void: float,
    lambda_void: float,
) -> tuple[float, np.ndarray]:
    """Grid-based occupancy void term: energy and gradient.

    Smooth occupancy of grid cell g:
        m_g = sum_i exp(-||x_i - c_g||^2 / 2 sigma_void^2)

    Energy:
        E = lambda_void * sum_g (m_g - m_bar)^2    where m_bar = mean_g(m_g)

    Gradient w.r.t. x_i:
        grad_i E = 2 * lambda_void
                   * sum_g (m_g - m_bar) * K_ig * (-(x_i - c_g) / sigma_void^2)

    where K_ig = exp(-||x_i - c_g||^2 / 2 sigma_void^2).

    Note: d(m_bar)/d(x_i) is included via the chain rule on m_bar = mean(m_g),
    but because m_bar is the mean of all m_g, its contribution sums to zero
    when all cells are counted — so we can equivalently center the residual
    and differentiate only through m_g. This is confirmed below:

        d/d(x_i) sum_g (m_g - m_bar)^2
        = 2 sum_g (m_g - m_bar) * d(m_g - m_bar)/d(x_i)
        = 2 sum_g (m_g - m_bar) * [d(m_g)/d(x_i) - (1/G) sum_{g'} d(m_{g'})/d(x_i)]
        = 2 sum_g (m_g - m_bar) * d(m_g)/d(x_i)
          - (2/G) [sum_g (m_g - m_bar)] * sum_{g'} d(m_{g'})/d(x_i)
        = 2 sum_g (m_g - m_bar) * d(m_g)/d(x_i)   [since sum_g (m_g - m_bar) = 0]

    So differentiating only through m_g with the centered residual is exact.

    Parameters
    ----------
    positions : (N, 2)
    grid_centers : (G, 2) — fixed domain grid, from make_grid_centers()
    sigma_void : float — must be >> within-bundle spacing to act at bundle scale
    lambda_void : float — strength; 0 = off

    Returns
    -------
    energy : float
    grad : (N, 2)
    """
    if lambda_void == 0.0:
        return 0.0, np.zeros_like(positions)

    N = positions.shape[0]
    G = grid_centers.shape[0]
    inv_sigma2 = 1.0 / (sigma_void ** 2)

    # diff[i, g] = x_i - c_g    shape (N, G, 2)
    diff = positions[:, None, :] - grid_centers[None, :, :]    # (N, G, 2)
    sq_dist = (diff ** 2).sum(axis=-1)                         # (N, G)

    # K[i, g] = exp(-sq_dist_ig / 2 sigma_void^2)
    K = np.exp(-0.5 * inv_sigma2 * sq_dist)                    # (N, G)

    # m_g = sum_i K[i, g]
    m = K.sum(axis=0)                                          # (G,)
    m_bar = m.mean()
    residual = m - m_bar                                       # (G,)  centered

    # Energy
    energy = float(lambda_void * (residual ** 2).sum())

    # Gradient:
    # grad_i = 2 * lambda_void * sum_g residual_g * d(m_g)/d(x_i)
    # d(m_g)/d(x_i) = -inv_sigma2 * K[i,g] * (x_i - c_g)     shape (2,)
    # => grad_i = -2 * lambda_void * inv_sigma2
    #             * sum_g [ residual_g * K[i,g] * diff[i,g,:] ]
    # = -2 * lambda_void * inv_sigma2 * (K * residual)[i,:] @ diff[i,:,:]
    weighted = K * residual[None, :]                           # (N, G)
    # sum over g: (N, G) x (N, G, 2) -> (N, 2)
    grad = -2.0 * lambda_void * inv_sigma2 * (weighted[:, :, None] * diff).sum(axis=1)

    return energy, grad


# ===========================================================================
# Section 1: Shape adapters (same contract as slice_sandbox.py)
# ===========================================================================

def _pack(pos2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2) -> positions (N, 1, 2), mask (N, 1) all-True."""
    return pos2d[:, None, :].copy(), np.ones((pos2d.shape[0], 1), dtype=bool)


def _unpack(positions_3d: np.ndarray) -> np.ndarray:
    return positions_3d[:, 0, :]


def _uniform_coherence(N: int) -> np.ndarray:
    """(N, N, 1) coherence — all off-diagonal = 1 (no label information)."""
    C = np.ones((N, N), dtype=float)
    np.fill_diagonal(C, 0.0)
    return C[:, :, None]


# ===========================================================================
# Section 2: Confinement (sandbox scaffolding only)
# ===========================================================================

def confinement(
    positions: np.ndarray,   # (N, 1, 2)
    mask: np.ndarray,        # (N, 1) bool
    domain: tuple[float, float] = DOMAIN,
    pad: float = DOMAIN_PAD,
    lambda_conf: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Soft elastic confinement to the padded sandbox domain.

    E = lambda_conf * sum_i sum_d max(0, |x_id| - (domain_half + pad))^2

    This is sandbox scaffolding — it gives 'spread out' a bounded meaning
    by preventing outward drift. It is NOT part of the core trajectory model.
    """
    domain_half = (domain[1] - domain[0]) / 2.0
    limit = domain_half + pad

    pos2d = _unpack(positions)    # (N, 2)
    obs_idx = np.flatnonzero(mask[:, 0])
    pos_obs = pos2d[obs_idx]      # (n_obs, 2)

    excess = np.maximum(0.0, np.abs(pos_obs) - limit)   # (n_obs, 2)
    energy = float(lambda_conf * (excess ** 2).sum())

    grad = np.zeros_like(positions)
    g = 2.0 * lambda_conf * excess * np.sign(pos_obs)   # (n_obs, 2)
    grad[obs_idx, 0, :] = g
    return energy, grad


# ===========================================================================
# Section 3: Synthetic datasets
# ===========================================================================

@dataclass
class VoidDataset:
    positions: np.ndarray    # (N, 2)
    labels: np.ndarray       # (N,) int
    name: str
    domain: tuple[float, float] = field(default_factory=lambda: DOMAIN)


def make_crowded_one_side(
    n_per_bundle: int = 30,
    noise: float = 0.25,
    random_seed: int = 42,
) -> VoidDataset:
    """4 compact bundles, all packed into the left half of the domain."""
    rng = np.random.default_rng(random_seed)
    centers = np.array([
        [-4.0,  2.5],
        [-4.0, -2.5],
        [-2.0,  0.5],
        [-2.0, -0.5],
    ])
    return _make_bundle_dataset(centers, n_per_bundle, noise, rng, "crowded_one_side")


def make_well_spaced(
    n_per_bundle: int = 30,
    noise: float = 0.25,
    random_seed: int = 42,
) -> VoidDataset:
    """4 compact bundles already approximately evenly spaced in the domain."""
    rng = np.random.default_rng(random_seed)
    centers = np.array([
        [-3.0,  3.0],
        [ 3.0,  3.0],
        [-3.0, -3.0],
        [ 3.0, -3.0],
    ])
    return _make_bundle_dataset(centers, n_per_bundle, noise, rng, "well_spaced")


def make_mixed_sizes(
    noise: float = 0.25,
    random_seed: int = 42,
) -> VoidDataset:
    """3 bundles with different sizes (N=10, 30, 60), crowded on one side."""
    rng = np.random.default_rng(random_seed)
    sizes = [10, 30, 60]
    centers = np.array([
        [-3.5,  2.0],
        [-3.5, -2.0],
        [-1.5,  0.0],
    ])
    parts = []
    labels = []
    for g, (n, c) in enumerate(zip(sizes, centers)):
        pts = rng.normal(0, noise, size=(n, 2)) + c
        parts.append(pts)
        labels.extend([g] * n)
    positions = np.vstack(parts)
    return VoidDataset(
        positions=positions,
        labels=np.array(labels, dtype=int),
        name="mixed_sizes",
    )


def make_line_crowded(
    n_per_bundle: int = 30,
    noise: float = 0.25,
    random_seed: int = 42,
) -> VoidDataset:
    """4 compact bundles arranged along a horizontal line, too close together."""
    rng = np.random.default_rng(random_seed)
    centers = np.array([
        [-3.0, 0.0],
        [-1.0, 0.0],
        [ 1.0, 0.0],
        [ 3.0, 0.0],
    ]) * 0.6   # compress: 0.6× makes them too close along x
    return _make_bundle_dataset(centers, n_per_bundle, noise, rng, "line_crowded")


def _make_bundle_dataset(
    centers: np.ndarray,
    n_per_bundle: int,
    noise: float,
    rng: np.random.Generator,
    name: str,
) -> VoidDataset:
    parts = []
    labels = []
    for g, c in enumerate(centers):
        pts = rng.normal(0, noise, size=(n_per_bundle, 2)) + c
        parts.append(pts)
        labels.extend([g] * n_per_bundle)
    return VoidDataset(
        positions=np.vstack(parts),
        labels=np.array(labels, dtype=int),
        name=name,
    )


ALL_DATASETS = [
    make_crowded_one_side,
    make_well_spaced,
    make_mixed_sizes,
    make_line_crowded,
]


# ===========================================================================
# Section 4: Metrics
# ===========================================================================

def _bundle_centroids(positions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """(n_bundles, 2) — mean position of each bundle."""
    groups = sorted(set(labels.tolist()))
    return np.array([positions[labels == g].mean(axis=0) for g in groups])


def _within_bundle_spread(positions: np.ndarray, labels: np.ndarray) -> float:
    groups = sorted(set(labels.tolist()))
    stds = [positions[labels == g].std() for g in groups]
    return float(np.mean(stds))


def _local_radius_p95(positions_curr: np.ndarray, positions_ref: np.ndarray, k: int = 5) -> float:
    """95th percentile of (current kNN radius / initial kNN radius) per point."""
    n = positions_curr.shape[0]
    k_eff = min(k, n - 1)
    ratios = []
    for pts, pts_r in [(positions_curr, positions_ref)]:
        for pos, ref in [(positions_curr, positions_ref)]:
            diff_c = pos[:, None, :] - pos[None, :, :]
            sq_c = (diff_c ** 2).sum(axis=-1); np.fill_diagonal(sq_c, np.inf)
            r_curr = np.sqrt(np.partition(sq_c, k_eff - 1, axis=1)[:, :k_eff]).mean(axis=1)

            diff_r = ref[:, None, :] - ref[None, :, :]
            sq_r = (diff_r ** 2).sum(axis=-1); np.fill_diagonal(sq_r, np.inf)
            r_ref = np.sqrt(np.partition(sq_r, k_eff - 1, axis=1)[:, :k_eff]).mean(axis=1)

            mask_valid = r_ref > 1e-12
            ratios = (r_curr[mask_valid] / r_ref[mask_valid]).tolist()
            break
        break
    if not ratios:
        return float("nan")
    return float(np.percentile(ratios, 95))


def _centroid_metrics(positions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Pairwise bundle-centroid distances."""
    centroids = _bundle_centroids(positions, labels)
    if len(centroids) < 2:
        return {"mean_centroid_dist": float("nan"), "centroid_spacing_cv": float("nan")}
    dists = [np.linalg.norm(centroids[i] - centroids[j])
             for i, j in combinations(range(len(centroids)), 2)]
    arr = np.array(dists)
    return {
        "mean_centroid_dist": float(arr.mean()),
        "centroid_spacing_cv": float(arr.std() / (arr.mean() + 1e-12)),
    }


def _domain_escape_frac(positions: np.ndarray, domain: tuple[float, float], pad: float) -> float:
    limit = (domain[1] - domain[0]) / 2.0 + pad
    outside = (np.abs(positions) > limit).any(axis=1)
    return float(outside.mean())


def compute_metrics(
    positions: np.ndarray,
    positions_ref: np.ndarray,
    labels: np.ndarray,
    domain: tuple[float, float] = DOMAIN,
    domain_pad: float = DOMAIN_PAD,
) -> dict:
    spread_curr = _within_bundle_spread(positions, labels)
    spread_ref = _within_bundle_spread(positions_ref, labels)
    m = {
        "within_bundle_spread_ratio": spread_curr / (spread_ref + 1e-12),
        "local_radius_ratio_p95": _local_radius_p95(positions, positions_ref),
        "collapse_score": (
            positions.std() / (positions_ref.std() + 1e-12)
        ),
        "domain_escape_frac": _domain_escape_frac(positions, domain, domain_pad),
    }
    m.update(_centroid_metrics(positions, labels))
    return m


# ===========================================================================
# Section 5: Run loop
# ===========================================================================

@dataclass
class VoidRunConfig:
    # Repulsion (local-spacing calibrated)
    repulsion_strength_mult: float = 0.005
    k_attract: int = 20
    sigma_frac: float = 0.5          # sigma = sigma_frac * scale_ref (global scale)
    # Grid-based void term
    lambda_void: float = 0.0         # 0 = off; start with 0.01–0.1
    sigma_void_frac: float = 3.0     # sigma_void = sigma_void_frac * scale_ref
                                     # must be >> s_local to act at bundle scale
    void_grid_shape: tuple = (16, 16)  # coarse grid resolution
    # Confinement (sandbox scaffolding only)
    lambda_conf: float = 0.5
    use_confinement: bool = True
    # Optimizer
    lr: float = 5e-4
    alpha: float = 0.9               # momentum
    n_iter: int = 300
    save_every: int = 10


@dataclass
class VoidRunResult:
    dataset: VoidDataset
    config: VoidRunConfig
    positions_history: np.ndarray    # (n_snapshots, N, 2)
    snapshot_iters: list[int]
    initial_metrics: dict
    final_metrics: dict
    metrics_history: list[dict]


def run_void(
    dataset: VoidDataset,
    config: VoidRunConfig,
    verbose: bool = True,
) -> VoidRunResult:
    """Run the void-term force loop on a 2D bundle dataset."""
    pos2d = dataset.positions.copy()
    N = pos2d.shape[0]
    labels = dataset.labels

    # Derive scales from initial geometry
    scale_ref = float(pos2d.std())
    sigma = config.sigma_frac * scale_ref
    positions_3d, mask = _pack(pos2d)
    s_local = estimate_local_spacing_ref(positions_3d, mask, k=5)
    epsilon_r = config.repulsion_strength_mult * s_local ** 2
    sigma_void = config.sigma_void_frac * scale_ref

    # Build fixed grid once from domain — never updated during optimization
    grid_centers = make_grid_centers(dataset.domain, config.void_grid_shape)

    if verbose:
        print(f"  [{dataset.name}] scale_ref={scale_ref:.3f}  s_local={s_local:.4f}  "
              f"sigma={sigma:.4f}  sigma_void={sigma_void:.4f}  "
              f"epsilon_r={epsilon_r:.6f}  lambda_void={config.lambda_void:.4f}  "
              f"grid={config.void_grid_shape}")

    # Initial coherence: uniform (no label info — void term should work label-free)
    coherence = _uniform_coherence(N)

    # Initial metrics
    initial_metrics = compute_metrics(pos2d, pos2d, labels, dataset.domain)

    positions_3d, mask = _pack(pos2d)
    velocities = np.zeros_like(positions_3d)
    history = []
    snapshot_iters = []
    metrics_history = []

    for n in range(config.n_iter):
        _, g_att = attraction(positions_3d, mask, coherence, sigma,
                              k_attract=config.k_attract)
        _, g_rep = repulsion(positions_3d, mask, epsilon_r, eta=1e-4)
        # Grid-based void term (sandbox-only). Operates on 2D positions directly.
        _, g_void_2d = grid_void_energy_and_grad(
            _unpack(positions_3d), grid_centers, sigma_void, config.lambda_void
        )
        g_void = g_void_2d[:, None, :]   # (N, 2) -> (N, 1, 2) to match gradient shape
        if config.use_confinement:
            _, g_conf = confinement(positions_3d, mask,
                                    domain=dataset.domain,
                                    pad=DOMAIN_PAD,
                                    lambda_conf=config.lambda_conf)
        else:
            g_conf = np.zeros_like(positions_3d)

        grad = g_att + g_rep + g_void + g_conf
        velocities = config.alpha * velocities - config.lr * grad
        positions_3d = positions_3d + velocities

        if n % config.save_every == 0:
            snap = _unpack(positions_3d).copy()
            history.append(snap)
            snapshot_iters.append(n)
            m = compute_metrics(snap, pos2d, labels, dataset.domain)
            e_void_snap, _ = grid_void_energy_and_grad(
                snap, grid_centers, sigma_void, config.lambda_void
            )
            m["energy_void"] = e_void_snap
            m["iter"] = n
            metrics_history.append(m)

    pos_final = _unpack(positions_3d)
    final_metrics = compute_metrics(pos_final, pos2d, labels, dataset.domain)

    if verbose:
        print(
            f"    spread_ratio={final_metrics['within_bundle_spread_ratio']:.3f}  "
            f"p95={final_metrics['local_radius_ratio_p95']:.3f}  "
            f"mean_centroid_dist: {initial_metrics['mean_centroid_dist']:.3f} -> "
            f"{final_metrics['mean_centroid_dist']:.3f}  "
            f"centroid_cv: {initial_metrics['centroid_spacing_cv']:.3f} -> "
            f"{final_metrics['centroid_spacing_cv']:.3f}  "
            f"escape={final_metrics['domain_escape_frac']:.3f}"
        )

    return VoidRunResult(
        dataset=dataset,
        config=config,
        positions_history=np.stack(history) if history else np.empty((0, N, 2)),
        snapshot_iters=snapshot_iters,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        metrics_history=metrics_history,
    )


# ===========================================================================
# Section 6: Plots
# ===========================================================================

def _color_map(labels: np.ndarray) -> dict:
    groups = sorted(set(labels.tolist()))
    return {g: _BUNDLE_COLORS[i % len(_BUNDLE_COLORS)] for i, g in enumerate(groups)}


def plot_before_after(result: VoidRunResult, output_dir: Path) -> None:
    """Side-by-side scatter: initial vs final, domain box overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pos_init = result.dataset.positions
    pos_final = result.positions_history[-1] if len(result.positions_history) else pos_init
    labels = result.dataset.labels
    cmap = _color_map(labels)
    d0, d1 = result.dataset.domain

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pos, title in [
        (axes[0], pos_init, "initial"),
        (axes[1], pos_final, "final"),
    ]:
        for g, c in cmap.items():
            m = labels == g
            ax.scatter(pos[m, 0], pos[m, 1], s=18, alpha=0.7, color=c, label=f"bundle {g}")
            centroid = pos[m].mean(axis=0)
            ax.scatter(*centroid, s=120, marker="+", color="black", lw=2, zorder=5)
        # Draw domain box
        rect = plt.Rectangle((d0, d0), d1 - d0, d1 - d0,
                              fill=False, edgecolor="gray", lw=1.5, ls="--", zorder=0)
        ax.add_patch(rect)
        ax.set_xlim(d0 - 1, d1 + 1)
        ax.set_ylim(d0 - 1, d1 + 1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    cfg = result.config
    fm = result.final_metrics
    im = result.initial_metrics
    fig.suptitle(
        f"{result.dataset.name} | "
        f"λ_void={cfg.lambda_void} σ_void_frac={cfg.sigma_void_frac} "
        f"λ_conf={cfg.lambda_conf}\n"
        f"spread_ratio={fm['within_bundle_spread_ratio']:.3f}  "
        f"p95={fm['local_radius_ratio_p95']:.3f}  "
        f"centroid_dist: {im['mean_centroid_dist']:.2f}→{fm['mean_centroid_dist']:.2f}  "
        f"centroid_cv: {im['centroid_spacing_cv']:.3f}→{fm['centroid_spacing_cv']:.3f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "before_after.png", dpi=130)
    plt.close(fig)


def plot_metrics_history(result: VoidRunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not result.metrics_history:
        return
    df = pd.DataFrame(result.metrics_history)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax_flat = axes.ravel()

    for ax, col, ylabel, good_dir in [
        (ax_flat[0], "mean_centroid_dist",         "mean centroid dist",       "↑ good for crowded"),
        (ax_flat[1], "centroid_spacing_cv",         "centroid spacing CV",      "↓ good (more uniform)"),
        (ax_flat[2], "within_bundle_spread_ratio",  "spread ratio (local)",     "≈1.0 good"),
        (ax_flat[3], "local_radius_ratio_p95",      "local radius ratio p95",   "≈1.0 good"),
    ]:
        if col in df.columns:
            ax.plot(df["iter"], df[col], lw=2, color="#2166AC")
            ax.axhline(1.0, color="gray", lw=0.8, ls=":")
            ax.set_title(f"{col}\n({good_dir})", fontsize=8)
            ax.set_xlabel("iter", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.2)

    cfg = result.config
    fig.suptitle(
        f"{result.dataset.name} | λ_void={cfg.lambda_void} σ_void_frac={cfg.sigma_void_frac}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_history.png", dpi=120)
    plt.close(fig)


def animate_void(result: VoidRunResult, output_dir: Path, frame_interval: int = 2) -> None:
    """GIF showing position evolution over iterations."""
    if not _PIL_AVAILABLE or len(result.positions_history) == 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = result.dataset.labels
    cmap = _color_map(labels)
    d0, d1 = result.dataset.domain
    pos_init = result.dataset.positions
    all_pos = result.positions_history.reshape(-1, 2)
    xlim = (min(d0 - 0.5, all_pos[:, 0].min() - 0.5), max(d1 + 0.5, all_pos[:, 0].max() + 0.5))
    ylim = (min(d0 - 0.5, all_pos[:, 1].min() - 0.5), max(d1 + 0.5, all_pos[:, 1].max() + 0.5))

    # Metric series for live panel
    df = pd.DataFrame(result.metrics_history) if result.metrics_history else pd.DataFrame()

    frames = []
    indices = range(0, len(result.snapshot_iters), frame_interval)

    for snap_idx in indices:
        iter_n = result.snapshot_iters[snap_idx]
        pos = result.positions_history[snap_idx]

        fig, (ax_pos, ax_met) = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

        # Left: scatter with domain box and centroids
        for g, c in cmap.items():
            m = labels == g
            ax_pos.scatter(pos[m, 0], pos[m, 1], s=15, alpha=0.7, color=c)
            ax_pos.scatter(*pos[m].mean(axis=0), s=100, marker="+", color="black", lw=2, zorder=5)
        rect = plt.Rectangle((d0, d0), d1 - d0, d1 - d0,
                              fill=False, edgecolor="gray", lw=1.5, ls="--")
        ax_pos.add_patch(rect)
        ax_pos.set_xlim(xlim)
        ax_pos.set_ylim(ylim)
        ax_pos.set_aspect("equal")
        ax_pos.set_title(
            f"{result.dataset.name} | iter {iter_n}\n"
            f"λ_void={result.config.lambda_void}  σ_void_frac={result.config.sigma_void_frac}",
            fontsize=9,
        )
        ax_pos.grid(True, alpha=0.2)

        # Right: live metrics
        if len(df) > 0:
            shown = df[df["iter"] <= iter_n]
            if len(shown) > 0:
                ax_met.plot(shown["iter"], shown["mean_centroid_dist"],
                            color="#2166AC", lw=2, label="centroid_dist ↑")
                ax2 = ax_met.twinx()
                ax2.plot(shown["iter"], shown["within_bundle_spread_ratio"],
                         color="#B2182B", lw=1.5, ls="--", label="spread_ratio ≈1")
                ax2.axhline(1.0, color="#B2182B", lw=0.8, ls=":", alpha=0.5)
                ax2.set_ylabel("spread_ratio", fontsize=8, color="#B2182B")
                ax2.set_ylim(0.5, 2.0)
                ax_met.legend(fontsize=7, loc="upper left")
                ax2.legend(fontsize=7, loc="upper right")
        ax_met.set_xlim(0, result.config.n_iter)
        ax_met.set_xlabel("iter", fontsize=8)
        ax_met.set_ylabel("mean centroid dist", fontsize=8)
        ax_met.set_title("Live metrics", fontsize=9)
        ax_met.grid(True, alpha=0.2)

        fig.tight_layout()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        frames.append(Image.fromarray(img))
        plt.close(fig)

    if frames:
        gif_path = output_dir / f"animation_{result.dataset.name}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=120, loop=0)
        print(f"  Saved: {gif_path}")


# ===========================================================================
# Section 7: Sweep
# ===========================================================================

def run_void_sweep(
    datasets: list[VoidDataset],
    lambda_void_values: list[float],
    sigma_void_fracs: list[float],
    base_config: VoidRunConfig,
    output_dir: Path,
    no_animation: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep lambda_void and sigma_void_frac across all datasets."""
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        for lv in lambda_void_values:
            for sv_frac in sigma_void_fracs:
                cfg = VoidRunConfig(
                    repulsion_strength_mult=base_config.repulsion_strength_mult,
                    k_attract=base_config.k_attract,
                    sigma_frac=base_config.sigma_frac,
                    lambda_void=lv,
                    sigma_void_frac=sv_frac,
                    void_grid_shape=base_config.void_grid_shape,
                    lambda_conf=base_config.lambda_conf,
                    use_confinement=base_config.use_confinement,
                    lr=base_config.lr,
                    alpha=base_config.alpha,
                    n_iter=base_config.n_iter,
                    save_every=base_config.save_every,
                )
                cond_dir = output_dir / ds.name / f"lv{lv}_sv{sv_frac}"
                cond_dir.mkdir(parents=True, exist_ok=True)

                result = run_void(ds, cfg, verbose=verbose)
                plot_before_after(result, cond_dir)
                plot_metrics_history(result, cond_dir)
                if not no_animation:
                    animate_void(result, cond_dir)

                fm = result.final_metrics
                im = result.initial_metrics
                rows.append({
                    "dataset": ds.name,
                    "lambda_void": lv,
                    "sigma_void_frac": sv_frac,
                    "within_bundle_spread_ratio": fm["within_bundle_spread_ratio"],
                    "local_radius_ratio_p95": fm["local_radius_ratio_p95"],
                    "mean_centroid_dist_initial": im["mean_centroid_dist"],
                    "mean_centroid_dist_final": fm["mean_centroid_dist"],
                    "centroid_dist_improvement": (
                        fm["mean_centroid_dist"] / (im["mean_centroid_dist"] + 1e-12)
                    ),
                    "centroid_spacing_cv_initial": im["centroid_spacing_cv"],
                    "centroid_spacing_cv_final": fm["centroid_spacing_cv"],
                    "collapse_score": fm["collapse_score"],
                    "domain_escape_frac": fm["domain_escape_frac"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "void_sweep_summary.csv", index=False)
    _plot_sweep_summary(df, output_dir)
    return df


def _plot_sweep_summary(df: pd.DataFrame, output_dir: Path) -> None:
    datasets = df["dataset"].unique()
    n_ds = len(datasets)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax_flat = axes.ravel()

    metrics = [
        ("centroid_dist_improvement",    "centroid dist improvement\n(>1 = spread more, ↑ good for crowded)"),
        ("centroid_spacing_cv_final",    "centroid spacing CV final\n(↓ = more uniform)"),
        ("within_bundle_spread_ratio",   "within_bundle_spread_ratio\n(target ≈1.0)"),
        ("local_radius_ratio_p95",       "local_radius_ratio_p95\n(target ≈1.0)"),
        ("collapse_score",               "collapse_score\n(target ≈1.0)"),
        ("domain_escape_frac",           "domain_escape_frac\n(target ≈0.0)"),
    ]

    colors = ["#2166AC", "#B2182B", "#4DAC26", "#F1A340"]
    eps_vals = sorted(df["lambda_void"].unique())

    for ax, (col, title) in zip(ax_flat, metrics):
        for di, ds_name in enumerate(datasets):
            sub = df[df["dataset"] == ds_name]
            # Plot vs lambda_void for first sigma_void_frac
            sv0 = sub["sigma_void_frac"].iloc[0]
            sub0 = sub[sub["sigma_void_frac"] == sv0].sort_values("lambda_void")
            ax.plot(sub0["lambda_void"], sub0[col],
                    "o-", color=colors[di % len(colors)], label=ds_name, lw=1.5)
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("lambda_void", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Grid void term sweep: lambda_void vs metrics (sigma_void_frac fixed at first value)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "void_sweep_summary.png", dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'void_sweep_summary.png'}")


# ===========================================================================
# Section 8: Main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Void term diagnostic sandbox.")
    p.add_argument("--output-dir",
                   default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/void_sandbox_v1")
    p.add_argument("--n-per-bundle", type=int, default=30)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--lambda-void-values", nargs="+", type=float,
                   default=[0.0, 0.01, 0.05, 0.1],
                   help="Grid void term strengths to sweep (lambda_void).")
    p.add_argument("--void-grid-shape", nargs=2, type=int, default=[16, 16],
                   help="Coarse grid resolution for void term (G_x G_y).")
    p.add_argument("--sigma-void-fracs", nargs="+", type=float,
                   default=[2.0, 3.0, 5.0],
                   help="sigma_void = frac × scale_ref.")
    p.add_argument("--lambda-conf", type=float, default=0.5,
                   help="Confinement strength (sandbox scaffolding).")
    p.add_argument("--no-confinement", action="store_true",
                   help="Disable confinement (beware: outward drift).")
    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--datasets", nargs="+",
                   choices=["crowded_one_side", "well_spaced", "mixed_sizes", "line_crowded"],
                   default=["crowded_one_side", "well_spaced", "mixed_sizes", "line_crowded"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_map = {
        "crowded_one_side": lambda: make_crowded_one_side(args.n_per_bundle, random_seed=args.seed),
        "well_spaced":      lambda: make_well_spaced(args.n_per_bundle, random_seed=args.seed),
        "mixed_sizes":      lambda: make_mixed_sizes(random_seed=args.seed),
        "line_crowded":     lambda: make_line_crowded(args.n_per_bundle, random_seed=args.seed),
    }
    datasets = [ds_map[name]() for name in args.datasets]

    base_config = VoidRunConfig(
        n_iter=args.n_iter,
        lambda_conf=args.lambda_conf,
        use_confinement=not args.no_confinement,
        void_grid_shape=tuple(args.void_grid_shape),
    )

    print(f"\n=== Void Sandbox Sweep ===")
    print(f"Datasets: {[ds.name for ds in datasets]}")
    print(f"lambda_void: {args.lambda_void_values}")
    print(f"sigma_void_frac: {args.sigma_void_fracs}")
    print(f"grid_shape: {args.void_grid_shape}")
    print(f"n_iter={args.n_iter}  lambda_conf={args.lambda_conf}  "
          f"confinement={'on' if not args.no_confinement else 'OFF'}\n")

    df = run_void_sweep(
        datasets=datasets,
        lambda_void_values=args.lambda_void_values,
        sigma_void_fracs=args.sigma_void_fracs,
        base_config=base_config,
        output_dir=output_dir,
        no_animation=args.no_animation,
        verbose=True,
    )

    print("\n=== Summary ===")
    for ds_name in df["dataset"].unique():
        print(f"\n-- {ds_name} --")
        sub = df[df["dataset"] == ds_name]
        print(sub[[
            "lambda_void", "sigma_void_frac",
            "centroid_dist_improvement", "centroid_spacing_cv_final",
            "within_bundle_spread_ratio", "local_radius_ratio_p95",
            "domain_escape_frac",
        ]].to_string(index=False))

    print(f"\nOutputs in: {output_dir}")


if __name__ == "__main__":
    main()
