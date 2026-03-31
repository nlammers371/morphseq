"""
slice_sandbox.py
----------------
Synthetic 2D slice sandbox for tuning the attraction/repulsion force law.

Tests the kNN-local force model on simple 2-cluster datasets before applying
it to real trajectory data. This answers the core question:

  Can we define an attraction/repulsion update rule on a single ordered 2D
  slice that tightens local neighborhoods without collapsing everything?

Run (smoke test):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/slice_sandbox.py \\
      --output-dir /tmp/slice_sandbox_test \\
      --variants separated \\
      --n-per-cluster 30 --n-iter 50 \\
      --k-values 5 10 --sigma-fracs 0.5 1.0 --eps-mults 0.6 \\
      --no-per-run-plots

Run (full sweep):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/slice_sandbox.py \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/slice_sandbox_v1
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — import force functions from the condensation module
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from trajectory_cosmology.condensation.forces import attraction, repulsion


# ===========================================================================
# Section 1: Shape adapters
# ===========================================================================

def _pack(pos2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2) -> positions (N, 1, 2), mask (N, 1) bool all-True."""
    N = pos2d.shape[0]
    return pos2d[:, None, :].copy(), np.ones((N, 1), dtype=bool)


def _unpack(positions_3d: np.ndarray) -> np.ndarray:
    """(N, 1, 2) -> (N, 2)."""
    return positions_3d[:, 0, :]


def _make_coherence(labels: np.ndarray, mode: str) -> np.ndarray:
    """Build (N, N, 1) static coherence tensor.

    'uniform' : all off-diagonal entries = 1  (no label knowledge)
    'oracle'  : 1 for same-label pairs, 0 for cross-label pairs
    """
    N = len(labels)
    if mode == "oracle":
        C2d = (labels[:, None] == labels[None, :]).astype(float)
    else:  # uniform
        C2d = np.ones((N, N), dtype=float)
    np.fill_diagonal(C2d, 0.0)
    return C2d[:, :, None]  # (N, N, 1)


# ===========================================================================
# Section 2: Synthetic data generator
# ===========================================================================

@dataclass
class SyntheticDataset:
    pos: np.ndarray        # (N, 2) float
    labels: np.ndarray     # (N,) int  (0 or 1)
    variant: str
    n_per_cluster: int
    true_separation: float  # ground-truth centroid distance


def make_two_cluster_dataset(
    variant: str,
    n_per_cluster: int = 60,
    random_seed: int = 42,
) -> SyntheticDataset:
    """Generate a 2-cluster 2D dataset.

    Variants
    --------
    separated   : well-separated Gaussians
    moderate    : moderate separation
    overlapping : weak separation / heavy overlap
    elongated   : one axis-aligned cluster, one orthogonal elongated cluster
    """
    rng = np.random.default_rng(random_seed)

    configs: dict[str, tuple] = {
        "separated":  ((-3, 0), (3, 0),   np.eye(2),           np.eye(2)),
        "moderate":   ((-1.5, 0), (1.5, 0), np.eye(2),         np.eye(2)),
        "overlapping":((-0.5, 0), (0.5, 0), np.eye(2),         np.eye(2)),
        "elongated":  ((-3, 0), (3, 0),
                       np.diag([3.0, 0.3]),
                       np.diag([0.3, 3.0])),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(configs)}")

    cA, cB, covA, covB = configs[variant]
    pts_A = rng.multivariate_normal(cA, covA, size=n_per_cluster)
    pts_B = rng.multivariate_normal(cB, covB, size=n_per_cluster)

    pos = np.vstack([pts_A, pts_B]).astype(float)
    labels = np.array([0] * n_per_cluster + [1] * n_per_cluster, dtype=int)
    true_sep = float(np.linalg.norm(np.array(cA) - np.array(cB)))

    return SyntheticDataset(
        pos=pos,
        labels=labels,
        variant=variant,
        n_per_cluster=n_per_cluster,
        true_separation=true_sep,
    )


# ===========================================================================
# Section 3: Metrics helpers
# ===========================================================================

def radial_spread(pos: np.ndarray) -> float:
    """RMS distance from centroid — reference spatial scale."""
    center = pos.mean(axis=0)
    return float(np.sqrt(((pos - center) ** 2).sum(axis=1).mean()))


def _cluster_metrics(pos: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute cluster structure metrics for a (N, 2) point cloud."""
    groups = sorted(set(labels.tolist()))
    centroids = np.array([pos[labels == g].mean(axis=0) for g in groups])

    # centroid distance (mean pairwise for >2 clusters; just the one for 2)
    cdiffs = centroids[:, None, :] - centroids[None, :, :]
    cdists = np.sqrt((cdiffs ** 2).sum(axis=-1))
    iu = np.triu_indices(len(groups), k=1)
    centroid_distance = float(cdists[iu].mean()) if len(iu[0]) > 0 else 0.0

    # within-cluster RMS radius
    within_rms_list = []
    for g, c in zip(groups, centroids):
        pts = pos[labels == g]
        rms = float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean()))
        within_rms_list.append(rms)
    within_cluster_rms = float(np.mean(within_rms_list))

    separation_ratio = centroid_distance / (within_cluster_rms + 1e-8)

    # global spread (collapse detector)
    global_spread = radial_spread(pos)

    return {
        "centroid_distance": centroid_distance,
        "within_cluster_rms": within_cluster_rms,
        "separation_ratio": separation_ratio,
        "global_spread": global_spread,
    }


# ===========================================================================
# Section 4: Single-run dataclass and runner
# ===========================================================================

@dataclass
class SliceRunConfig:
    sigma: float
    epsilon_r: float
    k_attract: int | None    # None = all-pairs; 0 = repulsion-only baseline
    subtract_mean: bool
    coherence_mode: str      # 'uniform' | 'oracle'
    n_iter: int = 300
    lr: float = 5e-4
    alpha: float = 0.9       # momentum coefficient
    eta: float = 1e-4        # repulsion soft-core stabilizer

    @property
    def repulsion_only(self) -> bool:
        """k_attract == 0 is treated as the repulsion-only baseline."""
        return self.k_attract == 0


@dataclass
class SliceRunResult:
    pos_initial: np.ndarray       # (N, 2) copy
    pos_final: np.ndarray         # (N, 2)
    metrics_history: pd.DataFrame # one row per iteration
    config: SliceRunConfig
    labels: np.ndarray            # (N,)

    @property
    def sep_ratio_initial(self) -> float:
        row0 = self.metrics_history.iloc[0]
        return float(row0["separation_ratio"])

    @property
    def sep_ratio_final(self) -> float:
        return float(self.metrics_history["separation_ratio"].iloc[-1])

    @property
    def sep_ratio_best(self) -> float:
        return float(self.metrics_history["separation_ratio"].max())

    @property
    def iter_best_sep_ratio(self) -> int:
        return int(self.metrics_history["separation_ratio"].idxmax())

    @property
    def collapse_score(self) -> float:
        gs_init = self.metrics_history["global_spread"].iloc[0]
        gs_final = self.metrics_history["global_spread"].iloc[-1]
        return float(gs_final / (gs_init + 1e-12))


def run_slice(
    pos0: np.ndarray,
    labels: np.ndarray,
    config: SliceRunConfig,
) -> SliceRunResult:
    """Run the 2D force sandbox for a single parameter configuration.

    Uses attraction + repulsion from forces.py (no elasticity, fidelity, or
    temporal coherence — this is the single-slice tuning sandbox).

    k_attract == 0 is a special repulsion-only baseline (no attraction).
    """
    positions_3d, mask = _pack(pos0)
    coherence = _make_coherence(labels, config.coherence_mode)
    velocities = np.zeros_like(positions_3d)

    rows: list[dict] = []
    prev_pos_snapshot: np.ndarray | None = None
    snapshot_iter: int = -1

    for i in range(config.n_iter):
        # --- Forces ---
        if config.repulsion_only:
            e_att = 0.0
            g_att = np.zeros_like(positions_3d)
        else:
            e_att, g_att = attraction(
                positions_3d, mask, coherence,
                sigma=config.sigma,
                k_attract=config.k_attract,
                subtract_mean=config.subtract_mean,
            )

        e_rep, g_rep = repulsion(
            positions_3d, mask,
            epsilon_r=config.epsilon_r,
            eta=config.eta,
        )

        grad = g_att + g_rep

        # --- Momentum step ---
        velocities = config.alpha * velocities - config.lr * grad
        positions_3d = positions_3d + velocities

        # --- Metrics ---
        pos2d = _unpack(positions_3d)
        cm = _cluster_metrics(pos2d, labels)

        # rms_displacement every 10 iters
        if i % 10 == 0:
            if prev_pos_snapshot is not None:
                rms_disp = float(
                    np.sqrt(((pos2d - prev_pos_snapshot) ** 2).sum(axis=1).mean())
                )
            else:
                rms_disp = float("nan")
            prev_pos_snapshot = pos2d.copy()
            snapshot_iter = i
        else:
            rms_disp = float("nan")

        rows.append({
            "iter": i,
            "e_att": e_att,
            "e_rep": e_rep,
            **cm,
            "rms_displacement": rms_disp,
        })

    metrics_history = pd.DataFrame(rows)

    return SliceRunResult(
        pos_initial=pos0.copy(),
        pos_final=_unpack(positions_3d),
        metrics_history=metrics_history,
        config=config,
        labels=labels,
    )


# ===========================================================================
# Section 5: Per-run plots
# ===========================================================================

_CLUSTER_COLORS = ["#2166AC", "#B2182B", "#4DAC26", "#F1A340"]


def _resolve_color_map(labels: np.ndarray) -> dict[int, str]:
    groups = sorted(set(labels.tolist()))
    return {g: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, g in enumerate(groups)}


def _scatter_panel(
    ax: plt.Axes,
    pos: np.ndarray,
    labels: np.ndarray,
    color_map: dict[int, str],
    title: str,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
) -> None:
    for g, c in color_map.items():
        mask = labels == g
        ax.scatter(pos[mask, 0], pos[mask, 1], s=18, alpha=0.7, color=c, label=f"cluster {g}")
        centroid = pos[mask].mean(axis=0)
        ax.scatter(*centroid, s=200, marker="+", color="black", linewidths=2, zorder=5)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, markerscale=1.2)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_run(result: SliceRunResult, output_dir: Path) -> None:
    """Save scatter.png and metrics.png for one run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    color_map = _resolve_color_map(result.labels)
    h = result.metrics_history

    # --- Shared axis limits ---
    all_pos = np.vstack([result.pos_initial, result.pos_final])
    pad = 0.5
    xlim = (all_pos[:, 0].min() - pad, all_pos[:, 0].max() + pad)
    ylim = (all_pos[:, 1].min() - pad, all_pos[:, 1].max() + pad)

    config_label = (
        f"k={cfg.k_attract}  σ={cfg.sigma:.3f}  εr={cfg.epsilon_r:.4f}"
        f"  sub={cfg.subtract_mean}  coh={cfg.coherence_mode}"
    )
    sr_before = result.sep_ratio_initial
    sr_after = result.sep_ratio_final

    # --- scatter.png ---
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(config_label, fontsize=9)
    _scatter_panel(ax_before, result.pos_initial, result.labels, color_map,
                   f"Initial  sep_ratio={sr_before:.2f}", xlim=xlim, ylim=ylim)
    _scatter_panel(ax_after, result.pos_final, result.labels, color_map,
                   f"Final (iter {cfg.n_iter})  sep_ratio={sr_after:.2f}  "
                   f"best={result.sep_ratio_best:.2f}@{result.iter_best_sep_ratio}",
                   xlim=xlim, ylim=ylim)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter.png", dpi=120)
    plt.close(fig)

    # --- metrics.png ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(config_label, fontsize=9)
    ax_flat = axes.ravel()

    # Panel 0: separation_ratio (primary)
    ax = ax_flat[0]
    ax.plot(h["iter"], h["separation_ratio"], color="#2166AC", lw=2, label="sep_ratio")
    ax.axhline(sr_before, color="#2166AC", lw=1, ls="--", alpha=0.5, label="initial")
    ax.axvline(result.iter_best_sep_ratio, color="gray", lw=1, ls=":", alpha=0.7)
    ax.set_title("separation_ratio (primary)", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 1: centroid_distance
    ax = ax_flat[1]
    ax.plot(h["iter"], h["centroid_distance"], color="#4DAC26")
    ax.set_title("centroid_distance", fontsize=9)

    # Panel 2: within_cluster_rms
    ax = ax_flat[2]
    ax.plot(h["iter"], h["within_cluster_rms"], color="#F1A340")
    ax.set_title("within_cluster_rms", fontsize=9)

    # Panel 3: energies
    ax = ax_flat[3]
    ax.plot(h["iter"], h["e_att"], color="#B2182B", label="e_att")
    ax.plot(h["iter"], h["e_rep"], color="#4DAC26", label="e_rep")
    ax.set_title("energies", fontsize=9)
    ax.legend(fontsize=7)

    # Panel 4: global_spread
    ax = ax_flat[4]
    ax.plot(h["iter"], h["global_spread"], color="#762A83")
    ax.set_title(f"global_spread  (collapse_score={result.collapse_score:.3f})", fontsize=9)

    # Panel 5: hide
    ax_flat[5].set_visible(False)

    for ax in ax_flat[:5]:
        ax.set_xlabel("iteration", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "metrics.png", dpi=120)
    plt.close(fig)


# ===========================================================================
# Section 6: Parameter sweep
# ===========================================================================

@dataclass
class SweepConfig:
    sigma_fracs: list[float]
    epsilon_r_mults: list[float]
    k_values: list[int | None]        # None = all-pairs; 0 = repulsion-only
    subtract_mean_values: list[bool]
    coherence_modes: list[str]
    n_iter: int = 300
    lr: float = 5e-4
    alpha: float = 0.9


def run_sweep(
    dataset: SyntheticDataset,
    sweep_config: SweepConfig,
    output_dir: Path,
    save_per_run_plots: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all parameter combinations. Returns a summary DataFrame."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pos0 = dataset.pos
    labels = dataset.labels
    scale_ref = radial_spread(pos0)
    initial_metrics = _cluster_metrics(pos0, labels)
    sep_before = initial_metrics["separation_ratio"]
    global_spread_initial = initial_metrics["global_spread"]

    combos = list(itertools.product(
        sweep_config.sigma_fracs,
        sweep_config.epsilon_r_mults,
        sweep_config.k_values,
        sweep_config.subtract_mean_values,
        sweep_config.coherence_modes,
    ))

    rows: list[dict] = []
    n_total = len(combos)

    for run_idx, (sf, em, k, sub, coh) in enumerate(combos):
        # Repulsion-only baseline: k==0 means no attraction
        sigma = sf * scale_ref
        epsilon_r = em * 0.6 * sigma ** 2

        cfg = SliceRunConfig(
            sigma=sigma,
            epsilon_r=epsilon_r,
            k_attract=k,
            subtract_mean=sub,
            coherence_mode=coh,
            n_iter=sweep_config.n_iter,
            lr=sweep_config.lr,
            alpha=sweep_config.alpha,
        )

        result = run_slice(pos0, labels, cfg)

        run_id = f"{run_idx:04d}"
        run_dir = output_dir / f"run_{run_id}"

        # Always save metrics CSV
        run_dir.mkdir(parents=True, exist_ok=True)
        result.metrics_history.to_csv(run_dir / "metrics_history.csv", index=False)

        if save_per_run_plots:
            plot_run(result, run_dir)

        final = result.metrics_history.iloc[-1]
        rows.append({
            "variant": dataset.variant,
            "run_id": run_id,
            "sigma_frac": sf,
            "eps_mult": em,
            "sigma": sigma,
            "epsilon_r": epsilon_r,
            "k_attract": k,
            "subtract_mean": sub,
            "coherence_mode": coh,
            "n_iter": sweep_config.n_iter,
            "sep_ratio_before": sep_before,
            "sep_ratio_final": result.sep_ratio_final,
            "sep_ratio_gain": result.sep_ratio_final - sep_before,
            "sep_ratio_best": result.sep_ratio_best,
            "iter_best_sep_ratio": result.iter_best_sep_ratio,
            "sep_ratio_gain_best": result.sep_ratio_best - sep_before,
            "centroid_distance_final": float(final["centroid_distance"]),
            "within_cluster_rms_final": float(final["within_cluster_rms"]),
            "global_spread_final": float(final["global_spread"]),
            "global_spread_initial": global_spread_initial,
            "collapse_score": result.collapse_score,
            "e_att_final": float(final["e_att"]),
            "e_rep_final": float(final["e_rep"]),
        })

        if verbose and (run_idx % max(1, n_total // 20) == 0 or run_idx == n_total - 1):
            print(
                f"  [{run_idx+1}/{n_total}] k={k} sf={sf:.1f} em={em:.1f}"
                f" sub={sub} coh={coh}"
                f" -> sep_ratio {sep_before:.2f} -> {result.sep_ratio_final:.2f}"
                f"  collapse={result.collapse_score:.3f}"
            )

    return pd.DataFrame(rows)


# ===========================================================================
# Section 7: Sweep summary plots
# ===========================================================================

def plot_sweep_summary(
    summary: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """Generate sweep_heatmap.png and sweep_top{N}.png."""
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = sorted(
        summary["k_attract"].unique(),
        key=lambda x: -1 if x is None else (float("inf") if x == 0 else x),
    )
    coh_modes = sorted(summary["coherence_mode"].unique())
    sigma_fracs = sorted(summary["sigma_frac"].unique())
    eps_mults = sorted(summary["eps_mult"].unique())

    nrows = len(k_values)
    ncols = len(coh_modes)

    # --- sweep_heatmap.png ---
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    fig.suptitle("sep_ratio_gain: σ_frac (rows) × eps_mult (cols)", fontsize=11)

    vmax = summary["sep_ratio_gain"].abs().max()
    vmax = max(vmax, 0.1)

    for ri, k in enumerate(k_values):
        for ci, coh in enumerate(coh_modes):
            ax = axes[ri][ci]
            sub_df = summary[(summary["k_attract"] == k) & (summary["coherence_mode"] == coh)]
            # aggregate over subtract_mean by taking max gain
            heat = np.full((len(sigma_fracs), len(eps_mults)), float("nan"))
            for si, sf in enumerate(sigma_fracs):
                for ei, em in enumerate(eps_mults):
                    cell = sub_df[
                        (np.isclose(sub_df["sigma_frac"], sf)) &
                        (np.isclose(sub_df["eps_mult"], em))
                    ]
                    if len(cell) > 0:
                        heat[si, ei] = cell["sep_ratio_gain"].max()

            im = ax.imshow(
                heat, aspect="auto", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
                origin="upper",
            )
            ax.set_xticks(range(len(eps_mults)))
            ax.set_xticklabels([f"{em:.1f}" for em in eps_mults], fontsize=8)
            ax.set_yticks(range(len(sigma_fracs)))
            ax.set_yticklabels([f"{sf:.1f}" for sf in sigma_fracs], fontsize=8)
            k_label = "all-pairs" if k is None else ("repulsion-only" if k == 0 else f"k={k}")
            ax.set_title(f"{k_label} | coh={coh}", fontsize=9)
            ax.set_xlabel("eps_mult", fontsize=8)
            ax.set_ylabel("sigma_frac", fontsize=8)

            for si in range(len(sigma_fracs)):
                for ei in range(len(eps_mults)):
                    v = heat[si, ei]
                    if not np.isnan(v):
                        ax.text(ei, si, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color="black")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "sweep_heatmap.png", dpi=120)
    plt.close(fig)

    # --- sweep_top{N}.png ---
    top = summary.nlargest(top_n, "sep_ratio_final")

    coh_color = {
        "uniform": "#2166AC",
        "oracle": "#B2182B",
    }
    bar_colors = [coh_color.get(c, "#808080") for c in top["coherence_mode"]]

    ylabels = [
        f"k={row.k_attract} sf={row.sigma_frac:.1f} em={row.eps_mult:.1f}"
        f" sub={row.subtract_mean} coh={row.coherence_mode}"
        for row in top.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    ax.barh(range(len(top)), top["sep_ratio_final"].values, color=bar_colors, alpha=0.85)
    ax.axvline(top["sep_ratio_before"].iloc[0], color="black", lw=1.5, ls="--",
               label="initial sep_ratio")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("sep_ratio_final", fontsize=9)
    ax.set_title(f"Top {top_n} runs by sep_ratio_final", fontsize=10)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=coh_color["uniform"], label="uniform coherence"),
        Patch(facecolor=coh_color["oracle"], label="oracle coherence"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"sweep_top{top_n}.png", dpi=120)
    plt.close(fig)


# ===========================================================================
# Section 8: Main entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic 2D slice sandbox for force parameter tuning."
    )
    p.add_argument(
        "--output-dir",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/slice_sandbox_v1",
    )
    p.add_argument(
        "--variants", nargs="+",
        choices=["separated", "moderate", "overlapping", "elongated"],
        default=["separated", "moderate", "overlapping", "elongated"],
    )
    p.add_argument("--n-per-cluster", type=int, default=60)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument(
        "--k-values", nargs="+",
        default=["5", "10", "20", "none"],
        help="k for kNN attraction; 'none' = all-pairs; '0' = repulsion-only baseline",
    )
    p.add_argument(
        "--sigma-fracs", nargs="+", type=float,
        default=[0.3, 0.5, 0.7, 1.0],
    )
    p.add_argument(
        "--eps-mults", nargs="+", type=float,
        default=[0.3, 0.6, 1.2],
    )
    p.add_argument("--no-per-run-plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parse k_values: "none" -> None, "0" -> 0 (repulsion-only), else int
    k_values: list[int | None] = []
    for v in args.k_values:
        if v.lower() == "none":
            k_values.append(None)
        else:
            k_values.append(int(v))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_cfg = SweepConfig(
        sigma_fracs=args.sigma_fracs,
        epsilon_r_mults=args.eps_mults,
        k_values=k_values,
        subtract_mean_values=[False, True],
        coherence_modes=["uniform", "oracle"],
        n_iter=args.n_iter,
        lr=args.lr,
        alpha=args.alpha,
    )

    all_summaries: list[pd.DataFrame] = []

    for variant in args.variants:
        print(f"\n=== Variant: {variant} ===")
        dataset = make_two_cluster_dataset(
            variant,
            n_per_cluster=args.n_per_cluster,
            random_seed=args.seed,
        )
        print(
            f"  N={len(dataset.pos)}  true_sep={dataset.true_separation:.2f}"
            f"  scale_ref={radial_spread(dataset.pos):.3f}"
        )

        variant_dir = output_dir / variant
        summary = run_sweep(
            dataset, sweep_cfg, variant_dir,
            save_per_run_plots=not args.no_per_run_plots,
            verbose=True,
        )
        summary.to_csv(variant_dir / "summary.csv", index=False)
        plot_sweep_summary(summary, variant_dir, top_n=args.top_n)
        all_summaries.append(summary)

        top10 = summary.nlargest(10, "sep_ratio_final")[
            ["run_id", "sigma_frac", "eps_mult", "k_attract", "subtract_mean",
             "coherence_mode", "sep_ratio_before", "sep_ratio_final",
             "sep_ratio_best", "iter_best_sep_ratio", "collapse_score"]
        ]
        print(f"\n  Top 10 by sep_ratio_final ({variant}):")
        print(top10.to_string(index=False))

    # Combined summary
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(output_dir / "all_variants_summary.csv", index=False)
    print(f"\nSaved all_variants_summary.csv ({len(combined)} rows)")
    print("\nTop 10 overall by sep_ratio_final:")
    print(
        combined.nlargest(10, "sep_ratio_final")[
            ["variant", "sigma_frac", "eps_mult", "k_attract",
             "subtract_mean", "coherence_mode",
             "sep_ratio_final", "sep_ratio_best", "collapse_score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
