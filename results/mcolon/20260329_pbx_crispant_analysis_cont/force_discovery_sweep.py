"""
force_discovery_sweep.py
------------------------
Phase B: Force discovery using the Y-shaped bifurcating trunk benchmark.

Purpose
-------
The bifurcating trunk sandbox showed that at the default parameter settings,
all four conditions (isotropic, fidelity, void, elasticity) produce nearly
identical geometry metrics. That is the discovery: at those scales, the added
forces don't move the needle. This script asks: at what strength does each
force start to matter, and what breaks first?

Design
------
For each force family, sweep its dimensionless multiplier over a log range.
All other forces stay at the isotropic baseline (coherence + repulsion only).
For each (force, multiplier) point:
  - reload the same shared initialization
  - run run_temporal() on the Y-dataset
  - record: trunk_linearity_early, branch_sep_late, within_branch_spread_ratio,
    coherence_selectivity, collapse_score, n_iter_converged

Output
------
  sweep_results.csv          — full numeric table
  sweep_<family>.png         — per-force sweep curve (4 metrics vs multiplier)
  sweep_all_families.png     — 2x2 grid comparing all families on same axes

The Y-axis answer: which force changes things, at what scale, and in which
direction? That tells us which default is sensible and which forces to
investigate further on real data.

Run (smoke test — fast):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/force_discovery_sweep.py \\
      --output-dir /tmp/force_sweep_smoke \\
      --n-per-branch 20 --n-iter 100 --n-points 5 --seed 42

Run (full discovery):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/force_discovery_sweep.py \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/force_discovery_v1
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from bifurcating_trunk_sandbox import (
    make_bifurcating_trunk,
    save_initialization,
    load_initialization,
    trunk_summary_metrics,
)
from temporal_sandbox import (
    TemporalRunConfig,
    TemporalRunResult,
    run_temporal,
)


# ===========================================================================
# Section 1: Sweep definitions
# ===========================================================================

@dataclass
class ForceSweep:
    """One force family sweep: name, multiplier field, range to explore."""
    family: str          # display name, e.g. "repulsion"
    field: str           # TemporalRunConfig field name
    values: list[float]  # multiplier values to sweep (log-spaced recommended)
    baseline_value: float  # the reference "off" value (usually 0.0 or default)
    label: str           # axis label, e.g. "λ_rep (repulsion strength mult)"
    note: str = ""       # optional one-line note about what this controls


def build_sweeps(n_points: int = 9) -> list[ForceSweep]:
    """Define all force family sweeps.

    Each sweep holds one force multiplier variable; all others stay at the
    isotropic baseline (repulsion_strength_mult=0.005, everything else off).

    n_points : number of log-spaced values between the min and max of each range.
    """
    def logspace(lo, hi, n):
        return list(np.logspace(np.log10(lo), np.log10(hi), n))

    return [
        ForceSweep(
            family="repulsion",
            field="repulsion_strength_mult",
            values=logspace(1e-4, 0.5, n_points),
            baseline_value=0.005,
            label="λ_rep (ε_r = λ_rep × s_local²)",
            note="Too small → collapse. Too large → bundle inflation / explosion.",
        ),
        ForceSweep(
            family="fidelity",
            field="fidelity_strength_mult",
            values=[0.0] + logspace(1e-4, 5.0, n_points),
            baseline_value=0.0,
            label="λ_fid (μ₀ = λ_fid / s_local²)",
            note="Too small → no effect. Too large → freezes positions, kills coherence.",
        ),
        ForceSweep(
            family="elasticity_stretch",
            field="stretch_strength_mult",
            values=[0.0] + logspace(1e-4, 2.0, n_points),
            baseline_value=0.0,
            label="λ_str (λ_stretch = λ_str / s_step²)",
            note="Too large → trajectories collapse to mean; kills branching.",
        ),
        ForceSweep(
            family="elasticity_bend",
            field="bend_strength_mult",
            values=[0.0] + logspace(1e-4, 2.0, n_points),
            baseline_value=0.0,
            label="λ_bnd (λ_bend = λ_bnd / s_bend²)",
            note="Too large → trajectories straighten; kills curvature at branch point.",
        ),
        ForceSweep(
            family="void_proxy",
            field="epsilon_void",
            values=[0.0] + logspace(1e-4, 0.1, n_points),
            baseline_value=0.0,
            label="ε_void (pairwise Gaussian void)",
            note="Broad pairwise repulsion. Too large → bundles pushed off-screen.",
        ),
    ]


# ===========================================================================
# Section 2: One sweep point
# ===========================================================================

def run_sweep_point(
    init_path: Path,
    variant: str,
    n_per_branch: int,
    n_time: int,
    split_full: int,
    sweep: ForceSweep,
    value: float,
    n_iter: int,
    k_attract: int,
    delta: int,
    lr: float,
    verbose: bool = False,
) -> dict:
    """Run one (force_family, multiplier_value) point on the Y-benchmark.

    Returns a flat dict with the key metrics.
    """
    dataset = load_initialization(init_path, variant=variant, n_per_cluster=n_per_branch)

    # Build config: start from isotropic baseline, override one field
    cfg = TemporalRunConfig(
        k_attract=k_attract,
        delta=delta,
        lr=lr,
        n_iter=n_iter,
        mu0=0.0,
        epsilon_void=0.0,
        lambda_stretch=0.0,
        lambda_bend=0.0,
        fidelity_strength_mult=0.0,
        stretch_strength_mult=0.0,
        bend_strength_mult=0.0,
    )
    setattr(cfg, sweep.field, value)

    result: TemporalRunResult = run_temporal(dataset, cfg, save_snapshots=False, verbose=verbose)

    # Trunk-specific metrics
    summary = trunk_summary_metrics(
        result.cond_result.positions,
        dataset.positions,
        dataset.labels,
        split_full=split_full,
    )

    return {
        "family": sweep.family,
        "field": sweep.field,
        "value": value,
        "trunk_linearity_early": summary["trunk_linearity_early"],
        "branch_sep_late": summary["branch_sep_late"],
        "within_branch_spread_ratio": summary["within_branch_spread_ratio"],
        "coherence_selectivity": result.final_metrics.get("coherence_selectivity", float("nan")),
        "collapse_score": result.collapse_score,
        "n_iter": result.cond_result.n_iter,
        "converged": result.cond_result.converged,
    }


# ===========================================================================
# Section 3: Visualization
# ===========================================================================

_METRIC_LABELS = {
    "trunk_linearity_early": "Trunk linearity\n(early t, >0.7 = line)",
    "branch_sep_late": "Branch separation\n(late t, >2.0 = clear split)",
    "within_branch_spread_ratio": "Within-branch spread ratio\n(~1.0 = preserved, >2 = inflated)",
    "coherence_selectivity": "Coherence selectivity\n(>1 = distinguishes branches)",
    "collapse_score": "Collapse score\n(1.0 = no change, <1 = contracted)",
}

_METRICS_TO_PLOT = [
    "trunk_linearity_early",
    "branch_sep_late",
    "within_branch_spread_ratio",
    "coherence_selectivity",
]

_BASELINE_COLOR = "#888888"
_SWEEP_COLOR = "#2166AC"


def _add_baseline_band(ax, df_baseline: pd.DataFrame, metric: str) -> None:
    """Draw a thin horizontal line at the isotropic baseline value."""
    if df_baseline is None or df_baseline.empty:
        return
    val = df_baseline[metric].iloc[0]
    ax.axhline(val, color=_BASELINE_COLOR, linewidth=1.0, linestyle="--",
               label=f"baseline={val:.3f}", zorder=1)


def plot_sweep_family(
    df_sweep: pd.DataFrame,
    sweep: ForceSweep,
    df_baseline: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """4-panel figure: one metric per panel, x=multiplier value (log), y=metric."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    # drop zero values for log x-axis (plot separately as a dashed marker)
    df_nonzero = df_sweep[df_sweep["value"] > 0].copy()
    df_zero = df_sweep[df_sweep["value"] == 0.0].copy()

    for ax, metric in zip(axes, _METRICS_TO_PLOT):
        _add_baseline_band(ax, df_baseline, metric)

        if not df_nonzero.empty:
            ax.plot(df_nonzero["value"], df_nonzero[metric],
                    "o-", color=_SWEEP_COLOR, linewidth=1.5, markersize=5, label="sweep")

        # Mark the zero point on the left margin
        if not df_zero.empty:
            zero_val = df_zero[metric].iloc[0]
            ax.scatter([df_nonzero["value"].min() * 0.3], [zero_val],
                       marker="<", color=_SWEEP_COLOR, s=60, zorder=5, label="value=0")

        ax.set_xscale("log")
        ax.set_xlabel(sweep.label, fontsize=8)
        ax.set_ylabel(_METRIC_LABELS[metric], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Force discovery sweep: {sweep.family}\n"
        f"({sweep.note})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_families(
    df_all: pd.DataFrame,
    sweeps: list[ForceSweep],
    df_baseline: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Grid figure: rows = metrics, cols = force families.

    Each cell shows how a metric changes as one force's strength increases.
    All cells share the y-axis scale within a row so comparisons are direct.
    """
    families = [s.family for s in sweeps]
    n_families = len(families)
    n_metrics = len(_METRICS_TO_PLOT)

    fig, axes = plt.subplots(
        n_metrics, n_families,
        figsize=(3.5 * n_families, 3.2 * n_metrics),
        squeeze=False,
    )

    # Compute per-metric y limits across all families for consistent scaling
    ylims: dict[str, tuple[float, float]] = {}
    for metric in _METRICS_TO_PLOT:
        vals = df_all[metric].dropna()
        if vals.empty:
            ylims[metric] = (0, 1)
            continue
        lo, hi = vals.min(), vals.max()
        pad = max((hi - lo) * 0.15, 0.05)
        ylims[metric] = (lo - pad, hi + pad)

    for col, sweep in enumerate(sweeps):
        df_fam = df_all[df_all["family"] == sweep.family].copy()
        df_nonzero = df_fam[df_fam["value"] > 0]
        df_zero = df_fam[df_fam["value"] == 0.0]

        for row, metric in enumerate(_METRICS_TO_PLOT):
            ax = axes[row][col]
            _add_baseline_band(ax, df_baseline, metric)

            if not df_nonzero.empty:
                ax.plot(df_nonzero["value"], df_nonzero[metric],
                        "o-", color=_SWEEP_COLOR, linewidth=1.5, markersize=4)

            if not df_zero.empty and not df_nonzero.empty:
                zero_val = df_zero[metric].iloc[0]
                ax.scatter(
                    [df_nonzero["value"].min() * 0.35], [zero_val],
                    marker="<", color=_SWEEP_COLOR, s=40, zorder=5
                )

            ax.set_ylim(ylims[metric])
            ax.set_xscale("log")
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(sweep.family, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(_METRIC_LABELS[metric], fontsize=7)
            if row == n_metrics - 1:
                ax.set_xlabel("multiplier value", fontsize=7)

    fig.suptitle(
        "Force discovery sweep — Y-shaped benchmark\n"
        "Each column: one force, all others at isotropic baseline",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Section 4: Activation threshold analysis
# ===========================================================================

_THRESHOLD_TAUS = [0.05, 0.10]   # 5% and 10% relative deviation from baseline

_REGIME_COLORS = {
    "inert":      "#CCDDEE",   # light blue — safe default zone
    "onset":      "#FFE0A0",   # amber — force beginning to matter
    "dominant":   "#F4C0C0",   # light red — geometry visibly altered
}


def compute_activation_thresholds(
    df_all: pd.DataFrame,
    df_baseline: pd.DataFrame,
    taus: list[float] = None,
) -> pd.DataFrame:
    """Find the onset multiplier at which each force × metric exceeds tau deviation.

    For each (family, metric, tau):
        ΔM(λ) = |M(λ) - M_baseline| / (|M_baseline| + ε)
        onset_lambda = smallest λ > 0 where ΔM(λ) > tau

    Returns a DataFrame with columns:
        family, metric, tau, onset_lambda, onset_delta_M, regime_description
    """
    if taus is None:
        taus = _THRESHOLD_TAUS

    eps = 1e-8
    rows = []

    for family, df_fam in df_all.groupby("family"):
        df_nonzero = df_fam[df_fam["value"] > 0].sort_values("value")
        if df_nonzero.empty:
            continue

        for metric in _METRICS_TO_PLOT:
            if metric not in df_nonzero.columns or metric not in df_baseline.columns:
                continue
            m_base = float(df_baseline[metric].iloc[0])

            delta_M = (df_nonzero[metric] - m_base).abs() / (abs(m_base) + eps)

            for tau in taus:
                # Find first λ where deviation exceeds tau
                exceeded = df_nonzero[delta_M > tau]
                if exceeded.empty:
                    onset_lambda = float("nan")
                    onset_delta = float(delta_M.max()) if not delta_M.empty else float("nan")
                    regime = "inert"
                else:
                    onset_lambda = float(exceeded["value"].iloc[0])
                    onset_delta = float(delta_M[delta_M > tau].iloc[0])
                    regime = "onset"

                rows.append({
                    "family": family,
                    "metric": metric,
                    "tau": tau,
                    "onset_lambda": onset_lambda,
                    "onset_delta_M": onset_delta,
                    "baseline_value": m_base,
                    "regime": regime,
                })

    return pd.DataFrame(rows)


def plot_threshold_summary(
    df_all: pd.DataFrame,
    df_baseline: pd.DataFrame,
    df_thresh: pd.DataFrame,
    sweeps: list[ForceSweep],
    output_path: Path,
    tau_primary: float = 0.05,
) -> None:
    """Regime-shaded sweep figure.

    Layout: rows = metrics, cols = force families.
    Each cell shows the sweep curve with:
      - inert zone shaded light blue (λ < onset at tau_primary)
      - onset zone shaded amber (onset_5% ≤ λ < onset_10%)
      - dominant zone shaded red (λ ≥ onset_10%)
      - vertical dashed lines at 5% and 10% onset thresholds
    """
    n_families = len(sweeps)
    n_metrics = len(_METRICS_TO_PLOT)

    fig, axes = plt.subplots(
        n_metrics, n_families,
        figsize=(3.5 * n_families, 3.2 * n_metrics),
        squeeze=False,
    )

    # Per-metric y limits (same as all-families plot)
    ylims: dict[str, tuple[float, float]] = {}
    for metric in _METRICS_TO_PLOT:
        vals = df_all[metric].dropna()
        if vals.empty:
            ylims[metric] = (0, 1)
            continue
        lo, hi = vals.min(), vals.max()
        pad = max((hi - lo) * 0.15, 0.05)
        ylims[metric] = (lo - pad, hi + pad)

    for col, sweep in enumerate(sweeps):
        df_fam = df_all[df_all["family"] == sweep.family].copy()
        df_nonzero = df_fam[df_fam["value"] > 0].sort_values("value")
        if df_nonzero.empty:
            continue

        x_min = df_nonzero["value"].min()
        x_max = df_nonzero["value"].max()

        for row, metric in enumerate(_METRICS_TO_PLOT):
            ax = axes[row][col]

            # Baseline reference
            m_base = float(df_baseline[metric].iloc[0])
            ax.axhline(m_base, color=_BASELINE_COLOR, linewidth=1.0, linestyle="--", zorder=2)

            # Get onset thresholds for this family × metric
            t5 = df_thresh[
                (df_thresh["family"] == sweep.family) &
                (df_thresh["metric"] == metric) &
                (df_thresh["tau"] == 0.05)
            ]
            t10 = df_thresh[
                (df_thresh["family"] == sweep.family) &
                (df_thresh["metric"] == metric) &
                (df_thresh["tau"] == 0.10)
            ]
            onset_5 = float(t5["onset_lambda"].iloc[0]) if not t5.empty else float("nan")
            onset_10 = float(t10["onset_lambda"].iloc[0]) if not t10.empty else float("nan")

            # Shade regimes
            if not np.isnan(onset_5):
                # Inert zone: x_min → onset_5
                ax.axvspan(x_min * 0.5, onset_5, color=_REGIME_COLORS["inert"], alpha=0.5, zorder=0)
                ax.axvline(onset_5, color="#E08000", linewidth=1.0, linestyle=":", zorder=3,
                           label=f"5% onset={onset_5:.2g}")

            if not np.isnan(onset_10):
                # Onset zone: onset_5 → onset_10
                lo = onset_5 if not np.isnan(onset_5) else x_min
                ax.axvspan(lo, onset_10, color=_REGIME_COLORS["onset"], alpha=0.5, zorder=0)
                ax.axvline(onset_10, color="#CC2200", linewidth=1.0, linestyle=":", zorder=3,
                           label=f"10% onset={onset_10:.2g}")
                # Dominant zone: onset_10 → x_max
                ax.axvspan(onset_10, x_max * 2, color=_REGIME_COLORS["dominant"], alpha=0.4, zorder=0)
            elif not np.isnan(onset_5):
                # Only 5% triggered — shade onset → x_max as onset zone
                ax.axvspan(onset_5, x_max * 2, color=_REGIME_COLORS["onset"], alpha=0.5, zorder=0)
            else:
                # Fully inert — shade entire range light blue
                ax.axvspan(x_min * 0.5, x_max * 2, color=_REGIME_COLORS["inert"], alpha=0.5, zorder=0)

            # Sweep curve
            ax.plot(df_nonzero["value"], df_nonzero[metric],
                    "o-", color=_SWEEP_COLOR, linewidth=1.8, markersize=4, zorder=4)

            ax.set_xlim(x_min * 0.4, x_max * 2.5)
            ax.set_ylim(ylims[metric])
            ax.set_xscale("log")
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(sweep.family, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(_METRIC_LABELS[metric], fontsize=7)
            if row == n_metrics - 1:
                ax.set_xlabel("multiplier value", fontsize=7)
            if not t5.empty and not np.isnan(onset_5):
                ax.legend(fontsize=5.5, loc="best")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_REGIME_COLORS["inert"], alpha=0.6, label="inert (<5% change)"),
        Patch(facecolor=_REGIME_COLORS["onset"], alpha=0.6, label="onset (5–10% change)"),
        Patch(facecolor=_REGIME_COLORS["dominant"], alpha=0.6, label="dominant (>10% change)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Activation threshold analysis — Y-shaped benchmark\n"
        "Shading shows inert / onset / dominant regimes per force × metric",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Section 5: CLI + main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B: force discovery sweep + activation threshold analysis on Y-benchmark")
    p.add_argument("--output-dir", default=str(
        _HERE / "results" / "force_discovery_v1"), type=str)
    p.add_argument("--n-per-branch", type=int, default=40)
    p.add_argument("--n-time", type=int, default=13)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--n-points", type=int, default=9,
                   help="Number of log-spaced multiplier values per sweep (excluding zero)")
    p.add_argument("--k-attract", type=int, default=20)
    p.add_argument("--delta", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--split-start", type=int, default=4)
    p.add_argument("--split-full", type=int, default=8)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for dataset generation")
    p.add_argument("--families", type=str, default=None,
                   help="Comma-separated list of family names to run (default: all)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate dataset and save shared initialization ---
    init_path = output_dir / "initialization.npz"
    if not init_path.exists():
        print("Generating Y-shaped dataset and saving initialization...")
        dataset = make_bifurcating_trunk(
            n_per_branch=args.n_per_branch,
            n_time=args.n_time,
            split_start=args.split_start,
            split_full=args.split_full,
            random_seed=args.seed,
        )
        save_initialization(dataset, init_path)
        print(f"  Saved: {init_path}")
    else:
        print(f"  Reusing existing initialization: {init_path}")

    # --- Run isotropic baseline (single point for reference lines) ---
    print("\nRunning isotropic baseline...")
    baseline_row = run_sweep_point(
        init_path=init_path,
        variant="bifurcating_trunk",
        n_per_branch=args.n_per_branch,
        n_time=args.n_time,
        split_full=args.split_full,
        sweep=ForceSweep("baseline", "repulsion_strength_mult", [0.005], 0.005, "baseline"),
        value=0.005,
        n_iter=args.n_iter,
        k_attract=args.k_attract,
        delta=args.delta,
        lr=args.lr,
        verbose=False,
    )
    df_baseline = pd.DataFrame([baseline_row])
    print(f"  Baseline: trunk_lin={baseline_row['trunk_linearity_early']:.3f}  "
          f"branch_sep={baseline_row['branch_sep_late']:.3f}  "
          f"spread={baseline_row['within_branch_spread_ratio']:.3f}  "
          f"coh_sel={baseline_row['coherence_selectivity']:.3f}")

    # --- Build sweeps ---
    sweeps = build_sweeps(n_points=args.n_points)
    if args.families is not None:
        keep = set(args.families.split(","))
        sweeps = [s for s in sweeps if s.family in keep]

    # --- Run all sweeps ---
    all_rows: list[dict] = []

    for sweep in sweeps:
        print(f"\n=== Sweep: {sweep.family} ({len(sweep.values)} points) ===")
        print(f"  {sweep.note}")
        for i, value in enumerate(sweep.values):
            print(f"  [{i+1}/{len(sweep.values)}] {sweep.field}={value:.5g} ...", end=" ", flush=True)
            try:
                row = run_sweep_point(
                    init_path=init_path,
                    variant="bifurcating_trunk",
                    n_per_branch=args.n_per_branch,
                    n_time=args.n_time,
                    split_full=args.split_full,
                    sweep=sweep,
                    value=value,
                    n_iter=args.n_iter,
                    k_attract=args.k_attract,
                    delta=args.delta,
                    lr=args.lr,
                    verbose=False,
                )
                all_rows.append(row)
                print(f"lin={row['trunk_linearity_early']:.3f}  "
                      f"sep={row['branch_sep_late']:.3f}  "
                      f"spread={row['within_branch_spread_ratio']:.3f}  "
                      f"sel={row['coherence_selectivity']:.3f}  "
                      f"n_iter={row['n_iter']}")
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({
                    "family": sweep.family, "field": sweep.field, "value": value,
                    "trunk_linearity_early": float("nan"),
                    "branch_sep_late": float("nan"),
                    "within_branch_spread_ratio": float("nan"),
                    "coherence_selectivity": float("nan"),
                    "collapse_score": float("nan"),
                    "n_iter": -1,
                    "converged": False,
                })

    if not all_rows:
        print("No results collected. Exiting.")
        return

    df_all = pd.DataFrame(all_rows)
    csv_path = output_dir / "sweep_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # --- Plot per-family sweeps ---
    print("\nGenerating plots...")
    for sweep in sweeps:
        df_fam = df_all[df_all["family"] == sweep.family]
        if df_fam.empty:
            continue
        plot_sweep_family(
            df_fam, sweep, df_baseline,
            output_dir / f"sweep_{sweep.family}.png",
        )

    # --- Plot all-families comparison ---
    plot_all_families(df_all, sweeps, df_baseline, output_dir / "sweep_all_families.png")

    # --- Activation threshold analysis ---
    print("\nComputing activation thresholds...")
    df_thresh = compute_activation_thresholds(df_all, df_baseline)
    thresh_csv = output_dir / "activation_thresholds.csv"
    df_thresh.to_csv(thresh_csv, index=False)
    print(f"  Saved: {thresh_csv}")

    plot_threshold_summary(
        df_all, df_baseline, df_thresh, sweeps,
        output_dir / "threshold_regimes.png",
    )

    # --- Print threshold table ---
    print("\n=== Activation Threshold Summary ===")
    print("(onset_lambda = smallest multiplier where |ΔM| > tau relative to isotropic baseline)")
    print(f"  {'family':<22} {'metric':<32} {'tau':>5}  {'onset_lambda':>14}  regime")
    print("  " + "-" * 80)
    for tau in _THRESHOLD_TAUS:
        df_t = df_thresh[df_thresh["tau"] == tau].sort_values(["family", "metric"])
        print(f"\n  tau={tau:.0%}:")
        for _, r in df_t.iterrows():
            ol = f"{r['onset_lambda']:.3g}" if not np.isnan(r["onset_lambda"]) else "  (inert)"
            print(f"    {r['family']:<22} {r['metric']:<32} {r['tau']:>5.0%}  {ol:>14}  {r['regime']}")

    # --- Print discovery summary ---
    print("\n=== Full Sweep Results ===")
    print(f"{'family':<22} {'value':>10}  {'lin':>6} {'sep':>6} {'spread':>7} {'sel':>7}")
    print("-" * 70)
    for _, row in df_all.iterrows():
        print(f"  {row['family']:<20} {row['value']:<10.4g}  "
              f"lin={row['trunk_linearity_early']:.3f}  "
              f"sep={row['branch_sep_late']:.3f}  "
              f"spread={row['within_branch_spread_ratio']:.3f}  "
              f"sel={row['coherence_selectivity']:.3f}")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
