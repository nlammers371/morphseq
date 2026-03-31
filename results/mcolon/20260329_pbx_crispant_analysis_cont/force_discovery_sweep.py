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
# Section 4: CLI + main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B: force discovery sweep on Y-benchmark")
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

    # --- Print discovery summary ---
    print("\n=== Discovery Summary ===")
    print(f"{'family':<22} {'field':<28} {'value':>10}  {'lin':>6} {'sep':>6} {'spread':>7} {'sel':>7}")
    print("-" * 90)
    for _, row in df_all.iterrows():
        print(f"  {row['family']:<20} {str(row['value']):<10}  "
              f"lin={row['trunk_linearity_early']:.3f}  "
              f"sep={row['branch_sep_late']:.3f}  "
              f"spread={row['within_branch_spread_ratio']:.3f}  "
              f"sel={row['coherence_selectivity']:.3f}")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
