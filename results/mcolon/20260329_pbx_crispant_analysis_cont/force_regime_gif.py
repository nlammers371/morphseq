"""
force_regime_gif.py
-------------------
Phase B: Regime visualization GIFs for the Y-shaped benchmark.

For each force family, pick three representative multiplier values from the
sweep results — no_change, moderate, extreme — derived automatically from
sweep_results.csv using the activation threshold framework. Then run those
four conditions (baseline + 3 regimes) and generate a rotating 3D GIF with
4 columns, all sharing the same initialization, axis limits, and camera path.

Regime definitions (derived from sweep data, not hardcoded):
  no_change  — largest multiplier where ALL key metrics stay within tau_nochange
               of the isotropic baseline (default tau=0.02, i.e. 2%)
  moderate   — first multiplier where at least one metric exceeds tau_moderate
               deviation (default tau=0.05, i.e. 5%), but total distortion is
               still bounded (no metric > tau_extreme)
  extreme    — first multiplier where at least one metric exceeds tau_extreme
               (default tau=0.15, i.e. 15%), OR the largest sweep value if
               the force never fully reaches tau_extreme

These thresholds are calibrated to the sweep range and the geometry of the
Y-benchmark. They can be tuned via CLI flags.

Run (smoke test):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/force_regime_gif.py \\
      --sweep-csv results/mcolon/20260329_pbx_crispant_analysis_cont/results/force_discovery_v1/sweep_results.csv \\
      --output-dir /tmp/regime_smoke \\
      --n-per-branch 20 --n-iter 80 --n-frames 36 --seed 42 \\
      --families repulsion,elasticity_stretch

Run (full):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/force_regime_gif.py \\
      --sweep-csv results/mcolon/20260329_pbx_crispant_analysis_cont/results/force_discovery_v1/sweep_results.csv \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/force_regime_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from bifurcating_trunk_sandbox import (
    make_bifurcating_trunk,
    save_initialization,
    load_initialization,
    trunk_summary_metrics,
    _draw_3d_trunk,
)
from temporal_sandbox import (
    TemporalRunConfig,
    TemporalRunResult,
    run_temporal,
    _fig_to_rgb,
)
from force_discovery_sweep import ForceSweep, build_sweeps


# ===========================================================================
# Section 1: Regime selection from sweep data
# ===========================================================================

# Primary metrics used for regime detection.
# Each metric has a "direction": -1 means lower=worse (e.g. branch_sep collapses),
# +1 means higher=worse (e.g. spread inflates). We measure absolute relative deviation.
_REGIME_METRICS = [
    "branch_sep_late",
    "within_branch_spread_ratio",
    "trunk_linearity_early",
]

_REGIMES = ["baseline", "no_change", "moderate", "extreme"]

_REGIME_COLORS = {
    "baseline":  "#333333",
    "no_change": "#2166AC",
    "moderate":  "#E08000",
    "extreme":   "#CC2200",
}


def _max_relative_deviation(row: pd.Series, baseline: pd.Series, eps: float = 1e-8) -> float:
    """Max |ΔM| / (|M_base| + ε) across all regime metrics."""
    devs = []
    for m in _REGIME_METRICS:
        if m not in row or m not in baseline:
            continue
        m_base = float(baseline[m].iloc[0])
        devs.append(abs(float(row[m]) - m_base) / (abs(m_base) + eps))
    return max(devs) if devs else 0.0


def pick_regimes(
    df_sweep: pd.DataFrame,
    df_baseline: pd.Series,
    tau_nochange: float = 0.02,
    tau_moderate: float = 0.05,
    tau_extreme: float = 0.15,
) -> dict[str, dict]:
    """Derive no_change / moderate / extreme picks from sweep data for one family.

    Parameters
    ----------
    df_sweep    : rows for one family, sorted by value ascending
    df_baseline : single-row DataFrame with the isotropic baseline metrics
    tau_nochange: max relative deviation for a point to be considered "no change"
    tau_moderate: min deviation for a point to qualify as "moderate"
    tau_extreme : min deviation for a point to qualify as "extreme"

    Returns
    -------
    dict with keys "no_change", "moderate", "extreme", each a dict with:
        value, field, branch_sep_late, within_branch_spread_ratio,
        trunk_linearity_early, max_deviation, rationale
    """
    df = df_sweep[df_sweep["value"] > 0].sort_values("value").copy()
    df["_dev"] = df.apply(lambda r: _max_relative_deviation(r, df_baseline), axis=1)

    field = df["field"].iloc[0] if not df.empty else ""

    def _make_row(row: pd.Series, regime: str, rationale: str) -> dict:
        return {
            "regime": regime,
            "field": field,
            "value": float(row["value"]),
            "branch_sep_late": float(row.get("branch_sep_late", float("nan"))),
            "within_branch_spread_ratio": float(row.get("within_branch_spread_ratio", float("nan"))),
            "trunk_linearity_early": float(row.get("trunk_linearity_early", float("nan"))),
            "max_deviation": float(row["_dev"]),
            "rationale": rationale,
        }

    picks: dict[str, dict] = {}

    # --- no_change: largest value where all metrics stay within tau_nochange ---
    inert = df[df["_dev"] <= tau_nochange]
    if not inert.empty:
        row = inert.iloc[-1]  # largest inert value
        picks["no_change"] = _make_row(
            row, "no_change",
            f"largest λ with max deviation ≤{tau_nochange:.0%} (deviation={row['_dev']:.3f})"
        )
    else:
        # All points exceed tau — pick the smallest (least invasive)
        row = df.iloc[0]
        picks["no_change"] = _make_row(
            row, "no_change",
            f"all points exceed τ={tau_nochange:.0%}; using smallest value as proxy"
        )

    # --- moderate: first value that exceeds tau_moderate (onset) ---
    onset = df[df["_dev"] > tau_moderate]
    if not onset.empty:
        row = onset.iloc[0]
        picks["moderate"] = _make_row(
            row, "moderate",
            f"first λ exceeding {tau_moderate:.0%} deviation (deviation={row['_dev']:.3f})"
        )
    else:
        # Never reaches tau_moderate — use largest sweep value
        row = df.iloc[-1]
        picks["moderate"] = _make_row(
            row, "moderate",
            f"force never exceeds τ={tau_moderate:.0%}; using largest sweep value"
        )

    # --- extreme: first value that exceeds tau_extreme, or largest if never reached ---
    dominant = df[df["_dev"] > tau_extreme]
    if not dominant.empty:
        row = dominant.iloc[0]
        picks["extreme"] = _make_row(
            row, "extreme",
            f"first λ exceeding {tau_extreme:.0%} deviation (deviation={row['_dev']:.3f})"
        )
    else:
        row = df.iloc[-1]
        picks["extreme"] = _make_row(
            row, "extreme",
            f"force never exceeds τ={tau_extreme:.0%}; using largest sweep value"
        )

    return picks


def derive_all_picks(
    df_all: pd.DataFrame,
    sweeps: list[ForceSweep],
    tau_nochange: float = 0.02,
    tau_moderate: float = 0.05,
    tau_extreme: float = 0.15,
) -> dict[str, dict[str, dict]]:
    """Derive regime picks for all families.

    Returns dict: family -> {regime -> pick_dict}
    Also computes a pseudo-baseline row from the zero/minimum-value sweep rows.
    """
    # Build baseline: the row with value == 0 (or smallest value) per family,
    # or the zero-value rows from families that include 0.
    # We use the first row of fidelity at value=0 as the canonical baseline
    # (all families share the same x0 and the baseline config is identical).
    baseline_rows = df_all[
        (df_all["family"] == "fidelity") & (df_all["value"] == 0.0)
    ]
    if baseline_rows.empty:
        # Fall back to first row of any family at its minimum value
        baseline_rows = df_all.groupby("family").first().reset_index().head(1)

    all_picks: dict[str, dict[str, dict]] = {}
    for sweep in sweeps:
        df_fam = df_all[df_all["family"] == sweep.family].copy()
        if df_fam.empty:
            continue
        all_picks[sweep.family] = pick_regimes(
            df_fam, baseline_rows,
            tau_nochange=tau_nochange,
            tau_moderate=tau_moderate,
            tau_extreme=tau_extreme,
        )
    return all_picks


def print_picks_table(all_picks: dict[str, dict[str, dict]]) -> None:
    """Print a human-readable summary of the derived threshold picks."""
    print("\n=== Derived Regime Picks ===")
    print(f"  {'family':<22} {'regime':<12} {'value':>10}  {'dev':>6}  rationale")
    print("  " + "-" * 90)
    for family, picks in all_picks.items():
        for regime in ("no_change", "moderate", "extreme"):
            p = picks.get(regime)
            if p is None:
                continue
            print(f"  {family:<22} {regime:<12} {p['value']:>10.4g}  "
                  f"{p['max_deviation']:>6.3f}  {p['rationale'][:55]}")
        print()


def save_picks_csv(all_picks: dict[str, dict[str, dict]], output_path: Path) -> None:
    rows = []
    for family, picks in all_picks.items():
        for regime, p in picks.items():
            rows.append({"family": family, **p})
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


# ===========================================================================
# Section 2: Run regime conditions
# ===========================================================================

def run_regime_conditions(
    init_path: Path,
    sweep: ForceSweep,
    picks: dict[str, dict],
    n_per_branch: int,
    n_time: int,
    n_iter: int,
    k_attract: int,
    delta: int,
    lr: float,
) -> dict[str, TemporalRunResult]:
    """Run baseline + 3 regime conditions for one force family.

    Returns dict: regime -> TemporalRunResult
    """
    results: dict[str, TemporalRunResult] = {}

    # Baseline: isotropic default (repulsion_strength_mult=0.005, all others off)
    print(f"  [baseline] ...")
    dataset = load_initialization(init_path, variant="bifurcating_trunk",
                                  n_per_cluster=n_per_branch)
    baseline_cfg = TemporalRunConfig(
        k_attract=k_attract, delta=delta, lr=lr, n_iter=n_iter,
        fidelity_init_strength=0.0, epsilon_void=0.0, lambda_stretch=0.0, lambda_bend=0.0,
        fidelity_strength_mult=0.0, stretch_strength_mult=0.0, bend_strength_mult=0.0,
    )
    results["baseline"] = run_temporal(dataset, baseline_cfg, save_snapshots=False, verbose=False)
    bm = results["baseline"].final_metrics
    print(f"    sep={bm.get('sep_ratio_mean', float('nan')):.3f}  "
          f"spread={results['baseline'].final_metrics.get('within_bundle_spread_ratio', float('nan')):.3f}")

    # 3 regime conditions
    for regime in ("no_change", "moderate", "extreme"):
        p = picks.get(regime)
        if p is None:
            continue
        value = p["value"]
        print(f"  [{regime}] {sweep.field}={value:.4g} ...")
        dataset = load_initialization(init_path, variant="bifurcating_trunk",
                                      n_per_cluster=n_per_branch)
        cfg = TemporalRunConfig(
            k_attract=k_attract, delta=delta, lr=lr, n_iter=n_iter,
            fidelity_init_strength=0.0, epsilon_void=0.0, lambda_stretch=0.0, lambda_bend=0.0,
            fidelity_strength_mult=0.0, stretch_strength_mult=0.0, bend_strength_mult=0.0,
        )
        setattr(cfg, sweep.field, value)
        results[regime] = run_temporal(dataset, cfg, save_snapshots=False, verbose=False)
        fm = results[regime].final_metrics
        print(f"    sep={fm.get('sep_ratio_mean', float('nan')):.3f}  "
              f"spread={fm.get('within_bundle_spread_ratio', float('nan')):.3f}  "
              f"n_iter={results[regime].cond_result.n_iter}")

    return results


# ===========================================================================
# Section 3: Regime GIF
# ===========================================================================

def make_regime_gif(
    results: dict[str, TemporalRunResult],
    picks: dict[str, dict],
    sweep: ForceSweep,
    output_path: Path,
    n_frames: int = 72,
    elev: float = 25,
    fps_ms: int = 80,
) -> None:
    """4-column rotating 3D GIF: baseline | no_change | moderate | extreme.

    All panels share axis limits and rotate in sync.
    """
    regimes = [r for r in _REGIMES if r in results]
    n_cols = len(regimes)

    # --- Compute shared axis limits across all conditions ---
    all_positions = []
    for r in results.values():
        all_positions.append(r.cond_result.positions)
        all_positions.append(r.dataset.positions)  # always include x0
    all_pos = np.concatenate(all_positions, axis=0)  # (N_total, T, 2)
    pad = 0.6
    xlim = (float(all_pos[:, :, 0].min()) - pad, float(all_pos[:, :, 0].max()) + pad)
    zlim = (float(all_pos[:, :, 1].min()) - pad, float(all_pos[:, :, 1].max()) + pad)
    dataset_ref = next(iter(results.values())).dataset
    time_values = dataset_ref.time_values
    ylim = (float(time_values[0]) - 0.5, float(time_values[-1]) + 0.5)
    labels = dataset_ref.labels

    # --- Column titles ---
    def _col_title(regime: str) -> str:
        if regime == "baseline":
            return f"baseline\n(λ_rep=0.005)"
        p = picks.get(regime, {})
        val = p.get("value", float("nan"))
        dev = p.get("max_deviation", float("nan"))
        return f"{regime}\n{sweep.field.split('_mult')[0].split('_strength')[0]}={val:.3g}\n(Δ={dev:.1%})"

    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    frames: list[Image.Image] = []

    for az in azimuths:
        fig = plt.figure(figsize=(4.5 * n_cols, 5.5), dpi=100)
        for col, regime in enumerate(regimes):
            ax = fig.add_subplot(1, n_cols, col + 1, projection="3d")
            pos = results[regime].cond_result.positions
            title = _col_title(regime)
            _draw_3d_trunk(ax, pos, labels, time_values, title=title)
            # Override with shared limits
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.view_init(elev=elev, azim=az)
            # Color-code border by regime
            for spine in ax.spines.values():
                spine.set_edgecolor(_REGIME_COLORS.get(regime, "#333333"))

        fig.suptitle(f"Force family: {sweep.family}  |  {sweep.note}",
                     fontsize=9, y=1.01)
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
    print(f"  Saved: {output_path}  ({len(frames)} frames)")


# ===========================================================================
# Section 4: CLI + main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate regime GIFs from force discovery sweep results"
    )
    p.add_argument(
        "--sweep-csv",
        type=str,
        default=str(_HERE / "results" / "force_discovery_v1" / "sweep_results.csv"),
        help="Path to sweep_results.csv from force_discovery_sweep.py",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_HERE / "results" / "force_regime_v1"),
    )
    p.add_argument("--n-per-branch", type=int, default=40)
    p.add_argument("--n-time", type=int, default=13)
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--n-frames", type=int, default=72)
    p.add_argument("--fps-ms", type=int, default=80)
    p.add_argument("--k-attract", type=int, default=20)
    p.add_argument("--delta", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--split-start", type=int, default=4)
    p.add_argument("--split-full", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tau-nochange", type=float, default=0.02,
        help="Max relative deviation for no_change regime (default 0.02 = 2%%)",
    )
    p.add_argument(
        "--tau-moderate", type=float, default=0.05,
        help="Min relative deviation for moderate regime (default 0.05 = 5%%)",
    )
    p.add_argument(
        "--tau-extreme", type=float, default=0.15,
        help="Min relative deviation for extreme regime (default 0.15 = 15%%)",
    )
    p.add_argument(
        "--families", type=str, default=None,
        help="Comma-separated list of families to run (default: all in sweep CSV)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_csv = Path(args.sweep_csv)
    if not sweep_csv.exists():
        print(f"ERROR: sweep CSV not found: {sweep_csv}")
        print("Run force_discovery_sweep.py first to generate it.")
        return

    # --- Load sweep results ---
    df_all = pd.read_csv(sweep_csv)
    print(f"Loaded sweep results: {len(df_all)} rows, "
          f"families: {sorted(df_all['family'].unique())}")

    # --- Build sweep metadata (for field names, notes) ---
    sweeps = build_sweeps()
    sweep_by_family = {s.family: s for s in sweeps}

    if args.families is not None:
        keep = set(args.families.split(","))
        sweeps = [s for s in sweeps if s.family in keep]

    # --- Derive regime picks from sweep data ---
    print(f"\nDeriving regime picks "
          f"(τ_nochange={args.tau_nochange:.0%}, "
          f"τ_moderate={args.tau_moderate:.0%}, "
          f"τ_extreme={args.tau_extreme:.0%}) ...")
    all_picks = derive_all_picks(
        df_all, sweeps,
        tau_nochange=args.tau_nochange,
        tau_moderate=args.tau_moderate,
        tau_extreme=args.tau_extreme,
    )
    print_picks_table(all_picks)
    save_picks_csv(all_picks, output_dir / "threshold_picks.csv")

    # --- Shared initialization ---
    # Use the discovery run's init if it exists alongside the CSV, otherwise generate fresh
    init_path = sweep_csv.parent / "initialization.npz"
    if not init_path.exists():
        print(f"\nNo initialization.npz found next to sweep CSV. Generating fresh...")
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
        print(f"\nReusing initialization from sweep: {init_path}")

    # --- Generate one GIF per family ---
    for sweep in sweeps:
        picks = all_picks.get(sweep.family)
        if picks is None:
            print(f"\nNo picks for {sweep.family} — skipping.")
            continue

        print(f"\n=== {sweep.family} ===")
        results = run_regime_conditions(
            init_path=init_path,
            sweep=sweep,
            picks=picks,
            n_per_branch=args.n_per_branch,
            n_time=args.n_time,
            n_iter=args.n_iter,
            k_attract=args.k_attract,
            delta=args.delta,
            lr=args.lr,
        )

        gif_path = output_dir / f"regime_{sweep.family}.gif"
        print(f"  Rendering GIF ({args.n_frames} frames) ...")
        make_regime_gif(
            results=results,
            picks=picks,
            sweep=sweep,
            output_path=gif_path,
            n_frames=args.n_frames,
            fps_ms=args.fps_ms,
        )

    print(f"\nDone. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
