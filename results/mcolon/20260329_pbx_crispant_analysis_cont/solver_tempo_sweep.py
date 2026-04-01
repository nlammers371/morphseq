"""
solver_tempo_sweep.py
---------------------
Calibrate the solver tempo (learning_rate × n_iter) on the Y-benchmark.

Goal
----
Find a (lr, n_iter) regime where:
  1. STABLE   — no explosion, oscillation, or spread inflation
  2. BANDWIDTH — weak forces have enough iterations to interact before being judged inert
  3. ECONOMICAL — cheap enough for routine use

A force that looks inert may just be under-integrated. This sweep distinguishes
"truly inert" from "not enough iterations."

Design
------
Grid:
  lr     : [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
  n_iter : [50, 100, 200, 300, 500, 800]

Three conditions per cell (all from same shared initialization):
  A: isotropic baseline     — coherence + repulsion only
  B: + weak fidelity        — fidelity_init_strength=1.0, fidelity_half_life_iters=70
  C: + weak elasticity      — stretch_strength_mult=0.001

Outputs
-------
  solver_tempo_sweep/
  ├── summary.csv                    — scalar metrics per (lr, n_iter, condition)
  ├── heatmap_branch_sep.png         — 2D grid: branch_sep_late, one panel per condition
  ├── heatmap_stability.png          — 2D grid: spread_ratio (stability proxy)
  ├── curves_lr<X>_niter<Y>.png      — per-iteration metric curves for selected cells
  └── regime_diagnosis.png           — annotated good/bad/unstable/under-integrated regions

Run (smoke test — fast, small grid):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/solver_tempo_sweep.py \\
      --output-dir /tmp/solver_tempo_smoke \\
      --n-per-branch 20 --seed 42 \\
      --lr-values 1e-5,1e-4,1e-3 --n-iter-values 50,200,500

Run (full):
    conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260329_pbx_crispant_analysis_cont/solver_tempo_sweep.py \\
      --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/solver_tempo_v1
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
morphseq_root = _here.parents[3]
sys.path.insert(0, str(morphseq_root))
sys.path.insert(0, str(_here))

from temporal_sandbox import (
    TemporalRunConfig,
    run_temporal,
    gamma_from_half_life_iters,
)
from bifurcating_trunk_sandbox import (
    make_bifurcating_trunk,
    save_initialization,
    load_initialization,
    trunk_summary_metrics,
)


# ---------------------------------------------------------------------------
# Sweep conditions
# ---------------------------------------------------------------------------

@dataclass
class SweepCondition:
    name: str
    label: str
    config_overrides: dict   # kwargs layered on top of base config


CONDITIONS = [
    SweepCondition(
        name="A_isotropic",
        label="A: isotropic",
        config_overrides=dict(fidelity_init_strength=0.0, epsilon_void=0.0),
    ),
    SweepCondition(
        name="B_fidelity",
        label="B: + weak fidelity\n(init_str=1.0, half_life=70 iters)",
        config_overrides=dict(
            fidelity_init_strength=1.0,
            fidelity_half_life_iters=70.0,
            epsilon_void=0.0,
        ),
    ),
    SweepCondition(
        name="C_elasticity",
        label="C: + weak elasticity\n(stretch_mult=0.001)",
        config_overrides=dict(
            fidelity_init_strength=0.0,
            epsilon_void=0.0,
            stretch_strength_mult=0.001,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Single cell runner
# ---------------------------------------------------------------------------

def run_cell(
    init_path: Path,
    lr: float,
    n_iter: int,
    condition: SweepCondition,
    n_per_branch: int,
    split_full: int,
    base_config_kwargs: dict,
) -> dict:
    """Run one (lr, n_iter, condition) cell. Returns scalar summary dict."""
    ds = load_initialization(init_path, variant="bifurcating_trunk",
                             n_per_cluster=n_per_branch)
    pos_init_fixed = ds.positions.copy()

    cfg = TemporalRunConfig(
        **base_config_kwargs,
        lr=lr,
        n_iter=n_iter,
        **condition.config_overrides,
    )

    result = run_temporal(ds, cfg, save_snapshots=True, verbose=False)
    pos_final = result.cond_result.positions

    summary = trunk_summary_metrics(pos_final, pos_init_fixed, ds.labels,
                                    split_full=split_full)

    # Stability: check spread ratio and max displacement
    metrics_h = result.cond_result.metrics_history
    disp_max_final = metrics_h[-1].get("disp_max_rel", float("nan")) if metrics_h else float("nan")
    disp_rms_final = metrics_h[-1].get("disp_rms_rel", float("nan")) if metrics_h else float("nan")
    # First-step displacement (stability indicator — iter 0 is the most revealing)
    disp_rms_step1 = metrics_h[0].get("disp_rms_rel", float("nan")) if metrics_h else float("nan")

    # Energy at final iter
    total_energy_final = metrics_h[-1].get("total", float("nan")) if metrics_h else float("nan")

    # Did any metric blow up? Check within_branch_spread_ratio > 3 as explosion proxy
    spread_ratio = summary.get("within_branch_spread_ratio", float("nan"))
    exploded = bool(spread_ratio > 3.0 or np.isnan(spread_ratio))

    # Per-iteration curves saved for later plotting
    metrics_df = pd.DataFrame(metrics_h)

    return {
        "lr": lr,
        "n_iter": n_iter,
        "condition": condition.name,
        "condition_label": condition.label,
        "branch_sep_late": summary["branch_sep_late"],
        "trunk_linearity_early": summary["trunk_linearity_early"],
        "within_branch_spread_ratio": spread_ratio,
        "disp_rms_step1": disp_rms_step1,
        "disp_rms_final": disp_rms_final,
        "disp_max_final": disp_max_final,
        "total_energy_final": total_energy_final,
        "exploded": exploded,
        "n_converged_iter": result.cond_result.n_iter,
        "_metrics_df": metrics_df,  # dropped before CSV save
    }


# ---------------------------------------------------------------------------
# Plotting: per-iteration curves for a set of cells
# ---------------------------------------------------------------------------

def plot_curves(
    rows: list[dict],
    lr_values: list[float],
    n_iter_values: list[int],
    output_dir: Path,
) -> None:
    """For each (lr, n_iter) cell, plot per-iteration curves for all 3 conditions."""
    color_map = {
        "A_isotropic": "#4D4D4D",
        "B_fidelity":  "#2166AC",
        "C_elasticity": "#D6604D",
    }
    # Group rows by (lr, n_iter)
    from collections import defaultdict
    cell_rows: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cell_rows[(r["lr"], r["n_iter"])].append(r)

    # One figure per cell — only plot cells that are informative (avoid huge output)
    # Select: all lr values × subset of n_iter
    selected_n_iters = n_iter_values  # plot all

    for lr in lr_values:
        for n_iter in selected_n_iters:
            cell = cell_rows.get((lr, n_iter), [])
            if not cell:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=False)
            ax_sep, ax_lin, ax_disp = axes

            for r in cell:
                mdf = r["_metrics_df"]
                if mdf.empty:
                    continue
                color = color_map.get(r["condition"], "gray")
                label = r["condition"]
                iters = mdf.get("iter", pd.Series(range(len(mdf))))

                if "disp_rms_rel" in mdf.columns:
                    ax_disp.semilogy(iters, mdf["disp_rms_rel"], color=color, lw=1.5,
                                     label=label)

            # Scalar final metrics as horizontal lines on sep/lin panels
            for r in cell:
                color = color_map.get(r["condition"], "gray")
                ax_sep.axhline(r["branch_sep_late"], color=color, lw=2.0, ls="-",
                               label=f"{r['condition']} sep={r['branch_sep_late']:.3f}")
                ax_lin.axhline(r["trunk_linearity_early"], color=color, lw=2.0, ls="-",
                               label=f"{r['condition']} lin={r['trunk_linearity_early']:.3f}")

            ax_sep.set_ylabel("branch_sep_late", fontsize=8, fontweight="bold")
            ax_lin.set_ylabel("trunk_lin_early", fontsize=8, fontweight="bold")
            ax_disp.set_ylabel("disp_rms_rel (log)", fontsize=8, fontweight="bold")
            ax_disp.set_xlabel("iteration", fontsize=8)

            for ax in axes:
                ax.legend(fontsize=7, loc="best")
                ax.grid(True, alpha=0.2, lw=0.5)

            lr_str = f"{lr:.0e}"
            exploded_any = any(r["exploded"] for r in cell)
            status = "  [EXPLODED]" if exploded_any else ""
            fig.suptitle(f"lr={lr_str}  n_iter={n_iter}{status}",
                         fontsize=10, fontweight="bold")
            fig.tight_layout()
            fname = output_dir / f"curves_lr{lr_str}_niter{n_iter}.png"
            fig.savefig(fname, dpi=110)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting: heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(
    df: pd.DataFrame,
    lr_values: list[float],
    n_iter_values: list[int],
    output_dir: Path,
) -> None:
    """2D heatmaps: rows=lr, cols=n_iter, one panel per condition."""
    metrics_to_plot = [
        ("branch_sep_late",            "branch_sep_late",            "Blues",   False),
        ("trunk_linearity_early",      "trunk_lin_early",            "Oranges", False),
        ("within_branch_spread_ratio", "spread_ratio\n(>3=exploded)", "RdYlGn_r", False),
        ("disp_rms_step1",             "disp_rms step1\n(stability)", "Reds",    True),
    ]

    for metric_col, metric_label, cmap, log_scale in metrics_to_plot:
        fig, axes = plt.subplots(1, len(CONDITIONS),
                                 figsize=(5 * len(CONDITIONS), 4.5),
                                 squeeze=False)
        fig.suptitle(f"{metric_label}  —  solver tempo sweep",
                     fontsize=11, fontweight="bold")

        for col_idx, cond in enumerate(CONDITIONS):
            ax = axes[0][col_idx]
            sub = df[df["condition"] == cond.name]

            grid = np.full((len(lr_values), len(n_iter_values)), float("nan"))
            for r_idx, lr in enumerate(lr_values):
                for c_idx, n_iter in enumerate(n_iter_values):
                    match = sub[(sub["lr"] == lr) & (sub["n_iter"] == n_iter)]
                    if not match.empty:
                        grid[r_idx, c_idx] = match[metric_col].values[0]

            if log_scale:
                with np.errstate(divide="ignore", invalid="ignore"):
                    grid = np.log10(np.where(grid > 0, grid, np.nan))
                label = f"log10({metric_label})"
            else:
                label = metric_label

            vmin = np.nanmin(grid)
            vmax = np.nanmax(grid)
            im = ax.imshow(grid, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, origin="upper")

            ax.set_xticks(range(len(n_iter_values)))
            ax.set_xticklabels([str(n) for n in n_iter_values], fontsize=8, rotation=30)
            ax.set_yticks(range(len(lr_values)))
            ax.set_yticklabels([f"{lr:.0e}" for lr in lr_values], fontsize=8)
            ax.set_xlabel("n_iter", fontsize=8)
            ax.set_ylabel("lr", fontsize=8)
            ax.set_title(cond.name, fontsize=8, fontweight="bold")

            # Annotate cells
            for r_idx in range(len(lr_values)):
                for c_idx in range(len(n_iter_values)):
                    v = grid[r_idx, c_idx]
                    if not np.isnan(v):
                        txt = f"{v:.2f}" if not log_scale else f"10^{v:.1f}"
                        ax.text(c_idx, r_idx, txt, ha="center", va="center",
                                fontsize=6, color="black")

            plt.colorbar(im, ax=ax, label=label, shrink=0.8)

        fig.tight_layout()
        safe_col = metric_col.replace("/", "_")
        fig.savefig(output_dir / f"heatmap_{safe_col}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: heatmap_{safe_col}.png")


# ---------------------------------------------------------------------------
# Regime diagnosis plot
# ---------------------------------------------------------------------------

def plot_regime_diagnosis(
    df: pd.DataFrame,
    lr_values: list[float],
    n_iter_values: list[int],
    output_dir: Path,
) -> None:
    """Annotated regime map: classify each (lr, n_iter) cell as good/bad/unstable/under-integrated.

    Classification per cell (using condition A isotropic as stability reference):
      UNSTABLE:          spread_ratio > 3  OR  disp_rms_step1 > 0.5
      GOOD:              spread_ratio <= 1.5  AND  branch_sep >= 1.35  AND  disp_rms_final < 0.01
      UNDER-INTEGRATED:  stable but branch_sep < 1.35 at small n_iter  (forces not expressing)
      OK:                everything else
    """
    cmap_colors = {
        "UNSTABLE":         "#B2182B",
        "GOOD":             "#4DAC26",
        "UNDER-INTEGRATED": "#F7B267",
        "OK":               "#92C5DE",
    }

    fig, axes = plt.subplots(1, len(CONDITIONS),
                             figsize=(5 * len(CONDITIONS), 4.5), squeeze=False)
    fig.suptitle("Solver tempo regime diagnosis  (lr × n_iter)",
                 fontsize=11, fontweight="bold")

    for col_idx, cond in enumerate(CONDITIONS):
        ax = axes[0][col_idx]
        sub = df[df["condition"] == cond.name]

        for r_idx, lr in enumerate(lr_values):
            for c_idx, n_iter in enumerate(n_iter_values):
                match = sub[(sub["lr"] == lr) & (sub["n_iter"] == n_iter)]
                if match.empty:
                    continue
                row = match.iloc[0]

                spread = row["within_branch_spread_ratio"]
                sep    = row["branch_sep_late"]
                disp1  = row["disp_rms_step1"]
                dispf  = row["disp_rms_final"]

                if spread > 3.0 or disp1 > 0.5 or np.isnan(spread):
                    regime = "UNSTABLE"
                elif spread <= 1.5 and sep >= 1.35 and dispf < 0.01:
                    regime = "GOOD"
                elif spread <= 2.0 and n_iter <= 100 and sep < 1.35:
                    regime = "UNDER-INTEGRATED"
                else:
                    regime = "OK"

                color = cmap_colors[regime]
                rect = plt.Rectangle((c_idx - 0.5, r_idx - 0.5), 1, 1,
                                     facecolor=color, alpha=0.75, edgecolor="white", lw=0.8)
                ax.add_patch(rect)
                ax.text(c_idx, r_idx,
                        f"{regime[:4]}\nsep={sep:.2f}\nspd={spread:.2f}",
                        ha="center", va="center", fontsize=5.5, color="black")

        ax.set_xlim(-0.5, len(n_iter_values) - 0.5)
        ax.set_ylim(-0.5, len(lr_values) - 0.5)
        ax.set_xticks(range(len(n_iter_values)))
        ax.set_xticklabels([str(n) for n in n_iter_values], fontsize=8, rotation=30)
        ax.set_yticks(range(len(lr_values)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lr_values], fontsize=8)
        ax.set_xlabel("n_iter", fontsize=8)
        ax.set_ylabel("lr", fontsize=8)
        ax.set_title(cond.name, fontsize=8, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=k, alpha=0.75)
                      for k, c in cmap_colors.items()]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.0), frameon=True)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_dir / "regime_diagnosis.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: regime_diagnosis.png")


# ---------------------------------------------------------------------------
# Force distinguishability plot
# ---------------------------------------------------------------------------

def plot_distinguishability(
    df: pd.DataFrame,
    lr_values: list[float],
    n_iter_values: list[int],
    output_dir: Path,
) -> None:
    """Show delta(branch_sep) = condition - isotropic as heatmap.

    This directly answers: at which (lr, n_iter) does each force become
    distinguishable from the isotropic baseline?
    """
    iso = df[df["condition"] == "A_isotropic"].set_index(["lr", "n_iter"])

    non_iso = [c for c in CONDITIONS if c.name != "A_isotropic"]
    fig, axes = plt.subplots(1, len(non_iso),
                             figsize=(5.5 * len(non_iso), 4.5), squeeze=False)
    fig.suptitle("Force distinguishability: Δbranch_sep vs isotropic baseline\n"
                 "Positive = force helps; near-zero = force inert at this tempo",
                 fontsize=10, fontweight="bold")

    for col_idx, cond in enumerate(non_iso):
        ax = axes[0][col_idx]
        sub = df[df["condition"] == cond.name].set_index(["lr", "n_iter"])

        grid = np.full((len(lr_values), len(n_iter_values)), float("nan"))
        for r_idx, lr in enumerate(lr_values):
            for c_idx, n_iter in enumerate(n_iter_values):
                try:
                    sep_cond = sub.loc[(lr, n_iter), "branch_sep_late"]
                    sep_iso  = iso.loc[(lr, n_iter), "branch_sep_late"]
                    grid[r_idx, c_idx] = sep_cond - sep_iso
                except KeyError:
                    pass

        vabs = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.01)
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r",
                       vmin=-vabs, vmax=vabs, origin="upper")

        ax.set_xticks(range(len(n_iter_values)))
        ax.set_xticklabels([str(n) for n in n_iter_values], fontsize=8, rotation=30)
        ax.set_yticks(range(len(lr_values)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lr_values], fontsize=8)
        ax.set_xlabel("n_iter", fontsize=8)
        ax.set_ylabel("lr", fontsize=8)
        ax.set_title(f"Δsep: {cond.name} − isotropic", fontsize=8, fontweight="bold")

        for r_idx in range(len(lr_values)):
            for c_idx in range(len(n_iter_values)):
                v = grid[r_idx, c_idx]
                if not np.isnan(v):
                    ax.text(c_idx, r_idx, f"{v:+.3f}", ha="center", va="center",
                            fontsize=6.5, color="black")

        plt.colorbar(im, ax=ax, label="Δbranch_sep", shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_dir / "distinguishability.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: distinguishability.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solver tempo sweep: lr × n_iter calibration.")
    p.add_argument("--output-dir",
                   default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/solver_tempo_v1")
    p.add_argument("--n-per-branch", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-full", type=int, default=8)
    p.add_argument("--lr-values", type=str, default="1e-5,3e-5,1e-4,3e-4,1e-3",
                   help="Comma-separated learning rate values")
    p.add_argument("--n-iter-values", type=str, default="50,100,200,300,500,800",
                   help="Comma-separated iteration counts")
    p.add_argument("--k-attract", type=int, default=20)
    p.add_argument("--delta", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lr_values    = [float(x) for x in args.lr_values.split(",")]
    n_iter_values = [int(x) for x in args.n_iter_values.split(",")]

    print(f"lr values:    {lr_values}")
    print(f"n_iter values: {n_iter_values}")
    print(f"Conditions:   {[c.name for c in CONDITIONS]}")
    total_runs = len(lr_values) * len(n_iter_values) * len(CONDITIONS)
    print(f"Total runs:   {total_runs}")

    # Generate and save shared initialization
    print("\nGenerating bifurcating trunk dataset...")
    dataset = make_bifurcating_trunk(
        n_per_branch=args.n_per_branch,
        n_time=13,
        split_start=4,
        split_full=args.split_full,
        random_seed=args.seed,
    )
    init_path = output_dir / "initialization.npz"
    save_initialization(dataset, init_path)
    print(f"  Saved initialization: {init_path}")

    base_config_kwargs = dict(
        k_attract=args.k_attract,
        delta=args.delta,
    )

    rows = []
    run_idx = 0
    for lr in lr_values:
        for n_iter in n_iter_values:
            for cond in CONDITIONS:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] lr={lr:.0e}  n_iter={n_iter}  {cond.name}")
                row = run_cell(
                    init_path=init_path,
                    lr=lr,
                    n_iter=n_iter,
                    condition=cond,
                    n_per_branch=args.n_per_branch,
                    split_full=args.split_full,
                    base_config_kwargs=base_config_kwargs,
                )
                print(f"  sep={row['branch_sep_late']:.4f}  "
                      f"lin={row['trunk_linearity_early']:.4f}  "
                      f"spread={row['within_branch_spread_ratio']:.3f}  "
                      f"disp1={row['disp_rms_step1']:.4f}  "
                      f"{'EXPLODED' if row['exploded'] else 'ok'}")
                rows.append(row)

    # Strip internal dataframes before saving CSV
    csv_rows = [{k: v for k, v in r.items() if k != "_metrics_df"} for r in rows]
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\n  Saved: summary.csv  ({len(df)} rows)")

    print("\nGenerating heatmaps...")
    plot_heatmaps(df, lr_values, n_iter_values, output_dir)

    print("Generating regime diagnosis...")
    plot_regime_diagnosis(df, lr_values, n_iter_values, output_dir)

    print("Generating distinguishability plot...")
    plot_distinguishability(df, lr_values, n_iter_values, output_dir)

    print("Generating per-cell curves...")
    plot_curves(rows, lr_values, n_iter_values, output_dir)

    # Print summary table
    print("\n=== Regime summary (branch_sep_late, A: isotropic) ===")
    iso_df = df[df["condition"] == "A_isotropic"][
        ["lr", "n_iter", "branch_sep_late", "within_branch_spread_ratio",
         "disp_rms_step1", "exploded"]
    ]
    print(iso_df.to_string(index=False))

    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
