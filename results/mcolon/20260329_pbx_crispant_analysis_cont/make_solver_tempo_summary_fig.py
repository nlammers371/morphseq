"""
make_solver_tempo_summary_fig.py
---------------------------------
Reads solver_tempo/summary.csv and produces a focused summary figure
explaining why lr=1e-4, n_iter=500 was chosen as the calibrated regime.

Output: force_calibration_v1/solver_tempo_selection.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LR_ORDER = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
LR_LABELS = {1e-5: "1e-5", 3e-5: "3e-5", 1e-4: "1e-4 ✓", 3e-4: "3e-4", 1e-3: "1e-3"}
LR_COLORS = {1e-5: "#4393c3", 3e-5: "#92c5de", 1e-4: "#d6604d", 3e-4: "#f4a582", 1e-3: "#b2182b"}

CHOSEN_LR = 1e-4
CHOSEN_NITER = 500


def load_data(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    # Normalize lr column (float parsing)
    df["lr"] = df["lr"].astype(float)
    df["n_iter"] = df["n_iter"].astype(int)
    return df


def plot_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Three-panel summary figure.

    Panel 1: branch_sep_late vs n_iter for each lr — isotropic condition.
             Shows where metrics plateau (convergence).
    Panel 2: Distinguishability = sep(fidelity) - sep(isotropic) vs n_iter per lr.
             Shows where forces become "readable" to the solver.
    Panel 3: disp_rms_step1 vs lr — shows where lr is too aggressive (explosion risk).
             Horizontal band marks 5× isotropic disp as "danger zone".
    """
    iso = df[df["condition"] == "A_isotropic"].copy()
    fid = df[df["condition"] == "B_fidelity"].copy()

    # Compute distinguishability
    merged = iso[["lr", "n_iter", "branch_sep_late"]].merge(
        fid[["lr", "n_iter", "branch_sep_late"]],
        on=["lr", "n_iter"], suffixes=("_iso", "_fid")
    )
    merged["delta_sep"] = merged["branch_sep_late_fid"] - merged["branch_sep_late_iso"]

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    n_iter_vals = sorted(iso["n_iter"].unique())

    # --- Panel 1: branch_sep_late vs n_iter (isotropic) ---
    for lr_val in LR_ORDER:
        sub = iso[np.isclose(iso["lr"], lr_val)].sort_values("n_iter")
        if sub.empty:
            continue
        lw = 2.5 if np.isclose(lr_val, CHOSEN_LR) else 1.2
        ls = "-" if np.isclose(lr_val, CHOSEN_LR) else "--"
        ax1.plot(sub["n_iter"], sub["branch_sep_late"],
                 color=LR_COLORS[lr_val], lw=lw, ls=ls, marker="o", ms=4,
                 label=LR_LABELS[lr_val])

    ax1.set_xlabel("n_iter", fontsize=9)
    ax1.set_ylabel("branch_sep_late (isotropic)", fontsize=9)
    ax1.set_title("Metric plateau: isotropic condition", fontsize=9, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7, title="lr", title_fontsize=7)

    # Annotate chosen point
    chosen_iso = iso[np.isclose(iso["lr"], CHOSEN_LR) & (iso["n_iter"] == CHOSEN_NITER)]
    if not chosen_iso.empty:
        ax1.scatter(chosen_iso["n_iter"], chosen_iso["branch_sep_late"],
                    s=80, zorder=5, color=LR_COLORS[CHOSEN_LR], edgecolors="black", lw=1.5)

    # --- Panel 2: distinguishability vs n_iter ---
    for lr_val in LR_ORDER:
        sub = merged[np.isclose(merged["lr"], lr_val)].sort_values("n_iter")
        if sub.empty:
            continue
        lw = 2.5 if np.isclose(lr_val, CHOSEN_LR) else 1.2
        ls = "-" if np.isclose(lr_val, CHOSEN_LR) else "--"
        ax2.plot(sub["n_iter"], sub["delta_sep"],
                 color=LR_COLORS[lr_val], lw=lw, ls=ls, marker="o", ms=4,
                 label=LR_LABELS[lr_val])

    ax2.axhline(0.0, color="gray", lw=1.0, ls=":")
    ax2.set_xlabel("n_iter", fontsize=9)
    ax2.set_ylabel("Δbranch_sep (fidelity − isotropic)", fontsize=9)
    ax2.set_title("Distinguishability: fidelity vs isotropic", fontsize=9, fontweight="bold")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7, title="lr", title_fontsize=7)

    # Annotate chosen point
    chosen_m = merged[np.isclose(merged["lr"], CHOSEN_LR) & (merged["n_iter"] == CHOSEN_NITER)]
    if not chosen_m.empty:
        ax2.scatter(chosen_m["n_iter"], chosen_m["delta_sep"],
                    s=80, zorder=5, color=LR_COLORS[CHOSEN_LR], edgecolors="black", lw=1.5)

    # --- Panel 3: disp_rms_step1 vs lr (at chosen n_iter) ---
    step1 = iso[iso["n_iter"] == CHOSEN_NITER].sort_values("lr")
    if step1.empty:
        # fallback to n_iter=500 or max available
        step1 = iso[iso["n_iter"] == iso["n_iter"].max()].sort_values("lr")

    ax3.plot(step1["lr"], step1["disp_rms_step1"], "o-", color="#555", lw=1.8, ms=5)

    # Danger zone: disp_rms_step1 > 5× minimum
    min_disp = step1["disp_rms_step1"].min()
    danger_thresh = min_disp * 5.0
    ax3.axhline(danger_thresh, color="#b2182b", lw=1.2, ls="--", alpha=0.8,
                label=f"5× min ({danger_thresh:.2e})")
    ax3.axvspan(CHOSEN_LR * 0.7, CHOSEN_LR * 1.5, alpha=0.12, color="green",
                label=f"chosen lr={CHOSEN_LR:.0e}")

    # Annotate chosen
    chosen_s = step1[np.isclose(step1["lr"], CHOSEN_LR)]
    if not chosen_s.empty:
        ax3.scatter(chosen_s["lr"], chosen_s["disp_rms_step1"],
                    s=80, zorder=5, color=LR_COLORS[CHOSEN_LR], edgecolors="black", lw=1.5)

    ax3.set_xscale("log")
    ax3.set_xlabel("learning rate (lr)", fontsize=9)
    ax3.set_ylabel("disp_rms at step 1", fontsize=9)
    ax3.set_title(f"Step-size safety (n_iter={CHOSEN_NITER})", fontsize=9, fontweight="bold")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=7)

    fig.suptitle(
        f"Solver tempo selection — chosen: lr={CHOSEN_LR:.0e}, n_iter={CHOSEN_NITER}\n"
        "Goal: slow, stable regime where weak forces have bandwidth to interact",
        fontsize=10, fontweight="bold", y=1.01,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv",
                   default="results/mcolon/20260329_pbx_crispant_analysis_cont/"
                           "results/force_calibration_v1/solver_tempo/summary.csv")
    p.add_argument("--output",
                   default="results/mcolon/20260329_pbx_crispant_analysis_cont/"
                           "results/force_calibration_v1/solver_tempo_selection.png")
    args = p.parse_args()

    df = load_data(Path(args.summary_csv))
    plot_summary(df, Path(args.output))


if __name__ == "__main__":
    main()
