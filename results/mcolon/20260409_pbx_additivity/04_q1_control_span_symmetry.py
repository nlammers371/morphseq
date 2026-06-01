"""
Q1: Control-span symmetry check.

Per bootstrap rep, fit THREE targets in the same span V = [v_1 | v_4]:
  - z_ctrl  = mean(non_B) - mean(non_A)   <- control noise (the thing we're testing)
  - z_pbx1b = mu_pbx1b - mu_non_A         <- real pbx1b signal (reference)
  - z_pbx4  = mu_pbx4  - mu_non_A         <- real pbx4 signal (reference)

The control target is interpretable only when compared to the real-signal
coefficients in the same span and same rep. If control alpha/beta overlap
the real-signal distributions, the machinery cannot distinguish biology from noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR       = Path(__file__).resolve().parent
REPO_ROOT        = SCRIPT_DIR.parents[2]
PBX_ANALYSIS_DIR = REPO_ROOT / "results" / "mcolon" / "20260407_pbx_analysis_cont"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PBX_ANALYSIS_DIR))

from common import load_combined_pbx_dataframe
from analyze.utils.binning import add_time_bins

EXPERIMENT_IDS = ["20251207_pbx", "20260304", "20260306"]
GENOTYPES      = ["wik_ab", "inj_ctrl", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant"]
TIME_COL  = "stage_hpf"
CLASS_COL = "genotype"
ID_COL    = "embryo_id"
BIN_WIDTH = 4.0
N_BOOTSTRAP = 500
RANDOM_SEED = 42
MIN_N = 3

NON_INJ = "wik_ab"
PBX1B   = "pbx1b_crispant"
PBX4    = "pbx4_crispant"

RESULTS_DIR = SCRIPT_DIR / "results" / "q1_control_span_symmetry"
FIGURES_DIR = SCRIPT_DIR / "figures" / "q1_control_span_symmetry"

# target label -> display label, color
TARGETS = {
    "z_ctrl":  ("control noise",  "#808080"),
    "z_pbx1b": ("pbx1b (ref)",    "#9467BD"),
    "z_pbx4":  ("pbx4 (ref)",     "#F7B267"),
}


def _ols(V: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float, float]:
    c, _, _, _ = np.linalg.lstsq(V, z, rcond=None)
    r = z - V @ c
    denom = float(np.dot(z, z))
    r2 = 1.0 - float(np.dot(r, r)) / denom if denom > 0 else float("nan")
    return c, r2, float(np.linalg.norm(r))


def _load_and_bin() -> tuple[pd.DataFrame, list[str]]:
    df = load_combined_pbx_dataframe(experiment_ids=EXPERIMENT_IDS, genotypes=GENOTYPES)
    df = add_time_bins(df, time_col=TIME_COL, bin_width=BIN_WIDTH, bin_col="time_bin")
    feat_cols = [c for c in df.columns if "z_mu_b" in c]
    if not feat_cols:
        raise ValueError("No VAE feature columns found matching 'z_mu_b'.")
    binned = df.groupby([ID_COL, CLASS_COL, "time_bin"], as_index=False)[feat_cols].mean()
    return binned, feat_cols


def _valid_bins(binned: pd.DataFrame) -> list[int]:
    required = [NON_INJ, PBX1B, PBX4]
    valid = []
    for tb, grp in binned.groupby("time_bin"):
        counts = grp.groupby(CLASS_COL)[ID_COL].nunique()
        if all(counts.get(g, 0) >= MIN_N for g in required):
            valid.append(int(tb))
    return sorted(valid)


def _run_bootstrap(binned: pd.DataFrame, feat_cols: list[str], valid_bins: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []

    for tb in valid_bins:
        grp     = binned[binned["time_bin"] == tb]
        non_ids = grp.loc[grp[CLASS_COL] == NON_INJ, ID_COL].unique()
        feats   = grp.set_index(ID_COL)[feat_cols]

        # Full-group means for the real reference vectors (fixed per bin)
        mu_pbx1b = grp.loc[grp[CLASS_COL] == PBX1B, feat_cols].mean().to_numpy(dtype=float)
        mu_pbx4  = grp.loc[grp[CLASS_COL] == PBX4,  feat_cols].mean().to_numpy(dtype=float)

        for rep in range(N_BOOTSTRAP):
            if len(non_ids) < 2:
                continue
            perm = rng.permutation(len(non_ids))
            mid  = max(1, len(non_ids) // 2)
            non_A_ids = non_ids[perm[:mid]]
            non_B_ids = non_ids[perm[mid:]]

            mu_non_A = feats.loc[feats.index.isin(non_A_ids)].mean().to_numpy(dtype=float)
            mu_non_B = feats.loc[feats.index.isin(non_B_ids)].mean().to_numpy(dtype=float)

            # Span built from real group means, origin at non_A
            v_1 = mu_pbx1b - mu_non_A
            v_4 = mu_pbx4  - mu_non_A
            V   = np.column_stack([v_1, v_4])

            # Three targets fitted in the same span
            targets_z = {
                "z_ctrl":  mu_non_B - mu_non_A,   # control noise
                "z_pbx1b": v_1,                    # real pbx1b signal
                "z_pbx4":  v_4,                    # real pbx4 signal
            }
            for tname, z in targets_z.items():
                c, r2, res_norm = _ols(V, z)
                rows.append({
                    "time_bin":      tb,
                    "rep":           rep,
                    "target":        tname,
                    "alpha":         float(c[0]),
                    "beta":          float(c[1]),
                    "R2_span":       r2,
                    "residual_norm": res_norm,
                })

    return pd.DataFrame(rows)


def _violin(ax: plt.Axes, data: np.ndarray, pos: float, color: str, alpha: float = 0.6) -> None:
    """Single-series violinplot with explicit color — avoids multi-body color bug."""
    if len(data) < 2:
        return
    vp = ax.violinplot([data], positions=[pos], showmeans=True, showextrema=False)
    vp["bodies"][0].set_facecolor(color)
    vp["bodies"][0].set_edgecolor(color)
    vp["bodies"][0].set_alpha(alpha)
    vp["cmeans"].set_color(color)
    vp["cmeans"].set_linewidth(2.0)


def _plot_over_time(
    results: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Line + ribbon plot: mean ± SD per target over time bins. One panel, readable at any n_bins."""
    from matplotlib.patches import Patch
    target_keys = list(TARGETS.keys())

    fig, ax = plt.subplots(figsize=(12, 4))
    for t in target_keys:
        label, color = TARGETS[t]
        sub = results[results["target"] == t].copy()
        stats = (
            sub.groupby("time_bin")[col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("time_bin")
        )
        x   = stats["time_bin"].to_numpy(dtype=float) + BIN_WIDTH / 2.0
        mu  = stats["mean"].to_numpy(dtype=float)
        sd  = stats["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, mu, color=color, linewidth=2.0, label=label)
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18)

    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time bin center (hpf)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot(results: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: alpha and beta on same panel ────────────────────────────────
    # Solid line = alpha (pbx1b axis), dashed line = beta (pbx4 axis).
    # Dashed reference at 1.0 = "1 unit of that support vector".
    # Control noise ideally near 0 on both; real signals recover ~1 on their axis.
    fig, ax = plt.subplots(figsize=(12, 4))
    for tkey in list(TARGETS.keys()):
        label, color = TARGETS[tkey]
        sub = results[results["target"] == tkey].copy()
        for col, ls, coef_label in [("alpha", "-", "alpha"), ("beta", "--", "beta")]:
            stats = (
                sub.groupby("time_bin")[col]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("time_bin")
            )
            x  = stats["time_bin"].to_numpy(dtype=float) + BIN_WIDTH / 2.0
            mu = stats["mean"].to_numpy(dtype=float)
            sd = stats["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mu, color=color, linewidth=2.0, linestyle=ls,
                    label=f"{label} — {coef_label}")
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.12)

    ax.set_xlabel("Time bin center (hpf)")
    ax.set_ylabel("Coefficient (alpha solid, beta dashed)")
    ax.set_title(
        "Q1: alpha (—) and beta (- -) for each target over time\n"
        "Fit: z ≈ alpha·v_1 + beta·v_4,   v_1 = mu_pbx1b − mu_non,   v_4 = mu_pbx4 − mu_non\n"
        "control noise mapping to pbx1b axis = bias to investigate"
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "q1_alpha_beta_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'q1_alpha_beta_combined.png'}")

    _plot_over_time(
        results, col="R2_span",
        ylabel="R²_span = 1 − ||r||² / ||z||²",
        title=("Q1: How much of each target is explained by the pbx1b/pbx4 span (R²_span)?\n"
               "R²_span = 1 − ||z − V·c||² / ||z||²,   c = argmin ||z − V·c||\n"
               "real signals (purple/amber) → near 1 | control noise (gray) → low"),
        out_path=FIGURES_DIR / "q1_r2_control_vs_real.png",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    binned, feat_cols = _load_and_bin()
    valid_bins = _valid_bins(binned)
    print(f"Valid time bins: {valid_bins}")

    results = _run_bootstrap(binned, feat_cols, valid_bins)
    results.to_csv(RESULTS_DIR / "q1_control_span_symmetry.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'q1_control_span_symmetry.csv'}")

    _plot(results)


if __name__ == "__main__":
    main()
