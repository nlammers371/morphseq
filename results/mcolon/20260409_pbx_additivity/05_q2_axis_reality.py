"""
Q2: Axis reality check.

Are the pbx1b and pbx4 single-gene axes stronger and more structured
than axes that arise from control-only sampling noise?

The matched reference task for each metric:
  - norm:   real pbx1b/pbx4 norms vs distribution of fake-axis norms (inj vs non splits)
  - cosine: real cosine(v_1, v_4) vs distribution of fake-axis cosines
  - cond:   real condition number of [v_1|v_4] vs fake condition numbers

Each real metric is a single value plotted against the bootstrap null distribution
of the same metric computed from control-only splits. This makes the comparison
concrete: is the real structure outside the noise floor?
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

NON_INJ  = "wik_ab"
INJ_CTRL = "inj_ctrl"
PBX1B    = "pbx1b_crispant"
PBX4     = "pbx4_crispant"

RESULTS_DIR = SCRIPT_DIR / "results" / "q2_axis_reality"
FIGURES_DIR = SCRIPT_DIR / "figures" / "q2_axis_reality"


def _load_and_bin() -> tuple[pd.DataFrame, list[str]]:
    df = load_combined_pbx_dataframe(experiment_ids=EXPERIMENT_IDS, genotypes=GENOTYPES)
    df = add_time_bins(df, time_col=TIME_COL, bin_width=BIN_WIDTH, bin_col="time_bin")
    feat_cols = [c for c in df.columns if "z_mu_b" in c]
    if not feat_cols:
        raise ValueError("No VAE feature columns found matching 'z_mu_b'.")
    binned = df.groupby([ID_COL, CLASS_COL, "time_bin"], as_index=False)[feat_cols].mean()
    return binned, feat_cols


def _valid_bins(binned: pd.DataFrame) -> list[int]:
    required = [NON_INJ, INJ_CTRL, PBX1B, PBX4]
    valid = []
    for tb, grp in binned.groupby("time_bin"):
        counts = grp.groupby(CLASS_COL)[ID_COL].nunique()
        if all(counts.get(g, 0) >= MIN_N for g in required):
            valid.append(int(tb))
    return sorted(valid)


def _axis_summaries(v1: np.ndarray, v2: np.ndarray) -> dict:
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    denom = norm1 * norm2
    cosine = float(np.dot(v1, v2) / denom) if denom > 0 else float("nan")
    V = np.column_stack([v1, v2])
    _, sv, _ = np.linalg.svd(V, full_matrices=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return {"norm_v1": norm1, "norm_v2": norm2, "cosine": cosine, "cond": cond}


def _run_q2a(binned: pd.DataFrame, feat_cols: list[str], valid_bins: list[int]) -> pd.DataFrame:
    """Real axis summaries — one row per bin, no bootstrap."""
    rows = []
    for tb in valid_bins:
        grp      = binned[binned["time_bin"] == tb]
        mu_non   = grp.loc[grp[CLASS_COL] == NON_INJ, feat_cols].mean().to_numpy(dtype=float)
        mu_pbx1b = grp.loc[grp[CLASS_COL] == PBX1B,   feat_cols].mean().to_numpy(dtype=float)
        mu_pbx4  = grp.loc[grp[CLASS_COL] == PBX4,    feat_cols].mean().to_numpy(dtype=float)
        v1 = mu_pbx1b - mu_non
        v4 = mu_pbx4  - mu_non
        s  = _axis_summaries(v1, v4)
        rows.append({"time_bin": tb, "norm_pbx1b": s["norm_v1"], "norm_pbx4": s["norm_v2"],
                     "cosine": s["cosine"], "cond": s["cond"]})
    return pd.DataFrame(rows)


def _run_q2b(binned: pd.DataFrame, feat_cols: list[str], valid_bins: list[int]) -> pd.DataFrame:
    """Fake-axis null distribution — same metrics from control-only splits."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for tb in valid_bins:
        grp     = binned[binned["time_bin"] == tb]
        inj_ids = grp.loc[grp[CLASS_COL] == INJ_CTRL, ID_COL].unique()
        non_ids = grp.loc[grp[CLASS_COL] == NON_INJ,  ID_COL].unique()
        feats   = grp.set_index(ID_COL)[feat_cols]

        for rep in range(N_BOOTSTRAP):
            if len(inj_ids) < 2 or len(non_ids) < 2:
                continue
            perm_inj = rng.permutation(len(inj_ids))
            perm_non = rng.permutation(len(non_ids))
            mid_inj  = max(1, len(inj_ids) // 2)
            mid_non  = max(1, len(non_ids) // 2)

            mu_inj_A = feats.loc[feats.index.isin(inj_ids[perm_inj[:mid_inj]])].mean().to_numpy(dtype=float)
            mu_inj_B = feats.loc[feats.index.isin(inj_ids[perm_inj[mid_inj:]])].mean().to_numpy(dtype=float)
            mu_non_A = feats.loc[feats.index.isin(non_ids[perm_non[:mid_non]])].mean().to_numpy(dtype=float)
            mu_non_B = feats.loc[feats.index.isin(non_ids[perm_non[mid_non:]])].mean().to_numpy(dtype=float)

            # Two fake axes: each is an inj_ctrl-half minus a non_inj-half
            v_fake1 = mu_inj_A - mu_non_A
            v_fake2 = mu_inj_B - mu_non_B
            s = _axis_summaries(v_fake1, v_fake2)
            rows.append({"time_bin": tb, "rep": rep,
                         "norm_fake1": s["norm_v1"], "norm_fake2": s["norm_v2"],
                         "cosine": s["cosine"], "cond": s["cond"]})
    return pd.DataFrame(rows)


def _plot_time_series(
    q2a: pd.DataFrame,
    q2b: pd.DataFrame,
    real_cols: list[str],
    fake_cols: list[str],
    colors: list[str],
    real_labels: list[str],
    ylabel: str,
    title: str,
    out_path: "Path",
    hline: float | None = None,
) -> None:
    """
    Time-series line+ribbon for Q2.

    For each real_col/color/label:  plot q2a mean (single value per bin) as a line.
    For fake_cols pooled together:  plot q2b mean ± SD ribbon in gray.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 4))

    # ── fake-axis null ribbon (pooled across fake_cols) ──────────────────────
    fake_long = pd.concat(
        [q2b[["time_bin", c]].rename(columns={c: "_val"}) for c in fake_cols],
        ignore_index=True,
    )
    fake_stats = (
        fake_long.groupby("time_bin")["_val"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("time_bin")
    )
    xf = fake_stats["time_bin"].to_numpy(dtype=float) + BIN_WIDTH / 2.0
    muf = fake_stats["mean"].to_numpy(dtype=float)
    sdf = fake_stats["std"].fillna(0.0).to_numpy(dtype=float)
    ax.fill_between(xf, muf - sdf, muf + sdf, color="#AAAAAA", alpha=0.30, label="fake null (mean ± SD)")
    ax.plot(xf, muf, color="#AAAAAA", linewidth=1.5, linestyle="--")

    # ── real values per bin ───────────────────────────────────────────────────
    bins_sorted = sorted(q2a["time_bin"].unique())
    x_real = np.array(bins_sorted, dtype=float) + BIN_WIDTH / 2.0
    for col, color, label in zip(real_cols, colors, real_labels):
        y_real = np.array([float(q2a.loc[q2a["time_bin"] == tb, col].iloc[0]) for tb in bins_sorted])
        ax.plot(x_real, y_real, color=color, linewidth=2.5, marker="o", markersize=4, label=label)

    if hline is not None:
        ax.axhline(hline, color="#444444", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Time bin center (hpf)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot(q2a: pd.DataFrame, q2b: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: norm over time ───────────────────────────────────────────────
    # v_1 = mu_pbx1b - mu_non,  v_4 = mu_pbx4 - mu_non
    # Fake axes: v_fake1 = mu_inj_A - mu_non_A,  v_fake2 = mu_inj_B - mu_non_B
    # Real norms should sit above fake-axis null at meaningful time bins.
    _plot_time_series(
        q2a, q2b,
        real_cols=["norm_pbx1b", "norm_pbx4"],
        fake_cols=["norm_fake1", "norm_fake2"],
        colors=["#9467BD", "#F7B267"],
        real_labels=["||v_1|| = ||mu_pbx1b - mu_non||", "||v_4|| = ||mu_pbx4 - mu_non||"],
        ylabel="Axis norm ||v||",
        title=("Q2: Axis norms over time — real vs fake-axis null\n"
               "v_1 = mu_pbx1b − mu_non,   v_4 = mu_pbx4 − mu_non\n"
               "fake axes: v_fake = mu_inj_half − mu_non_half   (gray ribbon = mean ± SD)"),
        out_path=FIGURES_DIR / "q2_norm_comparison.png",
        hline=0.0,
    )

    # ── Figure 2: cosine over time ─────────────────────────────────────────────
    # cos(v_1, v_4) — real vs fake-axis null cosine(v_fake1, v_fake2).
    # Lower real cosine = more independent axes = better decomposition.
    _plot_time_series(
        q2a, q2b,
        real_cols=["cosine"],
        fake_cols=["cosine"],
        colors=["#B2182B"],
        real_labels=["cos(v_1, v_4) = (v_1·v_4) / (||v_1|| ||v_4||)"],
        ylabel="Cosine similarity",
        title=("Q2: cos(v_1, v_4) over time — real vs fake-axis null\n"
               "real cos = (v_1·v_4) / (||v_1|| ||v_4||)\n"
               "real line below null → axes more independent than noise (good)"),
        out_path=FIGURES_DIR / "q2_cosine_comparison.png",
        hline=0.0,
    )

    # ── Figure 3: condition number over time ───────────────────────────────────
    # kappa(V) = sigma_max / sigma_min of V = [v_1 | v_4].
    # Lower kappa = more separable axes = more stable coefficient attribution.
    q2b_finite = q2b.copy()
    q2b_finite["cond"] = q2b_finite["cond"].replace([np.inf, -np.inf], np.nan)
    _plot_time_series(
        q2a, q2b_finite,
        real_cols=["cond"],
        fake_cols=["cond"],
        colors=["#B2182B"],
        real_labels=["kappa(V) = sigma_max / sigma_min,   V = [v_1 | v_4]"],
        ylabel="Condition number kappa(V)",
        title=("Q2: Condition number over time — real vs fake-axis null\n"
               "kappa([v_1|v_4]) = sigma_max / sigma_min\n"
               "real line below null → axes more separable than noise (good)"),
        out_path=FIGURES_DIR / "q2_cond_comparison.png",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    binned, feat_cols = _load_and_bin()
    valid_bins = _valid_bins(binned)
    print(f"Valid time bins: {valid_bins}")

    q2a = _run_q2a(binned, feat_cols, valid_bins)
    q2a.to_csv(RESULTS_DIR / "q2a_real_axes.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'q2a_real_axes.csv'}")

    q2b = _run_q2b(binned, feat_cols, valid_bins)
    q2b.to_csv(RESULTS_DIR / "q2b_fake_axes.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'q2b_fake_axes.csv'}")

    _plot(q2a, q2b)


if __name__ == "__main__":
    main()
