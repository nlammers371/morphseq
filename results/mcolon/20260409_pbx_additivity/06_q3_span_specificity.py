"""
Q3: Span specificity check.

The matched reference task: fit each target in BOTH the real span AND a fake
control span built from the same bootstrap rep. This directly answers:
"Does the real span do something a noise span cannot?"

Per bootstrap rep:
  - Real span  V_real = [mean(pbx1b_A)-mean(non_A) | mean(pbx4_A)-mean(non_A)]
  - Fake span  V_fake = [mean(inj_A)-mean(non_A)   | mean(inj_B)-mean(non_B)]

Targets (from B halves):
  - pbx1b_heldout = mean(pbx1b_B) - mean(non_B)
  - pbx4_heldout  = mean(pbx4_B)  - mean(non_B)
  - inj_ctrl      = mean(inj_B)   - mean(non_B)   (same inj_B used to build fake span)

For each target, record R2_span under both real and fake span.
The key comparison: real_span_R2 vs fake_span_R2, for each target type.

Expected:
  pbx1b_heldout: real R2 >> fake R2
  pbx4_heldout:  real R2 >> fake R2
  inj_ctrl:      real R2 ≈ fake R2  (neither span has a special claim on control noise)
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

RESULTS_DIR = SCRIPT_DIR / "results" / "q3_span_specificity"
FIGURES_DIR = SCRIPT_DIR / "figures" / "q3_span_specificity"

TARGET_COLORS = {
    "pbx1b_heldout": "#9467BD",
    "pbx4_heldout":  "#F7B267",
    "inj_ctrl":      "#2166AC",
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
    required = [NON_INJ, INJ_CTRL, PBX1B, PBX4]
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
        grp = binned[binned["time_bin"] == tb]
        non_ids   = grp.loc[grp[CLASS_COL] == NON_INJ,  ID_COL].unique()
        inj_ids   = grp.loc[grp[CLASS_COL] == INJ_CTRL, ID_COL].unique()
        pbx1b_ids = grp.loc[grp[CLASS_COL] == PBX1B,    ID_COL].unique()
        pbx4_ids  = grp.loc[grp[CLASS_COL] == PBX4,     ID_COL].unique()
        feats     = grp.set_index(ID_COL)[feat_cols]

        def mean_of(ids: np.ndarray) -> np.ndarray:
            return feats.loc[feats.index.isin(ids)].mean().to_numpy(dtype=float)

        def split(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            perm = rng.permutation(len(ids))
            mid  = max(1, len(ids) // 2)
            return ids[perm[:mid]], ids[perm[mid:]]

        for rep in range(N_BOOTSTRAP):
            if any(len(ids) < 2 for ids in [non_ids, inj_ids, pbx1b_ids, pbx4_ids]):
                continue

            non_A,    non_B    = split(non_ids)
            pbx1b_A,  pbx1b_B  = split(pbx1b_ids)
            pbx4_A,   pbx4_B   = split(pbx4_ids)
            inj_A,    inj_B    = split(inj_ids)

            mu_non_A   = mean_of(non_A)
            mu_non_B   = mean_of(non_B)
            mu_pbx1b_A = mean_of(pbx1b_A)
            mu_pbx4_A  = mean_of(pbx4_A)
            mu_pbx1b_B = mean_of(pbx1b_B)
            mu_pbx4_B  = mean_of(pbx4_B)
            mu_inj_A   = mean_of(inj_A)
            mu_inj_B   = mean_of(inj_B)

            # Real span: built from pbx1b/pbx4 A halves
            V_real = np.column_stack([mu_pbx1b_A - mu_non_A, mu_pbx4_A - mu_non_A])

            # Fake span: built from inj_ctrl halves — matched control reference
            V_fake = np.column_stack([mu_inj_A - mu_non_A, mu_inj_B - mu_non_B])

            # Held-out targets from B halves
            targets = {
                "pbx1b_heldout": mu_pbx1b_B - mu_non_B,
                "pbx4_heldout":  mu_pbx4_B  - mu_non_B,
                "inj_ctrl":      mu_inj_B   - mu_non_B,
            }

            for tname, z in targets.items():
                c_real, r2_real, res_real = _ols(V_real, z)
                c_fake, r2_fake, res_fake = _ols(V_fake, z)
                rows.append({
                    "time_bin":          tb,
                    "rep":               rep,
                    "target":            tname,
                    # real span
                    "alpha_real":        float(c_real[0]),
                    "beta_real":         float(c_real[1]),
                    "R2_real":           r2_real,
                    "resid_real":        res_real,
                    # fake span (baseline)
                    "alpha_fake":        float(c_fake[0]),
                    "beta_fake":         float(c_fake[1]),
                    "R2_fake":           r2_fake,
                    "resid_fake":        res_fake,
                    # delta: real advantage over fake
                    "delta_R2":          r2_real - r2_fake,
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
    targets: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    ylabel: str,
    title: str,
    out_path: Path,
    hline: float | None = 0.0,
) -> None:
    """Line + ribbon (mean ± SD) per target over time. One readable panel for any n_bins."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for t in targets:
        sub = results[results["target"] == t].copy()
        stats = (
            sub.groupby("time_bin")[col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("time_bin")
        )
        x  = stats["time_bin"].to_numpy(dtype=float) + BIN_WIDTH / 2.0
        mu = stats["mean"].to_numpy(dtype=float)
        sd = stats["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, mu, color=colors[t], linewidth=2.0, label=labels[t])
        ax.fill_between(x, mu - sd, mu + sd, color=colors[t], alpha=0.18)

    if hline is not None:
        ax.axhline(hline, color="#444444", linestyle="--", linewidth=1.0)
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
    targets = ["pbx1b_heldout", "pbx4_heldout", "inj_ctrl"]
    labels  = {
        "pbx1b_heldout": "pbx1b held-out (real)",
        "pbx4_heldout":  "pbx4 held-out (real)",
        "inj_ctrl":      "inj_ctrl (control)",
    }

    # ── Figure 1: R2_real vs R2_fake over time, one panel per target ──────────
    # Fit: z_target ≈ V·c + r,  R²_span = 1 − ||r||² / ||z_target||²
    # Real span V_real = [mu_pbx1b_A − mu_non_A | mu_pbx4_A − mu_non_A]
    # Fake span V_fake = [mu_inj_A − mu_non_A   | mu_inj_B − mu_non_B]
    # pbx targets: real R² >> fake R² = span is specific | inj_ctrl: real ≈ fake = expected
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 4 * len(targets)), sharex=True)
    for ax, tname in zip(axes, targets):
        color = TARGET_COLORS[tname]
        sub = results[results["target"] == tname].copy()
        for col, ls, lw, clr, lbl in [
            ("R2_real", "-",  2.0, color,     f"{labels[tname]} — real span V=[v_1|v_4]"),
            ("R2_fake", "--", 1.5, "#AAAAAA",  "fake span V=[inj_A−non_A | inj_B−non_B]"),
        ]:
            stats = (
                sub.groupby("time_bin")[col]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("time_bin")
            )
            x  = stats["time_bin"].to_numpy(dtype=float) + BIN_WIDTH / 2.0
            mu = stats["mean"].to_numpy(dtype=float)
            sd = stats["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mu, color=clr, linewidth=lw, linestyle=ls, label=lbl)
            ax.fill_between(x, mu - sd, mu + sd, color=clr, alpha=0.15)
        ax.axhline(0.0, color="#444444", linestyle=":", linewidth=0.8)
        ax.set_ylabel("R²_span = 1 − ||r||²/||z||²")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Time bin center (hpf)")
    fig.suptitle("Q3: R²_span in real span (solid) vs fake span (dashed gray)\n"
                 "Fit: z ≈ V·c + r,   R²_span = 1 − ||r||²/||z||²\n"
                 "pbx targets: real >> fake = span is biologically specific | inj_ctrl: real ≈ fake = expected")
    fig.tight_layout()
    out = FIGURES_DIR / "q3_r2_real_vs_fake.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 2: ΔR² over time ────────────────────────────────────────────────
    _plot_over_time(
        results, col="delta_R2",
        targets=targets, colors=TARGET_COLORS, labels=labels,
        ylabel="ΔR²_span = R²_real − R²_fake",
        title=("Q3: ΔR²_span over time — real span advantage over fake span\n"
               "ΔR² = R²(real span) − R²(fake span),   fake span built from inj_ctrl splits\n"
               "pbx targets: positive ΔR² = real span is specific | inj_ctrl: ΔR² near zero = expected"),
        out_path=FIGURES_DIR / "q3_delta_r2.png",
    )

    # ── Figure 3 & 4: alpha/beta coefficient recovery ─────────────────────────
    for coef, axis_name, expectation in [
        ("alpha_real", "alpha", "pbx1b_heldout → ~1 | pbx4_heldout & inj_ctrl → ~0"),
        ("beta_real",  "beta",  "pbx4_heldout → ~1  | pbx1b_heldout & inj_ctrl → ~0"),
    ]:
        _plot_over_time(
            results, col=coef,
            targets=targets, colors=TARGET_COLORS, labels=labels,
            ylabel=f"{axis_name} (real span)",
            title=(f"Q3: {axis_name} coefficient recovery over time\n"
                   f"Fit: z ≈ alpha·v_1 + beta·v_4,   targets are held-out B-halves\n"
                   f"{expectation}"),
            out_path=FIGURES_DIR / f"q3_{axis_name}_real_span.png",
        )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    binned, feat_cols = _load_and_bin()
    valid_bins = _valid_bins(binned)
    print(f"Valid time bins: {valid_bins}")

    results = _run_bootstrap(binned, feat_cols, valid_bins)
    results.to_csv(RESULTS_DIR / "q3_span_specificity.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'q3_span_specificity.csv'}")

    _plot(results)


if __name__ == "__main__":
    main()
