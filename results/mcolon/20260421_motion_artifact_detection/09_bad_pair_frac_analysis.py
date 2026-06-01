"""
09_bad_pair_frac_analysis.py
============================
Follow-up analysis to pick a practical bad_pair_frac threshold that complements
ncc_p05 when ncc_p05 is noisy in borderline cases.

Inputs:
  07_embryo_ncc_output/embryo_ncc_summaries.csv

Outputs (under figures/threshold_bins/v3_bad_pair_frac_analysis):
  - bad_pair_by_ncc_tranche.csv
  - bad_pair_threshold_sweep.csv
  - bad_pair_hist_by_ncc_tranche.png
  - bad_pair_threshold_sweep.png
  - BAD_PAIR_FRAC_OBSERVATIONS.md

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python
    results/mcolon/20260421_motion_artifact_detection/09_bad_pair_frac_analysis.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "07_embryo_ncc_output/embryo_ncc_summaries.csv"
OUT_DIR = BASE_DIR / "figures/threshold_bins/v3_bad_pair_frac_analysis"

GOOD_ANCHOR_MIN_NCC_P05 = 0.90
LIKELY_MOTION_MAX_NCC_P05 = 0.80
GRAY_MIN = 0.80
GRAY_MAX = 0.85


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    required = {"ncc_p05", "bad_pair_frac", "well", "t"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")
    return df.dropna(subset=["ncc_p05", "bad_pair_frac"]).copy()


def assign_tranche(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.80, 0.85, 0.90, np.inf]
    labels = ["<0.80", "0.80-0.85", "0.85-0.90", ">=0.90"]
    out = df.copy()
    out["ncc_tranche"] = pd.cut(out["ncc_p05"], bins=bins, labels=labels, right=False)
    return out


def tranche_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby("ncc_tranche", observed=True)["bad_pair_frac"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            q75=lambda s: s.quantile(0.75),
            q90=lambda s: s.quantile(0.90),
            q95=lambda s: s.quantile(0.95),
            q99=lambda s: s.quantile(0.99),
        )
        .reset_index()
    )
    return grp


def threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.00, 0.305, 0.005), 3)

    good_anchor = df["ncc_p05"] >= GOOD_ANCHOR_MIN_NCC_P05
    likely_motion = df["ncc_p05"] < LIKELY_MOTION_MAX_NCC_P05
    gray_zone = (df["ncc_p05"] >= GRAY_MIN) & (df["ncc_p05"] < GRAY_MAX)

    rows = []
    for thr in thresholds:
        flagged = df["bad_pair_frac"] > thr

        overall_flag_frac = flagged.mean()
        good_anchor_flag_frac = (flagged & good_anchor).sum() / max(good_anchor.sum(), 1)
        likely_motion_flag_frac = (flagged & likely_motion).sum() / max(likely_motion.sum(), 1)
        gray_zone_flag_frac = (flagged & gray_zone).sum() / max(gray_zone.sum(), 1)

        youden_proxy = likely_motion_flag_frac - good_anchor_flag_frac

        rows.append(
            {
                "bad_pair_frac_threshold": thr,
                "overall_flag_frac": overall_flag_frac,
                "good_anchor_flag_frac_ncc_p05_ge_0p90": good_anchor_flag_frac,
                "likely_motion_flag_frac_ncc_p05_lt_0p80": likely_motion_flag_frac,
                "gray_zone_flag_frac_0p80_to_0p85": gray_zone_flag_frac,
                "youden_proxy": youden_proxy,
            }
        )

    return pd.DataFrame(rows)


def choose_thresholds(df: pd.DataFrame, sweep: pd.DataFrame) -> dict[str, float]:
    good = df.loc[df["ncc_p05"] >= GOOD_ANCHOR_MIN_NCC_P05, "bad_pair_frac"]

    q95 = float(good.quantile(0.95)) if len(good) else np.nan
    q99 = float(good.quantile(0.99)) if len(good) else np.nan

    best_idx = sweep["youden_proxy"].idxmax()
    best_proxy_thr = float(sweep.loc[best_idx, "bad_pair_frac_threshold"])

    return {
        "good_anchor_q95": q95,
        "good_anchor_q99": q99,
        "best_proxy_threshold": best_proxy_thr,
    }


def plot_hist(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "bad_pair_hist_by_ncc_tranche.png"

    plt.figure(figsize=(9, 5))
    for tranche, color in [
        (">=0.90", "#2ca02c"),
        ("0.85-0.90", "#1f77b4"),
        ("0.80-0.85", "#ff7f0e"),
        ("<0.80", "#d62728"),
    ]:
        vals = df.loc[df["ncc_tranche"] == tranche, "bad_pair_frac"].dropna()
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=40, alpha=0.35, density=True, label=f"{tranche} (n={len(vals)})", color=color)

    plt.xlabel("bad_pair_frac")
    plt.ylabel("density")
    plt.title("bad_pair_frac distribution by ncc_p05 tranche")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_sweep(sweep: pd.DataFrame, picks: dict[str, float]) -> Path:
    out = OUT_DIR / "bad_pair_threshold_sweep.png"

    x = sweep["bad_pair_frac_threshold"]
    plt.figure(figsize=(9, 5))
    plt.plot(x, sweep["good_anchor_flag_frac_ncc_p05_ge_0p90"], label="Flag rate in good anchor (ncc_p05 >= 0.90)", color="#2ca02c")
    plt.plot(x, sweep["likely_motion_flag_frac_ncc_p05_lt_0p80"], label="Flag rate in likely-motion anchor (ncc_p05 < 0.80)", color="#d62728")
    plt.plot(x, sweep["gray_zone_flag_frac_0p80_to_0p85"], label="Flag rate in gray zone (0.80-0.85)", color="#ff7f0e")

    for key, color in [
        ("good_anchor_q95", "#2ca02c"),
        ("good_anchor_q99", "#1f77b4"),
        ("best_proxy_threshold", "#d62728"),
    ]:
        val = picks.get(key)
        if val is not None and np.isfinite(val):
            plt.axvline(val, color=color, ls="--", lw=1.2, alpha=0.9, label=f"{key}={val:.3f}")

    plt.xlabel("bad_pair_frac threshold")
    plt.ylabel("fraction flagged")
    plt.title("bad_pair_frac threshold sweep across ncc_p05 cohorts")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def render_example_rows(df: pd.DataFrame) -> str:
    examples = [("A10", 98), ("E11", 79), ("D05", 17), ("C04", 11), ("E11", 230)]
    lines = ["| Well | t | ncc_p05 | bad_pair_frac |", "|------|---|---------|---------------|"]
    for well, t in examples:
        sub = df[(df["well"] == well) & (df["t"] == t)]
        if sub.empty:
            lines.append(f"| {well} | {t} | NA | NA |")
        else:
            row = sub.iloc[0]
            lines.append(f"| {well} | {t} | {row['ncc_p05']:.3f} | {row['bad_pair_frac']:.3f} |")
    return "\n".join(lines)


def write_observations(df: pd.DataFrame, tranche: pd.DataFrame, picks: dict[str, float], sweep: pd.DataFrame) -> Path:
    out = OUT_DIR / "BAD_PAIR_FRAC_OBSERVATIONS.md"

    good = df[df["ncc_p05"] >= GOOD_ANCHOR_MIN_NCC_P05]["bad_pair_frac"]
    gray = df[(df["ncc_p05"] >= GRAY_MIN) & (df["ncc_p05"] < GRAY_MAX)]["bad_pair_frac"]

    best_row = sweep.loc[sweep["youden_proxy"].idxmax()]

    text = f"""# bad_pair_frac Threshold Analysis (2026-04-23)

## Anchor cohort summary

- Good anchor (`ncc_p05 >= 0.90`): n={len(good)}
  - mean bad_pair_frac = {good.mean():.4f}
  - median bad_pair_frac = {good.median():.4f}
  - 95th percentile = {picks['good_anchor_q95']:.4f}
  - 99th percentile = {picks['good_anchor_q99']:.4f}

- Gray zone (`0.80 <= ncc_p05 < 0.85`): n={len(gray)}
  - mean bad_pair_frac = {gray.mean():.4f}
  - median bad_pair_frac = {gray.median():.4f}

## Suggested threshold candidates

- **Conservative candidate**: `bad_pair_frac > {picks['good_anchor_q99']:.3f}`
  - Anchored to 99th percentile of clearly good images.
  - Minimizes false positives on clean embryos.

- **Balanced candidate**: `bad_pair_frac > {picks['good_anchor_q95']:.3f}`
  - Anchored to 95th percentile of clearly good images.
  - More sensitive in borderline motion cases.

- **Proxy-separation best** (using `ncc_p05 < 0.80` as likely-motion and `>=0.90` as good):
  - `bad_pair_frac > {picks['best_proxy_threshold']:.3f}`
  - good-anchor flag rate = {best_row['good_anchor_flag_frac_ncc_p05_ge_0p90']:.3f}
  - likely-motion flag rate = {best_row['likely_motion_flag_frac_ncc_p05_lt_0p80']:.3f}
  - gray-zone flag rate = {best_row['gray_zone_flag_frac_0p80_to_0p85']:.3f}

## User-provided examples

{render_example_rows(df)}

## Practical next rule to test

Use a two-stage rule in borderline cases:

1. Primary fail: `ncc_p05 < 0.85`
2. Secondary confirmation in borderline zone: if `0.80 <= ncc_p05 < 0.85`, require `bad_pair_frac` above a chosen threshold from this analysis.

See CSV/plots in this folder for tranche-level tradeoffs.
"""
    out.write_text(text)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df = assign_tranche(df)

    tranche = tranche_summary(df)
    sweep = threshold_sweep(df)
    picks = choose_thresholds(df, sweep)

    tranche_csv = OUT_DIR / "bad_pair_by_ncc_tranche.csv"
    sweep_csv = OUT_DIR / "bad_pair_threshold_sweep.csv"
    tranche.to_csv(tranche_csv, index=False)
    sweep.to_csv(sweep_csv, index=False)

    hist_png = plot_hist(df)
    sweep_png = plot_sweep(sweep, picks)
    obs_md = write_observations(df, tranche, picks, sweep)

    print(f"Saved tranche summary: {tranche_csv}")
    print(f"Saved threshold sweep: {sweep_csv}")
    print(f"Saved hist plot: {hist_png}")
    print(f"Saved sweep plot: {sweep_png}")
    print(f"Saved observations: {obs_md}")


if __name__ == "__main__":
    main()
