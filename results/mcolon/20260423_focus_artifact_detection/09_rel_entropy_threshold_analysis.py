"""
09_rel_entropy_threshold_analysis.py
====================================
Threshold search for mask-aware focus QC using rel_entropy_mean.

Inputs:
  07_focus_output/focus_summaries.csv

Outputs (under figures/threshold_bins/v1_rel_entropy_analysis):
  - rel_entropy_by_category.csv
  - rel_entropy_threshold_sweep.csv
  - rel_entropy_hist_by_category.png
  - rel_entropy_threshold_sweep.png
  - REL_ENTROPY_THRESHOLD_OBSERVATIONS.md

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/09_rel_entropy_threshold_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "07_focus_output/focus_summaries.csv"
OUT_DIR = BASE_DIR / "figures/threshold_bins/v1_rel_entropy_analysis"

GOOD_ANCHOR = "Great Images"
BAD_ANCHOR = "Bad Images"
GRAY_ANCHOR = "Okay Images"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    required = {"rel_entropy_mean", "well", "t"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")
    if "category" not in df.columns:
        raise ValueError(
            "focus_summaries.csv is missing the 'category' column. "
            "Run 07_focus_analysis.py to regenerate the summary CSV with labels merged."
        )
    return df.dropna(subset=["rel_entropy_mean"]).copy()


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = (
        df.groupby("category", observed=True)["rel_entropy_mean"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            q05=lambda s: s.quantile(0.05),
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            q95=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )
    return grp


def tail_summary(df: pd.DataFrame) -> pd.DataFrame:
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows: list[dict] = []

    for category, grp in df.groupby("category", observed=True):
        vals = grp["rel_entropy_mean"].dropna()
        row = {"category": category, "n": int(len(vals))}
        for q in quantiles:
            row[f"q{int(q * 100):02d}"] = float(vals.quantile(q)) if len(vals) else np.nan
        rows.append(row)

    vals = df["rel_entropy_mean"].dropna()
    row = {"category": "ALL", "n": int(len(vals))}
    for q in quantiles:
        row[f"q{int(q * 100):02d}"] = float(vals.quantile(q)) if len(vals) else np.nan
    rows.append(row)
    return pd.DataFrame(rows)


def threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    lo = float(df["rel_entropy_mean"].min())
    hi = float(df["rel_entropy_mean"].max())
    span = hi - lo
    thresholds = np.round(np.linspace(lo - 0.10 * span, hi + 0.10 * span, 241), 4)

    good_anchor = df["category"] == GOOD_ANCHOR
    bad_anchor = df["category"] == BAD_ANCHOR
    gray_zone = df["category"] == GRAY_ANCHOR

    rows = []
    for thr in thresholds:
        flagged = df["rel_entropy_mean"] < thr

        good_flag_frac = (flagged & good_anchor).sum() / max(good_anchor.sum(), 1)
        bad_flag_frac = (flagged & bad_anchor).sum() / max(bad_anchor.sum(), 1)
        gray_flag_frac = (flagged & gray_zone).sum() / max(gray_zone.sum(), 1)
        overall_flag_frac = flagged.mean()

        rows.append(
            {
                "rel_entropy_threshold": thr,
                "overall_flag_frac": overall_flag_frac,
                "good_anchor_flag_frac_great": good_flag_frac,
                "bad_anchor_flag_frac_bad": bad_flag_frac,
                "gray_zone_flag_frac_okay": gray_flag_frac,
                "youden_proxy": bad_flag_frac - good_flag_frac,
            }
        )

    return pd.DataFrame(rows)


def choose_thresholds(df: pd.DataFrame, sweep: pd.DataFrame) -> dict[str, float]:
    good = df.loc[df["category"] == GOOD_ANCHOR, "rel_entropy_mean"]
    bad = df.loc[df["category"] == BAD_ANCHOR, "rel_entropy_mean"]

    q05 = float(good.quantile(0.05)) if len(good) else np.nan
    q01 = float(good.quantile(0.01)) if len(good) else np.nan
    bad_q50 = float(bad.quantile(0.50)) if len(bad) else np.nan
    bad_q95 = float(bad.quantile(0.95)) if len(bad) else np.nan

    best_idx = sweep["youden_proxy"].idxmax()
    best_proxy_thr = float(sweep.loc[best_idx, "rel_entropy_threshold"])

    return {
        "good_anchor_q05": q05,
        "good_anchor_q01": q01,
        "bad_anchor_q50": bad_q50,
        "bad_anchor_q95": bad_q95,
        "best_proxy_threshold": best_proxy_thr,
    }


def plot_hist(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "rel_entropy_hist_by_category.png"
    plt.figure(figsize=(9.5, 5.2))

    for category, color in [
        (GOOD_ANCHOR, "#2ca02c"),
        (GRAY_ANCHOR, "#ff7f0e"),
        (BAD_ANCHOR, "#d62728"),
    ]:
        vals = df.loc[df["category"] == category, "rel_entropy_mean"].dropna()
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=40, alpha=0.35, density=True, label=f"{category} (n={len(vals)})", color=color)

    plt.xlabel("rel_entropy_mean")
    plt.ylabel("density")
    plt.title("rel_entropy_mean distribution by label category")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_tail_zoom(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "rel_entropy_tail_zoom.png"
    vals = df["rel_entropy_mean"].dropna()
    if vals.empty:
        return out

    lo = float(vals.quantile(0.01))
    hi = float(vals.quantile(0.40))
    plt.figure(figsize=(9.5, 5.2))

    for category, color in [
        (GOOD_ANCHOR, "#2ca02c"),
        (GRAY_ANCHOR, "#ff7f0e"),
        (BAD_ANCHOR, "#d62728"),
    ]:
        sub = df.loc[df["category"] == category, "rel_entropy_mean"].dropna()
        if len(sub) == 0:
            continue
        plt.hist(
            sub,
            bins=35,
            range=(lo, hi),
            alpha=0.35,
            density=True,
            label=f"{category} (n={len(sub)})",
            color=color,
        )

    good_q05 = float(df.loc[df["category"] == GOOD_ANCHOR, "rel_entropy_mean"].quantile(0.05))
    bad_q95 = float(df.loc[df["category"] == BAD_ANCHOR, "rel_entropy_mean"].quantile(0.95))
    plt.axvline(good_q05, color="#2ca02c", ls="--", lw=1.2, label=f"{GOOD_ANCHOR} q05")
    plt.axvline(bad_q95, color="#d62728", ls="--", lw=1.2, label=f"{BAD_ANCHOR} q95")
    plt.xlabel("rel_entropy_mean")
    plt.ylabel("density")
    plt.title("Lower-tail focus separation by label category")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_tail_scatter(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "rel_entropy_tail_scatter.png"
    if "ncc_min" not in df.columns:
        return out

    tail_cut = float(df["rel_entropy_mean"].quantile(0.20))
    tail = df[(df["rel_entropy_mean"] <= tail_cut) & df["ncc_min"].notna()].copy()
    if tail.empty:
        return out

    plt.figure(figsize=(7.4, 6))
    colors = {GOOD_ANCHOR: "#2ca02c", GRAY_ANCHOR: "#ff7f0e", BAD_ANCHOR: "#d62728"}
    for category, grp in tail.groupby("category", observed=True):
        color = colors.get(category, "#444444")
        plt.scatter(
            grp["ncc_min"],
            grp["rel_entropy_mean"],
            s=28,
            alpha=0.8,
            color=color,
            edgecolors="k",
            linewidths=0.3,
            label=f"{category} (n={len(grp)})",
        )

    plt.axhline(tail_cut, color="#666666", ls="--", lw=1, label=f"tail cutoff q20={tail_cut:.3f}")
    plt.axvline(0.85, color="red", ls="--", lw=1, label="ncc_min=0.85")
    plt.xlabel("ncc_min")
    plt.ylabel("rel_entropy_mean")
    plt.title("Tail cohort: rel_entropy_mean vs ncc_min")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def write_tail_examples(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "rel_entropy_tail_examples.csv"
    cols = ["well", "t", "p", "category", "rel_entropy_mean", "rel_entropy_min", "entropy_mean"]
    if "ncc_min" in df.columns:
        cols.insert(5, "ncc_min")

    low = df.nsmallest(15, "rel_entropy_mean")[cols].copy()
    low.insert(0, "rank_group", "lowest")
    high = df.nlargest(15, "rel_entropy_mean")[cols].copy()
    high.insert(0, "rank_group", "highest")
    pd.concat([low, high], ignore_index=True).to_csv(out, index=False)
    return out


def plot_sweep(sweep: pd.DataFrame, picks: dict[str, float]) -> Path:
    out = OUT_DIR / "rel_entropy_threshold_sweep.png"

    x = sweep["rel_entropy_threshold"]
    plt.figure(figsize=(9.5, 5.2))
    plt.plot(x, sweep["good_anchor_flag_frac_great"], label=f"Flag rate in {GOOD_ANCHOR}", color="#2ca02c")
    plt.plot(x, sweep["bad_anchor_flag_frac_bad"], label=f"Flag rate in {BAD_ANCHOR}", color="#d62728")
    plt.plot(x, sweep["gray_zone_flag_frac_okay"], label=f"Flag rate in {GRAY_ANCHOR}", color="#ff7f0e")

    for key, color in [
        ("good_anchor_q05", "#2ca02c"),
        ("good_anchor_q01", "#1f77b4"),
        ("bad_anchor_q95", "#d62728"),
        ("best_proxy_threshold", "#9467bd"),
    ]:
        val = picks.get(key)
        if val is not None and np.isfinite(val):
            plt.axvline(val, color=color, ls="--", lw=1.2, alpha=0.9, label=f"{key}={val:.3f}")

    plt.xlabel("rel_entropy_mean threshold")
    plt.ylabel("fraction flagged")
    plt.title("rel_entropy_mean threshold sweep across label cohorts")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def render_example_rows(df: pd.DataFrame) -> str:
    examples = [
        ("B10", 97),
        ("B10", 92),
        ("C04", 28),
        ("C04", 112),
        ("E11", 79),
    ]
    cols = ["well", "t", "rel_entropy_mean", "entropy_mean"]
    if "ncc_min" in df.columns:
        cols.insert(2, "ncc_min")

    lines = ["| Well | t | ncc_min | rel_entropy_mean | entropy_mean |", "|------|---|--------|------------------|--------------|"]
    for well, t in examples:
        sub = df[(df["well"] == well) & (df["t"] == t)]
        if sub.empty:
            lines.append(f"| {well} | {t} | NA | NA | NA |")
            continue
        row = sub.iloc[0]
        ncc = f"{row['ncc_min']:.3f}" if "ncc_min" in sub.columns and pd.notna(row.get("ncc_min")) else "NA"
        lines.append(
            f"| {well} | {t} | {ncc} | {row['rel_entropy_mean']:.3f} | {row['entropy_mean']:.3f} |"
        )
    return "\n".join(lines)


def write_observations(df: pd.DataFrame, category: pd.DataFrame, picks: dict[str, float], sweep: pd.DataFrame) -> Path:
    out = OUT_DIR / "REL_ENTROPY_THRESHOLD_OBSERVATIONS.md"
    good = df[df["category"] == GOOD_ANCHOR]["rel_entropy_mean"]
    bad = df[df["category"] == BAD_ANCHOR]["rel_entropy_mean"]
    gray = df[df["category"] == GRAY_ANCHOR]["rel_entropy_mean"]

    best_row = sweep.loc[sweep["youden_proxy"].idxmax()]

    text = f"""# rel_entropy_mean Threshold Analysis (2026-04-23)

## Cohort summary

- {GOOD_ANCHOR}: n={len(good)}
  - mean rel_entropy_mean = {good.mean():.4f}
  - median rel_entropy_mean = {good.median():.4f}
  - 5th percentile = {picks['good_anchor_q05']:.4f}
  - 1st percentile = {picks['good_anchor_q01']:.4f}

- {BAD_ANCHOR}: n={len(bad)}
  - mean rel_entropy_mean = {bad.mean():.4f}
  - median rel_entropy_mean = {bad.median():.4f}
  - 95th percentile = {picks['bad_anchor_q95']:.4f}

- {GRAY_ANCHOR}: n={len(gray)}
  - mean rel_entropy_mean = {gray.mean():.4f}
  - median rel_entropy_mean = {gray.median():.4f}

## Suggested threshold candidates

- **Conservative candidate**: `rel_entropy_mean < {picks['good_anchor_q05']:.3f}`
  - Flags only the lower tail of clearly good images.

- **Stricter candidate**: `rel_entropy_mean < {picks['good_anchor_q01']:.3f}`
  - Very low false-positive rate on good anchors.

- **Proxy-separation best** (using `{BAD_ANCHOR}` vs `{GOOD_ANCHOR}`):
  - `rel_entropy_mean < {picks['best_proxy_threshold']:.3f}`
  - good-anchor flag rate = {best_row['good_anchor_flag_frac_great']:.3f}
  - bad-anchor flag rate = {best_row['bad_anchor_flag_frac_bad']:.3f}
  - gray-zone flag rate = {best_row['gray_zone_flag_frac_okay']:.3f}

## Practical rule to test

Treat `rel_entropy_mean` as a WARN/FAIL axis:

1. `rel_entropy_mean` below the chosen threshold is a focus warning.
2. If the threshold is set conservatively, a second pass can inspect borderline cases visually in the ranked figure.

## User-provided examples

{render_example_rows(df)}

## Notes

- Lower `rel_entropy_mean` means blurrier / worse focus.
- This analysis is mask-aware and uses a background entropy estimate from tiles outside the embryo mask.
- The tail plots are the main sanity check for whether the score is actually separating focus quality.
"""
    out.write_text(text)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    category = category_summary(df)
    tails = tail_summary(df)
    sweep = threshold_sweep(df)
    picks = choose_thresholds(df, sweep)

    category_csv = OUT_DIR / "rel_entropy_by_category.csv"
    tail_csv = OUT_DIR / "rel_entropy_tail_quantiles.csv"
    sweep_csv = OUT_DIR / "rel_entropy_threshold_sweep.csv"
    category.to_csv(category_csv, index=False)
    tails.to_csv(tail_csv, index=False)
    sweep.to_csv(sweep_csv, index=False)

    hist_path = plot_hist(df)
    tail_zoom_path = plot_tail_zoom(df)
    tail_scatter_path = plot_tail_scatter(df)
    tail_examples_csv = write_tail_examples(df)
    sweep_path = plot_sweep(sweep, picks)
    md_path = write_observations(df, category, picks, sweep)

    print(f"Saved category summary: {category_csv}")
    print(f"Saved tail quantiles: {tail_csv}")
    print(f"Saved threshold sweep: {sweep_csv}")
    print(f"Saved histogram: {hist_path}")
    print(f"Saved tail zoom: {tail_zoom_path}")
    print(f"Saved tail scatter: {tail_scatter_path}")
    print(f"Saved tail examples: {tail_examples_csv}")
    print(f"Saved sweep plot: {sweep_path}")
    print(f"Saved observations: {md_path}")


if __name__ == "__main__":
    main()
