"""
07_focus_tail_analysis.py
==========================
Fast tail inspection for focus signal quality using precomputed motion outputs.

This avoids recomputing stacks. It uses:
  - results/mcolon/20260421_motion_artifact_detection/slice_metrics_relative.csv
  - results/mcolon/20260421_motion_artifact_detection/stack_metrics.csv

Outputs:
  - 07_focus_output_tail/focus_tail_summary.csv
  - 07_focus_output_tail/rel_entropy_tail_hist.png
  - 07_focus_output_tail/rel_entropy_tail_zoom.png
  - 07_focus_output_tail/rel_entropy_tail_scatter.png
  - 07_focus_output_tail/rel_entropy_tail_quantiles.csv
  - 07_focus_output_tail/rel_entropy_tail_examples.csv

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/07_focus_tail_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
MOTION_DIR = BASE_DIR.parent / "20260421_motion_artifact_detection"
SLICE_REL_CSV = MOTION_DIR / "slice_metrics_relative.csv"
STACK_CSV = MOTION_DIR / "stack_metrics.csv"
LOOKUP_CSV = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv")

OUT_DIR = BASE_DIR / "07_focus_output_tail"
CSV_PATH = OUT_DIR / "focus_tail_summary.csv"

GOOD_ANCHOR = "Great Images"
BAD_ANCHOR = "Bad Images"
GRAY_ANCHOR = "Okay Images"


def load_slice_relative() -> pd.DataFrame:
    df = pd.read_csv(SLICE_REL_CSV)
    required = {"well", "time_int", "embryo", "rel_entropy"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {SLICE_REL_CSV}: {missing}")
    return df


def load_stack_metrics() -> pd.DataFrame:
    df = pd.read_csv(STACK_CSV)
    required = {"well", "time_int", "embryo", "ncc_min", "bad_pair_frac_ncc", "longest_bad_ncc_run"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {STACK_CSV}: {missing}")
    return df


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LOOKUP_CSV)
    df["time_int"] = df["time_int"].astype(int)
    df["category"] = df["category"].astype(str)
    return df[["well", "time_int", "category"]].drop_duplicates()


def build_summary() -> pd.DataFrame:
    rel = load_slice_relative()
    stack = load_stack_metrics()
    labels = load_labels()

    rel_summary = (
        rel.groupby(["well", "time_int", "embryo"], observed=True)["rel_entropy"]
        .agg(
            rel_entropy_mean="mean",
            rel_entropy_min="min",
            rel_entropy_std="std",
            n_z="count",
        )
        .reset_index()
    )

    df = rel_summary.merge(
        stack[[
            "well",
            "time_int",
            "embryo",
            "ncc_min",
            "ncc_median",
            "ncc_std",
            "bad_pair_frac_ncc",
            "longest_bad_ncc_run",
            "max_phase_shift_px",
        ]],
        on=["well", "time_int", "embryo"],
        how="left",
    )
    df = df.merge(labels, on=["well", "time_int"], how="left")
    df = df.sort_values(["rel_entropy_mean", "ncc_min"], ascending=[True, True]).reset_index(drop=True)
    return df


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
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


def plot_tail_hist(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "rel_entropy_tail_hist.png"
    plt.figure(figsize=(9.5, 5.2))

    for category, color in [
        (GOOD_ANCHOR, "#2ca02c"),
        (GRAY_ANCHOR, "#ff7f0e"),
        (BAD_ANCHOR, "#d62728"),
    ]:
        vals = df.loc[df["category"] == category, "rel_entropy_mean"].dropna()
        if len(vals) == 0:
            continue
        plt.hist(vals, bins=45, alpha=0.35, density=True, label=f"{category} (n={len(vals)})", color=color)

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
        plt.hist(sub, bins=35, range=(lo, hi), alpha=0.35, density=True,
                 label=f"{category} (n={len(sub)})", color=color)

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


def write_examples(df: pd.DataFrame) -> Path:
    out = OUT_DIR / "rel_entropy_tail_examples.csv"
    cols = ["well", "time_int", "embryo", "category", "rel_entropy_mean", "rel_entropy_min", "ncc_min", "bad_pair_frac_ncc"]
    low = df.nsmallest(15, "rel_entropy_mean")[cols].copy()
    low.insert(0, "rank_group", "lowest")
    high = df.nlargest(15, "rel_entropy_mean")[cols].copy()
    high.insert(0, "rank_group", "highest")
    pd.concat([low, high], ignore_index=True).to_csv(out, index=False)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_summary()
    cat = category_summary(df)
    tails = tail_summary(df)

    summary_csv = CSV_PATH
    cat_csv = OUT_DIR / "rel_entropy_by_category.csv"
    tail_csv = OUT_DIR / "rel_entropy_tail_quantiles.csv"
    df.to_csv(summary_csv, index=False)
    cat.to_csv(cat_csv, index=False)
    tails.to_csv(tail_csv, index=False)

    hist = plot_tail_hist(df)
    zoom = plot_tail_zoom(df)
    scatter = plot_tail_scatter(df)
    examples = write_examples(df)

    print(f"Saved summary -> {summary_csv}")
    print(f"Saved category summary -> {cat_csv}")
    print(f"Saved tail quantiles -> {tail_csv}")
    print(f"Saved histogram -> {hist}")
    print(f"Saved tail zoom -> {zoom}")
    print(f"Saved tail scatter -> {scatter}")
    print(f"Saved tail examples -> {examples}")


if __name__ == "__main__":
    main()
