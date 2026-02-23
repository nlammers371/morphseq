"""Validate z-score divergence computation and understand when it's appropriate.

The z-score normalization transforms abs_difference to have mean=0, std=1 within
each metric. This is useful for multi-metric comparison on the same axis, but can
be misleading if interpreted as "statistical significance" - it amplifies noise
when the raw divergence is consistently small.

This script:
1. Computes divergence for curvature (Homo vs WT, proper genotype comparison)
2. Shows raw abs_difference vs abs_difference_zscore
3. Plots both to visualize when z-score amplifies trivial differences
4. Computes effect sizes (Cohen's d) at each timepoint for context

Key insight: z-score normalization is a VISUALIZATION tool for comparing metrics
with different scales, NOT a statistical test. If raw abs_difference is flat,
z-score "spikes" are artifacts of normalizing noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.difference_detection.comparison import compute_metric_divergence


def cohens_d_per_timepoint(
    df: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    metric_col: str,
    time_col: str = "predicted_stage_hpf",
    time_resolution: float = 1.0,
) -> pd.DataFrame:
    """Compute Cohen's d effect size per timepoint."""
    df = df[df[group_col].isin([group1, group2])].copy()
    df["time_rounded"] = (df[time_col] // time_resolution) * time_resolution

    results = []
    for t in sorted(df["time_rounded"].unique()):
        sub = df[df["time_rounded"] == t]
        g1_vals = sub[sub[group_col] == group1][metric_col].dropna().values
        g2_vals = sub[sub[group_col] == group2][metric_col].dropna().values

        if len(g1_vals) < 2 or len(g2_vals) < 2:
            continue

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(g1_vals) - 1) * np.var(g1_vals, ddof=1) + (len(g2_vals) - 1) * np.var(g2_vals, ddof=1))
            / (len(g1_vals) + len(g2_vals) - 2)
        )
        d = (np.mean(g1_vals) - np.mean(g2_vals)) / pooled_std if pooled_std > 0 else 0.0

        # t-test (for reference)
        t_stat, pval = scipy_stats.ttest_ind(g1_vals, g2_vals)

        results.append({
            "hpf": t,
            "cohens_d": d,
            "g1_mean": np.mean(g1_vals),
            "g2_mean": np.mean(g2_vals),
            "abs_diff": abs(np.mean(g1_vals) - np.mean(g2_vals)),
            "n_g1": len(g1_vals),
            "n_g2": len(g2_vals),
            "ttest_pval": pval,
        })

    return pd.DataFrame(results)


def zscore_normalize(series: pd.Series) -> pd.Series:
    """Z-score normalize (mean=0, std=1)."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return series - mean
    return (series - mean) / std


def main() -> int:
    data_path = (
        PROJECT_ROOT
        / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    )
    output_dir = (
        Path(__file__).parent / "output" / "cep290" / "zscore_divergence_validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path, low_memory=False)

    if "genotype" not in df.columns:
        raise ValueError("genotype column not found")

    # Filter to Homo vs WT (proper comparison)
    df_filtered = df[df["genotype"].isin(["cep290_homozygous", "cep290_wildtype"])].copy()

    metric_col = "baseline_deviation_normalized"
    print(f"\nComputing divergence for {metric_col}...")

    # Compute divergence using the API
    divergence = compute_metric_divergence(
        df_filtered,
        group_col="genotype",
        group1="cep290_homozygous",
        group2="cep290_wildtype",
        metric_col=metric_col,
        time_col="predicted_stage_hpf",
        embryo_id_col="embryo_id",
    )

    # Apply z-score normalization
    divergence["abs_difference_zscore"] = zscore_normalize(divergence["abs_difference"])

    # Compute Cohen's d per timepoint (for context)
    print("Computing Cohen's d per timepoint...")
    cohens_d_df = cohens_d_per_timepoint(
        df_filtered,
        group_col="genotype",
        group1="cep290_homozygous",
        group2="cep290_wildtype",
        metric_col=metric_col,
        time_col="predicted_stage_hpf",
        time_resolution=1.0,
    )

    # Merge for comparison
    merged = divergence.merge(cohens_d_df, on="hpf", how="left")

    # Save
    merged.to_csv(output_dir / "divergence_with_effect_sizes.csv", index=False)
    print(f"Saved: {output_dir / 'divergence_with_effect_sizes.csv'}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY: Raw vs Z-score Divergence")
    print("=" * 70)
    print(f"\nRaw abs_difference:")
    print(f"  Mean: {divergence['abs_difference'].mean():.4f}")
    print(f"  Std:  {divergence['abs_difference'].std():.4f}")
    print(f"  Min:  {divergence['abs_difference'].min():.4f}")
    print(f"  Max:  {divergence['abs_difference'].max():.4f}")

    print(f"\nZ-score normalized abs_difference:")
    print(f"  Mean: {divergence['abs_difference_zscore'].mean():.4f} (should be ~0)")
    print(f"  Std:  {divergence['abs_difference_zscore'].std():.4f} (should be ~1)")
    print(f"  Min:  {divergence['abs_difference_zscore'].min():.4f}")
    print(f"  Max:  {divergence['abs_difference_zscore'].max():.4f}")

    print(f"\nEffect sizes (Cohen's d) over time:")
    print(f"  Mean |d|: {cohens_d_df['cohens_d'].abs().mean():.3f}")
    print(f"  Max  |d|: {cohens_d_df['cohens_d'].abs().max():.3f}")
    print(f"  Timepoints with |d| > 0.5 (moderate): {(cohens_d_df['cohens_d'].abs() > 0.5).sum()}")
    print(f"  Timepoints with |d| > 0.8 (large): {(cohens_d_df['cohens_d'].abs() > 0.8).sum()}")

    # Focus on early timepoints (12-24 hpf)
    early = merged[(merged["hpf"] >= 12) & (merged["hpf"] <= 24)]
    print(f"\nEarly timepoints (12-24 hpf):")
    print(f"  Raw abs_difference mean: {early['abs_difference'].mean():.4f}")
    print(f"  Z-score abs_diff mean: {early['abs_difference_zscore'].mean():.4f}")
    print(f"  Cohen's d mean: {early['cohens_d'].abs().mean():.3f}")

    late = merged[merged["hpf"] >= 80]
    print(f"\nLate timepoints (≥80 hpf):")
    print(f"  Raw abs_difference mean: {late['abs_difference'].mean():.4f}")
    print(f"  Z-score abs_diff mean: {late['abs_difference_zscore'].mean():.4f}")
    print(f"  Cohen's d mean: {late['cohens_d'].abs().mean():.3f}")

    # Plot
    print("\nCreating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel A: Raw abs_difference
    ax = axes[0]
    ax.plot(divergence["hpf"], divergence["abs_difference"], "o-", color="#2E7D32", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Raw abs_difference\n(curvature units)", fontsize=11)
    ax.set_title("A. Raw Divergence (Homo vs WT curvature)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Panel B: Z-score normalized
    ax = axes[1]
    ax.plot(divergence["hpf"], divergence["abs_difference_zscore"], "o-", color="#D32F2F", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(1, color="orange", linestyle=":", alpha=0.5, label="±1 SD")
    ax.axhline(-1, color="orange", linestyle=":", alpha=0.5)
    ax.set_ylabel("Z-score normalized\nabs_difference", fontsize=11)
    ax.set_title("B. Z-score Normalized (mean=0, std=1)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel C: Cohen's d effect size
    ax = axes[2]
    ax.plot(cohens_d_df["hpf"], cohens_d_df["cohens_d"].abs(), "o-", color="#1976D2", linewidth=2)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.5, label="Small (0.2)")
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.5, label="Moderate (0.5)")
    ax.axhline(0.8, color="red", linestyle=":", alpha=0.5, label="Large (0.8)")
    ax.set_ylabel("|Cohen's d|", fontsize=11)
    ax.set_xlabel("Time (hpf)", fontsize=11)
    ax.set_title("C. Effect Size (proper statistical measure)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "divergence_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close(fig)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Z-score normalization is WORKING CORRECTLY but can be MISLEADING:

1. What it does:
   - Transforms abs_difference to have mean=0, std=1 across timepoints
   - Useful for plotting multiple metrics with different scales on same axis

2. Why it's misleading for single-metric interpretation:
   - If raw abs_difference is consistently small (Panel A flat), z-score 
     "spikes" (Panel B) are just normalized noise, NOT real divergence
   - Z-score > 1 means "1 SD above average divergence for this metric over time"
     but does NOT mean "statistically significant difference between groups"

3. Proper interpretation:
   - Panel C (Cohen's d) is the correct effect size measure per timepoint
   - Panel A (raw difference) shows actual magnitude in original units
   - Panel B (z-score) is for visual comparison across metrics ONLY

RECOMMENDATION:
- For Panel B in 3-panel figures, plot RAW abs_difference (Panel A style)
- Use z-score ONLY when overlaying multiple metrics with different scales
- Always report effect sizes (Cohen's d) or classification AUROC for inference
""")

    print(f"\nAll outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
