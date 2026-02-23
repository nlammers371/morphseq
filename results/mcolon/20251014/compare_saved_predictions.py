"""
Compare saved baseline vs balanced prediction CSV files to identify coverage
and value differences at the embryo × time_bin level.

This script helps explain why downstream visualisations may show different
numbers of plotted points despite identical overall prediction counts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

RESULTS_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014")
DATA_DIR = RESULTS_DIR / "imbalance_methods" / "data"
OUTPUT_DIR = RESULTS_DIR / "imbalance_methods" / "diagnostics" / "prediction_csv_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISONS: Dict[str, List[Tuple[str, str]]] = {
    "b9d2": [
        ("b9d2_wildtype", "b9d2_heterozygous"),
        ("b9d2_wildtype", "b9d2_homozygous"),
        ("b9d2_heterozygous", "b9d2_homozygous"),
    ],
    "cep290": [
        ("cep290_wildtype", "cep290_heterozygous"),
        ("cep290_wildtype", "cep290_homozygous"),
        ("cep290_heterozygous", "cep290_homozygous"),
    ],
    "tmem67": [
        ("tmem67_wildtype", "tmem67_heterozygote"),
        ("tmem67_wildtype", "tmem67_homozygous"),
        ("tmem67_heterozygote", "tmem67_homozygous"),
    ],
}

KEY_COLS = ["embryo_id", "time_bin"]


# -----------------------------------------------------------------------------
# CORE ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

def load_predictions(gene: str, group1: str, group2: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline and balanced prediction CSVs for a gene comparison."""
    safe_name = f"{group1.split('_')[-1]}_vs_{group2.split('_')[-1]}"
    gene_dir = DATA_DIR / gene

    baseline_path = gene_dir / f"embryo_probs_baseline_{safe_name}.csv"
    balanced_path = gene_dir / f"embryo_probs_class_weight_{safe_name}.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline prediction file: {baseline_path}")
    if not balanced_path.exists():
        raise FileNotFoundError(f"Missing balanced prediction file: {balanced_path}")

    df_baseline = pd.read_csv(baseline_path)
    df_balanced = pd.read_csv(balanced_path)

    return df_baseline, df_balanced


def check_duplicate_keys(df: pd.DataFrame, method: str) -> None:
    """Warn if any embryo × time_bin keys are duplicated."""
    duplicate_mask = df.duplicated(subset=KEY_COLS, keep=False)
    n_duplicates = duplicate_mask.sum()
    if n_duplicates:
        dup_df = df.loc[duplicate_mask, KEY_COLS + ["true_label"]].copy()
        dup_path = OUTPUT_DIR / f"duplicates_{method}.csv"
        dup_df.to_csv(dup_path, index=False)
        print(f"    WARNING: {method} has {n_duplicates} duplicate rows. Saved details to {dup_path}")


def merged_key_report(df_baseline: pd.DataFrame, df_balanced: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Merge on embryo_id/time_bin to identify missing keys."""
    merged = pd.merge(
        df_baseline[KEY_COLS],
        df_balanced[KEY_COLS],
        on=KEY_COLS,
        how="outer",
        indicator=True,
    )

    missing_in_balanced = merged["_merge"].eq("left_only").sum()
    missing_in_baseline = merged["_merge"].eq("right_only").sum()

    print(f"    Key comparison ({tag}):")
    print(f"      Rows only in baseline : {missing_in_balanced}")
    print(f"      Rows only in balanced : {missing_in_baseline}")

    if missing_in_balanced or missing_in_baseline:
        diff_path = OUTPUT_DIR / f"missing_keys_{tag}.csv"
        merged.to_csv(diff_path, index=False)
        print(f"      Saved missing key details to {diff_path}")

    return merged


def per_embryo_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate counts per embryo to understand coverage."""
    agg = (
        df.groupby(["true_label", "embryo_id"])
        .agg(
            n_rows=("time_bin", "size"),
            n_unique_time_bins=("time_bin", "nunique"),
            time_min=("time_bin", "min"),
            time_max=("time_bin", "max"),
        )
        .reset_index()
    )
    return agg


def compare_per_embryo(
    counts_baseline: pd.DataFrame,
    counts_balanced: pd.DataFrame,
    tag: str,
) -> pd.DataFrame:
    """Join per-embryo coverage tables and report differences."""
    comparison = pd.merge(
        counts_baseline,
        counts_balanced,
        on=["true_label", "embryo_id"],
        how="outer",
        suffixes=("_baseline", "_balanced"),
    )

    for col in ["n_rows", "n_unique_time_bins"]:
        comparison[f"{col}_baseline"] = comparison[f"{col}_baseline"].fillna(0).astype(int)
        comparison[f"{col}_balanced"] = comparison[f"{col}_balanced"].fillna(0).astype(int)
        comparison[f"{col}_diff"] = comparison[f"{col}_balanced"] - comparison[f"{col}_baseline"]

    for col in ["time_min", "time_max"]:
        comparison[f"{col}_baseline"] = comparison[f"{col}_baseline"].fillna(np.nan)
        comparison[f"{col}_balanced"] = comparison[f"{col}_balanced"].fillna(np.nan)

    diff_rows = comparison[(comparison["n_rows_diff"] != 0) | (comparison["n_unique_time_bins_diff"] != 0)]
    n_diff = len(diff_rows)

    print(f"    Per-embryo coverage differences ({tag}): {n_diff}")
    if n_diff:
        diff_path = OUTPUT_DIR / f"coverage_differences_{tag}.csv"
        diff_rows.sort_values(["true_label", "n_rows_diff"], ascending=[True, False]).to_csv(diff_path, index=False)
        print(f"      Saved detailed coverage differences to {diff_path}")

    summary = comparison.groupby("true_label")[["n_rows_diff", "n_unique_time_bins_diff"]].agg(["mean", "min", "max"])
    if not summary.empty:
        print("    Coverage diff summary by true_label:")
        print(summary)

    return comparison


def compare_prediction_values(
    df_baseline: pd.DataFrame,
    df_balanced: pd.DataFrame,
    tag: str,
) -> None:
    """Compare prediction values for matching embryo × time_bin rows."""
    merged = pd.merge(
        df_baseline,
        df_balanced,
        on=KEY_COLS,
        how="inner",
        suffixes=("_baseline", "_balanced"),
    )

    print(f"    Value comparison ({tag}):")
    print(f"      Matched rows: {len(merged)}")

    if merged.empty:
        print("      No overlapping embryo × time_bin rows to compare.")
        return

    value_cols = ["pred_proba", "confidence", "signed_margin"]

    for col in value_cols:
        diff_col = merged[f"{col}_balanced"] - merged[f"{col}_baseline"]
        max_abs_diff = diff_col.abs().max()
        mean_abs_diff = diff_col.abs().mean()
        print(f"      {col}: mean |Δ|={mean_abs_diff:.6f}, max |Δ|={max_abs_diff:.6f}")

    # Optional: save extreme differences
    largest_diffs = {}
    for col in value_cols:
        diff_col = (merged[f"{col}_balanced"] - merged[f"{col}_baseline"]).abs()
        largest_idx = diff_col.nlargest(5).index
        largest_diffs[col] = merged.loc[largest_idx, KEY_COLS + [f"{col}_baseline", f"{col}_balanced"]]

    extremes_records = []
    for col, df in largest_diffs.items():
        if df.empty:
            continue
        df = df.copy()
        df["metric"] = col
        extremes_records.append(df)

    if extremes_records:
        extremes = pd.concat(extremes_records, ignore_index=True)
        extremes_path = OUTPUT_DIR / f"value_differences_{tag}.csv"
        extremes.to_csv(extremes_path, index=False)
        print(f"      Saved top absolute differences to {extremes_path}")


def analyze_comparison(gene: str, group1: str, group2: str) -> None:
    """Run the full comparison workflow for a given gene comparison."""
    tag = f"{gene}_{group1.split('_')[-1]}_vs_{group2.split('_')[-1]}"
    print("\n" + "=" * 80)
    print(f"COMPARING SAVED PREDICTIONS: {gene.upper()} :: {group1} vs {group2}")
    print("=" * 80)

    df_baseline, df_balanced = load_predictions(gene, group1, group2)

    print(f"  Loaded baseline rows: {len(df_baseline)} (embryos={df_baseline['embryo_id'].nunique()})")
    print(f"  Loaded balanced rows: {len(df_balanced)} (embryos={df_balanced['embryo_id'].nunique()})")

    check_duplicate_keys(df_baseline, f"baseline_{tag}")
    check_duplicate_keys(df_balanced, f"balanced_{tag}")

    merged_keys = merged_key_report(df_baseline, df_balanced, tag=tag)
    matched_rows = merged_keys["_merge"].eq("both").sum()
    print(f"    Matched embryo × time_bin keys: {matched_rows}")

    counts_baseline = per_embryo_counts(df_baseline)
    counts_balanced = per_embryo_counts(df_balanced)

    comparison = compare_per_embryo(counts_baseline, counts_balanced, tag=tag)
    comparison_path = OUTPUT_DIR / f"per_embryo_coverage_{tag}.csv"
    comparison.sort_values(["true_label", "embryo_id"]).to_csv(comparison_path, index=False)
    print(f"    Saved per-embryo coverage table to {comparison_path}")

    compare_prediction_values(df_baseline, df_balanced, tag=tag)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs balanced saved prediction CSV files."
    )
    parser.add_argument(
        "--gene",
        choices=sorted(COMPARISONS.keys()),
        help="Gene to analyze. If omitted, all genes are processed.",
    )
    parser.add_argument(
        "--group1",
        help="First group for comparison (requires --gene).",
    )
    parser.add_argument(
        "--group2",
        help="Second group for comparison (requires --gene).",
    )
    return parser.parse_args()


def determine_comparisons(args: argparse.Namespace) -> Iterable[Tuple[str, str, str]]:
    if args.gene and args.group1 and args.group2:
        return [(args.gene, args.group1, args.group2)]

    if args.gene:
        return [(args.gene, g1, g2) for g1, g2 in COMPARISONS[args.gene]]

    combos: List[Tuple[str, str, str]] = []
    for gene, pairs in COMPARISONS.items():
        combos.extend((gene, g1, g2) for g1, g2 in pairs)
    return combos


def main() -> None:
    args = parse_args()
    comparisons = list(determine_comparisons(args))

    if not comparisons:
        print("No comparisons selected. Check CLI arguments.")
        return

    print(f"Output directory: {OUTPUT_DIR}")
    for gene, group1, group2 in comparisons:
        try:
            analyze_comparison(gene, group1, group2)
        except FileNotFoundError as exc:
            print(f"  Skipping comparison due to missing data: {exc}")

    print("\n" + "=" * 80)
    print("PREDICTION CSV COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
