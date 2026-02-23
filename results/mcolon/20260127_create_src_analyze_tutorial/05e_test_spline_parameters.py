"""
Test different spline_fit_wrapper parameters to find what works
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"

# Load data
proj = pd.read_csv(OUTPUT_DIR / "figures/04/projection_results/combined_projection_bootstrap.csv")
dfs = []
for exp in ["20260122", "20260124"]:
    df_exp = pd.read_csv(meta_dir / f"qc_staged_{exp}.csv", low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp
    dfs.append(df_exp)

df = pd.concat(dfs, ignore_index=True)
proj_cols = ["embryo_id", "cluster_label", "membership"]
proj = proj[proj_cols].drop_duplicates(subset="embryo_id")
df = df.merge(proj, on="embryo_id", how="inner")
df = df[df["cluster_label"].notna()].copy()

coords = ["baseline_deviation_normalized", "total_length_um"]

from analyze.spline_fitting import spline_fit_wrapper

print("="*80)
print("TESTING SPLINE PARAMETERS")
print("="*80)
print(f"Data: {len(df)} rows, {df['embryo_id'].nunique()} embryos")
print(f"Clusters: {sorted(df['cluster_label'].unique())}")
print()

# Test 1: Single cluster, 1 bootstrap
print("TEST 1: Single cluster (Not Penetrant), 1 bootstrap")
df_test = df[df["cluster_label"] == "Not Penetrant"].copy()
result = spline_fit_wrapper(
    df_test,
    group_by=None,
    pca_cols=coords,
    stage_col="predicted_stage_hpf",
    n_bootstrap=1,
    n_spline_points=200,
)
print(f"  Result: {result.shape}, NaNs: {result[coords[0]].isna().sum()}/200")
print()

# Test 2: All clusters with group_by, 1 bootstrap
print("TEST 2: All clusters with group_by, 1 bootstrap")
result = spline_fit_wrapper(
    df,
    group_by="cluster_label",
    pca_cols=coords,
    stage_col="predicted_stage_hpf",
    n_bootstrap=1,
    n_spline_points=200,
)
print(f"  Result: {result.shape}")
for cluster in result["cluster_label"].unique():
    nans = result[result["cluster_label"] == cluster][coords[0]].isna().sum()
    print(f"    {cluster}: {nans}/200 NaNs")
print()

# Test 3: All clusters with group_by, 10 bootstraps
print("TEST 3: All clusters with group_by, 10 bootstraps")
result = spline_fit_wrapper(
    df,
    group_by="cluster_label",
    pca_cols=coords,
    stage_col="predicted_stage_hpf",
    n_bootstrap=10,
    n_spline_points=200,
)
print(f"  Result: {result.shape}")
for cluster in result["cluster_label"].unique():
    nans = result[result["cluster_label"] == cluster][coords[0]].isna().sum()
    print(f"    {cluster}: {nans}/200 NaNs")
print()

# Test 4: Different bandwidth
print("TEST 4: All clusters, 1 bootstrap, bandwidth=0.1")
result = spline_fit_wrapper(
    df,
    group_by="cluster_label",
    pca_cols=coords,
    stage_col="predicted_stage_hpf",
    n_bootstrap=1,
    bandwidth=0.1,
    n_spline_points=200,
)
print(f"  Result: {result.shape}")
for cluster in result["cluster_label"].unique():
    nans = result[result["cluster_label"] == cluster][coords[0]].isna().sum()
    print(f"    {cluster}: {nans}/200 NaNs")
print()

print("="*80)
print("PARAMETER TESTING COMPLETE")
print("="*80)
