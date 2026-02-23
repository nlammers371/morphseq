"""
Debug LocalPrincipalCurve fitting to find why it returns NaN
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    dfs.append(df_exp)

df = pd.concat(dfs, ignore_index=True)
proj_cols = ["embryo_id", "cluster_label"]
proj = proj[proj_cols].drop_duplicates(subset="embryo_id")
df = df.merge(proj, on="embryo_id", how="inner")
df = df[df["cluster_label"].notna()].copy()

coords = ["baseline_deviation_normalized", "total_length_um"]

print("="*80)
print("DEBUGGING LocalPrincipalCurve FITTING")
print("="*80)

# Test on Intermediate cluster (one that fails)
cluster = "Intermediate"
df_cluster = df[df["cluster_label"] == cluster].copy()
print(f"\nTesting cluster: {cluster}")
print(f"  Rows: {len(df_cluster)}")
print(f"  Embryos: {df_cluster['embryo_id'].nunique()}")

from analyze.spline_fitting.lpc_model import LocalPrincipalCurve

# Setup parameters (same as spline_fit_wrapper)
stage_col = "predicted_stage_hpf"
bandwidth = 0.5
h = None
max_iter = 2500
tol = 1e-5
angle_penalty_exp = 1
n_spline_points = 200
time_window = 2
bootstrap_size = min(len(df_cluster), 2500)

# Extract coordinates
coord_array = df_cluster[coords].values
print(f"\nCoordinate array shape: {coord_array.shape}")
print(f"  Coord range: [{coord_array.min():.3f}, {coord_array.max():.3f}]")
print(f"  NaN count: {np.isnan(coord_array).sum()}")

# Compute anchor points
min_time = df_cluster[stage_col].min()
early_mask = (df_cluster[stage_col] >= min_time) & (df_cluster[stage_col] < min_time + time_window)
early_points = df_cluster.loc[early_mask, coords].values

max_time = df_cluster[stage_col].max()
late_mask = df_cluster[stage_col] >= (max_time - time_window)
late_points = df_cluster.loc[late_mask, coords].values

print(f"\nAnchor points:")
print(f"  Time range: [{min_time:.1f}, {max_time:.1f}] hpf")
print(f"  Early points (within {time_window} hpf of start): {len(early_points)}")
print(f"  Late points (within {time_window} hpf of end): {len(late_points)}")

# Try fitting LPC with different bootstrap samples
print(f"\nTrying {3} bootstrap iterations...")
rng = np.random.RandomState(42)
obs_weights = np.ones(len(coord_array)) / len(coord_array)

for boot_iter in range(3):
    print(f"\n--- Bootstrap iteration {boot_iter + 1} ---")

    # Sample data
    subset_indices = rng.choice(len(coord_array), size=bootstrap_size, replace=True, p=obs_weights)
    coord_subset = coord_array[subset_indices, :]
    print(f"  Subset shape: {coord_subset.shape}")
    print(f"  Subset range: [{coord_subset.min():.3f}, {coord_subset.max():.3f}]")

    # Random anchor points
    start_idx = rng.choice(len(early_points))
    stop_idx = rng.choice(len(late_points))
    start_point = early_points[start_idx, :]
    stop_point = late_points[stop_idx, :]
    print(f"  Start point: {start_point}")
    print(f"  Stop point: {stop_point}")

    # Fit LPC
    lpc = LocalPrincipalCurve(
        bandwidth=bandwidth,
        h=h,
        max_iter=max_iter,
        tol=tol,
        angle_penalty_exp=angle_penalty_exp
    )

    try:
        lpc.fit(
            coord_subset,
            start_points=start_point[None, :],
            end_point=stop_point[None, :],
            num_points=n_spline_points
        )

        # Check result
        if hasattr(lpc, 'cubic_splines') and lpc.cubic_splines is not None:
            spline = lpc.cubic_splines[0]
            print(f"  ✓ Fit succeeded!")
            print(f"    Spline shape: {spline.shape}")
            print(f"    Spline range: [{np.nanmin(spline):.3f}, {np.nanmax(spline):.3f}]")
            print(f"    NaN count: {np.isnan(spline).sum()}/{spline.size}")
            print(f"    First 3 points:\n{spline[:3]}")
        else:
            print(f"  ✗ Fit failed - no cubic_splines attribute")

    except Exception as e:
        print(f"  ✗ Fit failed with error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
