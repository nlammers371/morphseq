"""
Debug script to test spline fitting and create simple PNG visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"

print("=" * 80)
print("DEBUG: SPLINE FITTING")
print("=" * 80)

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

print(f"Data loaded: {len(df)} rows, {df['embryo_id'].nunique()} embryos")

# Test spline fitting on one cluster
test_cluster = "Not Penetrant"
df_test = df[df["cluster_label"] == test_cluster].copy()
print(f"\nTesting on cluster: {test_cluster}")
print(f"  Rows: {len(df_test)}")
print(f"  Embryos: {df_test['embryo_id'].nunique()}")

coords = ["baseline_deviation_normalized", "total_length_um"]
print(f"  Coordinates: {coords}")
print(f"  Time column: predicted_stage_hpf")

# Try simple spline fitting
from analyze.spline_fitting import spline_fit_wrapper

print("\n" + "=" * 80)
print("ATTEMPTING SPLINE FIT (1 bootstrap iteration)")
print("=" * 80)

try:
    spline_df = spline_fit_wrapper(
        df_test,
        group_by=None,  # Single group
        pca_cols=coords,
        stage_col="predicted_stage_hpf",
        n_bootstrap=1,  # Just 1 to test quickly
        bootstrap_size=2500,
        n_spline_points=200,
        time_window=2,
    )

    print(f"\n✓ Spline fitting succeeded!")
    print(f"  Shape: {spline_df.shape}")
    print(f"  Columns: {list(spline_df.columns)}")
    print(f"\nFirst few rows:")
    print(spline_df.head())
    print(f"\nData check:")
    print(spline_df[coords].describe())
    print(f"\nNaN count:")
    print(spline_df[coords].isna().sum())

except Exception as e:
    print(f"\n✗ Spline fitting failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    spline_df = None

# Create simple 2D visualization regardless
print("\n" + "=" * 80)
print("CREATING SIMPLE 2D VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: baseline_deviation vs time
ax = axes[0]
for cluster in sorted(df["cluster_label"].unique()):
    df_c = df[df["cluster_label"] == cluster]
    ax.scatter(
        df_c["predicted_stage_hpf"],
        df_c["baseline_deviation_normalized"],
        s=2,
        alpha=0.3,
        label=cluster
    )
ax.set_xlabel("Time (hpf)")
ax.set_ylabel("Baseline Deviation (Normalized)")
ax.set_title("Curvature Over Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: length vs time
ax = axes[1]
for cluster in sorted(df["cluster_label"].unique()):
    df_c = df[df["cluster_label"] == cluster]
    ax.scatter(
        df_c["predicted_stage_hpf"],
        df_c["total_length_um"],
        s=2,
        alpha=0.3,
        label=cluster
    )
ax.set_xlabel("Time (hpf)")
ax.set_ylabel("Total Length (µm)")
ax.set_title("Length Over Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Raw Data: Morphological Features Over Time", fontsize=14, fontweight='bold')
fig.tight_layout()

output_path = OUTPUT_DIR / "figures/05/debug_raw_trajectories_2d.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Saved: {output_path}")
plt.close()

# If spline worked, plot it too
if spline_df is not None and not spline_df[coords[0]].isna().all():
    print("\n" + "=" * 80)
    print("CREATING SPLINE OVERLAY VISUALIZATION")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Raw data
    ax.scatter(
        df_test[coords[0]],
        df_test[coords[1]],
        s=4,
        alpha=0.2,
        color='gray',
        label='Raw data'
    )

    # Spline
    ax.plot(
        spline_df[coords[0]],
        spline_df[coords[1]],
        color='red',
        linewidth=3,
        label='Fitted spline'
    )

    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])
    ax.set_title(f"Spline Fit: {test_cluster}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = OUTPUT_DIR / "figures/05/debug_spline_overlay.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
