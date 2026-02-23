"""
Tutorial 05c: Fix Spline Fitting (Fit Each Cluster Separately)

The grouped spline fitting with group_by parameter has issues.
This script fits each cluster separately to ensure reliable results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "05"
RESULTS_DIR = OUTPUT_DIR / "results"

meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"
SOURCE_EXPERIMENTS = ["20260122", "20260124"]

from analyze.spline_fitting import spline_fit_wrapper

print("=" * 80)
print("TUTORIAL 05C: FIXING SPLINE FITTING")
print("=" * 80)

# Load projection assignments
PROJECTION_DIR = OUTPUT_DIR / "figures" / "04" / "projection_results"
PROJECTION_CSV = PROJECTION_DIR / "combined_projection_bootstrap.csv"
print(f"Loading projections: {PROJECTION_CSV}")
proj = pd.read_csv(PROJECTION_CSV, low_memory=False)

# Load source trajectories
print("Loading source trajectories...")
source_dfs = []
for exp_id in SOURCE_EXPERIMENTS:
    df_exp = pd.read_csv(meta_dir / f"qc_staged_{exp_id}.csv", low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp_id
    source_dfs.append(df_exp)

df = pd.concat(source_dfs, ignore_index=True)

# Merge cluster labels
proj_cols = ["embryo_id", "cluster_label", "membership"]
proj = proj[proj_cols].drop_duplicates(subset="embryo_id")
df = df.merge(proj, on="embryo_id", how="inner")
df = df[df["cluster_label"].notna()].copy()

print(f"Trajectories with labels: {df['embryo_id'].nunique()} embryos")
print(f"Cluster labels: {sorted(df['cluster_label'].unique())}")

coords = ["baseline_deviation_normalized", "total_length_um"]
print(f"Using coordinates: {coords}")

# Fit splines for each cluster SEPARATELY
print("\n" + "=" * 80)
print("Fitting splines for each cluster separately...")
print("=" * 80)

clusters = sorted(df["cluster_label"].unique())
spline_dfs = []

for cluster in tqdm(clusters, desc="Fitting splines"):
    df_cluster = df[df["cluster_label"] == cluster].copy()
    print(f"\n{cluster}: {len(df_cluster)} rows, {df_cluster['embryo_id'].nunique()} embryos")

    try:
        spline_cluster = spline_fit_wrapper(
            df_cluster,
            group_by=None,  # Fit single spline per cluster
            pca_cols=coords,
            stage_col="predicted_stage_hpf",
            n_bootstrap=50,
            bootstrap_size=2500,
            n_spline_points=200,
            time_window=2,
        )

        # Add cluster label
        spline_cluster["cluster_label"] = cluster

        # Check for NaNs
        nan_count = spline_cluster[coords[0]].isna().sum()
        if nan_count > 0:
            print(f"  WARNING: {nan_count}/200 points are NaN")
        else:
            print(f"  ✓ All 200 points valid")

        spline_dfs.append(spline_cluster)

    except Exception as e:
        print(f"  ✗ ERROR: {e}")

# Combine all clusters
spline_df = pd.concat(spline_dfs, ignore_index=True)
print(f"\n✓ Fitted splines for {len(spline_dfs)} clusters")
print(f"  Total rows: {len(spline_df)}")
print(f"  Valid data points: {spline_df[coords[0]].notna().sum()}/{len(spline_df)}")

# Save results
csv_path = RESULTS_DIR / "05_projection_splines_by_cluster_FIXED.csv"
pkl_path = RESULTS_DIR / "05_projection_splines_by_cluster_FIXED.pkl"

spline_df.to_csv(csv_path, index=False)
print(f"\n✓ Saved: {csv_path}")

import pickle
with open(pkl_path, "wb") as f:
    pickle.dump(spline_df, f)
print(f"✓ Saved: {pkl_path}")

# Create 2D visualizations
print("\n" + "=" * 80)
print("Creating 2D visualizations...")
print("=" * 80)

n_cols = 2
n_rows = int(np.ceil(len(clusters) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

for idx, cluster in enumerate(clusters):
    r, c = divmod(idx, n_cols)
    ax = axes[r][c]

    df_c = df[df["cluster_label"] == cluster]
    spline_c = spline_df[spline_df["cluster_label"] == cluster]

    # Raw data
    ax.scatter(
        df_c[coords[0]],
        df_c[coords[1]],
        s=6,
        alpha=0.2,
        color='gray',
        label="points"
    )

    # Spline
    if not spline_c.empty and not spline_c[coords[0]].isna().all():
        ax.plot(
            spline_c[coords[0]],
            spline_c[coords[1]],
            color="red",
            linewidth=2,
            label="spline"
        )

    ax.set_title(cluster)
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for idx in range(len(clusters), n_rows * n_cols):
    r, c = divmod(idx, n_cols)
    axes[r][c].axis("off")

fig.suptitle("Spline Fits per Cluster (FIXED)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = FIGURES_DIR / "05_projection_splines_by_cluster_FIXED.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path}")
plt.close(fig)

# Create per-cluster temporal plots
print("\n" + "=" * 80)
print("Creating temporal trajectory plots...")
print("=" * 80)

for cluster in clusters:
    df_c = df[df["cluster_label"] == cluster]
    spline_c = spline_df[spline_df["cluster_label"] == cluster]

    if spline_c[coords[0]].isna().all():
        print(f"  Skipping {cluster} (no valid spline)")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: baseline_deviation vs time
    ax = axes[0]
    ax.scatter(
        df_c["predicted_stage_hpf"],
        df_c[coords[0]],
        s=4,
        alpha=0.3,
        color='gray',
        label='Raw data'
    )
    # Add spline (need to add time coordinate)
    time_min = df["predicted_stage_hpf"].min()
    time_max = df["predicted_stage_hpf"].max()
    spline_c_copy = spline_c.copy()
    spline_c_copy["predicted_stage_hpf"] = np.linspace(time_min, time_max, len(spline_c))
    ax.plot(
        spline_c_copy["predicted_stage_hpf"],
        spline_c_copy[coords[0]],
        color='red',
        linewidth=3,
        label='Spline fit'
    )
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel(coords[0])
    ax.set_title(f"{cluster}: Curvature Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: length vs time
    ax = axes[1]
    ax.scatter(
        df_c["predicted_stage_hpf"],
        df_c[coords[1]],
        s=4,
        alpha=0.3,
        color='gray',
        label='Raw data'
    )
    ax.plot(
        spline_c_copy["predicted_stage_hpf"],
        spline_c_copy[coords[1]],
        color='red',
        linewidth=3,
        label='Spline fit'
    )
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel(coords[1])
    ax.set_title(f"{cluster}: Length Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig_path = FIGURES_DIR / f"05_{cluster.replace(' ', '_')}_temporal_FIXED.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {fig_path.name}")
    plt.close(fig)

print("\n" + "=" * 80)
print("✓ Tutorial 05c complete - splines fixed and visualized!")
print("=" * 80)
