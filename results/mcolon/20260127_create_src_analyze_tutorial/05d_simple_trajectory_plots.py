"""
Tutorial 05d: Simple Trajectory Visualizations (No Spline Fitting)

Since spline fitting is failing, create simple visualizations of the raw
trajectories to show the phenotypic patterns. Plot rolling means instead
of complex spline fits.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "05"
RESULTS_DIR = OUTPUT_DIR / "results"

meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"
SOURCE_EXPERIMENTS = ["20260122", "20260124"]

print("=" * 80)
print("TUTORIAL 05D: SIMPLE TRAJECTORY VISUALIZATIONS")
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

# Create smooth trend lines using rolling mean per cluster
print("\n" + "=" * 80)
print("Computing rolling means for each cluster...")
print("=" * 80)

clusters = sorted(df["cluster_label"].unique())
cluster_colors = {
    "Not Penetrant": "#2ca02c",  # green
    "Low_to_High": "#1f77b4",    # blue
    "High_to_Low": "#ff7f0e",    # orange
    "Intermediate": "#d62728"     # red
}

# Sort by time for rolling mean
df = df.sort_values("predicted_stage_hpf").copy()

# Compute rolling means per cluster
window = 50  # points for rolling mean
smoothed_data = {}

for cluster in clusters:
    df_c = df[df["cluster_label"] == cluster].sort_values("predicted_stage_hpf").copy()

    # Compute rolling mean
    df_c["baseline_deviation_smoothed"] = df_c[coords[0]].rolling(
        window=window, center=True, min_periods=10
    ).mean()
    df_c["length_smoothed"] = df_c[coords[1]].rolling(
        window=window, center=True, min_periods=10
    ).mean()

    smoothed_data[cluster] = df_c

print(f"Computed rolling means for {len(clusters)} clusters")

# Create comprehensive visualizations
print("\n" + "=" * 80)
print("Creating visualizations...")
print("=" * 80)

# Plot 1: Temporal trajectories (both features vs time)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Baseline deviation vs time
ax = axes[0]
for cluster in clusters:
    df_c = smoothed_data[cluster]
    color = cluster_colors.get(cluster, "gray")

    # Raw data (faint)
    ax.scatter(
        df_c["predicted_stage_hpf"],
        df_c[coords[0]],
        s=1,
        alpha=0.1,
        color=color
    )

    # Smoothed trend
    ax.plot(
        df_c["predicted_stage_hpf"],
        df_c["baseline_deviation_smoothed"],
        color=color,
        linewidth=3,
        label=cluster,
        alpha=0.8
    )

ax.set_xlabel("Time (hpf)", fontsize=12)
ax.set_ylabel("Baseline Deviation (Normalized)", fontsize=12)
ax.set_title("Body Curvature Over Time", fontsize=14, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

# Length vs time
ax = axes[1]
for cluster in clusters:
    df_c = smoothed_data[cluster]
    color = cluster_colors.get(cluster, "gray")

    # Raw data (faint)
    ax.scatter(
        df_c["predicted_stage_hpf"],
        df_c[coords[1]],
        s=1,
        alpha=0.1,
        color=color
    )

    # Smoothed trend
    ax.plot(
        df_c["predicted_stage_hpf"],
        df_c["length_smoothed"],
        color=color,
        linewidth=3,
        label=cluster,
        alpha=0.8
    )

ax.set_xlabel("Time (hpf)", fontsize=12)
ax.set_ylabel("Total Length (µm)", fontsize=12)
ax.set_title("Body Length Over Time", fontsize=14, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig_path = FIGURES_DIR / "05_temporal_trajectories_smoothed.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path.name}")
plt.close()

# Plot 2: 2D Feature space with trajectories
fig, ax = plt.subplots(figsize=(10, 8))

for cluster in clusters:
    df_c = smoothed_data[cluster]
    color = cluster_colors.get(cluster, "gray")

    # Raw data points
    ax.scatter(
        df_c[coords[0]],
        df_c[coords[1]],
        s=4,
        alpha=0.2,
        color=color,
        label=f"{cluster} (raw)"
    )

    # Smoothed trajectory
    valid = df_c[["baseline_deviation_smoothed", "length_smoothed"]].notna().all(axis=1)
    ax.plot(
        df_c.loc[valid, "baseline_deviation_smoothed"],
        df_c.loc[valid, "length_smoothed"],
        color=color,
        linewidth=3,
        alpha=0.8,
        label=f"{cluster} (trend)"
    )

ax.set_xlabel(coords[0], fontsize=12)
ax.set_ylabel(coords[1], fontsize=12)
ax.set_title("Morphological Trajectories (2D Feature Space)", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig_path = FIGURES_DIR / "05_feature_space_trajectories.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path.name}")
plt.close()

# Plot 3: Per-cluster detailed views (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, cluster in enumerate(clusters):
    ax = axes[idx]
    df_c = smoothed_data[cluster]
    color = cluster_colors.get(cluster, "gray")

    # 2D plot: curvature vs length
    ax.scatter(
        df_c[coords[0]],
        df_c[coords[1]],
        c=df_c["predicted_stage_hpf"],
        cmap="viridis",
        s=10,
        alpha=0.5
    )

    # Add smoothed trajectory
    valid = df_c[["baseline_deviation_smoothed", "length_smoothed"]].notna().all(axis=1)
    if valid.sum() > 1:
        ax.plot(
            df_c.loc[valid, "baseline_deviation_smoothed"],
            df_c.loc[valid, "length_smoothed"],
            color="red",
            linewidth=2,
            label="Trend"
        )

    ax.set_title(f"{cluster} ({df_c['embryo_id'].nunique()} embryos)", fontweight="bold")
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-Cluster Trajectories (colored by time)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = FIGURES_DIR / "05_per_cluster_trajectories.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path.name}")
plt.close()

# Save smoothed data
print("\n" + "=" * 80)
print("Saving smoothed trajectory data...")
print("=" * 80)

smoothed_combined = pd.concat(smoothed_data.values(), ignore_index=True)
csv_path = RESULTS_DIR / "05_smoothed_trajectories.csv"
smoothed_combined.to_csv(csv_path, index=False)
print(f"✓ Saved: {csv_path}")

print("\n" + "=" * 80)
print("✓ Tutorial 05d complete - simple trajectory plots created!")
print("=" * 80)
print("\nSummary:")
print(f"  - {len(clusters)} clusters visualized")
print(f"  - Rolling window: {window} points")
print(f"  - 3 PNG visualizations created")
print(f"  - Smoothed data saved to CSV")
