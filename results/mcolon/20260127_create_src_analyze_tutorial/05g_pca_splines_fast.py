"""
Tutorial 05g: PCA-based Spline Fitting (Fast, n=1 bootstrap)

1. Compute PCA on morphological features
2. Fit splines on PC1, PC2, PC3 per cluster
3. Visualize with 3D plots
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "05"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"
SOURCE_EXPERIMENTS = ["20260122", "20260124"]

print("=" * 80)
print("TUTORIAL 05G: PCA-BASED SPLINE FITTING (FAST)")
print("=" * 80)

# Load projection assignments
PROJECTION_DIR = OUTPUT_DIR / "figures" / "04" / "projection_results"
PROJECTION_CSV = PROJECTION_DIR / "combined_projection_bootstrap.csv"
print(f"\nLoading projections: {PROJECTION_CSV}")
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

# ============================================================================
# Step 1: Compute PCA on morphological features
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: COMPUTE PCA")
print("=" * 80)

# Use two morphological features
morph_features = ["baseline_deviation_normalized", "total_length_um"]
print(f"Features: {morph_features}")

# Extract feature array
X = df[morph_features].values
print(f"Feature array shape: {X.shape}")

# Fit PCA (2D -> 2D, but could do 3D if we had more features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Add PCA coordinates to dataframe
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

print(f"\nPCA explained variance:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.1f}%")

pca_cols = ["PC1", "PC2"]

# ============================================================================
# Step 2: Fit splines per cluster on PCA coordinates (n=1 bootstrap)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: FIT SPLINES PER CLUSTER (n=1 bootstrap)")
print("=" * 80)

from analyze.spline_fitting.lpc_model import LocalPrincipalCurve
from tqdm import tqdm

def fit_spline_single_bootstrap(df_cluster, pca_cols, stage_col="predicted_stage_hpf"):
    """Fit single spline (n=1 bootstrap) with validation."""

    coord_array = df_cluster[pca_cols].values

    # Compute anchor points
    time_window = 2
    min_time = df_cluster[stage_col].min()
    early_mask = (df_cluster[stage_col] >= min_time) & (df_cluster[stage_col] < min_time + time_window)
    early_points = df_cluster.loc[early_mask, pca_cols].values

    max_time = df_cluster[stage_col].max()
    late_mask = df_cluster[stage_col] >= (max_time - time_window)
    late_points = df_cluster.loc[late_mask, pca_cols].values

    if len(early_points) == 0 or len(late_points) == 0:
        print(f"    Warning: No anchor points found")
        return None

    # Pick random anchors
    rng = np.random.RandomState(42)
    start_point = early_points[rng.choice(len(early_points)), :]
    stop_point = late_points[rng.choice(len(late_points)), :]

    # Fit LPC with retry logic
    max_retries = 5
    for attempt in range(max_retries):
        lpc = LocalPrincipalCurve(
            bandwidth=0.5,
            h=None,
            max_iter=2500,
            tol=1e-5,
            angle_penalty_exp=1
        )

        try:
            lpc.fit(
                coord_array,
                start_points=start_point[None, :],
                end_point=stop_point[None, :],
                num_points=200
            )

            # Validate result
            if (hasattr(lpc, 'cubic_splines') and
                lpc.cubic_splines is not None and
                len(lpc.cubic_splines) > 0 and
                lpc.cubic_splines[0] is not None):

                spline = lpc.cubic_splines[0]
                if not np.isnan(spline).any():
                    return spline

            # Failed, retry with different anchor
            start_point = early_points[rng.choice(len(early_points)), :]
            stop_point = late_points[rng.choice(len(late_points)), :]

        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            continue

    print(f"    Warning: All {max_retries} attempts failed")
    return None


clusters = sorted(df["cluster_label"].unique())
spline_results = []

for cluster in tqdm(clusters, desc="Fitting splines"):
    df_cluster = df[df["cluster_label"] == cluster].copy()
    print(f"\n{cluster}: {len(df_cluster)} rows, {df_cluster['embryo_id'].nunique()} embryos")

    spline = fit_spline_single_bootstrap(df_cluster, pca_cols)

    if spline is not None:
        # Create DataFrame
        spline_df = pd.DataFrame(spline, columns=pca_cols)
        spline_df["cluster_label"] = cluster
        spline_df["spline_point_index"] = range(len(spline_df))
        spline_results.append(spline_df)
        print(f"  ✓ Spline fit succeeded: {spline.shape}")
    else:
        print(f"  ✗ Spline fit failed")

# Combine all splines
if spline_results:
    spline_df_all = pd.concat(spline_results, ignore_index=True)
    print(f"\n✓ Fitted splines for {len(spline_results)}/{len(clusters)} clusters")
else:
    print("\n✗ No splines fitted successfully")
    sys.exit(1)

# Save spline results
csv_path = RESULTS_DIR / "05_pca_splines.csv"
spline_df_all.to_csv(csv_path, index=False)
print(f"\n✓ Saved: {csv_path}")

# ============================================================================
# Step 3: Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATE VISUALIZATIONS")
print("=" * 80)

# Plot 1: PCA space with splines (2D)
fig, ax = plt.subplots(figsize=(10, 8))

cluster_colors = {
    "Not Penetrant": "#2ca02c",
    "Low_to_High": "#1f77b4",
    "High_to_Low": "#ff7f0e",
    "Intermediate": "#d62728"
}

for cluster in clusters:
    df_c = df[df["cluster_label"] == cluster]
    spline_c = spline_df_all[spline_df_all["cluster_label"] == cluster]
    color = cluster_colors.get(cluster, "gray")

    # Raw data
    ax.scatter(
        df_c["PC1"],
        df_c["PC2"],
        s=4,
        alpha=0.2,
        color=color,
        label=f"{cluster} (data)"
    )

    # Spline
    if not spline_c.empty:
        ax.plot(
            spline_c["PC1"],
            spline_c["PC2"],
            color=color,
            linewidth=3,
            alpha=0.8,
            label=f"{cluster} (spline)"
        )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title("PCA Space with Spline Trajectories", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

fig_path = FIGURES_DIR / "05_pca_splines_2d.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path.name}")
plt.close()

# Plot 2: Per-cluster detailed views
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, cluster in enumerate(clusters):
    ax = axes[idx]
    df_c = df[df["cluster_label"] == cluster]
    spline_c = spline_df_all[spline_df_all["cluster_label"] == cluster]
    color = cluster_colors.get(cluster, "gray")

    # Raw data
    ax.scatter(
        df_c["PC1"],
        df_c["PC2"],
        c=df_c["predicted_stage_hpf"],
        cmap="viridis",
        s=10,
        alpha=0.5
    )

    # Spline
    if not spline_c.empty:
        ax.plot(
            spline_c["PC1"],
            spline_c["PC2"],
            color="red",
            linewidth=2,
            label="Spline"
        )

    ax.set_title(f"{cluster} ({df_c['embryo_id'].nunique()} embryos)", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-Cluster PCA Trajectories (colored by time)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = FIGURES_DIR / "05_pca_splines_per_cluster.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path.name}")
plt.close()

# Plot 3: Original feature space with back-projected splines
print("\nBack-projecting splines to original feature space...")

# Transform spline coordinates back to original space
for cluster in clusters:
    spline_c = spline_df_all[spline_df_all["cluster_label"] == cluster]
    if not spline_c.empty:
        spline_pca = spline_c[pca_cols].values
        spline_orig = pca.inverse_transform(spline_pca)
        spline_df_all.loc[spline_c.index, "baseline_deviation_normalized_proj"] = spline_orig[:, 0]
        spline_df_all.loc[spline_c.index, "total_length_um_proj"] = spline_orig[:, 1]

fig, ax = plt.subplots(figsize=(10, 8))

for cluster in clusters:
    df_c = df[df["cluster_label"] == cluster]
    spline_c = spline_df_all[spline_df_all["cluster_label"] == cluster]
    color = cluster_colors.get(cluster, "gray")

    # Raw data
    ax.scatter(
        df_c["baseline_deviation_normalized"],
        df_c["total_length_um"],
        s=4,
        alpha=0.2,
        color=color,
        label=f"{cluster} (data)"
    )

    # Back-projected spline
    if not spline_c.empty and "baseline_deviation_normalized_proj" in spline_c.columns:
        ax.plot(
            spline_c["baseline_deviation_normalized_proj"],
            spline_c["total_length_um_proj"],
            color=color,
            linewidth=3,
            alpha=0.8,
            label=f"{cluster} (spline)"
        )

ax.set_xlabel("Baseline Deviation (Normalized)", fontsize=12)
ax.set_ylabel("Total Length (µm)", fontsize=12)
ax.set_title("Original Feature Space with Back-Projected Splines", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

fig_path = FIGURES_DIR / "05_original_space_splines.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {fig_path.name}")
plt.close()

print("\n" + "=" * 80)
print("✓ Tutorial 05g complete - PCA-based splines!")
print("=" * 80)
print(f"\nGenerated:")
print(f"  - PCA spline data: {csv_path.name}")
print(f"  - 3 visualization PNGs")
