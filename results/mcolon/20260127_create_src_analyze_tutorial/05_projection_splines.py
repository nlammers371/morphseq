"""
Tutorial 05: 3D PCA on VAE Embeddings + Splines per Cluster

Steps:
1. Load latent embeddings from build06_output, filter use_embryo_flag
2. Merge cluster labels from Tutorial 04 projection
3. PCA(n_components=3) on z_mu_b_20..z_mu_b_99 (80 dims -> 3 PCs)
4. Fit one LPC spline per cluster via spline_fit_wrapper
5. Single combined 3D HTML plot with per-cluster splines
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "05"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_dir = project_root / "morphseq_playground" / "metadata" / "build06_output"

print("=" * 80)
print("TUTORIAL 05: 3D PCA ON VAE EMBEDDINGS + SPLINES")
print("=" * 80)

# ============================================================================
# 1. Load latent data from build06, filter, merge cluster labels
# ============================================================================
PROJECTION_CSV = OUTPUT_DIR / "figures/04/projection_results/combined_projection_bootstrap.csv"
proj = pd.read_csv(PROJECTION_CSV, low_memory=False)

source_dfs = []
for exp_id in ["20260122", "20260124"]:
    fpath = data_dir / f"df03_final_output_with_latents_{exp_id}.csv"
    df_exp = pd.read_csv(fpath, low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp_id
    source_dfs.append(df_exp)

df = pd.concat(source_dfs, ignore_index=True)

proj = proj[["embryo_id", "cluster_label", "membership"]].drop_duplicates(subset="embryo_id")
df = df.merge(proj, on="embryo_id", how="inner")
df = df[df["cluster_label"].notna()].copy()

print(f"Loaded {df['embryo_id'].nunique()} embryos, {len(df)} timepoints")
print(f"Clusters: {sorted(df['cluster_label'].unique())}")

# ============================================================================
# 2. PCA(n_components=3) on z_mu_b columns (80 dims)
# ============================================================================
z_cols = [f"z_mu_b_{i}" for i in range(20, 100)]
missing = [c for c in z_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing z_mu_b columns: {missing[:5]}... ({len(missing)} total)")

print(f"\nRunning PCA on {len(z_cols)} z_mu_b columns...")

pca = PCA(n_components=3)
X_pca = pca.fit_transform(df[z_cols].values)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df["PC3"] = X_pca[:, 2]

evr = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, PC3={evr[2]*100:.1f}%")

pca_cols = ["PC1", "PC2", "PC3"]
stage_col = "predicted_stage_hpf"

# ============================================================================
# 3. Fit one spline per cluster via spline_fit_wrapper
# ============================================================================
from analyze.spline_fitting.bootstrap import spline_fit_wrapper

print("\nFitting splines per cluster...")

spline_df = spline_fit_wrapper(
    df,
    pca_cols=pca_cols,
    group_by="cluster_label",
    stage_col=stage_col,
    n_bootstrap=1,
    n_spline_points=200,
)

print(f"\nSpline result: {len(spline_df)} rows, "
      f"{spline_df['cluster_label'].nunique()} clusters")

# Save spline CSV
csv_path = RESULTS_DIR / "05_pca_splines_by_cluster.csv"
spline_df.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# ============================================================================
# 4. Combined 3D HTML plot with per-cluster splines
# ============================================================================
print("\n" + "=" * 80)
print("CREATING 3D VISUALIZATION")
print("=" * 80)

from analyze.spline_fitting.viz import plot_3d_with_spline

cluster_colors = {
    "Not Penetrant": "#2ca02c",
    "Low_to_High": "#1f77b4",
    "High_to_Low": "#ff7f0e",
    "Intermediate": "#d62728",
}

fig = plot_3d_with_spline(
    df,
    coords=pca_cols,
    spline=spline_df,
    spline_group_by="cluster_label",
    color_by="cluster_label",
    color_palette=cluster_colors,
    spline_width=6,
    title="Embedding PCA Splines by Cluster",
)

html_path = FIGURES_DIR / "05_embedding_pca_splines.html"
fig.write_html(str(html_path))
print(f"Saved: {html_path}")

print("\n✓ Tutorial 05 complete.")
