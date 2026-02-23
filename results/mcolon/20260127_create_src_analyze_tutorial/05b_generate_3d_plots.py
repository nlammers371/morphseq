"""
Tutorial 05b: Generate 3D Spline Visualizations

Creates interactive 3D plotly visualizations for the splines generated in Tutorial 05.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "05"
RESULTS_DIR = OUTPUT_DIR / "results"

meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"
SOURCE_EXPERIMENTS = ["20260122", "20260124"]

from analyze.spline_fitting.viz import plot_3d_with_spline

print("=" * 80)
print("TUTORIAL 05B: 3D SPLINE VISUALIZATIONS")
print("=" * 80)

# Load spline dataframe
spline_csv = RESULTS_DIR / "05_projection_splines_by_cluster.csv"
print(f"Loading splines: {spline_csv}")
spline_df = pd.read_csv(spline_csv)

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

coords = ["baseline_deviation_normalized", "total_length_um"]
coords_3d = coords + ["predicted_stage_hpf"]

print(f"\nGenerating 3D visualizations for {spline_df['cluster_label'].nunique()} clusters...")
clusters = sorted(df["cluster_label"].unique())

# Add time coordinate to spline dataframe
time_min = df["predicted_stage_hpf"].min()
time_max = df["predicted_stage_hpf"].max()

for cluster in clusters:
    spline_c = spline_df[spline_df["cluster_label"] == cluster].copy()

    # Add evenly-spaced time points
    spline_c["predicted_stage_hpf"] = np.linspace(time_min, time_max, len(spline_c))

    df_c = df[df["cluster_label"] == cluster]

    # Create 3D plot with spline overlay
    fig = plot_3d_with_spline(
        df_c,
        coords=coords_3d,
        spline=spline_c[coords_3d] if not spline_c.empty else None,
        color_by="experiment_id",
        spline_color="red",
        spline_width=6,
        title=f"Cluster: {cluster}"
    )

    # Save interactive HTML
    html_path = FIGURES_DIR / f"05_{cluster.replace(' ', '_')}_3d_spline.html"
    fig.write_html(str(html_path))
    print(f"  ✓ Saved: {html_path.name}")

print("\n✓ Tutorial 05b complete - 3D visualizations generated!")
