# %% [markdown]
# # CEP290 Cross-Experiment Analysis Tutorial
#
# End-to-end analysis of CEP290 crispant experiments using `src/analyze`.
#
# **Pipeline**: Feature plotting → PCA → DTW clustering → Cross-experiment
# projection → Classification tests
#
# **Data**: Two CEP290 crispant experiments (20260122, 20260124) from `build06_output`
# (includes curvature metrics + VAE latents).
#
# **Note**: Bootstrap/permutation counts are reduced for speed. For production
# analyses, increase `n_bootstrap` to 100+ and `n_permutations` to 200+.

# %% Cell 0: Setup & Data Loading
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project paths
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

TUTORIAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TUTORIAL_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "notebook"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load both CEP290 experiments (build06 has curvature + z_mu_b latents)
data_dir = project_root / "morphseq_playground" / "metadata" / "build06_output"
source_dfs = {}
for exp_id in ["20260122", "20260124"]:
    df_exp = pd.read_csv(data_dir / f"df03_final_output_with_latents_{exp_id}.csv", low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp_id
    source_dfs[exp_id] = df_exp

df = pd.concat(source_dfs.values(), ignore_index=True)

print(f"Loaded {df['embryo_id'].nunique()} embryos, {len(df)} timepoints")
print(f"Experiments: {sorted(df['experiment_id'].unique())}")
print(f"Genotypes: {sorted(df['genotype'].unique())}")
print(f"Time range: {df['predicted_stage_hpf'].min():.1f}–{df['predicted_stage_hpf'].max():.1f} hpf")

# Shared color palette
from analyze.trajectory_analysis.viz.styling import get_color_for_genotype

color_lookup = {gt: get_color_for_genotype(gt) for gt in df["genotype"].unique()}

# %% Cell 1: Feature-Over-Time Plotting
# Key function: plot_feature_over_time()
from analyze.viz.plotting import plot_feature_over_time

# Example 1: Single feature — curvature by genotype (both backends)
figs = plot_feature_over_time(
    df,
    features="baseline_deviation_normalized",
    color_by="genotype",
    color_lookup=color_lookup,
    backend="both",
)
figs["plotly"].write_html(FIGURES_DIR / "notebook_01_curvature.html")
figs["matplotlib"].savefig(FIGURES_DIR / "notebook_01_curvature.png", dpi=300, bbox_inches="tight")
plt.close(figs["matplotlib"])

# Example 2: Multi-feature faceted (curvature + length × genotype)
figs = plot_feature_over_time(
    df,
    features=["baseline_deviation_normalized", "total_length_um"],
    color_by="genotype",
    color_lookup=color_lookup,
    facet_col="genotype",
    backend="matplotlib",
)
plt.savefig(FIGURES_DIR / "notebook_01_multi_feature.png", dpi=300, bbox_inches="tight")
plt.close(figs)

print("Cell 1 done: feature-over-time plots saved")

# %% Cell 2: PCA & 3D Scatter
# Key functions: fit_transform_pca(), plot_3d_scatter()
from analyze.utils import fit_transform_pca
from analyze.viz.plotting import plot_3d_scatter

df_pca, pca, scaler, z_mu_cols = fit_transform_pca(df, n_components=3)
pca_cols = ["PCA_1", "PCA_2", "PCA_3"]

var = pca.explained_variance_ratio_
print(f"PCA variance explained: PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%, PC3={var[2]*100:.1f}% (total={var.sum()*100:.1f}%)")

# 3D scatter with individual + mean trajectories (categorical: genotype)
fig = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by="genotype",
    color_palette=color_lookup,
    show_lines=False,
    show_mean=False,
)
fig.write_html(FIGURES_DIR / "notebook_02_pca_3d.html")

# 3D scatter with continuous coloring by developmental time
fig_time = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by="predicted_stage_hpf",
    color_continuous=True,
    colorscale="Viridis",
    colorbar_title="predicted_stage_hpf",
    show_lines=False,
    show_mean=False,
    hover_cols=["genotype", "experiment_id"],
)
fig_time.write_html(FIGURES_DIR / "notebook_02_pca_3d_by_time.html")

print("Cell 2 done: PCA scatter saved")

# %% Cell 3: DTW Clustering
# Key functions: compute_trajectory_distances(), run_k_selection_with_plots()
from analyze.trajectory_analysis.utilities.dtw_utils import compute_trajectory_distances
from analyze.trajectory_analysis.clustering import run_k_selection_with_plots

# HPF coverage check
from analyze.viz.hpf_coverage import experiment_hpf_coverage, plot_hpf_overlap_quick

bins_mid, cover_df, cov_count = experiment_hpf_coverage(
    df, experiment_col="experiment_id", hpf_col="predicted_stage_hpf",
    embryo_col="embryo_id", bin_width=0.5, min_embryos_per_bin=3,
)
hpf_start, hpf_end = plot_hpf_overlap_quick(
    bins_mid, cov_count, cover_df=cover_df, min_experiments=2,
    show_heatmap=True,
    coverage_plot_path=FIGURES_DIR / "notebook_03_hpf_coverage.png",
    heatmap_path=FIGURES_DIR / "notebook_03_hpf_heatmap.png",
    show=False,
)
# Override — batch effects in staging make auto-detection unreliable
hpf_start, hpf_end = 25, 50
time_window = (hpf_start, hpf_end)
print(f"HPF window: {hpf_start}–{hpf_end}")

# Compute MD-DTW distance matrix
FEATURES = ["baseline_deviation_normalized"]
D, embryo_ids, time_grid = compute_trajectory_distances(
    df, metrics=FEATURES, time_col="predicted_stage_hpf",
    time_window=time_window, embryo_id_col="embryo_id",
    normalize=True, sakoe_chiba_radius=20, verbose=True,
)
print(f"Distance matrix: {D.shape}, range [{D.min():.2f}, {D.max():.2f}]")

# K-selection (reduced bootstrap for speed)
df_filtered = df[df["embryo_id"].isin(embryo_ids)].copy()
k_selection_dir = FIGURES_DIR / "notebook_03_k_selection"

k_results = run_k_selection_with_plots(
    df=df_filtered, D=D, embryo_ids=embryo_ids,
    output_dir=k_selection_dir,
    k_range=[2, 3, 4, 5, 6],
    n_bootstrap=20,  # Production: use 100+
    method="kmedoids",
    plotting_metrics=["baseline_deviation_normalized", "total_length_um"],
    x_col="predicted_stage_hpf",
    iqr_multiplier=2, verbose=True,
)
print(f"Best k: {k_results['best_k']}")

# Apply k=3 clusters to full dataframe
cluster_assignments = pd.read_csv(k_selection_dir / "cluster_assignments.csv")
df_clustered = df.merge(
    cluster_assignments[["embryo_id", "clustering_k_3"]],
    on="embryo_id", how="left",
).rename(columns={"clustering_k_3": "cluster"})

clustered_path = RESULTS_DIR / "notebook_df_clustered_k3.csv"
df_clustered.to_csv(clustered_path, index=False)
print(f"Cell 3 done: k-selection + clustering saved")

# %% Cell 4: Cross-Experiment Projection
# Key function: run_bootstrap_projection_with_plots()
from analyze.trajectory_analysis import run_bootstrap_projection_with_plots

# Load reference cluster definitions (CEP290 mutants, 7 experiments)
CEP290_REF_DIR = (
    project_root / "results" / "mcolon"
    / "20251229_cep290_phenotype_extraction" / "final_data"
)
df_ref_data = pd.read_csv(CEP290_REF_DIR / "embryo_data_with_labels.csv", low_memory=False)
labels_valid = pd.read_csv(CEP290_REF_DIR / "embryo_cluster_labels.csv", low_memory=False)
labels_valid = labels_valid.drop_duplicates(subset="embryo_id")
labels_valid = labels_valid[labels_valid["cluster_categories"].notna()].copy()

df_ref = df_ref_data[df_ref_data["embryo_id"].isin(labels_valid["embryo_id"])].copy()

print(f"Reference: {df_ref['embryo_id'].nunique()} embryos")
print(f"Clusters: {labels_valid['cluster_categories'].value_counts().to_dict()}")

# Project each experiment onto reference (combined approach)
df_combined = pd.concat(source_dfs.values(), ignore_index=True)

PROJECTION_DIR = FIGURES_DIR / "notebook_04_projection"
projection_results = run_bootstrap_projection_with_plots(
    source_df=df_combined,
    reference_df=df_ref,
    labels_df=labels_valid,
    output_dir=PROJECTION_DIR,
    run_name="combined_projection",
    id_col="embryo_id",
    time_col="predicted_stage_hpf",
    cluster_col="cluster_categories",
    category_col=None,
    metrics=["baseline_deviation_normalized"],
    sakoe_chiba_radius=20,
    n_bootstrap=20,  # Production: use 100+
    frac=0.8,
    bootstrap_on="reference",
    method="nearest_neighbor",
    classification="2d",
    normalize=True,
    verbose=True,
    save_outputs=True,
)

df_proj = projection_results["assignments_df"]
genotype_map = dict(zip(df_combined["embryo_id"], df_combined["genotype"]))
experiment_map = dict(zip(df_combined["embryo_id"], df_combined["experiment_id"]))
df_proj["genotype"] = df_proj["embryo_id"].map(genotype_map)
df_proj["experiment_id"] = df_proj["embryo_id"].map(experiment_map)

# Save projection results
df_proj.to_csv(PROJECTION_DIR / "combined_projection_bootstrap.csv", index=False)

# Proportion summary
print("\nCluster proportions by genotype:")
prop = df_proj.groupby("genotype")["cluster_label"].value_counts(normalize=True).unstack(fill_value=0)
print((prop * 100).round(1))

# Proportion plot by experiment and genotype (matches tutorial 04 projection script)
from analyze.viz.plotting import plot_proportions

df_embryo_proj = df_proj.drop_duplicates(subset="embryo_id")
fig = plot_proportions(
    df_embryo_proj,
    color_by_grouping="cluster_label",
    row_by="genotype",
    col_by="experiment_id",
    count_by="embryo_id",
    normalize=True,
    bar_mode="grouped",
    title="Cluster Distribution by Experiment and Genotype",
    show_counts=True,
)
plt.savefig(FIGURES_DIR / "notebook_04_proportion_by_experiment.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Cell 4 done: cross-experiment projection saved")

# %% Cell 5: Classification Tests
# Key functions: run_classification_test(), plot_feature_comparison_grid()
from analyze.difference_detection import run_classification_test
from analyze.difference_detection.classification_test_viz import plot_feature_comparison_grid

# Load projection labels — use build06 data (has z_mu_b latents)
proj_labels = df_proj[["embryo_id", "cluster_label", "membership"]].drop_duplicates(subset="embryo_id")
df_target = df.merge(proj_labels, on="embryo_id", how="inner")
df_target = df_target[df_target["cluster_label"].notna()].copy()

print(f"Classification data: {df_target['embryo_id'].nunique()} embryos")
print(f"Clusters: {df_target.groupby('cluster_label')['embryo_id'].nunique().to_dict()}")

FEATURE_SETS = {
    "curvature": ["baseline_deviation_normalized"],
    "length": ["total_length_um"],
    "embedding": "z_mu_b",
}
FEATURE_LABELS = {
    "curvature": "Curvature",
    "length": "Body Length",
    "embedding": "VAE Embedding",
}
CLUSTER_COLORS = {
    "Not Penetrant": "#4477AA",
    "Low_to_High": "#EE6677",
    "Intermediate": "#228833",
    "High_to_Low": "#CCBB44",
}

# Mode 1: One-vs-Rest
print("\nMode 1: One-vs-Rest classification")
ovr_results = {}
for feat_key, features in FEATURE_SETS.items():
    res = run_classification_test(
        df_target, groupby="cluster_label", groups="all", reference="rest",
        features=features, n_permutations=50, n_jobs=4, verbose=False,
    )
    ovr_results[feat_key] = res

fig = plot_feature_comparison_grid(
    results_by_feature=ovr_results, feature_labels=FEATURE_LABELS,
    cluster_colors=CLUSTER_COLORS, title="One-vs-Rest Classification by Feature Type",
    save_path=FIGURES_DIR / "notebook_05_ovr_comparison.png",
)
plt.close(fig)

# Mode 2: Each cluster vs Not Penetrant
print("Mode 2: Each cluster vs Not Penetrant")
non_wt = [c for c in df_target["cluster_label"].unique() if c != "Not Penetrant"]
vs_wt_results = {}
for feat_key, features in FEATURE_SETS.items():
    res = run_classification_test(
        df_target, groupby="cluster_label", groups=non_wt,
        reference="Not Penetrant", features=features,
        n_permutations=50, n_jobs=4, verbose=False,
    )
    vs_wt_results[feat_key] = res

fig = plot_feature_comparison_grid(
    results_by_feature=vs_wt_results, feature_labels=FEATURE_LABELS,
    cluster_colors=CLUSTER_COLORS, title="Cluster vs Not Penetrant by Feature Type",
    save_path=FIGURES_DIR / "notebook_05_vs_wt_comparison.png",
)
plt.close(fig)

# Combined summary table
summary_rows = []
for mode, results_dict, mode_label in [
    ("ovr", ovr_results, "One-vs-Rest"),
    ("vs_wt", vs_wt_results, "vs Not Penetrant"),
]:
    for feat_key, res in results_dict.items():
        s = res.summary()
        for _, row in s.iterrows():
            summary_rows.append({
                "mode": mode_label,
                "feature": FEATURE_LABELS.get(feat_key, feat_key),
                "positive": row["positive"],
                "negative": row["negative"],
                "max_auroc": row.get("max_auroc", np.nan),
                "min_pval": row.get("min_pval", np.nan),
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "notebook_classification_summary.csv", index=False)
print("\nClassification summary:")
print(summary_df.to_string(index=False))

# NOTE: Modes 3/4 (cross-experiment controls: crispant vs homozygous, ab vs
# wildtype) are omitted. They require combining crispant + mutant reference
# data in a shared feature space. In practice, VAE batch effects between
# experiments make these controls unreliable — embedding AUROCs were inflated
# (~0.8) even for expected-null comparisons. Curvature-only controls work
# better but are less informative. See Tutorial 06 for the full implementation.

print("\nCell 5 done: classification tests saved")

# %% [markdown]
# ## Key Takeaways
#
# 1. **Phenotype emergence**: CEP290 crispant penetrance becomes detectable
#    ~26-30 hpf, consistent with homozygous mutant timing.
# 2. **Embedding vs single metrics**: VAE embedding (z_mu_b) outperforms
#    curvature or length alone for early phenotype detection (higher AUROC).
# 3. **Cross-experiment controls**: Inconclusive due to VAE batch effects —
#    embedding-based comparisons are inflated across experiments. Curvature
#    alone captures most of the penetrance signal and is batch-robust.
# 4. **Projection**: Bootstrap projection onto well-characterized reference
#    clusters provides per-embryo confidence scores (max_p, entropy).
