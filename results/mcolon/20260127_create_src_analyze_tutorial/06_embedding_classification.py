"""
Tutorial 06: Embedding & Metric Classification

Demonstrates ``run_classification_test`` on three feature types:
  - curvature (baseline_deviation_normalized)
  - length (total_length_um)
  - embedding (z_mu_b latent space)

Modes:
  1. One-vs-Rest  — all 4 clusters vs rest
  2. Each cluster vs Not Penetrant (WT-like reference)
  3. Negative control — crispant vs homozygous mutant within Low_to_High
     (should be ~0.5 if phenotypic cluster truly captures genotype-independent
     morphology)
  4. Positive control — ab controls vs cep290_wildtype (should also be ~0.5)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "06"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_dir = project_root / "morphseq_playground" / "metadata" / "build06_output"
REFERENCE_CSV = (
    project_root
    / "results/mcolon/20251229_cep290_phenotype_extraction"
    / "final_data/embryo_data_with_labels.csv"
)

from analyze.difference_detection import (
    run_classification_test,
)
from analyze.difference_detection.classification_test_viz import (
    plot_feature_comparison_grid,
)

print("=" * 80)
print("TUTORIAL 06: EMBEDDING & METRIC CLASSIFICATION")
print("=" * 80)

# ============================================================================
# 1. Load target data (crispant experiments) + cluster labels
# ============================================================================
PROJECTION_CSV = OUTPUT_DIR / "figures/04/projection_results/combined_projection_bootstrap.csv"
proj = pd.read_csv(PROJECTION_CSV, low_memory=False)
proj = proj[["embryo_id", "cluster_label", "membership"]].drop_duplicates(subset="embryo_id")

source_dfs = []
for exp_id in ["20260122", "20260124"]:
    fpath = data_dir / f"df03_final_output_with_latents_{exp_id}.csv"
    df_exp = pd.read_csv(fpath, low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp_id
    source_dfs.append(df_exp)

df_target = pd.concat(source_dfs, ignore_index=True)
df_target = df_target.merge(proj, on="embryo_id", how="inner")
df_target = df_target[df_target["cluster_label"].notna()].copy()

print(f"\nTarget data: {df_target['embryo_id'].nunique()} embryos, {len(df_target)} timepoints")
print(f"Genotypes: {sorted(df_target['genotype'].unique())}")
print(f"Clusters: {sorted(df_target['cluster_label'].unique())}")
print(df_target.groupby("cluster_label")["embryo_id"].nunique())

# ============================================================================
# 2. Define feature sets and colors
# ============================================================================
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

# ============================================================================
# 3. Mode 1: One-vs-Rest (all clusters)
# ============================================================================
print("\n" + "=" * 60)
print("MODE 1: One-vs-Rest")
print("=" * 60)

ovr_results = {}
for feat_key, features in FEATURE_SETS.items():
    print(f"\n  Running OvR with {feat_key} features...")
    res = run_classification_test(
        df_target,
        groupby="cluster_label",
        groups="all",
        reference="rest",
        features=features,
        n_permutations=200,
        n_jobs=4,
        verbose=False,
    )
    ovr_results[feat_key] = res
    res.comparisons.to_csv(RESULTS_DIR / f"06_ovr_{feat_key}.csv", index=False)

fig = plot_feature_comparison_grid(
    results_by_feature=ovr_results,
    feature_labels=FEATURE_LABELS,
    cluster_colors=CLUSTER_COLORS,
    title="One-vs-Rest Classification by Feature Type",
    save_path=FIGURES_DIR / "06_ovr_feature_comparison.png",
)
plt.close(fig)

# Combined CSV
ovr_all = pd.concat(
    [r.comparisons.assign(feature_type=k) for k, r in ovr_results.items()],
    ignore_index=True,
)
ovr_all.to_csv(RESULTS_DIR / "06_ovr_results.csv", index=False)
print(f"\n  Saved OvR results: {RESULTS_DIR / '06_ovr_results.csv'}")

# ============================================================================
# 4. Mode 2: Each cluster vs Not Penetrant
# ============================================================================
print("\n" + "=" * 60)
print("MODE 2: Each Cluster vs Not Penetrant")
print("=" * 60)

non_wt_clusters = [c for c in df_target["cluster_label"].unique() if c != "Not Penetrant"]

vs_wt_results = {}
for feat_key, features in FEATURE_SETS.items():
    print(f"\n  Running vs-WT with {feat_key} features...")
    res = run_classification_test(
        df_target,
        groupby="cluster_label",
        groups=non_wt_clusters,
        reference="Not Penetrant",
        features=features,
        n_permutations=200,
        n_jobs=4,
        verbose=False,
    )
    vs_wt_results[feat_key] = res

fig = plot_feature_comparison_grid(
    results_by_feature=vs_wt_results,
    feature_labels=FEATURE_LABELS,
    cluster_colors=CLUSTER_COLORS,
    title="Cluster vs Not Penetrant by Feature Type",
    save_path=FIGURES_DIR / "06_vs_wt_feature_comparison.png",
)
plt.close(fig)

vs_wt_all = pd.concat(
    [r.comparisons.assign(feature_type=k) for k, r in vs_wt_results.items()],
    ignore_index=True,
)
vs_wt_all.to_csv(RESULTS_DIR / "06_vs_wt_results.csv", index=False)
print(f"\n  Saved vs-WT results: {RESULTS_DIR / '06_vs_wt_results.csv'}")

# ============================================================================
# 5. Mode 3: Negative control — crispant vs homozygous within Low_to_High
# ============================================================================
print("\n" + "=" * 60)
print("MODE 3: Crispant vs Homozygous Mutant (Low_to_High) — Negative Control")
print("=" * 60)

# Load reference data (homozygous mutants with cluster labels)
df_ref = pd.read_csv(REFERENCE_CSV, low_memory=False)
df_ref = df_ref[df_ref["use_embryo_flag"]].copy()

# Filter to homozygous Low_to_High
df_ref_lth = df_ref[
    (df_ref["genotype"] == "cep290_homozygous")
    & (df_ref["cluster_categories"] == "Low_to_High")
].copy()

# Filter target to crispant Low_to_High
df_target_lth = df_target[
    (df_target["genotype"] == "cep290_crispant")
    & (df_target["cluster_label"] == "Low_to_High")
].copy()

print(f"  Reference homozygous L2H: {df_ref_lth['embryo_id'].nunique()} embryos")
print(f"  Target crispant L2H: {df_target_lth['embryo_id'].nunique()} embryos")

# Combine — assign a common genotype column for comparison
df_ref_lth = df_ref_lth.assign(comparison_group="cep290_homozygous")
df_target_lth = df_target_lth.assign(comparison_group="cep290_crispant")

# Use only columns that exist in both
shared_cols = sorted(set(df_ref_lth.columns) & set(df_target_lth.columns))
df_neg_ctrl = pd.concat(
    [df_ref_lth[shared_cols], df_target_lth[shared_cols]], ignore_index=True
)

neg_ctrl_results = {}
for feat_key, features in FEATURE_SETS.items():
    print(f"\n  Running negative control with {feat_key} features...")
    res = run_classification_test(
        df_neg_ctrl,
        groupby="comparison_group",
        groups="cep290_crispant",
        reference="cep290_homozygous",
        features=features,
        n_permutations=200,
        n_jobs=4,
        verbose=False,
    )
    neg_ctrl_results[feat_key] = res

# --- Positive control: ab vs cep290_wildtype ---
print("\n  Positive control: ab vs cep290_wildtype (should be ~0.5)")
df_ref_wt = df_ref[df_ref["genotype"] == "cep290_wildtype"].copy()
df_target_ab = df_target[df_target["genotype"] == "ab"].copy()

df_ref_wt = df_ref_wt.assign(comparison_group="cep290_wildtype")
df_target_ab = df_target_ab.assign(comparison_group="ab")

shared_cols_pc = sorted(set(df_ref_wt.columns) & set(df_target_ab.columns))
df_pos_ctrl = pd.concat(
    [df_ref_wt[shared_cols_pc], df_target_ab[shared_cols_pc]], ignore_index=True
)

print(f"  Reference wildtype: {df_ref_wt['embryo_id'].nunique()} embryos")
print(f"  Target ab: {df_target_ab['embryo_id'].nunique()} embryos")

pos_ctrl_results = {}
for feat_key, features in FEATURE_SETS.items():
    print(f"\n  Running positive control with {feat_key} features...")
    res = run_classification_test(
        df_pos_ctrl,
        groupby="comparison_group",
        groups="ab",
        reference="cep290_wildtype",
        features=features,
        n_permutations=200,
        n_jobs=4,
        verbose=False,
    )
    pos_ctrl_results[feat_key] = res

# Plot negative + positive control side by side using plot_multiple_aurocs
from analyze.difference_detection.classification_test_viz import plot_multiple_aurocs

CTRL_COLORS = {
    "Crispant vs Homozygous (L2H)": "#EE6677",
    "ab vs Wildtype (all)": "#4477AA",
}

fig, axes = plt.subplots(1, len(FEATURE_SETS), figsize=(6 * len(FEATURE_SETS), 5))
for idx, feat_key in enumerate(FEATURE_SETS):
    ax = axes[idx]

    # Build dict of DFs for this panel
    dfs_dict = {}
    # Negative control
    neg_res = neg_ctrl_results[feat_key]
    for (pos, neg), sub_df in neg_res.items():
        dfs_dict["Crispant vs Homozygous (L2H)"] = sub_df

    # Positive control
    pos_res = pos_ctrl_results[feat_key]
    for (pos, neg), sub_df in pos_res.items():
        dfs_dict["ab vs Wildtype (all)"] = sub_df

    plot_multiple_aurocs(
        auroc_dfs_dict=dfs_dict,
        colors_dict=CTRL_COLORS,
        title=FEATURE_LABELS[feat_key],
        ax=ax,
        ylim=(0.3, 1.05),
    )

fig.suptitle(
    "Control Comparisons (expect AUROC ≈ 0.5)",
    fontsize=16, fontweight="bold", y=1.02,
)
fig.tight_layout()
save_path = FIGURES_DIR / "06_crispant_vs_mutant_lth.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"\n  Saved: {save_path}")
plt.close(fig)

# Save control CSVs
ctrl_dfs = []
for feat_key in FEATURE_SETS:
    for label, res_dict in [("negative_ctrl", neg_ctrl_results), ("positive_ctrl", pos_ctrl_results)]:
        ctrl_dfs.append(
            res_dict[feat_key].comparisons.assign(feature_type=feat_key, control_type=label)
        )
ctrl_all = pd.concat(ctrl_dfs, ignore_index=True)
ctrl_all.to_csv(RESULTS_DIR / "06_crispant_vs_mutant_lth.csv", index=False)
print(f"  Saved: {RESULTS_DIR / '06_crispant_vs_mutant_lth.csv'}")

# ============================================================================
# 6. Summary table
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

summary_rows = []
for mode, results_dict, mode_label in [
    ("ovr", ovr_results, "One-vs-Rest"),
    ("vs_wt", vs_wt_results, "vs Not Penetrant"),
    ("neg_ctrl", neg_ctrl_results, "Crispant vs Homozygous (L2H)"),
    ("pos_ctrl", pos_ctrl_results, "ab vs Wildtype"),
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
                "n_sig_01": row.get("n_significant_01", 0),
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "06_summary.csv", index=False)
print(summary_df.to_string(index=False))
print(f"\nSaved: {RESULTS_DIR / '06_summary.csv'}")

print("\n" + "=" * 80)
print("DONE — Tutorial 06")
print("=" * 80)
