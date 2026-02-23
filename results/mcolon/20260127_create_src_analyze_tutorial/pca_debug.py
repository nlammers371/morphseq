"""
PCA debug script for combined experiments.

Purpose:
- Validate PCA + 3D plotting behavior on a combined object
- Use library APIs only (fit_transform_pca + plot_3d_scatter)
- Generate points-only 3D plots colored by:
  1) genotype
  2) experiment_id
  3) predicted_stage_hpf (continuous)
"""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from analyze.trajectory_analysis.viz.styling import get_color_for_genotype
from analyze.utils import fit_transform_pca
from analyze.viz.plotting import plot_3d_scatter


EXPERIMENT_IDS = ["20250305", "20251125", "20260122", "20260124"]
metadata_dir = project_root / "morphseq_playground" / "metadata"

output_dir = Path(__file__).resolve().parent / "output"
figures_dir = output_dir / "figures" / "pca_debug"
results_dir = output_dir / "results"
figures_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)


def find_latent_csv(exp_id: str) -> Optional[Path]:
    """Find df03 latent CSV for a single experiment across metadata builds."""
    pattern = f"df03_final_output_with_latents_{exp_id}.csv"
    candidates = sorted(metadata_dir.rglob(pattern))
    if not candidates:
        return None

    # Prefer build06 if available, else first sorted match.
    for candidate in candidates:
        if "build06_output" in str(candidate):
            return candidate
    return candidates[0]


def load_experiment_df(exp_id: str, csv_path: Path) -> pd.DataFrame:
    """Load and standardize one experiment dataframe."""
    df_exp = pd.read_csv(csv_path, low_memory=False)
    if "use_embryo_flag" in df_exp.columns:
        df_exp = df_exp[df_exp["use_embryo_flag"].astype(bool)].copy()

    # Enforce requested experiment labels in combined output.
    df_exp["experiment_id"] = exp_id
    return df_exp


loaded_dfs: List[pd.DataFrame] = []
loaded_ids: List[str] = []
missing_ids: List[str] = []

for exp_id in EXPERIMENT_IDS:
    csv_path = find_latent_csv(exp_id)
    if csv_path is None:
        missing_ids.append(exp_id)
        print(f"Missing latent CSV for {exp_id}")
        continue

    print(f"Loading {exp_id}: {csv_path}")
    df_exp = load_experiment_df(exp_id, csv_path)
    loaded_dfs.append(df_exp)
    loaded_ids.append(exp_id)

if not loaded_dfs:
    raise FileNotFoundError(
        "No latent CSVs found for requested experiments: "
        + ", ".join(EXPERIMENT_IDS)
    )

df = pd.concat(loaded_dfs, ignore_index=True)

required_cols = ["embryo_id", "genotype", "predicted_stage_hpf"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for PCA debug: {missing_cols}")

print("")
print(f"Requested experiments: {EXPERIMENT_IDS}")
print(f"Loaded experiments: {loaded_ids}")
if missing_ids:
    print(f"Skipped (missing latent CSV): {missing_ids}")
print(f"Loaded {df['embryo_id'].nunique()} embryos, {len(df)} timepoints")
print(f"Genotypes: {sorted(df['genotype'].dropna().unique())}")
print(
    f"Time range: {df['predicted_stage_hpf'].min():.1f}-"
    f"{df['predicted_stage_hpf'].max():.1f} hpf"
)

df_pca, pca, scaler, z_mu_cols = fit_transform_pca(df, n_components=3)
pca_cols = ["PCA_1", "PCA_2", "PCA_3"]
var = pca.explained_variance_ratio_

print(
    f"PCA variance explained: PC1={var[0]*100:.1f}%, "
    f"PC2={var[1]*100:.1f}%, PC3={var[2]*100:.1f}% "
    f"(total={var.sum()*100:.1f}%)"
)

run_tag = "_".join(EXPERIMENT_IDS)
pd.DataFrame(
    {
        "component": ["PC1", "PC2", "PC3"],
        "variance_explained": var,
        "cumulative_variance": var.cumsum(),
    }
).to_csv(results_dir / f"pca_debug_{run_tag}_variance.csv", index=False)

genotypes = sorted(df_pca["genotype"].dropna().unique())
genotype_color_lookup = {gt: get_color_for_genotype(gt) for gt in genotypes}

# Plot A: color by genotype
fig_genotype = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by="genotype",
    color_palette=genotype_color_lookup,
    show_lines=False,
    show_mean=False,
    hover_cols=["experiment_id"],
    title=f"PCA Debug ({run_tag}): colored by genotype",
)
fig_genotype_path = figures_dir / f"pca_debug_{run_tag}_by_genotype.html"
fig_genotype.write_html(fig_genotype_path)

# Plot B: color by experiment_id
fig_experiment = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by="experiment_id",
    color_order=[exp for exp in EXPERIMENT_IDS if exp in loaded_ids],
    show_lines=False,
    show_mean=False,
    hover_cols=["genotype"],
    title=f"PCA Debug ({run_tag}): colored by experiment_id",
)
fig_experiment_path = figures_dir / f"pca_debug_{run_tag}_by_experiment.html"
fig_experiment.write_html(fig_experiment_path)

# Plot C: color continuously by predicted_stage_hpf
fig_time = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by="predicted_stage_hpf",
    color_continuous=True,
    group_by="experiment_id",
    colorscale="Viridis",
    colorbar_title="predicted_stage_hpf",
    show_lines=False,
    show_mean=False,
    hover_cols=["genotype", "experiment_id"],
    title=f"PCA Debug ({run_tag}): colored by predicted_stage_hpf",
)
fig_time_path = figures_dir / f"pca_debug_{run_tag}_by_time.html"
fig_time.write_html(fig_time_path)

print(f"Saved: {fig_genotype_path}")
print(f"Saved: {fig_experiment_path}")
print(f"Saved: {fig_time_path}")
print("Done.")
