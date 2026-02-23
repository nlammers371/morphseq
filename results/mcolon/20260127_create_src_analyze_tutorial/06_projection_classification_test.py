"""
Tutorial 06: Classification Tests on Projection-Derived Groups

Runs time-resolved classification tests to evaluate group separability
within each experiment, using projection-derived cluster labels.

Analyses:
1) One-vs-rest for cluster_label (per experiment)
2) cep290_crispant vs ab (per experiment)
3) 20260122 only: Not Penetrant crispants vs Not Penetrant ab
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "06"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROJECTION_DIR = OUTPUT_DIR / "figures" / "04" / "projection_results"
SOURCE_EXPERIMENTS = ["20260122", "20260124"]
meta_dir = project_root / "morphseq_playground" / "metadata" / "build04_output"

from analyze.difference_detection import run_classification_test

def _select_features(df: pd.DataFrame):
    candidates = ["baseline_deviation_normalized", "total_length_um"]
    feats = [c for c in candidates if c in df.columns]
    if feats:
        return feats
    if "z_mu_b" in df.columns:
        return "z_mu_b"
    raise ValueError("No suitable features found for classification.")

print("=" * 80)
print("TUTORIAL 06: PROJECTION CLASSIFICATION TESTS")
print("=" * 80)

# Load and merge projection labels
exp_data = {}
for exp_id in SOURCE_EXPERIMENTS:
    proj_path = PROJECTION_DIR / f"{exp_id}_projection_bootstrap.csv"
    if not proj_path.exists():
        raise FileNotFoundError(f"Missing projection file: {proj_path}")
    proj = pd.read_csv(proj_path, low_memory=False)
    proj = proj[["embryo_id", "cluster_label", "membership"]].drop_duplicates(subset="embryo_id")

    df_exp = pd.read_csv(meta_dir / f"qc_staged_{exp_id}.csv", low_memory=False)
    df_exp = df_exp[df_exp["use_embryo_flag"]].copy()
    df_exp["experiment_id"] = exp_id

    df_exp = df_exp.merge(proj, on="embryo_id", how="inner")
    df_exp = df_exp[df_exp["cluster_label"].notna()].copy()
    exp_data[exp_id] = df_exp

print("Loaded experiments:", {k: v["embryo_id"].nunique() for k, v in exp_data.items()})

# ---------------------------------------------------------------------------
# 1) One-vs-rest per experiment (cluster_label)
# ---------------------------------------------------------------------------
for exp_id, df_exp in exp_data.items():
    print("\n" + "=" * 70)
    print(f"ONE-VS-REST: cluster_label (experiment {exp_id})")
    print("=" * 70)

    features = _select_features(df_exp)
    results_ovr = run_classification_test(
        df_exp,
        groupby="cluster_label",
        groups="all",
        reference="rest",
        features=features,
        time_col="predicted_stage_hpf",
        embryo_id_col="embryo_id",
        bin_width=4.0,
        n_splits=5,
        n_permutations=100,
        n_jobs=4,
        verbose=True,
    )

    out_path = RESULTS_DIR / f"{exp_id}_clusterlabel_ovr.csv"
    results_ovr.comparisons.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# 2) cep290_crispant vs ab (per experiment)
# ---------------------------------------------------------------------------
for exp_id, df_exp in exp_data.items():
    if "genotype" not in df_exp.columns:
        print(f"\nSkipping genotype comparison for {exp_id}: genotype column missing")
        continue

    genotypes = set(df_exp["genotype"].dropna().unique())
    if "cep290_crispant" not in genotypes or "ab" not in genotypes:
        print(f"\nSkipping genotype comparison for {exp_id}: missing 'cep290_crispant' or 'ab'")
        continue

    print("\n" + "=" * 70)
    print(f"GENOTYPE: cep290_crispant vs ab (experiment {exp_id})")
    print("=" * 70)

    features = _select_features(df_exp)
    results_geno = run_classification_test(
        df_exp,
        groupby="genotype",
        groups="cep290_crispant",
        reference="ab",
        features=features,
        time_col="predicted_stage_hpf",
        embryo_id_col="embryo_id",
        bin_width=4.0,
        n_splits=5,
        n_permutations=100,
        n_jobs=4,
        verbose=True,
    )

    out_path = RESULTS_DIR / f"{exp_id}_geno_crispant_vs_ab.csv"
    results_geno.comparisons.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# 3) 20260122: Not Penetrant crispant vs ab
# ---------------------------------------------------------------------------
if "20260122" in exp_data:
    df_122 = exp_data["20260122"]
    if "genotype" in df_122.columns:
        df_np = df_122[df_122["cluster_label"] == "Not Penetrant"].copy()
        genotypes = set(df_np["genotype"].dropna().unique())
        if "cep290_crispant" in genotypes and "ab" in genotypes:
            print("\n" + "=" * 70)
            print("20260122: Not Penetrant crispant vs ab")
            print("=" * 70)

            features = _select_features(df_np)
            results_np = run_classification_test(
                df_np,
                groupby="genotype",
                groups="cep290_crispant",
                reference="ab",
                features=features,
                time_col="predicted_stage_hpf",
                embryo_id_col="embryo_id",
                bin_width=4.0,
                n_splits=5,
                n_permutations=100,
                n_jobs=4,
                verbose=True,
            )

            out_path = RESULTS_DIR / "20260122_not_penetrant_crispant_vs_ab.csv"
            results_np.comparisons.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
        else:
            print("\nSkipping Not Penetrant genotype test: missing genotypes in 20260122")

print("\n✓ Tutorial 06 complete.")
