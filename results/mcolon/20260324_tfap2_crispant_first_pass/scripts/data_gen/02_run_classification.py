"""TFAP2 crispant classification analysis from the aggregated combined dataset."""

import sys
from pathlib import Path


def main() -> None:
    run_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = run_dir / "results"
    classification_dir = results_dir / "classification"
    results_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(run_dir))
    sys.path.insert(0, str(project_root / "src"))

    from analyze.classification import run_classification
    from scripts.common import EXPERIMENT_LABEL, FEATURES, load_aggregate_dataframe

    df = load_aggregate_dataframe(
        run_dir,
        required_cols={"embryo_id", "genotype", "experiment_id", "predicted_stage_hpf", *FEATURES},
    )

    embryo_df = df.drop_duplicates(subset="embryo_id")[["embryo_id", "genotype", "experiment_id"]].copy()
    genotype_counts = (
        embryo_df.groupby("genotype", observed=True)["embryo_id"]
        .nunique()
        .rename("n_embryos")
        .sort_values(ascending=False)
        .reset_index()
    )

    genotype_order = genotype_counts["genotype"].tolist()
    ref_genotype = "inj_ctrl" if "inj_ctrl" in genotype_order else None
    non_ctrl_genotypes = [g for g in genotype_order if g != "inj_ctrl"]

    class_feature_sets = {
        "curvature": ["baseline_deviation_normalized"],
        "length": ["total_length_um"],
        "embedding": "z_mu_b",
    }
    n_permutations = 100
    bin_width = 2.0
    min_samples_per_class = 3

    print("Running one-vs-all classification...")
    ovr_analysis = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        comparisons="all_vs_rest",
        features=class_feature_sets,
        n_jobs=-1,
        n_permutations=n_permutations,
        bin_width=bin_width,
        min_samples_per_group=min_samples_per_class,
        verbose=False,
    )

    ovr_scores = ovr_analysis.scores.copy()
    ovr_scores["positive"] = ovr_scores["positive_label"]
    ovr_scores["negative"] = ovr_scores["negative_label"]

    for feature_set in ovr_analysis.feature_sets:
        feat_scores = ovr_scores[ovr_scores["feature_set"] == feature_set].copy()
        out_path = classification_dir / f"{EXPERIMENT_LABEL}_one_vs_all_{feature_set}_comparisons.csv"
        feat_scores.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name}")

    if ref_genotype is not None and non_ctrl_genotypes:
        print("Running each-vs-inj_ctrl classification...")
        vs_ctrl_analysis = run_classification(
            df,
            class_col="genotype",
            id_col="embryo_id",
            time_col="predicted_stage_hpf",
            positive=non_ctrl_genotypes,
            negative=ref_genotype,
            features=class_feature_sets,
            n_jobs=-1,
            n_permutations=n_permutations,
            bin_width=bin_width,
            min_samples_per_group=min_samples_per_class,
            verbose=False,
        )

        vs_ctrl_scores = vs_ctrl_analysis.scores.copy()
        vs_ctrl_scores["positive"] = vs_ctrl_scores["positive_label"]
        vs_ctrl_scores["negative"] = vs_ctrl_scores["negative_label"]

        for feature_set in vs_ctrl_analysis.feature_sets:
            feat_scores = vs_ctrl_scores[vs_ctrl_scores["feature_set"] == feature_set].copy()
            out_path = classification_dir / f"{EXPERIMENT_LABEL}_each_vs_inj_ctrl_{feature_set}_comparisons.csv"
            feat_scores.to_csv(out_path, index=False)
            print(f"  Saved: {out_path.name}")

    print(f"\nClassification results saved to: {classification_dir}")


if __name__ == "__main__":
    main()
