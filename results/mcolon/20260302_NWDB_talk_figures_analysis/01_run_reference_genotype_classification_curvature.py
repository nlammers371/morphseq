"""
NWDB talk: CEP290 reference genotype classification (AUROC) for multiple feature sets.

This script:
1) Loads the CEP290 reference dataset (7 experiments) with valid cluster labels.
2) Filters to 24–120 hpf and excludes cep290_unknown.
3) Derives a 0–1-ish "curvature" feature from baseline_deviation_normalized.
4) Runs permutation-based classification tests:
   - Het vs WT
   - Homo vs WT
   for each feature set: curvature, total length, VAE embeddings (z_mu_b_*).
5) Caches the filtered dataframe and saves results to plot_dir/classification/*.

Run with the repo-mandated interpreter:
  PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
  "$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/01_run_reference_genotype_classification_curvature.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    out_dir: Path
    plot_dir: Path
    cache_dir: Path
    classification_dir: Path
    classification_subdir: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run CEP290 reference genotype AUROC classification (Het/Homo vs WT) for multiple feature sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="NWDB analysis folder; plot_dir/ will be created inside.",
    )
    p.add_argument(
        "--data-dir",
        default="results/mcolon/20251229_cep290_phenotype_extraction/final_data",
        help="Directory containing embryo_data_with_labels.csv and embryo_cluster_labels.csv.",
    )
    p.add_argument(
        "--t-min",
        type=float,
        default=24.0,
        help="Minimum HPF to include.",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=120.0,
        help="Maximum HPF to include.",
    )
    p.add_argument(
        "--bin-width",
        type=float,
        default=2.0,
        help="Time bin width (hpf).",
    )
    p.add_argument(
        "--classification-subdir",
        default="classification",
        help="Subdirectory under plot_dir/ to write classification bundles into (lets you keep multiple bin widths).",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=300,
        help="Number of permutations for null distributions.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallelism for classification (-1 = all cores).",
    )
    p.add_argument(
        "--null-save-mode",
        choices=["summary", "full"],
        default="summary",
        help="Null persistence mode. 'summary' saves mean/std per bin; 'full' additionally writes a small npz bundle.",
    )
    p.add_argument(
        "--results-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Serialization format for comparisons table.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing cached parquet and classification outputs.",
    )
    p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing outputs (raise if present).",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached parquet if available (when not overwriting).",
    )
    p.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Rebuild parquet cache from CSVs.",
    )
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Paths:
    out_dir = Path(args.out_dir).resolve()
    data_dir = (_PROJECT_ROOT / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir).resolve()
    plot_dir = out_dir / "plot_dir"
    cache_dir = plot_dir / "cache"
    classification_subdir = str(args.classification_subdir)
    classification_dir = plot_dir / classification_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        project_root=_PROJECT_ROOT,
        data_dir=data_dir,
        out_dir=out_dir,
        plot_dir=plot_dir,
        cache_dir=cache_dir,
        classification_dir=classification_dir,
        classification_subdir=classification_subdir,
    )


def _load_reference_df(
    data_dir: Path,
    t_min: float,
    t_max: float,
) -> pd.DataFrame:
    """
    Load and preprocess the CEP290 reference dataset to match the notebook snippet:
    - Filter embryos to those with valid cluster_categories labels.
    - Filter time to [t_min, t_max].
    - Exclude cep290_unknown.
    - Derive curvature from baseline_deviation_normalized.
    - Remap cluster_categories: Intermediate -> Low_to_High.
    """
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from analyze.utils.stats import normalize_arbitrary_feature

    labels_path = data_dir / "embryo_cluster_labels.csv"
    data_path = data_dir / "embryo_data_with_labels.csv"

    labels_valid = pd.read_csv(
        labels_path,
        usecols=["embryo_id", "cluster_categories"],
        low_memory=False,
    )
    labels_valid = labels_valid.drop_duplicates(subset="embryo_id")
    labels_valid = labels_valid[labels_valid["cluster_categories"].notna()].copy()

    keep_ids = set(labels_valid["embryo_id"].astype(str).tolist())

    def usecols(col: str) -> bool:
        if col in {
            "embryo_id",
            "genotype",
            "predicted_stage_hpf",
            "baseline_deviation_normalized",
            "total_length_um",
            "cluster_categories",
        }:
            return True
        return col.startswith("z_mu_b")

    df = pd.read_csv(
        data_path,
        usecols=usecols,
        low_memory=False,
    )

    df["embryo_id"] = df["embryo_id"].astype(str)
    df = df[df["embryo_id"].isin(keep_ids)].copy()

    df = df[df["predicted_stage_hpf"].notna()].copy()
    df = df[(df["predicted_stage_hpf"] >= float(t_min)) & (df["predicted_stage_hpf"] <= float(t_max))].copy()

    df = df[df["genotype"].astype(str) != "cep290_unknown"].copy()

    # Normalize to ~[0,1] using the 100th percentile (robust-ish, matches reference snippet)
    df["curvature"] = normalize_arbitrary_feature(
        df["baseline_deviation_normalized"],
        low=0,
        high_percentile=100,
        clip=False,
    )

    if "cluster_categories" in df.columns:
        df.loc[df["cluster_categories"] == "Intermediate", "cluster_categories"] = "Low_to_High"

    return df


def main() -> None:
    args = _parse_args()
    paths = _resolve_paths(args)

    sys.path.insert(0, str(paths.project_root / "src"))
    from analyze.classification.classification_test import run_classification_test

    cache_path = paths.cache_dir / f"cep290_ref_{int(args.t_min)}_{int(args.t_max)}_curvature_length_embedding.parquet"

    if cache_path.exists() and args.use_cache and not args.overwrite:
        df = pd.read_parquet(cache_path)
        print(f"Loaded cache: {cache_path}")
    else:
        if cache_path.exists() and not args.overwrite and not args.use_cache:
            raise FileExistsError(f"Cache exists and overwrite is disabled: {cache_path}")
        df = _load_reference_df(paths.data_dir, t_min=args.t_min, t_max=args.t_max)
        if cache_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite cache (use --overwrite): {cache_path}")
        df.to_parquet(cache_path, index=False)
        print(f"Wrote cache: {cache_path}")

    # Basic summary
    embryo_counts = df.groupby("genotype")["embryo_id"].nunique().to_dict()
    row_counts = df["genotype"].value_counts().to_dict()
    print(f"Reference rows (filtered): {len(df)}")
    print(f"Reference embryos by genotype: {embryo_counts}")
    print(f"Rows by genotype: {row_counts}")
    print(f"Time range: {args.t_min}–{args.t_max} hpf; bin_width={args.bin_width}")
    print(f"Permutations: {args.n_permutations}; n_jobs={args.n_jobs}")
    print(f"Null save mode: {args.null_save_mode}; results format: {args.results_format}")
    print()

    positives = ["cep290_heterozygous", "cep290_homozygous"]
    reference = "cep290_wildtype"

    feature_sets: dict[str, object] = {
        "curvature": ["curvature"],
        "length": ["total_length_um"],
        "embedding": "z_mu_b",  # auto-expands to z_mu_b_*
    }

    for feat_key, feat_spec in feature_sets.items():
        out_dir = paths.classification_dir / f"{feat_key}_het_homo_vs_wt"
        if out_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists and overwrite disabled: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Running classification: {feat_key} ===")
        res = run_classification_test(
            df=df,
            groupby="genotype",
            groups=positives,
            reference=reference,
            features=feat_spec,
            time_col="predicted_stage_hpf",
            embryo_id_col="embryo_id",
            bin_width=float(args.bin_width),
            n_splits=5,
            n_permutations=int(args.n_permutations),
            n_jobs=int(args.n_jobs),
            min_samples_per_class=3,
            within_bin_time_stratification=True,
            within_bin_time_strata_width=0.5,
            random_state=42,
            verbose=True,
            save_null=True,
            null_save_mode=str(args.null_save_mode),
        )
        res.save(out_dir, format=str(args.results_format), overwrite=bool(args.overwrite))

        summary = res.summary().sort_values(["min_pval", "max_auroc"], ascending=[True, False])
        summary_path = out_dir / "summary.csv"
        if summary_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite summary (use --overwrite): {summary_path}")
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {out_dir}")
        print(f"Saved: {summary_path}")
        print()

    print("Done.")
    print(f"Cache: {cache_path}")
    print(f"Classification outputs: {paths.classification_dir}")


if __name__ == "__main__":
    main()
