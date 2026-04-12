from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys
import warnings

import pandas as pd


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260407_pairwise_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*multi_class.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*liblinear.*multiclass classification.*deprecated.*")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.classification import run_classification

from common import (
    ALL_EXPERIMENT_IDS,
    CANONICAL_GENOTYPES,
    SENSITIVITY_GENOTYPES,
    load_combined_pbx_dataframe,
    pairwise_results_dir,
)


LAYER_EXPORTS = [
    "raw_contrast_scores_long",
    "contrast_support_long",
    "contrast_specificity_by_timebin",
    "raw_coordinates",
    "shrunk_coordinates",
    "residual_coordinates",
    "probe_index",
]
BASE_COORD_COLS = {"feature_set", "embryo_id", "genotype", "time_bin", "time_bin_center"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined PBX all-pairs contrast coordinates.")
    parser.add_argument("--include-wik-ab", action="store_true", help="Include wik_ab as a sensitivity run.")
    parser.add_argument(
        "--class-set",
        choices=["canonical", "wik_ab", "both"],
        default="both",
        help="Which class set(s) to generate. 'both' writes canonical and wik_ab-inclusive bundles.",
    )
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--experiment-ids",
        nargs="+",
        default=None,
        help="Explicit experiment IDs to include, e.g. 20260304 20260306.",
    )
    parser.add_argument(
        "--genotypes",
        nargs="+",
        default=None,
        help="Explicit genotype set to include, e.g. inj_ctrl wik_ab.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _prepare_coordinate_export(
    coordinates: pd.DataFrame,
    *,
    embryo_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if coordinates.empty:
        raise ValueError("Cannot export pairwise vectors from an empty coordinate table.")
    probe_cols = [c for c in coordinates.columns if c not in BASE_COORD_COLS]
    out = coordinates.copy()
    out = out.merge(embryo_meta, on="embryo_id", how="left", validate="many_to_one")
    out = out[["embryo_id", "time_bin_center", "genotype", *probe_cols, "experiment_id", "time_bin"]]
    out = out.sort_values(["time_bin_center", "embryo_id"]).reset_index(drop=True)
    return out, probe_cols


def export_sparse_vectors(
    coordinates: pd.DataFrame,
    *,
    embryo_meta: pd.DataFrame,
    output_path: Path,
) -> list[str]:
    sparse, probe_cols = _prepare_coordinate_export(coordinates, embryo_meta=embryo_meta)
    sparse.to_csv(output_path, index=False)
    return probe_cols


def export_legacy_zero_filled_vectors(
    coordinates: pd.DataFrame,
    *,
    embryo_meta: pd.DataFrame,
    output_path: Path,
) -> list[str]:
    dense, probe_cols = _prepare_coordinate_export(coordinates, embryo_meta=embryo_meta)
    dense[probe_cols] = dense[probe_cols].fillna(0.0)
    dense.to_csv(output_path, index=False)
    return probe_cols


def maybe_apply_smoke_subset(df: pd.DataFrame, *, bin_width: float, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    time_bins = (df["stage_hpf"].floordiv(bin_width) * bin_width).astype(int)
    keep_bins = sorted(time_bins.unique())[:4]
    return df.loc[time_bins.isin(keep_bins)].reset_index(drop=True)


def _run_one(
    *,
    include_wik_ab: bool,
    experiment_ids: list[str] | None,
    genotypes: list[str] | None,
    bin_width: float,
    n_permutations: int,
    n_jobs: int,
    smoke: bool,
    output_dir: Path | None,
) -> Path:
    selected_genotypes = list(genotypes) if genotypes is not None else (
        SENSITIVITY_GENOTYPES if include_wik_ab else CANONICAL_GENOTYPES
    )
    resolved_output_dir = output_dir or pairwise_results_dir(
        include_wik_ab=bool(include_wik_ab),
        bin_width=float(bin_width),
        n_permutations=int(n_permutations),
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_pbx_dataframe(experiment_ids=experiment_ids, genotypes=selected_genotypes)
    df = maybe_apply_smoke_subset(df, bin_width=float(bin_width), enabled=bool(smoke))
    embryo_meta = df[["embryo_id", "experiment_id"]].drop_duplicates().reset_index(drop=True)

    analysis = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="stage_hpf",
        comparisons="all_pairs",
        features={
            "vae": "z_mu_b",
            "length": ["total_length_um"],
            "curvature": ["baseline_deviation_um"],
            "shape": ["total_length_um", "baseline_deviation_um"],
        },
        bin_width=float(bin_width),
        n_permutations=n_permutations,
        n_jobs=int(n_jobs),
        min_samples_per_group=3,
        min_samples_per_member=2,
        random_state=42,
        verbose=False,
        save_null_arrays=True,
        save_contrast_coordinates=True,
        save_predictions=True,
        save_dir=resolved_output_dir,
        overwrite=True,
    )

    for layer_name in LAYER_EXPORTS:
        analysis.layers[layer_name].to_csv(resolved_output_dir / f"{layer_name}.csv", index=False)

    raw_probe_cols = export_sparse_vectors(
        analysis.layers["raw_coordinates"],
        embryo_meta=embryo_meta,
        output_path=resolved_output_dir / "pairwise_raw_vectors.csv",
    )
    shrunk_probe_cols = export_sparse_vectors(
        analysis.layers["shrunk_coordinates"],
        embryo_meta=embryo_meta,
        output_path=resolved_output_dir / "pairwise_shrunk_vectors.csv",
    )
    export_legacy_zero_filled_vectors(
        analysis.layers["raw_coordinates"],
        embryo_meta=embryo_meta,
        output_path=resolved_output_dir / "pairwise_raw_vectors_legacy_zero_filled.csv",
    )
    export_legacy_zero_filled_vectors(
        analysis.layers["shrunk_coordinates"],
        embryo_meta=embryo_meta,
        output_path=resolved_output_dir / "pairwise_shrunk_vectors_legacy_zero_filled.csv",
    )

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "experiment_ids": list(experiment_ids or ALL_EXPERIMENT_IDS),
        "genotypes": list(selected_genotypes),
        "results_dir": str(resolved_output_dir),
        "bin_width": float(bin_width),
        "n_permutations": int(n_permutations),
        "n_jobs": int(n_jobs),
        "smoke": bool(smoke),
        "canonical_representation": "pairwise_raw_vectors.csv",
        "shrinkage_warning": "Shrunk/weighted pairwise coordinates are experimental and not yet validated for real-world use. PBX currently defaults to raw coordinates.",
        "pairwise_vectors_semantics": "sparse_nan_preserving",
        "legacy_zero_filled_exports": True,
        "raw_vector_columns": raw_probe_cols,
        "shrunk_vector_columns": shrunk_probe_cols,
        "include_wik_ab": bool(include_wik_ab),
    }
    pd.Series(manifest, dtype="object").to_json(resolved_output_dir / "pairwise_manifest.json", indent=2)
    print(resolved_output_dir)
    return resolved_output_dir


def main() -> None:
    args = parse_args()
    n_permutations = 10 if args.smoke else int(args.n_permutations)

    if args.class_set == "canonical":
        targets = [False]
    elif args.class_set == "wik_ab":
        targets = [True]
    else:
        targets = [False, True]
    if args.include_wik_ab and True not in targets:
        targets.append(True)

    for include_wik_ab in targets:
        _run_one(
            include_wik_ab=include_wik_ab,
            experiment_ids=None if args.experiment_ids is None else list(args.experiment_ids),
            genotypes=None if args.genotypes is None else list(args.genotypes),
            bin_width=float(args.bin_width),
            n_permutations=n_permutations,
            n_jobs=int(args.n_jobs),
            smoke=bool(args.smoke),
            output_dir=args.output_dir if len(targets) == 1 else None,
        )


if __name__ == "__main__":
    main()
