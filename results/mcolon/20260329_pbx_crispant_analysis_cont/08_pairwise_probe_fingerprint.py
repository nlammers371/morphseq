from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys
import warnings

import pandas as pd


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_pairwise_probe_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*multi_class.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*liblinear.*multiclass classification.*deprecated.*")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification import run_classification

from common import BUILD06_DIR, CURRENT_EXPERIMENT_IDS, normalize_genotype


DEFAULT_GENOTYPES = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]
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
    parser = argparse.ArgumentParser(description="Build pairwise contrast-coordinate artifacts for PBX positioning.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_pairwise_bin4_perm500",
    )
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke configuration with 10 permutations and a few time bins.")
    return parser.parse_args()


def load_current_dataframe(genotypes: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for exp_id in CURRENT_EXPERIMENT_IDS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{exp_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing build06 file: {path}")
        part = pd.read_csv(path, low_memory=False)
        if "experiment_id" in part.columns:
            part = part[part["experiment_id"].astype(str) == exp_id].copy()
        else:
            part["experiment_id"] = exp_id
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()
    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(normalize_genotype)
    df = df[df["genotype"].isin(genotypes)].copy()
    if df.empty:
        raise ValueError("No rows remain after genotype filtering.")

    predicted = pd.to_numeric(df.get("predicted_stage_hpf"), errors="coerce")
    start_age = pd.to_numeric(df.get("start_age_hpf"), errors="coerce")
    relative_time_hpf = pd.to_numeric(df.get("relative_time_s"), errors="coerce") / 3600.0
    df["stage_hpf"] = predicted.where(predicted.notna(), start_age + relative_time_hpf)
    df = df[df["stage_hpf"].notna()].copy()
    return df.reset_index(drop=True)


def maybe_apply_smoke_subset(df: pd.DataFrame, *, bin_width: float, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    time_bins = (df["stage_hpf"].floordiv(bin_width) * bin_width).astype(int)
    keep_bins = sorted(time_bins.unique())[:3]
    return df.loc[time_bins.isin(keep_bins)].reset_index(drop=True)


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
    out["_time_bin"] = out["time_bin"].astype(int)
    out = out[["embryo_id", "time_bin_center", "genotype", *probe_cols, "experiment_id", "_time_bin", "time_bin"]]
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_current_dataframe(list(DEFAULT_GENOTYPES))
    df = maybe_apply_smoke_subset(df, bin_width=float(args.bin_width), enabled=bool(args.smoke))
    embryo_meta = df[["embryo_id", "experiment_id"]].drop_duplicates().reset_index(drop=True)

    n_permutations = 10 if args.smoke else int(args.n_permutations)
    analysis = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="stage_hpf",
        comparisons="all_pairs",
        features={"vae": "z_mu_b"},
        bin_width=float(args.bin_width),
        n_permutations=n_permutations,
        n_jobs=int(args.n_jobs),
        min_samples_per_group=3,
        min_samples_per_member=2,
        random_state=42,
        verbose=False,
        save_null_arrays=True,
        save_contrast_coordinates=True,
        save_dir=args.output_dir,
    )

    for layer_name in LAYER_EXPORTS:
        analysis.layers[layer_name].to_csv(args.output_dir / f"{layer_name}.csv", index=False)

    raw_probe_cols = export_sparse_vectors(
        analysis.layers["raw_coordinates"],
        embryo_meta=embryo_meta,
        output_path=args.output_dir / "pairwise_raw_vectors.csv",
    )
    shrunk_probe_cols = export_sparse_vectors(
        analysis.layers["shrunk_coordinates"],
        embryo_meta=embryo_meta,
        output_path=args.output_dir / "pairwise_shrunk_vectors.csv",
    )
    export_legacy_zero_filled_vectors(
        analysis.layers["raw_coordinates"],
        embryo_meta=embryo_meta,
        output_path=args.output_dir / "pairwise_raw_vectors_legacy_zero_filled.csv",
    )
    export_legacy_zero_filled_vectors(
        analysis.layers["shrunk_coordinates"],
        embryo_meta=embryo_meta,
        output_path=args.output_dir / "pairwise_shrunk_vectors_legacy_zero_filled.csv",
    )

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "experiment_ids": list(CURRENT_EXPERIMENT_IDS),
        "genotypes": list(DEFAULT_GENOTYPES),
        "results_dir": str(args.output_dir),
        "bin_width": float(args.bin_width),
        "n_permutations": int(n_permutations),
        "n_jobs": int(args.n_jobs),
        "smoke": bool(args.smoke),
        "raw_vector_columns": raw_probe_cols,
        "shrunk_vector_columns": shrunk_probe_cols,
        "pairwise_vectors_semantics": "sparse_nan_preserving",
        "legacy_zero_filled_exports": True,
    }
    pd.Series(manifest, dtype="object").to_json(args.output_dir / "pairwise_manifest.json", indent=2)
    print(args.output_dir)


if __name__ == "__main__":
    main()
