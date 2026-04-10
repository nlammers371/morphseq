from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT_IDS = ["20260213", "20260223", "20260224", "20260319", "20260320"]
EXPERIMENT_LABEL = "_".join(EXPERIMENT_IDS)
OVERLAP_FEATURE = "baseline_deviation_normalized"
FEATURES = ["total_length_um", OVERLAP_FEATURE]
MULTIMETRIC_FACET_FEATURES = [OVERLAP_FEATURE, "total_length_um"]
AGGREGATE_STEM = f"tfap2_combined_{EXPERIMENT_LABEL}"
AGGREGATE_PARQUET_NAME = f"{AGGREGATE_STEM}.parquet"
AGGREGATE_METADATA_NAME = f"{AGGREGATE_STEM}_metadata.json"
REQUIRED_BASE_COLS = {"embryo_id", "experiment_id", "genotype", "predicted_stage_hpf", *FEATURES}


def normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower()
    g = g.replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")

    g = g.replace("cispant", "crispant")
    g = g.replace("wik-ab", "wik_ab")

    if g in {
        "ab_inj_ctrl",
        "wik_ab_inj_ctrl",
        "wik_ab_ctrl_inj",
    }:
        return "inj_ctrl"

    if g in {"ab", "wik_ab"}:
        return "non_inj_ctrl"

    return g


def aggregate_paths(run_dir: str | Path) -> tuple[Path, Path]:
    base = Path(run_dir) / "results"
    return base / AGGREGATE_PARQUET_NAME, base / AGGREGATE_METADATA_NAME


def _load_experiment_frame(project_root: Path, exp_id: str) -> pd.DataFrame:
    data_path = (
        project_root
        / "morphseq_playground"
        / "metadata"
        / "build06_output"
        / f"df03_final_output_with_latents_{exp_id}.csv"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    part = pd.read_csv(data_path, low_memory=False)
    if "experiment_id" in part.columns:
        part = part[part["experiment_id"].astype(str) == exp_id].copy()
    else:
        part["experiment_id"] = exp_id
    return part


def build_aggregate_dataframe(project_root: str | Path) -> pd.DataFrame:
    project_root = Path(project_root)
    frames = [_load_experiment_frame(project_root, exp_id) for exp_id in EXPERIMENT_IDS]
    df = pd.concat(frames, ignore_index=True)

    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()

    df = df[df["embryo_id"].notna()].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["genotype"] = df["genotype"].fillna("unknown").map(normalize_genotype)
    return df


def validate_aggregate_dataframe(
    df: pd.DataFrame,
    *,
    required_cols: set[str] | None = None,
) -> None:
    required = set(REQUIRED_BASE_COLS)
    if required_cols is not None:
        required |= set(required_cols)

    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not any(col.startswith("z_mu_b") for col in df.columns):
        raise ValueError("Missing embedding columns with prefix 'z_mu_b'")


def load_aggregate_dataframe(
    run_dir: str | Path,
    *,
    required_cols: set[str] | None = None,
) -> pd.DataFrame:
    parquet_path, _ = aggregate_paths(run_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Aggregate dataset not found: {parquet_path}. "
            "Run scripts/data_gen/00_aggregate_experiments.py first."
        )

    df = pd.read_parquet(parquet_path)
    validate_aggregate_dataframe(df, required_cols=required_cols)
    return df


def write_aggregate_artifacts(project_root: str | Path, run_dir: str | Path) -> tuple[pd.DataFrame, Path, Path]:
    project_root = Path(project_root)
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = build_aggregate_dataframe(project_root)
    validate_aggregate_dataframe(df)

    parquet_path, metadata_path = aggregate_paths(run_dir)
    df.to_parquet(parquet_path, index=False)

    embryo_df = df.drop_duplicates(subset="embryo_id")[["embryo_id", "experiment_id", "genotype"]].copy()
    metadata = {
        "experiment_ids": EXPERIMENT_IDS,
        "experiment_label": EXPERIMENT_LABEL,
        "aggregate_parquet": str(parquet_path),
        "n_rows": int(len(df)),
        "n_embryos": int(embryo_df["embryo_id"].nunique()),
        "n_genotypes": int(embryo_df["genotype"].nunique()),
        "embryos_by_experiment": {
            str(exp): int(count)
            for exp, count in embryo_df.groupby("experiment_id", observed=True)["embryo_id"].nunique().items()
        },
        "embryos_by_genotype": {
            str(gt): int(count)
            for gt, count in embryo_df.groupby("genotype", observed=True)["embryo_id"].nunique().items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return df, parquet_path, metadata_path
