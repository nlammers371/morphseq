#!/usr/bin/env python
"""Build lean, standardized input CSVs for experiment-level analysis.

This script reads source datasets listed in
`results/mcolon/20260213_subtle_phenotype_methods/input_data/source_datasets.csv`
and writes:
- `results/mcolon/20260213_subtle_phenotype_methods/input_data/experiments/{dataset_id}/input_core.csv`
- `results/mcolon/20260213_subtle_phenotype_methods/input_data/datasets_manifest.csv`
- `results/mcolon/20260213_subtle_phenotype_methods/input_data/column_inventory.csv`

Design goals:
- Keep workflow explicit and simple (just load one CSV per dataset)
- Avoid hidden utility layers
- Preserve reproducibility via manifest metadata
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

def _find_repo_root(start: Path) -> Path:
    """Find repository root by walking up to the nearest directory with .git."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"Could not locate repository root from: {start}")


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = _find_repo_root(SCRIPT_PATH.parent)
INPUT_DIR = SCRIPT_PATH.parent
SOURCES_CSV = INPUT_DIR / "source_datasets.csv"
EXPERIMENTS_DIR = INPUT_DIR / "experiments"
MANIFEST_CSV = INPUT_DIR / "datasets_manifest.csv"
INVENTORY_CSV = INPUT_DIR / "column_inventory.csv"

CORE_COLUMNS = [
    "snip_id",
    "image_id",
    "embryo_id",
    "video_id",
    "experiment_id",
    "well_id",
    "experiment_date",
    "frame_index",
    "time_int",
    "raw_time_s",
    "relative_time_s",
    "predicted_stage_hpf",
    "start_age_hpf",
    "genotype",
    "chem_perturbation",
    "phenotype",
    "short_pert_name",
    "control_flag",
    "use_embryo_flag",
    "dead_flag2",
    "well_qc_flag",
    "sam2_qc_flag",
    "focus_flag",
    "bubble_flag",
    "no_yolk_flag",
    "sa_outlier_flag",
    "total_length_um",
    "baseline_deviation_um",
    "mean_curvature_per_um",
    "std_curvature_per_um",
    "max_curvature_per_um",
    "surface_area_um",
    "area_um2",
]

UNKNOWN_TOKENS = {
    "",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
    "unkown",
    "unlabeled",
}


@dataclass
class SourceDataset:
    dataset_id: str
    source_csv: Path
    description: str


def load_sources(path: Path) -> List[SourceDataset]:
    rows: List[SourceDataset] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"dataset_id", "source_csv", "description"}
        missing_headers = required.difference(reader.fieldnames or [])
        if missing_headers:
            raise ValueError(f"source_datasets.csv missing headers: {sorted(missing_headers)}")
        for row in reader:
            if not row["dataset_id"].strip():
                continue
            rows.append(
                SourceDataset(
                    dataset_id=row["dataset_id"].strip(),
                    source_csv=REPO_ROOT / row["source_csv"].strip(),
                    description=row["description"].strip(),
                )
            )
    return rows


def _norm_token(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_")


def _is_unknown_like(value: object) -> bool:
    token = _norm_token(value)
    if token in UNKNOWN_TOKENS:
        return True
    if token.endswith("_unknown") or token.endswith("_unkown"):
        return True
    return False


def _resolve_phenotype_series(df: pd.DataFrame) -> pd.Series:
    """Resolve derived phenotype labels with explicit priority."""
    if "cluster_categories" in df.columns:
        src = df["cluster_categories"].copy()
    elif "phenotype_label" in df.columns:
        src = df["phenotype_label"].copy()
    else:
        src = df.get("phenotype", pd.Series([""] * len(df), index=df.index)).copy()

    return src


def build_core_csv(source: SourceDataset) -> Dict[str, str]:
    output_dir = EXPERIMENTS_DIR / source.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "input_core.csv"

    if not source.source_csv.exists():
        raise FileNotFoundError(f"Source CSV missing: {source.source_csv}")

    df = pd.read_csv(source.source_csv, low_memory=False)
    source_columns = list(df.columns)
    if not source_columns:
        raise ValueError(f"No header found in source CSV: {source.source_csv}")

    missing_core = [col for col in CORE_COLUMNS if col not in source_columns]

    # Resolve derived phenotype labels first, then apply unknown filtering.
    phenotype_resolved = _resolve_phenotype_series(df)
    df = df.copy()
    df["phenotype"] = phenotype_resolved

    n_input_rows = int(len(df))
    genotype_unknown_mask = df.get("genotype", pd.Series([""] * len(df))).map(_is_unknown_like)
    phenotype_unknown_mask = df["phenotype"].map(_is_unknown_like)
    drop_mask = genotype_unknown_mask | phenotype_unknown_mask

    n_dropped_unknown = int(drop_mask.sum())
    if n_dropped_unknown > 0:
        df = df.loc[~drop_mask].copy()

    # Prepare output with fixed core schema.
    out_df = pd.DataFrame(index=df.index)
    for col in CORE_COLUMNS:
        if col in df.columns:
            out_df[col] = df[col]
        else:
            out_df[col] = ""

    row_count = int(len(out_df))
    out_df.to_csv(output_csv, index=False)

    return {
        "dataset_id": source.dataset_id,
        "description": source.description,
        "source_csv": str(source.source_csv.relative_to(REPO_ROOT)),
        "output_csv": str(output_csv.relative_to(REPO_ROOT)),
        "row_count": str(row_count),
        "source_column_count": str(len(source_columns)),
        "core_column_count": str(len(CORE_COLUMNS)),
        "missing_core_columns": "|".join(missing_core),
        "input_row_count": str(n_input_rows),
        "dropped_unknown_or_unlabeled": str(n_dropped_unknown),
        "phenotype_source_priority": "cluster_categories>phenotype_label>phenotype",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "ok" if not missing_core else "missing_core_columns",
    }, source_columns


def write_manifest(rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "dataset_id",
        "description",
        "source_csv",
        "output_csv",
        "row_count",
        "source_column_count",
        "core_column_count",
        "missing_core_columns",
        "input_row_count",
        "dropped_unknown_or_unlabeled",
        "phenotype_source_priority",
        "created_at_utc",
        "status",
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_column_inventory(source_columns_by_dataset: Dict[str, List[str]]) -> None:
    fieldnames = ["dataset_id", "column_name", "present_in_source", "in_core_schema"]
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_id, columns in sorted(source_columns_by_dataset.items()):
            column_set = set(columns)
            all_columns = sorted(column_set.union(CORE_COLUMNS))
            for col in all_columns:
                writer.writerow(
                    {
                        "dataset_id": dataset_id,
                        "column_name": col,
                        "present_in_source": "yes" if col in column_set else "no",
                        "in_core_schema": "yes" if col in CORE_COLUMNS else "no",
                    }
                )


def main() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_sources(SOURCES_CSV)
    if not sources:
        raise ValueError(
            "No datasets listed in "
            "results/mcolon/20260213_subtle_phenotype_methods/input_data/source_datasets.csv"
        )

    manifest_rows: List[Dict[str, str]] = []
    source_columns_by_dataset: Dict[str, List[str]] = {}

    for source in sources:
        manifest_row, source_columns = build_core_csv(source)
        manifest_rows.append(manifest_row)
        source_columns_by_dataset[source.dataset_id] = source_columns

    write_manifest(manifest_rows)
    write_column_inventory(source_columns_by_dataset)

    print("Built standardized input CSVs:")
    for row in manifest_rows:
        print(f"  - {row['dataset_id']}: {row['output_csv']} ({row['row_count']} rows)")
    print(f"Wrote manifest: {MANIFEST_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote inventory: {INVENTORY_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
