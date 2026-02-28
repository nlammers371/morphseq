from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.snip_processing import (
    NULLABLE_COLUMNS_SNIP_MANIFEST,
    REQUIRED_COLUMNS_SNIP_MANIFEST,
)


def rel_to_root(path: Path, *, output_root: Path) -> str:
    path = Path(path)
    output_root = Path(output_root)
    try:
        rel = path.relative_to(output_root)
        return rel.as_posix()
    except Exception:
        return str(path)


def resolve_from_root(path_str: str, *, output_root: Path) -> Path:
    p = Path(str(path_str))
    return p if p.is_absolute() else (Path(output_root) / p)


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def pipeline_version() -> str:
    """
    Best-effort pipeline version string for provenance.
    Falls back to "unknown" when git is unavailable.
    """
    try:
        here = Path(__file__).resolve()
        # repo_root = .../src/data_pipeline/snip_processing/io.py -> go up to repo root
        # /.../repo/src/data_pipeline/snip_processing/io.py
        repo_root = here.parents[3]
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return sha[:12]
    except Exception:
        return "unknown"


def validate_snip_manifest_df(df: pd.DataFrame) -> None:
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_SNIP_MANIFEST,
        "snip_manifest",
        nullable_columns=NULLABLE_COLUMNS_SNIP_MANIFEST,
    )

    # Conditional non-null requirements.
    if "is_valid" in df.columns:
        valid = df["is_valid"] == True  # noqa: E712
        if valid.any():
            must_have = [
                "processed_snip_path",
                "processed_file_size_bytes",
                "rotation_angle_rad",
                "rotation_angle_deg",
            ]
            for col in must_have:
                if df.loc[valid, col].isna().any():
                    n = int(df.loc[valid, col].isna().sum())
                    raise ValueError(f"Valid snips must have non-null {col}; found {n} null values.")
            if (df.loc[valid, "processed_file_size_bytes"].astype(float) <= 0).any():
                raise ValueError("Valid snips must have processed_file_size_bytes > 0.")

        invalid = df["is_valid"] == False  # noqa: E712
        if invalid.any():
            if df.loc[invalid, "error_message"].isna().any():
                raise ValueError("Invalid snips must have error_message populated.")

    if df["snip_id"].duplicated().any():
        dups = df.loc[df["snip_id"].duplicated(keep=False), ["snip_id"]].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate snip_id values in snip_manifest: {dups}")


@dataclass(frozen=True)
class SnipPaths:
    per_well_root: Path
    contracts_dir: Path
    processed_dir: Path
    raw_crops_dir: Path
    artifacts_dir: Path


def per_well_output_dirs(*, output_root: Path, experiment_id: str, well_id: str) -> SnipPaths:
    output_root = Path(output_root)
    per_well_root = output_root / "processed_snips" / str(experiment_id) / "per_well" / str(well_id)
    contracts_dir = per_well_root / "contracts"
    processed_dir = per_well_root / "processed"
    raw_crops_dir = per_well_root / "raw_crops"
    artifacts_dir = per_well_root / "artifacts"
    for p in (contracts_dir, processed_dir, raw_crops_dir, artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)
    return SnipPaths(
        per_well_root=per_well_root,
        contracts_dir=contracts_dir,
        processed_dir=processed_dir,
        raw_crops_dir=raw_crops_dir,
        artifacts_dir=artifacts_dir,
    )


def merged_output_dirs(*, output_root: Path, experiment_id: str) -> tuple[Path, Path]:
    output_root = Path(output_root)
    exp_root = output_root / "processed_snips" / str(experiment_id)
    contracts_dir = exp_root / "contracts"
    views_dir = exp_root / "views"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)
    return contracts_dir, views_dir
