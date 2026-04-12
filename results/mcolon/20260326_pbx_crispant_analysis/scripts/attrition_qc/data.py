from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BUILD04_DIR, CANONICAL_QC_FLAGS, DEFAULT_RAW_TO_ANALYSIS_GENOTYPE, EXPERIMENT_IDS, INFO_QC_FLAGS


REQUIRED_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "genotype",
    "embryo_id",
    "predicted_stage_hpf",
    "use_embryo_flag",
    "dead_flag",
    "dead_flag2",
    "sa_outlier_flag",
    "sam2_qc_flag",
    "frame_flag",
    "no_yolk_flag",
    "focus_flag",
    "bubble_flag",
]


def normalize_build04_genotype(genotype: str, mapping: dict[str, str] | None = None) -> str:
    mapping = mapping or DEFAULT_RAW_TO_ANALYSIS_GENOTYPE
    key = str(genotype).strip().lower().replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    key = key.replace("wik-ab", "wik_ab")
    return mapping.get(key, key)


def load_build04_dataframe(
    *,
    build_dir: Path = BUILD04_DIR,
    experiment_ids: list[str] | None = None,
    genotype_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    experiment_ids = experiment_ids or EXPERIMENT_IDS
    genotype_map = genotype_map or DEFAULT_RAW_TO_ANALYSIS_GENOTYPE
    frames: list[pd.DataFrame] = []
    allowed_raw = set(genotype_map.keys())
    for exp_id in experiment_ids:
        path = build_dir / f"qc_staged_{exp_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing build04 QC file: {path}")
        part = pd.read_csv(path, usecols=REQUIRED_COLUMNS, low_memory=False)
        part["source_experiment_id"] = str(exp_id)
        part["genotype_raw"] = part["genotype"].astype(str)
        part["genotype_norm_key"] = (
            part["genotype_raw"].str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("wik-ab", "wik_ab", regex=False)
        )
        part = part[part["genotype_norm_key"].isin(allowed_raw)].copy()
        part["genotype"] = part["genotype_norm_key"].map(genotype_map)
        frames.append(part)
    if not frames:
        raise ValueError("No build04 data loaded.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["embryo_id"].notna()].copy()
    df["experiment_date"] = df["experiment_date"].astype(str)
    for col in ["use_embryo_flag", "dead_flag", "dead_flag2", *CANONICAL_QC_FLAGS, *INFO_QC_FLAGS]:
        df[col] = df[col].fillna(False).astype(bool)
    return df.reset_index(drop=True)


def build_embryo_bin_status(
    df: pd.DataFrame,
    *,
    bin_width: float,
    time_col: str,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No rows provided to build_embryo_bin_status.")
    out = df.copy()
    out["time_bin_start"] = (np.floor(out[time_col] / float(bin_width)) * float(bin_width)).astype(int)
    out["time_bin_center"] = out["time_bin_start"].astype(float) + float(bin_width) / 2.0
    group_cols = ["experiment_date", "source_experiment_id", "experiment_id", "genotype", "embryo_id", "time_bin_start", "time_bin_center"]
    agg = (
        out.groupby(group_cols, as_index=False)
        .agg(
            n_frames=(time_col, "size"),
            stage_hpf_min=(time_col, "min"),
            stage_hpf_max=(time_col, "max"),
            use_embryo_flag=("use_embryo_flag", "max"),
            dead_flag=("dead_flag", "max"),
            dead_flag2=("dead_flag2", "max"),
            sa_outlier_flag=("sa_outlier_flag", "max"),
            sam2_qc_flag=("sam2_qc_flag", "max"),
            frame_flag=("frame_flag", "max"),
            no_yolk_flag=("no_yolk_flag", "max"),
            focus_flag=("focus_flag", "max"),
            bubble_flag=("bubble_flag", "max"),
        )
        .sort_values(["genotype", "experiment_date", "embryo_id", "time_bin_start"])
        .reset_index(drop=True)
    )
    agg["embryo_uid"] = agg["experiment_date"].astype(str) + "::" + agg["embryo_id"].astype(str)
    agg["dead_like"] = agg["dead_flag"] | agg["dead_flag2"]
    agg["canonical_qc_like"] = False
    for flag in CANONICAL_QC_FLAGS:
        agg["canonical_qc_like"] = agg["canonical_qc_like"] | agg[flag]
    agg["info_qc_like"] = False
    for flag in INFO_QC_FLAGS:
        agg["info_qc_like"] = agg["info_qc_like"] | agg[flag]
    agg["embryo_present"] = True
    agg["included"] = agg["use_embryo_flag"].astype(bool)
    agg["excluded"] = ~agg["included"]
    agg["excluded_dead_only"] = agg["excluded"] & agg["dead_like"] & ~agg["canonical_qc_like"]
    agg["excluded_qc_only"] = agg["excluded"] & ~agg["dead_like"] & agg["canonical_qc_like"]
    agg["excluded_dead_and_qc"] = agg["excluded"] & agg["dead_like"] & agg["canonical_qc_like"]
    agg["excluded_death_involved"] = agg["excluded_dead_only"] | agg["excluded_dead_and_qc"]
    agg["excluded_non_death_qc"] = agg["excluded_qc_only"]
    agg["excluded_other"] = agg["excluded"] & ~(
        agg["excluded_dead_only"] | agg["excluded_qc_only"] | agg["excluded_dead_and_qc"]
    )
    return agg

