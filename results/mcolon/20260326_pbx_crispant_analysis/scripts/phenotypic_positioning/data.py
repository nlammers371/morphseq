from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BUILD06_DIR, EXPERIMENT_IDS


def normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    if g in {
        "ab_inj_ctrl",
        "wik-ab_inj_ctrl",
        "wik-ab_ctrl_inj",
        "wik_ab_inj_ctrl",
        "wik_ab_ctrl_inj",
    }:
        return "inj_ctrl"
    return g.replace("wik-ab", "wik_ab")


def short_name(label: str) -> str:
    label = str(label).strip().lower()
    mapping = {
        "inj_ctrl": "inj_ctrl",
        "wik_ab": "wik_ab",
        "pbx1b_crispant": "pbx1b",
        "pbx4_crispant": "pbx4",
        "pbx1b_pbx4_crispant": "pbx1b+4",
    }
    return mapping.get(label, label.replace("_", " "))


def pair_id(group1: str, group2: str) -> str:
    return f"{group1}_vs_{group2}"


def load_dataframe(
    genotypes: list[str],
    *,
    build_dir: Path = BUILD06_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for exp_id in EXPERIMENT_IDS:
        data_path = build_dir / f"df03_final_output_with_latents_{exp_id}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file: {data_path}")
        part = pd.read_csv(data_path, low_memory=False)
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

    embryo_meta = (
        df[["embryo_id", "genotype", "experiment_id"]]
        .drop_duplicates()
        .rename(columns={"genotype": "true_label"})
        .reset_index(drop=True)
    )
    return df, embryo_meta


def resolve_feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    if not cols:
        raise ValueError(f"No embedding columns found with prefix {prefix!r}")
    return cols


def aggregate_features_by_time(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    time_col: str,
    bin_width: float,
) -> pd.DataFrame:
    out = df.copy()
    out["_time_bin"] = (np.floor(out[time_col] / float(bin_width)) * float(bin_width)).astype(int)
    out["time_bin_center"] = out["_time_bin"].astype(float) + float(bin_width) / 2.0
    group_cols = ["embryo_id", "genotype", "experiment_id", "_time_bin", "time_bin_center"]
    agg = out.groupby(group_cols, as_index=False)[feature_cols].mean()
    return agg.sort_values(["_time_bin", "embryo_id"]).reset_index(drop=True)
