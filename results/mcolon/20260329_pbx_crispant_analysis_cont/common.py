from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD06_DIR = REPO_ROOT / "morphseq_playground" / "metadata" / "build06_output"

CURRENT_EXPERIMENT_IDS = ["20260304", "20260306"]
BRIDGE_EXPERIMENT_ID = "20251207_pbx"
ALL_EXPERIMENT_IDS = [BRIDGE_EXPERIMENT_ID, *CURRENT_EXPERIMENT_IDS]

CURRENT_REFERENCE_LABEL = "current_ref"
BRIDGE_LABEL = "bridge_20251207_pbx"

SHARED_GENOTYPES = [
    "inj_ctrl",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]
BRIDGE_PLUS_WIK_AB_GENOTYPES = [*SHARED_GENOTYPES, "wik_ab"]

GENOTYPE_MAP = {
    "ab_inj_ctrl": "inj_ctrl",
    "wik-ab_inj_ctrl": "inj_ctrl",
    "wik-ab_ctrl_inj": "inj_ctrl",
    "wik_ab_inj_ctrl": "inj_ctrl",
    "wik_ab_ctrl_inj": "inj_ctrl",
    "crispr-inj-ctrl": "inj_ctrl",
    "crispr_pbx1": "pbx1b_crispant",
    "crispr-pbx1": "pbx1b_crispant",
    "crispr_pbx4": "pbx4_crispant",
    "crispr-pbx4": "pbx4_crispant",
    "crispr_pbx1+4": "pbx1b_pbx4_crispant",
    "crispr-pbx1+4": "pbx1b_pbx4_crispant",
}

SHORT_NAME_MAP = {
    "inj_ctrl": "inj_ctrl",
    "wik_ab": "wik_ab",
    "pbx1b_crispant": "pbx1b",
    "pbx4_crispant": "pbx4",
    "pbx1b_pbx4_crispant": "pbx1b+4",
}


def normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    g = GENOTYPE_MAP.get(g, g)
    return g.replace("wik-ab", "wik_ab")


def short_name(label: str) -> str:
    return SHORT_NAME_MAP.get(str(label), str(label).replace("_", " "))


def load_bridge_ready_dataframe(
    *,
    experiment_ids: Iterable[str] | None = None,
    genotypes: Iterable[str] | None = None,
    bridge_window: tuple[float, float] = (48.0, 80.0),
) -> pd.DataFrame:
    experiment_ids = list(experiment_ids or ALL_EXPERIMENT_IDS)
    genotypes = list(genotypes or SHARED_GENOTYPES)

    frames: list[pd.DataFrame] = []
    for exp_id in experiment_ids:
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

    df["genotype"] = df["genotype"].fillna("unknown").map(normalize_genotype)
    df = df[df["genotype"].isin(genotypes)].copy()
    if df.empty:
        raise ValueError("No rows remain after bridge genotype filtering.")

    predicted = pd.to_numeric(df.get("predicted_stage_hpf"), errors="coerce")
    start_age = pd.to_numeric(df.get("start_age_hpf"), errors="coerce")
    relative_time_hpf = pd.to_numeric(df.get("relative_time_s"), errors="coerce") / 3600.0
    reconstructed = start_age + relative_time_hpf
    df["stage_hpf_bridge"] = predicted.where(predicted.notna(), reconstructed)

    if df["stage_hpf_bridge"].isna().all():
        raise ValueError("Bridge loader could not construct any non-null stage values.")

    bridge_mask = df["experiment_id"].astype(str) == BRIDGE_EXPERIMENT_ID
    low, high = bridge_window
    df = df[~bridge_mask | df["stage_hpf_bridge"].between(low, high)].copy()
    df["source_group"] = df["experiment_id"].astype(str).map(
        lambda exp: BRIDGE_LABEL if exp == BRIDGE_EXPERIMENT_ID else CURRENT_REFERENCE_LABEL
    )
    return df.reset_index(drop=True)
