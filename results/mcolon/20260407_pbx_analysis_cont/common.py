from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD06_DIR = REPO_ROOT / "morphseq_playground" / "metadata" / "build06_output"
RESULTS_ROOT = REPO_ROOT / "results" / "mcolon" / "20260407_pbx_analysis_cont"

BRIDGE_EXPERIMENT_ID = "20251207_pbx"
CURRENT_EXPERIMENT_IDS = ["20260304", "20260306"]
ALL_EXPERIMENT_IDS = [BRIDGE_EXPERIMENT_ID, *CURRENT_EXPERIMENT_IDS]

CANONICAL_GENOTYPES = [
    "inj_ctrl",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]
SENSITIVITY_GENOTYPES = [*CANONICAL_GENOTYPES, "wik_ab"]

GENOTYPE_COLORS: dict[str, str] = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
    "pbx1b_crispant": "#9467BD",
    "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}

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


def normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    g = GENOTYPE_MAP.get(g, g)
    return g.replace("wik-ab", "wik_ab")


def load_combined_pbx_dataframe(
    *,
    experiment_ids: Iterable[str] | None = None,
    genotypes: Iterable[str] | None = None,
    bridge_window: tuple[float, float] = (48.0, 80.0),
) -> pd.DataFrame:
    experiment_ids = list(experiment_ids or ALL_EXPERIMENT_IDS)
    genotypes = list(genotypes or CANONICAL_GENOTYPES)

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
    df = df[df["embryo_id"].notna()].copy()

    df["genotype"] = df["genotype"].fillna("unknown").map(normalize_genotype)
    df = df[df["genotype"].isin(genotypes)].copy()
    if df.empty:
        raise ValueError("No rows remain after genotype filtering.")

    predicted = pd.to_numeric(df.get("predicted_stage_hpf"), errors="coerce")
    start_age = pd.to_numeric(df.get("start_age_hpf"), errors="coerce")
    relative_time_hpf = pd.to_numeric(df.get("relative_time_s"), errors="coerce") / 3600.0
    reconstructed = start_age + relative_time_hpf
    df["stage_hpf"] = predicted.where(predicted.notna(), reconstructed)
    if df["stage_hpf"].isna().all():
        raise ValueError("Could not construct any non-null stage_hpf values.")

    bridge_low, bridge_high = bridge_window
    bridge_mask = df["experiment_id"].astype(str) == BRIDGE_EXPERIMENT_ID
    df = df[~bridge_mask | df["stage_hpf"].between(bridge_low, bridge_high)].copy()
    return df.reset_index(drop=True)


def pairwise_results_dir(*, include_wik_ab: bool, bin_width: float, n_permutations: int) -> Path:
    stem = f"combined_pairwise_{'5class' if include_wik_ab else '4class'}_bin{int(bin_width)}_perm{int(n_permutations)}"
    return RESULTS_ROOT / "results" / "positioning" / "pairwise" / stem


def condensation_results_dir(
    *,
    variant: str,
    include_wik_ab: bool,
    bin_width: float,
    n_permutations: int,
) -> Path:
    stem = f"combined_{variant}_condensation_{'5class' if include_wik_ab else '4class'}_bin{int(bin_width)}_perm{int(n_permutations)}"
    return RESULTS_ROOT / "results" / "positioning" / "trajectory" / stem
