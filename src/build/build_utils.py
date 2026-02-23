"""
Stage reference generation utilities.

This reproduces the legacy notebook approach to create `metadata/stage_ref_df.csv`
used by Build04 stage inference:

Overview of legacy method (from jupyter/make_sa_stage_key.ipynb and data_qc/make_sa_key.ipynb):
- Start from `metadata/combined_metadata_files/embryo_metadata_df01.csv`
- Filter to embryos marked usable (`use_embryo_flag == 1`)
- Optionally restrict to wild-type/control cohorts using a perturbation key
- Round `predicted_stage_hpf` to integer bins, compute upper quantile (q90–q95)
  of `surface_area_um` per stage bin
- Fit a 4-parameter Hill/sigmoid curve to (stage -> surface area):
    sa(t) = offset + sa_max * t^n / (k^n + t^n)
- Evaluate the fitted curve on a dense stage grid (0..72 or 0..96 hpf)
- Write `metadata/stage_ref_df.csv` with columns `sa_um,stage_hpf`
- Optionally also write `metadata/stage_ref_params.csv` with the fitted parameters

Minimal reproducible snippet:
    from src.build.build_utils import generate_stage_ref_from_df01
    generate_stage_ref_from_df01(
        root="/path/to/morphseq",
        ref_dates=["20230620", "20240626"],  # optional; if omitted, uses all dates
        quantile=0.95,                          # q90 or q95 were used historically
        max_stage=96,                           # 72 was also used in earlier work
        pert_key_path=None                      # optional; provide to enforce WT/control filter
    )

Outputs:
- metadata/stage_ref_df.csv  (columns: sa_um, stage_hpf)
- metadata/stage_ref_params.csv  (columns: offset, sa_max, hill_coeff, inflection_point)

Notes:
- If `phenotype` and `control_flag` are not present in df01, pass a perturbation
  key CSV via `pert_key_path`. The function will construct `master_perturbation`
  from `chem_perturbation`/`genotype`, then merge the key to obtain phenotype/control.
- Column names are matched to those used by Build03A/Build04: `predicted_stage_hpf`,
  `surface_area_um`, `use_embryo_flag`.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def _ensure_master_perturbation(df: pd.DataFrame) -> pd.DataFrame:
    """Create/ensure `master_perturbation` column from chem_perturbation/genotype.

    Mirrors the logic used in Build04 prior to merging the perturbation key.
    """
    if "master_perturbation" in df.columns:
        return df
    df = df.copy()
    chem = df.get("chem_perturbation")
    if chem is None:
        # If chem_perturbation absent, default master_perturbation to genotype
        df["master_perturbation"] = df["genotype"].astype(str)
        return df
    df["chem_perturbation"] = df["chem_perturbation"].astype(str)
    df.loc[df["chem_perturbation"].isin(["nan", "None", "none", "", "NaN"]), "chem_perturbation"] = "None"
    df["master_perturbation"] = df["chem_perturbation"].copy()
    none_mask = df["master_perturbation"] == "None"
    if "genotype" in df.columns:
        df.loc[none_mask, "master_perturbation"] = df.loc[none_mask, "genotype"].astype(str).values
    return df


def _merge_pert_key(df: pd.DataFrame, pert_key_path: str) -> pd.DataFrame:
    """Merge perturbation key to provide `phenotype` and `control_flag`.

    Expects the key to have at least: master_perturbation, phenotype, control_flag.
    """
    df = _ensure_master_perturbation(df)
    key = pd.read_csv(pert_key_path)
    # Normalize expected columns
    if "master_perturbation" not in key.columns:
        raise ValueError("Perturbation key must include 'master_perturbation' column")
    if "phenotype" not in key.columns or "control_flag" not in key.columns:
        raise ValueError("Perturbation key must include 'phenotype' and 'control_flag' columns")
    merged = df.merge(key, how="left", on="master_perturbation", indicator=True)
    if (merged["_merge"] != "both").any():
        missing = np.unique(merged.loc[merged["_merge"] != "both", "master_perturbation"].astype(str))
        raise ValueError("Missing perturbations in key: " + ", ".join(missing.tolist()))
    merged.drop(columns=["_merge"], inplace=True)
    return merged


def _hill_sigmoid(params: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """Hill-type sigmoid: offset + sa_max * t^n / (k^n + t^n)."""
    offset, sa_max, n_hill, k_half = params
    return offset + sa_max * np.divide(np.power(t_vec, n_hill), np.power(k_half, n_hill) + np.power(t_vec, n_hill))


def generate_stage_ref_from_df01(
    root: str,
    df01_path: Optional[str] = None,
    pert_key_path: Optional[str] = None,
    ref_dates: Optional[Iterable[str]] = None,
    quantile: float = 0.95,
    max_stage: int = 96,
    outfile: Optional[str] = None,
    write_params: bool = True,
    initial_guess: Optional[Iterable[float]] = None,
    params_out_path: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate stage reference CSV by reproducing the legacy notebook logic.

    Parameters:
    - root: MorphSeq project root (contains metadata/, built_image_data/, etc.)
    - df01_path: Optional explicit path to embryo_metadata_df01.csv; if None uses
                 {root}/metadata/combined_metadata_files/embryo_metadata_df01.csv
    - pert_key_path: Optional path to perturbation_name_key.csv; if provided, the
                     function filters to WT/control using the key.
    - ref_dates: Optional iterable of experiment_date strings to restrict the
                 reference set. If None, uses all dates present in df01.
    - quantile: Upper quantile per rounded stage (0.90 or 0.95 typical)
    - max_stage: Upper bound of the stage grid to evaluate (72 or 96)
    - outfile: Optional explicit output CSV path; defaults to
               {root}/metadata/stage_ref_df.csv
    - write_params: Also write params CSV; path configurable via `params_out_path`
    - params_out_path: Optional explicit params CSV path; defaults to
               {root}/metadata/stage_ref_params.csv
    - initial_guess: Optional starting guess for [offset, sa_max, hill_coeff, k_half].

    Returns:
    - stage_ref_df: DataFrame with columns [sa_um, stage_hpf]
    - param_df: DataFrame with columns [offset, sa_max, hill_coeff, inflection_point]
    """
    if df01_path is None:
        df01_path = os.path.join(root, "metadata", "combined_metadata_files", "embryo_metadata_df01.csv")
    stage_ref_out = outfile or os.path.join(root, "metadata", "stage_ref_df.csv")
    params_out = params_out_path or os.path.join(root, "metadata", "stage_ref_params.csv")

    df = pd.read_csv(df01_path)
    if "experiment_date" in df.columns:
        df["experiment_date"] = df["experiment_date"].astype(str)
    if ref_dates is not None:
        ref_dates_set = set(map(str, ref_dates))
        df = df[df["experiment_date"].astype(str).isin(ref_dates_set)].copy()

    # Merge perturbation key if provided to enable WT/control filtering
    if pert_key_path:
        df = _merge_pert_key(df, pert_key_path)
        wt_ctrl_mask = (df.get("phenotype", "").astype(str).str.lower() == "wt") | (df.get("control_flag", False).astype(bool))
    else:
        # If we cannot ensure phenotype/control_flag, skip this filter
        wt_ctrl_mask = pd.Series(True, index=df.index)

    # Core filters matching the notebooks
    use_mask = (df.get("use_embryo_flag", 1) == 1)
    df_ref = df.loc[use_mask & wt_ctrl_mask, :].copy()

    if df_ref.empty:
        raise ValueError("No rows available after filtering for use_embryo_flag (and WT/control if pert_key provided).")

    # Compute per-stage upper quantile of surface area
    if "predicted_stage_hpf" not in df_ref.columns or "surface_area_um" not in df_ref.columns:
        raise ValueError("df01 missing required columns: 'predicted_stage_hpf' and 'surface_area_um'")

    df_ref["stage_group_hpf"] = np.round(df_ref["predicted_stage_hpf"]).astype(float)
    grouped = (
        df_ref.loc[:, ["stage_group_hpf", "surface_area_um"]]
        .groupby("stage_group_hpf")
        .quantile(quantile)
        .reset_index()
        .rename(columns={"surface_area_um": "sa_um", "stage_group_hpf": "stage_hpf"})
    )

    # Fit 4-parameter Hill function to (stage_hpf -> sa_um)
    t_vec = grouped["stage_hpf"].to_numpy()
    sa_vec = grouped["sa_um"].to_numpy()

    if initial_guess is None:
        # Reasonable defaults from notebooks
        # make_sa_stage_key.ipynb used [3e5, 1.9e6, 2, 24]; make_sa_key.ipynb used [4e5, 1e6, 2, 24]
        initial_guess = [4e5, 1.0e6, 2.0, 24.0]
    lb = (0.0, 0.0, 0.0, 0.0)
    ub = (np.inf, np.inf, np.inf, np.inf)

    def residuals(p: np.ndarray) -> np.ndarray:
        return _hill_sigmoid(p, t_vec) - sa_vec

    res = least_squares(residuals, x0=np.array(initial_guess, dtype=float), bounds=(lb, ub))
    params = res.x

    # Evaluate on dense stage grid
    t_full = np.linspace(0.0, float(max_stage), int(max_stage) + 1)
    sa_full = _hill_sigmoid(params, t_full)

    # Compose outputs; order columns as in docs example (sa_um first)
    stage_ref_df = pd.DataFrame({"sa_um": sa_full, "stage_hpf": t_full})
    param_df = pd.DataFrame([params], columns=["offset", "sa_max", "hill_coeff", "inflection_point"])

    # Write to disk
    # Ensure output directories exist
    os.makedirs(os.path.dirname(stage_ref_out), exist_ok=True)
    stage_ref_df.to_csv(stage_ref_out, index=False)
    if write_params:
        os.makedirs(os.path.dirname(params_out), exist_ok=True)
        param_df.to_csv(params_out, index=False)

    return stage_ref_df, param_df


__all__ = ["generate_stage_ref_from_df01"]


def reconstruct_perturbation_key_from_df02(
    root: str,
    df02_path: Optional[str] = None,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Recreate perturbation_name_key.csv from Build04 outputs (df02).

    Strategy:
    - Load embryo_metadata_df02.csv (Build04 output), which already contains the
      merged perturbation annotations: short_pert_name, phenotype, control_flag,
      pert_type, background alongside master_perturbation.
    - Take unique mappings per master_perturbation using majority vote/mode for
      each attribute to resolve occasional inconsistencies across rows.
    - Write metadata/perturbation_name_key.csv.

    Requirements in df02:
    - Columns: master_perturbation, short_pert_name, phenotype, control_flag,
               pert_type, background

    Returns: DataFrame of the reconstructed key.
    """
    if df02_path is None:
        df02_path = os.path.join(root, "metadata", "combined_metadata_files", "embryo_metadata_df02.csv")
    if out_path is None:
        out_path = os.path.join(root, "metadata", "perturbation_name_key.csv")

    df02 = pd.read_csv(df02_path)
    needed = [
        "master_perturbation",
        "short_pert_name",
        "phenotype",
        "control_flag",
        "pert_type",
        "background",
    ]
    missing = [c for c in needed if c not in df02.columns]
    if missing:
        raise ValueError(f"df02 missing required columns for key reconstruction: {', '.join(missing)}")

    # Normalize types
    df02 = df02.copy()
    # control_flag can be 0/1, True/False, or strings; cast to bool robustly
    cf = df02["control_flag"]
    if cf.dtype == object:
        df02["control_flag"] = cf.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])
    else:
        df02["control_flag"] = cf.astype(float).fillna(0).astype(int).astype(bool)

    def _mode(series: pd.Series) -> any:
        # Return most frequent non-null value; stable tie-break by lexical order
        s = series.dropna().astype(str)
        if s.empty:
            return np.nan
        counts = s.value_counts()
        top = counts[counts == counts.max()].index.tolist()
        return sorted(top)[0]

    agg = {
        "short_pert_name": _mode,
        "phenotype": _mode,
        "control_flag": lambda s: bool(pd.Series(s).astype(bool).mean() >= 0.5),
        "pert_type": _mode,
        "background": _mode,
    }

    key_df = (
        df02.groupby("master_perturbation", as_index=False)
        .agg(agg)
        .loc[:, needed]
        .sort_values(by=["short_pert_name", "master_perturbation"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.join(root, "metadata"), exist_ok=True)
    key_df.to_csv(out_path, index=False)

    return key_df


def bootstrap_perturbation_key_from_df01(
    root: str,
    df01: pd.DataFrame | None = None,
    out_path: str | None = None,
) -> pd.DataFrame:
    """Create a minimal perturbation_name_key.csv from df01 when missing.

    Strategy:
    - Ensure `master_perturbation` is constructed from `chem_perturbation`/`genotype`.
    - For each master, set:
        - short_pert_name: mode of df01.short_pert_name if present, else master
        - phenotype/control_flag/pert_type/background via safe defaults:
            - if master in {wik, ab, wt, wt_ab, wt_wik}: phenotype=wt,
              control_flag=True, pert_type=background, background=<master when in {wik,ab} else unknown>
            - else: phenotype=unknown, control_flag=False, pert_type=unknown, background=unknown
    - Write CSV to out_path if provided.
    """
    if df01 is None:
        df01_path = os.path.join(root, "metadata", "combined_metadata_files", "embryo_metadata_df01.csv")
        df01 = pd.read_csv(df01_path)

    df01 = _ensure_master_perturbation(df01)
    masters = df01["master_perturbation"].astype(str).str.strip()

    # Short names: use mode per master if available
    if "short_pert_name" in df01.columns:
        short_map = (
            df01.loc[:, ["master_perturbation", "short_pert_name"]]
            .astype(str)
            .groupby("master_perturbation")["short_pert_name"]
            .agg(lambda s: s.value_counts().index[0])
        )
    else:
        short_map = pd.Series(dtype=str)

    import time as _time
    timestamp = _time.strftime('%Y-%m-%d %H:%M:%S')
    rows = []
    ctrl_masters = {"wik", "ab", "wt", "wt_ab", "wt_wik"}
    for m in sorted(masters.unique()):
        m_l = str(m).strip()
        short = short_map.loc[m] if m in short_map.index else m_l
        if m_l.lower() in ctrl_masters:
            phenotype = "wt"
            control_flag = True
            pert_type = "background"
            background = m_l if m_l.lower() in {"wik", "ab"} else "unknown"
        else:
            phenotype = "unknown"
            control_flag = False
            pert_type = "unknown"
            background = "unknown"
        rows.append({
            "master_perturbation": m_l,
            "short_pert_name": short,
            "phenotype": phenotype,
            "control_flag": control_flag,
            "pert_type": pert_type,
            "background": background,
            "time_auto_constructed": timestamp,
        })

    key_df = pd.DataFrame(rows)[
        ["master_perturbation", "short_pert_name", "phenotype", "control_flag", "pert_type", "background", "time_auto_constructed"]
    ]

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        key_df.to_csv(out_path, index=False)

    return key_df


def _coerce_bool(val) -> bool:
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _mode_str(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return ""
    counts = s.value_counts()
    top = counts[counts == counts.max()].index.tolist()
    return sorted(top)[0]


def derive_perturbation_key_from_legacy(
    legacy_csv_path: str,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """Derive perturbation_name_key.csv from a legacy embryo stats CSV.

    Best-effort extraction of required columns using available legacy fields.
    This is intended to bootstrap a modern key; manual curation is still
    recommended for accuracy.

    Expected useful legacy columns (if present):
      - master_perturbation, short_pert_name, phenotype, control_flag,
        pert_type, background
      - chem_perturbation, genotype, strain/background-like columns

    Strategy:
      1) Compute master_perturbation from legacy columns if missing
      2) Fill short_pert_name from legacy or fall back to master_perturbation
      3) Use phenotype/control_flag/pert_type/background if available; otherwise
         apply conservative defaults (unknown/False) with minimal heuristics
      4) Aggregate by master_perturbation using mode to resolve inconsistencies

    Returns a DataFrame with columns:
      [master_perturbation, short_pert_name, phenotype, control_flag, pert_type, background]
    and writes it to out_path if provided.
    """
    df = pd.read_csv(legacy_csv_path)

    # Normalize candidate columns
    cols = {c.lower(): c for c in df.columns}

    def has(name: str) -> bool:
        return name in cols

    def col(name: str) -> str:
        return cols[name]

    # 1) master_perturbation
    if has("master_perturbation"):
        master = df[col("master_perturbation")].astype(str)
    else:
        chem = df[col("chem_perturbation")].astype(str) if has("chem_perturbation") else pd.Series(["None"] * len(df))
        geno = df[col("genotype")].astype(str) if has("genotype") else pd.Series(["unknown"] * len(df))
        # If chem is None-like, fall back to genotype
        chem_norm = chem.str.strip()
        none_mask = chem_norm.isin(["", "none", "None", "nan", "NaN"]) | chem_norm.isna()
        master = chem_norm.copy()
        master.loc[none_mask] = geno.loc[none_mask].astype(str).values
    master = master.fillna("").str.strip()

    # 2) short_pert_name
    if has("short_pert_name"):
        short = df[col("short_pert_name")].astype(str).fillna("").str.strip()
    else:
        short = master.copy()

    # 3) phenotype
    if has("phenotype"):
        phenotype = df[col("phenotype")].astype(str).fillna("").str.strip()
    else:
        phenotype = pd.Series([""] * len(df))

    # 4) control_flag
    if has("control_flag"):
        control_flag = df[col("control_flag")].map(_coerce_bool)
    else:
        # If phenotype says wt, assume control; else False
        control_flag = phenotype.str.lower().eq("wt")

    # 5) pert_type
    if has("pert_type"):
        pert_type = df[col("pert_type")].astype(str).fillna("").str.strip()
    else:
        # Minimal heuristic
        pt = []
        for m in master.str.lower().tolist():
            if m in {"inj-ctrl", "inj_ctrl", "control", "wt"}:
                pt.append("control")
            elif m in {"em", "egg water", "ew"}:
                pt.append("medium")
            else:
                pt.append("")
        pert_type = pd.Series(pt)

    # 6) background
    if has("background"):
        background = df[col("background")].astype(str).fillna("").str.strip()
    else:
        # Try common alternates
        if has("strain"):
            background = df[col("strain")].astype(str).fillna("").str.strip()
        elif has("background_strain"):
            background = df[col("background_strain")].astype(str).fillna("").str.strip()
        else:
            background = pd.Series([""] * len(df))

    # Compose minimal key rows
    key_df = pd.DataFrame({
        "master_perturbation": master,
        "short_pert_name": short,
        "phenotype": phenotype,
        "control_flag": control_flag.astype(bool),
        "pert_type": pert_type,
        "background": background,
    })

    # Clean whitespace and normalize empties to NaN for aggregation
    for c in ["master_perturbation", "short_pert_name", "phenotype", "pert_type", "background"]:
        key_df[c] = key_df[c].astype(str).str.strip()
        key_df.loc[key_df[c].isin(["", "nan", "NaN"]), c] = np.nan

    # Aggregate by master_perturbation using mode/majority
    agg = {
        "short_pert_name": _mode_str,
        "phenotype": _mode_str,
        "control_flag": lambda s: bool(pd.Series(s).astype(bool).mean() >= 0.5),
        "pert_type": _mode_str,
        "background": _mode_str,
    }
    key_df = (
        key_df.groupby("master_perturbation", as_index=False)
        .agg(agg)
        .sort_values(by=["short_pert_name", "master_perturbation"], na_position="last")
        .reset_index(drop=True)
    )

    # Final fill for any missing after aggregation
    key_df["short_pert_name"].fillna(key_df["master_perturbation"], inplace=True)
    key_df["phenotype"].fillna("unknown", inplace=True)
    key_df["pert_type"].fillna("unknown", inplace=True)
    key_df["background"].fillna("unknown", inplace=True)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        key_df.to_csv(out_path, index=False)

    return key_df
