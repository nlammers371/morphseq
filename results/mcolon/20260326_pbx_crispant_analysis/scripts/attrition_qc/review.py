from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from data_pipeline.quality_control.config import QC_DEFAULTS
from data_pipeline.quality_control.death_detection import detect_persistent_death_inflection

from phenotypic_positioning.data import short_name

from .config import BUILD04_DIR, DEFAULT_BIN_WIDTH, DEFAULT_RAW_TO_ANALYSIS_GENOTYPE, EXPERIMENT_IDS


REVIEW_REQUIRED_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "genotype",
    "embryo_id",
    "snip_id",
    "predicted_stage_hpf",
    "time_int",
    "fraction_alive",
    "dead_flag",
    "dead_flag2",
    "dead_inflection_time_int",
    "use_embryo_flag",
    "sa_outlier_flag",
    "sam2_qc_flag",
    "sam2_qc_flags",
    "frame_flag",
    "no_yolk_flag",
    "area_px",
]

GRANULAR_FLAG_LABELS = {
    "dead_flag": "Dead manual",
    "dead_flag2": "Dead inferred",
    "sa_outlier_flag": "Shape outlier",
    "no_yolk_flag": "No yolk",
    "sam2:MASK_ON_EDGE": "SAM2: Mask on edge",
    "sam2:DISCONTINUOUS_MASK": "SAM2: Discontinuous mask",
    "sam2:SMALL_MASK": "SAM2: Small mask",
    "sam2:UNSPECIFIED_SAM2_QC": "SAM2: Unspecified",
    "frame:BOUNDARY_TRUNCATION_LIKE": "Frame: Boundary-like",
    "frame:ZERO_OR_INVALID_MASK_AREA": "Frame: Zero/invalid area",
    "frame:MASK_ON_EDGE": "Frame: Mask on edge",
    "frame:DISCONTINUOUS_MASK": "Frame: Discontinuous",
    "frame:SMALL_MASK": "Frame: Small mask",
    "frame:OTHER_SAM2_RELATED": "Frame: Other SAM2-related",
}

GRANULAR_FLAG_COLORS = {
    "dead_flag": "#6a3d9a",
    "dead_flag2": "#b15928",
    "sa_outlier_flag": "#1b9e77",
    "no_yolk_flag": "#d95f02",
    "sam2:MASK_ON_EDGE": "#4c78a8",
    "sam2:DISCONTINUOUS_MASK": "#7b6fd0",
    "sam2:SMALL_MASK": "#9c89ff",
    "sam2:UNSPECIFIED_SAM2_QC": "#9e9e9e",
    "frame:BOUNDARY_TRUNCATION_LIKE": "#e6ab02",
    "frame:ZERO_OR_INVALID_MASK_AREA": "#ff9f1c",
    "frame:MASK_ON_EDGE": "#f2cf5b",
    "frame:DISCONTINUOUS_MASK": "#c9a227",
    "frame:SMALL_MASK": "#f6d55c",
    "frame:OTHER_SAM2_RELATED": "#8c6d1f",
}


def _normalize_genotype(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    norm = series.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    norm = norm.str.replace("wik-ab", "wik_ab", regex=False)
    return norm.map(mapping).fillna(norm)


def load_review_dataframe(
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
        part = pd.read_csv(path, usecols=REVIEW_REQUIRED_COLUMNS, low_memory=False)
        raw_norm = part["genotype"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("wik-ab", "wik_ab", regex=False)
        part = part[raw_norm.isin(allowed_raw)].copy()
        part["genotype"] = _normalize_genotype(part["genotype"], genotype_map)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["experiment_date"] = df["experiment_date"].astype(str)
    df["embryo_id"] = df["embryo_id"].astype(str)
    df["snip_id"] = df["snip_id"].astype(str)
    for col in ["predicted_stage_hpf", "time_int", "fraction_alive", "dead_inflection_time_int"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["dead_flag", "dead_flag2", "use_embryo_flag", "sa_outlier_flag", "sam2_qc_flag", "frame_flag", "no_yolk_flag"]:
        df[col] = df[col].fillna(False).astype(bool)
    df["sam2_qc_flags"] = df["sam2_qc_flags"].fillna("").astype(str)
    return df.dropna(subset=["predicted_stage_hpf", "time_int"]).reset_index(drop=True)


def _split_sam2_reasons(value: str) -> list[str]:
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return []
    normalized = raw
    for sep in ["|", ";"]:
        normalized = normalized.replace(sep, ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def summarize_sam2_reasons(
    df: pd.DataFrame,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> pd.DataFrame:
    work = df.copy()
    work = work[work["sam2_qc_flag"]].copy()
    if work.empty:
        return pd.DataFrame()
    work["time_bin_start"] = (np.floor(work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    work["time_bin_center"] = work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    work["embryo_bin_uid"] = (
        work["experiment_date"].astype(str)
        + "::"
        + work["embryo_id"].astype(str)
        + "::"
        + work["time_bin_start"].astype(str)
    )
    rows: list[dict[str, object]] = []
    for row in work.itertuples(index=False):
        reasons = _split_sam2_reasons(getattr(row, "sam2_qc_flags"))
        if not reasons:
            reasons = ["UNSPECIFIED_SAM2_QC"]
        for reason in reasons:
            rows.append(
                {
                    "experiment_date": row.experiment_date,
                    "genotype": row.genotype,
                    "embryo_id": row.embryo_id,
                    "time_bin_start": int(row.time_bin_start),
                    "time_bin_center": float(row.time_bin_center),
                    "embryo_bin_uid": row.embryo_bin_uid,
                    "reason": reason,
                }
            )
    exploded = pd.DataFrame(rows)

    all_bins = df.copy()
    all_bins["time_bin_start"] = (np.floor(all_bins["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    all_bins["time_bin_center"] = all_bins["time_bin_start"].astype(float) + float(bin_width) / 2.0
    all_bin_summary = (
        all_bins.groupby(["genotype", "time_bin_center"], as_index=False)
        .agg(
            embryo_bins_present=("embryo_id", "nunique"),
            embryo_bins_excluded=("use_embryo_flag", lambda s: int((~s.astype(bool)).sum())),
        )
    )
    reason_summary = (
        exploded.groupby(["genotype", "time_bin_center", "reason"], as_index=False)
        .agg(
            embryo_bins_with_reason=("embryo_bin_uid", "nunique"),
        )
    )
    summary = reason_summary.merge(all_bin_summary, on=["genotype", "time_bin_center"], how="left")
    summary = summary.assign(
        fraction_present=summary["embryo_bins_with_reason"] / summary["embryo_bins_present"].clip(lower=1),
        fraction_excluded=summary["embryo_bins_with_reason"] / summary["embryo_bins_excluded"].clip(lower=1),
    )
    return summary.sort_values(["reason", "genotype", "time_bin_center"]).reset_index(drop=True)


def _primary_frame_reason(row: pd.Series) -> str:
    area = pd.to_numeric(pd.Series([row.get("area_px")]), errors="coerce").iloc[0]
    sam2_raw = str(row.get("sam2_qc_flags", "")).strip()
    sam2_reasons = set(_split_sam2_reasons(sam2_raw))
    if pd.isna(area) or float(area) <= 0:
        return "ZERO_OR_INVALID_MASK_AREA"
    if "MASK_ON_EDGE" in sam2_reasons:
        return "MASK_ON_EDGE"
    if "DISCONTINUOUS_MASK" in sam2_reasons:
        return "DISCONTINUOUS_MASK"
    if "SMALL_MASK" in sam2_reasons:
        return "SMALL_MASK"
    if sam2_reasons:
        return "OTHER_SAM2_RELATED"
    return "BOUNDARY_TRUNCATION_LIKE"


def summarize_frame_reasons(
    df: pd.DataFrame,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> pd.DataFrame:
    work = df[df["frame_flag"]].copy()
    if work.empty:
        return pd.DataFrame()
    work["time_bin_start"] = (np.floor(work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    work["time_bin_center"] = work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    work["embryo_bin_uid"] = (
        work["experiment_date"].astype(str)
        + "::"
        + work["embryo_id"].astype(str)
        + "::"
        + work["time_bin_start"].astype(str)
    )
    work["frame_qc_reason"] = work.apply(_primary_frame_reason, axis=1)
    reason_summary = (
        work.groupby(["genotype", "time_bin_center", "frame_qc_reason"], as_index=False)
        .agg(embryo_bins_with_reason=("embryo_bin_uid", "nunique"))
    )

    all_bins = df.copy()
    all_bins["time_bin_start"] = (np.floor(all_bins["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    all_bins["time_bin_center"] = all_bins["time_bin_start"].astype(float) + float(bin_width) / 2.0
    all_bin_summary = (
        all_bins.groupby(["genotype", "time_bin_center"], as_index=False)
        .agg(
            embryo_bins_present=("embryo_id", "nunique"),
            embryo_bins_excluded=("use_embryo_flag", lambda s: int((~s.astype(bool)).sum())),
        )
    )
    summary = reason_summary.merge(all_bin_summary, on=["genotype", "time_bin_center"], how="left")
    summary["fraction_present"] = summary["embryo_bins_with_reason"] / summary["embryo_bins_present"].clip(lower=1)
    summary["fraction_excluded"] = summary["embryo_bins_with_reason"] / summary["embryo_bins_excluded"].clip(lower=1)
    return summary.sort_values(["frame_qc_reason", "genotype", "time_bin_center"]).reset_index(drop=True)


def _build_embryo_bin_base(
    df: pd.DataFrame,
    *,
    bin_width: float,
) -> pd.DataFrame:
    work = df.copy()
    work["time_bin_start"] = (np.floor(work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    work["time_bin_center"] = work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    work["embryo_bin_uid"] = (
        work["experiment_date"].astype(str)
        + "::"
        + work["embryo_id"].astype(str)
        + "::"
        + work["time_bin_start"].astype(str)
    )
    base = (
        work.groupby(["experiment_date", "genotype", "embryo_id", "time_bin_start", "time_bin_center", "embryo_bin_uid"], as_index=False)
        .agg(
            included=("use_embryo_flag", "any"),
            dead_flag=("dead_flag", "any"),
            dead_flag2=("dead_flag2", "any"),
            sa_outlier_flag=("sa_outlier_flag", "any"),
            no_yolk_flag=("no_yolk_flag", "any"),
        )
    )
    base["dead_like"] = base["dead_flag"] | base["dead_flag2"]
    base["excluded"] = ~base["included"]
    return base


def summarize_granular_exclusion_flags(
    df: pd.DataFrame,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> pd.DataFrame:
    base = _build_embryo_bin_base(df, bin_width=bin_width)
    rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        static_reasons: list[str] = []
        if row.dead_flag:
            static_reasons.append("dead_flag")
        if row.dead_flag2:
            static_reasons.append("dead_flag2")
        if row.sa_outlier_flag:
            static_reasons.append("sa_outlier_flag")
        if row.no_yolk_flag:
            static_reasons.append("no_yolk_flag")
        for reason in static_reasons:
            rows.append(
                {
                    "genotype": row.genotype,
                    "time_bin_center": row.time_bin_center,
                    "embryo_bin_uid": row.embryo_bin_uid,
                    "reason_key": reason,
                }
            )

    sam2_rows: list[dict[str, object]] = []
    sam2_work = df[df["sam2_qc_flag"]].copy()
    if not sam2_work.empty:
        sam2_work["time_bin_start"] = (np.floor(sam2_work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
        sam2_work["time_bin_center"] = sam2_work["time_bin_start"].astype(float) + float(bin_width) / 2.0
        sam2_work["embryo_bin_uid"] = (
            sam2_work["experiment_date"].astype(str)
            + "::"
            + sam2_work["embryo_id"].astype(str)
            + "::"
            + sam2_work["time_bin_start"].astype(str)
        )
        for row in sam2_work.itertuples(index=False):
            reasons = _split_sam2_reasons(getattr(row, "sam2_qc_flags")) or ["UNSPECIFIED_SAM2_QC"]
            for reason in reasons:
                sam2_rows.append(
                    {
                        "genotype": row.genotype,
                        "time_bin_center": row.time_bin_center,
                        "embryo_bin_uid": row.embryo_bin_uid,
                        "reason_key": f"sam2:{reason}",
                    }
                )

    frame_rows: list[dict[str, object]] = []
    frame_work = df[df["frame_flag"]].copy()
    if not frame_work.empty:
        frame_work["time_bin_start"] = (np.floor(frame_work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
        frame_work["time_bin_center"] = frame_work["time_bin_start"].astype(float) + float(bin_width) / 2.0
        frame_work["embryo_bin_uid"] = (
            frame_work["experiment_date"].astype(str)
            + "::"
            + frame_work["embryo_id"].astype(str)
            + "::"
            + frame_work["time_bin_start"].astype(str)
        )
        frame_work["frame_qc_reason"] = frame_work.apply(_primary_frame_reason, axis=1)
        for row in frame_work.itertuples(index=False):
            frame_rows.append(
                {
                    "genotype": row.genotype,
                    "time_bin_center": row.time_bin_center,
                    "embryo_bin_uid": row.embryo_bin_uid,
                    "reason_key": f"frame:{row.frame_qc_reason}",
                }
            )

    all_rows = rows + sam2_rows + frame_rows
    if not all_rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(all_rows).drop_duplicates()
    reason_summary = (
        exploded.groupby(["genotype", "time_bin_center", "reason_key"], as_index=False)
        .agg(embryo_bins_with_reason=("embryo_bin_uid", "nunique"))
    )
    denom = (
        base.groupby(["genotype", "time_bin_center"], as_index=False)
        .agg(
            embryo_bins_present=("embryo_bin_uid", "nunique"),
            embryo_bins_excluded=("excluded", "sum"),
        )
    )
    summary = reason_summary.merge(denom, on=["genotype", "time_bin_center"], how="left")
    summary = summary.assign(
        fraction_present=summary["embryo_bins_with_reason"] / summary["embryo_bins_present"].clip(lower=1),
        fraction_excluded=summary["embryo_bins_with_reason"] / summary["embryo_bins_excluded"].clip(lower=1),
    )
    summary["reason_label"] = summary["reason_key"].map(GRANULAR_FLAG_LABELS).fillna(summary["reason_key"])
    return summary.sort_values(["reason_key", "genotype", "time_bin_center"]).reset_index(drop=True)


def summarize_alive_granular_exclusion_flags(
    df: pd.DataFrame,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> pd.DataFrame:
    base = _build_embryo_bin_base(df, bin_width=bin_width)
    alive_bins = base[~base["dead_like"]].copy()
    if alive_bins.empty:
        return pd.DataFrame()
    alive_uids = set(alive_bins["embryo_bin_uid"])

    rows: list[dict[str, object]] = []
    for row in alive_bins.itertuples(index=False):
        static_reasons: list[str] = []
        if row.sa_outlier_flag:
            static_reasons.append("sa_outlier_flag")
        if row.no_yolk_flag:
            static_reasons.append("no_yolk_flag")
        for reason in static_reasons:
            rows.append(
                {
                    "genotype": row.genotype,
                    "time_bin_center": row.time_bin_center,
                    "embryo_bin_uid": row.embryo_bin_uid,
                    "reason_key": reason,
                }
            )

    sam2_rows: list[dict[str, object]] = []
    sam2_work = df[df["sam2_qc_flag"]].copy()
    sam2_work["time_bin_start"] = (np.floor(sam2_work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    sam2_work["time_bin_center"] = sam2_work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    sam2_work["embryo_bin_uid"] = (
        sam2_work["experiment_date"].astype(str)
        + "::"
        + sam2_work["embryo_id"].astype(str)
        + "::"
        + sam2_work["time_bin_start"].astype(str)
    )
    sam2_work = sam2_work[sam2_work["embryo_bin_uid"].isin(alive_uids)]
    for row in sam2_work.itertuples(index=False):
        reasons = _split_sam2_reasons(getattr(row, "sam2_qc_flags")) or ["UNSPECIFIED_SAM2_QC"]
        for reason in reasons:
            sam2_rows.append(
                {
                    "genotype": row.genotype,
                    "time_bin_center": row.time_bin_center,
                    "embryo_bin_uid": row.embryo_bin_uid,
                    "reason_key": f"sam2:{reason}",
                }
            )

    frame_rows: list[dict[str, object]] = []
    frame_work = df[df["frame_flag"]].copy()
    frame_work["time_bin_start"] = (np.floor(frame_work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    frame_work["time_bin_center"] = frame_work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    frame_work["embryo_bin_uid"] = (
        frame_work["experiment_date"].astype(str)
        + "::"
        + frame_work["embryo_id"].astype(str)
        + "::"
        + frame_work["time_bin_start"].astype(str)
    )
    frame_work = frame_work[frame_work["embryo_bin_uid"].isin(alive_uids)]
    frame_work["frame_qc_reason"] = frame_work.apply(_primary_frame_reason, axis=1)
    for row in frame_work.itertuples(index=False):
        frame_rows.append(
            {
                "genotype": row.genotype,
                "time_bin_center": row.time_bin_center,
                "embryo_bin_uid": row.embryo_bin_uid,
                "reason_key": f"frame:{row.frame_qc_reason}",
            }
        )

    all_rows = rows + sam2_rows + frame_rows
    if not all_rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(all_rows).drop_duplicates()
    reason_summary = (
        exploded.groupby(["genotype", "time_bin_center", "reason_key"], as_index=False)
        .agg(embryo_bins_with_reason=("embryo_bin_uid", "nunique"))
    )
    denom = (
        alive_bins.groupby(["genotype", "time_bin_center"], as_index=False)
        .agg(
            embryo_bins_present=("embryo_bin_uid", "nunique"),
            embryo_bins_excluded=("excluded", "sum"),
        )
    )
    summary = reason_summary.merge(denom, on=["genotype", "time_bin_center"], how="left")
    summary = summary.assign(
        fraction_present=summary["embryo_bins_with_reason"] / summary["embryo_bins_present"].clip(lower=1),
        fraction_excluded=summary["embryo_bins_with_reason"] / summary["embryo_bins_excluded"].clip(lower=1),
    )
    summary["reason_label"] = summary["reason_key"].map(GRANULAR_FLAG_LABELS).fillna(summary["reason_key"])
    return summary.sort_values(["reason_key", "genotype", "time_bin_center"]).reset_index(drop=True)


def plot_granular_exclusion_flags(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    denominator: str,
) -> None:
    if summary_df.empty:
        return
    if denominator not in {"present", "excluded"}:
        raise ValueError(f"Unsupported denominator: {denominator}")
    frac_col = "fraction_present" if denominator == "present" else "fraction_excluded"
    reasons = (
        summary_df.groupby(["reason_key", "reason_label"], as_index=False)["embryo_bins_with_reason"]
        .sum()
        .sort_values("embryo_bins_with_reason", ascending=False)
    )
    genotypes = summary_df["genotype"].drop_duplicates().tolist()
    n_cols = min(3, max(1, len(genotypes)))
    n_rows = int(np.ceil(len(genotypes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.0 * n_rows), squeeze=False, sharex=True, sharey=True)
    flat = axes.flatten()
    for ax, genotype in zip(flat, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype]
        for row in reasons.itertuples(index=False):
            sub = grp[grp["reason_key"] == row.reason_key].sort_values("time_bin_center")
            if sub.empty:
                continue
            ax.plot(
                sub["time_bin_center"],
                100.0 * sub[frac_col].fillna(0.0),
                marker="o",
                markersize=2.8,
                linewidth=1.5,
                color=GRANULAR_FLAG_COLORS.get(row.reason_key, "#666666"),
                label=row.reason_label,
            )
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel(f"Fraction of embryo-bins {denominator} (%)")
        ax.set_ylim(0, 100)
    for ax in flat[len(genotypes):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Granular exclusionary QC fraction over time ({denominator} denominator)", fontweight="bold", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_granular_exclusion_flags_alive(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    denominator: str,
) -> None:
    if summary_df.empty:
        return
    if denominator not in {"present", "excluded"}:
        raise ValueError(f"Unsupported denominator: {denominator}")
    frac_col = "fraction_present" if denominator == "present" else "fraction_excluded"
    title = "Granular exclusionary QC fraction over time (alive bins)"
    out_suffix = "alive_present_over_time" if denominator == "present" else "alive_excluded_over_time"
    suffix_label = "alive embryo-bins"
    genotypes = summary_df["genotype"].drop_duplicates().tolist()
    n_cols = min(3, max(1, len(genotypes)))
    n_rows = int(np.ceil(len(genotypes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.0 * n_rows), squeeze=False, sharex=True, sharey=True)
    flat = axes.flatten()
    reasons = (
        summary_df.groupby(["reason_key", "reason_label"], as_index=False)["embryo_bins_with_reason"]
        .sum()
        .sort_values("embryo_bins_with_reason", ascending=False)
    )
    for ax, genotype in zip(flat, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype]
        for row in reasons.itertuples(index=False):
            sub = grp[grp["reason_key"] == row.reason_key].sort_values("time_bin_center")
            if sub.empty:
                continue
            ax.plot(
                sub["time_bin_center"],
                100.0 * sub[frac_col].fillna(0.0),
                marker="o",
                markersize=2.8,
                linewidth=1.5,
                color=GRANULAR_FLAG_COLORS.get(row.reason_key, "#666666"),
                label=row.reason_label,
            )
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel(f"Fraction of {suffix_label} (%)")
        ax.set_ylim(0, 100)
    for ax in flat[len(genotypes):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"{title} ({denominator} denominator)", fontweight="bold", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_frame_reason_breakdown(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    denominator: str,
) -> None:
    if summary_df.empty:
        return
    if denominator not in {"present", "excluded"}:
        raise ValueError(f"Unsupported denominator: {denominator}")
    frac_col = "fraction_present" if denominator == "present" else "fraction_excluded"
    reasons = summary_df.groupby("frame_qc_reason")["embryo_bins_with_reason"].sum().sort_values(ascending=False).index.tolist()
    genotypes = summary_df["genotype"].drop_duplicates().tolist()
    n_cols = min(3, max(1, len(genotypes)))
    n_rows = int(np.ceil(len(genotypes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False, sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    color_map = {reason: cmap(i % 10) for i, reason in enumerate(reasons)}
    flat = axes.flatten()
    for ax, genotype in zip(flat, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype]
        for reason in reasons:
            sub = grp[grp["frame_qc_reason"] == reason].sort_values("time_bin_center")
            if sub.empty:
                continue
            ax.plot(
                sub["time_bin_center"],
                100.0 * sub[frac_col].fillna(0.0),
                marker="o",
                markersize=3.0,
                linewidth=1.6,
                color=color_map[reason],
                label=reason,
            )
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel(f"Frame reason fraction of {denominator} bins (%)")
        ax.set_ylim(0, 100)
    for ax in flat[len(genotypes):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(3, len(labels)), frameon=False, fontsize=8)
    fig.suptitle(f"Reconstructed frame QC breakdown over time ({denominator} denominator)", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sam2_reason_breakdown(
    summary_df: pd.DataFrame,
    *,
    output_path: Path,
    denominator: str,
) -> None:
    if summary_df.empty:
        return
    if denominator not in {"present", "excluded"}:
        raise ValueError(f"Unsupported denominator: {denominator}")
    frac_col = "fraction_present" if denominator == "present" else "fraction_excluded"
    reasons = summary_df.groupby("reason")["embryo_bins_with_reason"].sum().sort_values(ascending=False).index.tolist()
    genotypes = summary_df["genotype"].drop_duplicates().tolist()
    n_cols = min(3, max(1, len(genotypes)))
    n_rows = int(np.ceil(len(genotypes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False, sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    color_map = {reason: cmap(i % 10) for i, reason in enumerate(reasons)}
    flat = axes.flatten()
    for ax, genotype in zip(flat, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype]
        for reason in reasons:
            sub = grp[grp["reason"] == reason].sort_values("time_bin_center")
            if sub.empty:
                continue
            ax.plot(
                sub["time_bin_center"],
                100.0 * sub[frac_col].fillna(0.0),
                marker="o",
                markersize=3.0,
                linewidth=1.6,
                color=color_map[reason],
                label=reason,
            )
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel(f"SAM2 reason fraction of {denominator} bins (%)")
        ax.set_ylim(0, 100)
    for ax in flat[len(genotypes):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(4, len(labels)), frameon=False, fontsize=8)
    fig.suptitle(f"SAM2 QC reason breakdown over time ({denominator} denominator)", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_dead_flag_review_embryos(
    df: pd.DataFrame,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_per_genotype: int = 3,
) -> pd.DataFrame:
    work = df.copy()
    work["time_bin_start"] = (np.floor(work["predicted_stage_hpf"] / float(bin_width)) * float(bin_width)).astype(int)
    work["time_bin_center"] = work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    bin_summary = (
        work.groupby(["genotype", "time_bin_center"], as_index=False)
        .agg(
            embryo_bins_present=("embryo_id", "nunique"),
            dead_flag_bins=("dead_flag", lambda s: int(s.astype(bool).groupby(work.loc[s.index, "embryo_id"]).any().sum())),
        )
    )
    bin_summary["dead_flag_fraction_present"] = bin_summary["dead_flag_bins"] / bin_summary["embryo_bins_present"].clip(lower=1)
    selected_rows: list[pd.DataFrame] = []
    for genotype, grp in work.groupby("genotype"):
        top_bins = (
            bin_summary[bin_summary["genotype"] == genotype]
            .sort_values(["dead_flag_fraction_present", "dead_flag_bins", "time_bin_center"], ascending=[False, False, True])
            .head(2)
        )
        if top_bins.empty:
            continue
        genotype_candidates = work.merge(top_bins[["genotype", "time_bin_center"]], on=["genotype", "time_bin_center"], how="inner")
        genotype_candidates = genotype_candidates[genotype_candidates["dead_flag"]].copy()
        if genotype_candidates.empty:
            continue
        ranked = (
            genotype_candidates.groupby(["experiment_date", "genotype", "embryo_id"], as_index=False)
            .agg(
                n_dead_rows=("dead_flag", "sum"),
                first_hpf=("predicted_stage_hpf", "min"),
                max_hpf=("predicted_stage_hpf", "max"),
            )
            .sort_values(["n_dead_rows", "first_hpf"], ascending=[False, True])
            .head(max_per_genotype)
        )
        selected_rows.append(ranked)
    if not selected_rows:
        return pd.DataFrame()
    return pd.concat(selected_rows, ignore_index=True)


def _plot_one_death_panel(ax: plt.Axes, embryo_df: pd.DataFrame) -> None:
    data = embryo_df.sort_values("predicted_stage_hpf").copy()
    result = detect_persistent_death_inflection(data)
    x = data["predicted_stage_hpf"].to_numpy(dtype=float)
    y = data["fraction_alive"].to_numpy(dtype=float)
    ax.plot(x, y, color="#1f77b4", marker="o", markersize=3, linewidth=1.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("HPF")
    ax.set_ylabel("Fraction alive")
    ax.grid(alpha=0.2)

    dead_x = data.loc[data["dead_flag"], "predicted_stage_hpf"].to_numpy(dtype=float)
    if dead_x.size:
        ax.scatter(dead_x, np.full_like(dead_x, 1.0), color="red", s=12, label="dead_flag")
    dead2_x = data.loc[data["dead_flag2"], "predicted_stage_hpf"].to_numpy(dtype=float)
    if dead2_x.size:
        ax.scatter(dead2_x, np.full_like(dead2_x, 0.96), color="green", s=12, label="dead_flag2")

    title = str(data["embryo_id"].iloc[0]).split("::")[-1] if "::" in str(data["embryo_id"].iloc[0]) else str(data["embryo_id"].iloc[0])
    subtitle = "no detection"
    if result is not None:
        inflection_time_int = result["inflection_time"]
        inflection_row = data.loc[data["time_int"] == inflection_time_int]
        if not inflection_row.empty:
            inflection_hpf = float(inflection_row["predicted_stage_hpf"].iloc[0])
            ax.axvline(inflection_hpf, color="red", linestyle="--", linewidth=1.1)
            ax.axvspan(
                inflection_hpf - float(QC_DEFAULTS["dead_lead_time_hours"]),
                float(x.max()),
                color="#f3e2b2",
                alpha=0.45,
            )
        stats = result["persistence_stats"]
        subtitle = f"detected: {100.0 * float(stats['dead_fraction']):.0f}% post-inflection"
    ax.set_title(f"{title}\n{subtitle}", fontsize=9, fontweight="bold")


def plot_dead_flag_review(
    df: pd.DataFrame,
    selected_df: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    if selected_df.empty:
        return
    n = len(selected_df)
    n_cols = min(4, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)
    flat = axes.flatten()
    for ax, row in zip(flat, selected_df.itertuples(index=False)):
        embryo_df = df[(df["experiment_date"] == row.experiment_date) & (df["embryo_id"] == row.embryo_id)].copy()
        _plot_one_death_panel(ax, embryo_df)
    for ax in flat[n:]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=8)
    fig.suptitle(
        f"Dead flag review using current persistence method ({int(QC_DEFAULTS['persistence_threshold'] * 100)}% post-inflection threshold)",
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_granular_exclusion_flags",
    "plot_granular_exclusion_flags_alive",
    "plot_frame_reason_breakdown",
    "load_review_dataframe",
    "plot_dead_flag_review",
    "plot_sam2_reason_breakdown",
    "summarize_granular_exclusion_flags",
    "summarize_frame_reasons",
    "select_dead_flag_review_embryos",
    "summarize_dead_flag_agreement",
    "summarize_sam2_reasons",
]


def summarize_dead_flag_agreement(df: pd.DataFrame) -> pd.DataFrame:
    embryo_level = (
        df.groupby(["experiment_date", "genotype", "embryo_id"], as_index=False)
        .agg(
            any_dead_flag=("dead_flag", "max"),
            any_dead_flag2=("dead_flag2", "max"),
        )
    )
    summary = (
        embryo_level.groupby(["experiment_date", "genotype"], as_index=False)
        .agg(
            embryos=("embryo_id", "nunique"),
            dead_flag_embryos=("any_dead_flag", "sum"),
            dead_flag2_embryos=("any_dead_flag2", "sum"),
        )
    )
    summary["dead_flag_only_embryos"] = summary["dead_flag_embryos"] - summary["dead_flag2_embryos"]
    summary["dead_flag_fraction"] = summary["dead_flag_embryos"] / summary["embryos"].clip(lower=1)
    summary["dead_flag2_fraction"] = summary["dead_flag2_embryos"] / summary["embryos"].clip(lower=1)
    return summary.sort_values(["experiment_date", "genotype"]).reset_index(drop=True)
