from __future__ import annotations

import pandas as pd


EXCLUSIONARY_FLAG_COLUMNS = [
    "dead_flag",
    "dead_flag2",
    "sa_outlier_flag",
    "sam2_qc_flag",
    "frame_flag",
    "no_yolk_flag",
]


def summarize_attrition(status_df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        status_df.groupby(group_cols, as_index=False)
        .agg(
            embryos_present=("embryo_uid", "nunique"),
            embryos_included=("included", "sum"),
            embryos_excluded=("excluded", "sum"),
            excluded_death_involved=("excluded_death_involved", "sum"),
            excluded_non_death_qc=("excluded_non_death_qc", "sum"),
            excluded_dead_only=("excluded_dead_only", "sum"),
            excluded_qc_only=("excluded_qc_only", "sum"),
            excluded_dead_and_qc=("excluded_dead_and_qc", "sum"),
            excluded_other=("excluded_other", "sum"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary["included_fraction"] = summary["embryos_included"] / summary["embryos_present"].clip(lower=1)
    summary["excluded_fraction"] = summary["embryos_excluded"] / summary["embryos_present"].clip(lower=1)
    summary["excluded_death_involved_fraction"] = summary["excluded_death_involved"] / summary["embryos_present"].clip(lower=1)
    summary["excluded_non_death_qc_fraction"] = summary["excluded_non_death_qc"] / summary["embryos_present"].clip(lower=1)
    return summary


def summarize_overall_attrition(status_df: pd.DataFrame) -> pd.DataFrame:
    embryo_level = (
        status_df.groupby(["genotype", "embryo_uid"], as_index=False)
        .agg(
            ever_included=("included", "max"),
            ever_excluded=("excluded", "max"),
            ever_excluded_death_involved=("excluded_death_involved", "max"),
            ever_excluded_non_death_qc=("excluded_non_death_qc", "max"),
            ever_excluded_dead_only=("excluded_dead_only", "max"),
            ever_excluded_qc_only=("excluded_qc_only", "max"),
            ever_excluded_dead_and_qc=("excluded_dead_and_qc", "max"),
            n_time_bins_present=("time_bin_start", "nunique"),
            n_time_bins_included=("included", "sum"),
            n_time_bins_excluded=("excluded", "sum"),
        )
    )
    summary = (
        embryo_level.groupby("genotype", as_index=False)
        .agg(
            embryos_present=("embryo_uid", "nunique"),
            embryos_ever_included=("ever_included", "sum"),
            embryos_ever_excluded=("ever_excluded", "sum"),
            embryos_ever_excluded_death_involved=("ever_excluded_death_involved", "sum"),
            embryos_ever_excluded_non_death_qc=("ever_excluded_non_death_qc", "sum"),
            embryos_ever_excluded_dead_only=("ever_excluded_dead_only", "sum"),
            embryos_ever_excluded_qc_only=("ever_excluded_qc_only", "sum"),
            embryos_ever_excluded_dead_and_qc=("ever_excluded_dead_and_qc", "sum"),
            total_embryo_bins_present=("n_time_bins_present", "sum"),
            total_embryo_bins_included=("n_time_bins_included", "sum"),
            total_embryo_bins_excluded=("n_time_bins_excluded", "sum"),
        )
        .sort_values("genotype")
        .reset_index(drop=True)
    )
    summary["fraction_ever_included"] = summary["embryos_ever_included"] / summary["embryos_present"].clip(lower=1)
    summary["fraction_ever_excluded"] = summary["embryos_ever_excluded"] / summary["embryos_present"].clip(lower=1)
    summary["fraction_ever_excluded_death_involved"] = (
        summary["embryos_ever_excluded_death_involved"] / summary["embryos_present"].clip(lower=1)
    )
    summary["fraction_ever_excluded_non_death_qc"] = (
        summary["embryos_ever_excluded_non_death_qc"] / summary["embryos_present"].clip(lower=1)
    )
    summary["unique_experiments"] = summary["genotype"].map(status_df.groupby("genotype")["experiment_date"].nunique().to_dict())
    return summary


def summarize_alive_only_qc(status_df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    alive = status_df[~status_df["dead_like"]].copy()
    summary = (
        alive.groupby(group_cols, as_index=False)
        .agg(
            alive_embryos_present=("embryo_uid", "nunique"),
            alive_embryos_included=("included", "sum"),
            alive_embryos_excluded=("excluded", "sum"),
            alive_sa_outlier_flag_rate=("sa_outlier_flag", "mean"),
            alive_sam2_qc_flag_rate=("sam2_qc_flag", "mean"),
            alive_frame_flag_rate=("frame_flag", "mean"),
            alive_no_yolk_flag_rate=("no_yolk_flag", "mean"),
            alive_focus_flag_rate=("focus_flag", "mean"),
            alive_bubble_flag_rate=("bubble_flag", "mean"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary["alive_use_embryo_pass_rate"] = summary["alive_embryos_included"] / summary["alive_embryos_present"].clip(lower=1)
    summary["alive_use_embryo_fail_rate"] = summary["alive_embryos_excluded"] / summary["alive_embryos_present"].clip(lower=1)
    return summary


def summarize_exclusionary_flag_rates(status_df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        status_df.groupby(group_cols, as_index=False)
        .agg(
            embryos_present=("embryo_uid", "nunique"),
            embryos_excluded=("excluded", "sum"),
            dead_flag_count=("dead_flag", "sum"),
            dead_flag2_count=("dead_flag2", "sum"),
            sa_outlier_flag_count=("sa_outlier_flag", "sum"),
            sam2_qc_flag_count=("sam2_qc_flag", "sum"),
            frame_flag_count=("frame_flag", "sum"),
            no_yolk_flag_count=("no_yolk_flag", "sum"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    for flag in EXCLUSIONARY_FLAG_COLUMNS:
        count_col = f"{flag}_count"
        summary[f"{flag}_fraction_present"] = summary[count_col] / summary["embryos_present"].clip(lower=1)

    excluded = status_df[status_df["excluded"]].copy()
    if excluded.empty:
        for flag in EXCLUSIONARY_FLAG_COLUMNS:
            summary[f"{flag}_count_excluded"] = 0
            summary[f"{flag}_fraction_excluded"] = 0.0
        return summary

    excluded_summary = (
        excluded.groupby(group_cols, as_index=False)
        .agg(
            embryos_excluded=("embryo_uid", "nunique"),
            dead_flag_count_excluded=("dead_flag", "sum"),
            dead_flag2_count_excluded=("dead_flag2", "sum"),
            sa_outlier_flag_count_excluded=("sa_outlier_flag", "sum"),
            sam2_qc_flag_count_excluded=("sam2_qc_flag", "sum"),
            frame_flag_count_excluded=("frame_flag", "sum"),
            no_yolk_flag_count_excluded=("no_yolk_flag", "sum"),
        )
    )
    summary = summary.merge(
        excluded_summary,
        on=group_cols,
        how="left",
        suffixes=("", "_excluded_merge"),
    )
    summary["embryos_excluded"] = summary["embryos_excluded_excluded_merge"].fillna(summary["embryos_excluded"])
    summary = summary.drop(columns=["embryos_excluded_excluded_merge"])
    count_excluded_map = {
        "dead_flag": "dead_flag_count_excluded",
        "dead_flag2": "dead_flag2_count_excluded",
        "sa_outlier_flag": "sa_outlier_flag_count_excluded",
        "sam2_qc_flag": "sam2_qc_flag_count_excluded",
        "frame_flag": "frame_flag_count_excluded",
        "no_yolk_flag": "no_yolk_flag_count_excluded",
    }
    for flag, count_col in count_excluded_map.items():
        summary[count_col] = summary[count_col].fillna(0).astype(int)
        summary[f"{flag}_fraction_excluded"] = summary[count_col] / summary["embryos_excluded"].clip(lower=1)
    return summary
