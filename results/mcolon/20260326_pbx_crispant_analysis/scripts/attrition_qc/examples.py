from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze.viz.embryo_renderer import (
    build_embryo_track,
    draw_rounded_rect,
    put_text_with_outline_colors,
    render_embryo_sequence,
)
from phenotypic_positioning.data import short_name

from .config import (
    BUILD04_DIR,
    CANONICAL_QC_FLAGS,
    DEFAULT_BIN_WIDTH,
    DEFAULT_EXAMPLES_PER_GENOTYPE,
    DEFAULT_RAW_TO_ANALYSIS_GENOTYPE,
    DEFAULT_TARGET_FLAGS,
    DEFAULT_TIME_COL,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_FRAMES,
    DEFAULT_VIDEO_WINDOW_HPF,
    EXPERIMENT_IDS,
    INFO_QC_FLAGS,
    SNIP_ROOT,
)


FRAME_LEVEL_REQUIRED_COLUMNS = [
    "experiment_id",
    "experiment_date",
    "genotype",
    "embryo_id",
    "snip_id",
    "frame_index",
    "predicted_stage_hpf",
    "use_embryo_flag",
    "dead_flag",
    "dead_flag2",
    "sa_outlier_flag",
    "sam2_qc_flag",
    "sam2_qc_flags",
    "frame_flag",
    "no_yolk_flag",
    "focus_flag",
    "bubble_flag",
]

DISPLAY_FLAG_ORDER = [
    "dead_flag",
    "dead_flag2",
    "sa_outlier_flag",
    "sam2_qc_flag",
    "frame_flag",
    "no_yolk_flag",
    "focus_flag",
    "bubble_flag",
]


@dataclass(frozen=True)
class ExampleRenderConfig:
    bin_width: float = DEFAULT_BIN_WIDTH
    time_col: str = DEFAULT_TIME_COL
    window_hpf: float = DEFAULT_VIDEO_WINDOW_HPF
    fps: int = DEFAULT_VIDEO_FPS
    n_frames_out: int = DEFAULT_VIDEO_FRAMES


def _normalize_genotype_series(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    norm = series.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    norm = norm.str.replace("wik-ab", "wik_ab", regex=False)
    return norm.map(mapping).fillna(norm)


def load_build04_frame_qc_dataframe(
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
        part = pd.read_csv(path, usecols=FRAME_LEVEL_REQUIRED_COLUMNS, low_memory=False)
        part["source_experiment_id"] = str(exp_id)
        raw_norm = part["genotype"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("wik-ab", "wik_ab", regex=False)
        part = part[raw_norm.isin(allowed_raw)].copy()
        part["genotype"] = _normalize_genotype_series(part["genotype"], genotype_map)
        frames.append(part)
    if not frames:
        raise ValueError("No build04 frame-level QC data loaded.")
    df = pd.concat(frames, ignore_index=True)
    df["experiment_date"] = df["experiment_date"].astype(str)
    df["embryo_id"] = df["embryo_id"].astype(str)
    df["snip_id"] = df["snip_id"].astype(str)
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df = df.dropna(subset=["frame_index", "predicted_stage_hpf"]).copy()
    for col in ["use_embryo_flag", *DISPLAY_FLAG_ORDER]:
        df[col] = df[col].fillna(False).astype(bool)
    if "sam2_qc_flags" in df.columns:
        df["sam2_qc_flags"] = df["sam2_qc_flags"].fillna("").astype(str)
    else:
        df["sam2_qc_flags"] = ""
    return df.reset_index(drop=True)


def _active_flags(row: pd.Series, flag_cols: Iterable[str] = DISPLAY_FLAG_ORDER) -> list[str]:
    active: list[str] = []
    for col in flag_cols:
        if bool(row.get(col, False)):
            active.append(str(col))
    return active


def _sam2_qc_details(row: pd.Series) -> list[str]:
    if not bool(row.get("sam2_qc_flag", False)):
        return []
    raw = str(row.get("sam2_qc_flags", "")).strip()
    if not raw or raw.lower() == "nan":
        return []
    normalized = raw
    for sep in ["|", ";"]:
        normalized = normalized.replace(sep, ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    return parts


def build_flag_example_summary(
    frame_df: pd.DataFrame,
    *,
    target_flag: str,
    bin_width: float = DEFAULT_BIN_WIDTH,
    time_col: str = DEFAULT_TIME_COL,
) -> pd.DataFrame:
    if target_flag not in DISPLAY_FLAG_ORDER:
        raise ValueError(f"Unsupported target_flag: {target_flag}")
    work = frame_df.copy()
    work["time_bin_start"] = (np.floor(pd.to_numeric(work[time_col], errors="coerce") / float(bin_width)) * float(bin_width)).astype(int)
    work["time_bin_center"] = work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    work["dead_like"] = work["dead_flag"] | work["dead_flag2"]
    other_flags = [flag for flag in CANONICAL_QC_FLAGS if flag != target_flag]
    group_cols = ["experiment_date", "source_experiment_id", "experiment_id", "genotype", "embryo_id", "time_bin_start", "time_bin_center"]
    rows: list[dict[str, object]] = []
    for key, grp in work.groupby(group_cols, sort=True):
        grp = grp.sort_values([time_col, "frame_index"]).copy()
        included_in_bin = bool(grp["use_embryo_flag"].any())
        dead_like = bool(grp["dead_like"].any())
        target_any = bool(grp[target_flag].any())
        if not target_any:
            continue
        active_other_flags = [flag for flag in other_flags if bool(grp[flag].any())]
        active_canonical_flags = [flag for flag in CANONICAL_QC_FLAGS if bool(grp[flag].any())]
        representative_pool = grp[grp[target_flag]].copy()
        representative_pool["bin_distance"] = (representative_pool[time_col] - float(key[-1])).abs()
        representative = representative_pool.sort_values(
            ["bin_distance", "use_embryo_flag", "frame_index"],
            ascending=[True, True, True],
        ).iloc[0]
        active_flags = _active_flags(representative)
        row = {
            "experiment_date": key[0],
            "source_experiment_id": key[1],
            "experiment_id": key[2],
            "genotype": key[3],
            "embryo_id": key[4],
            "time_bin_start": key[5],
            "time_bin_center": key[6],
            "n_frames_in_bin": int(len(grp)),
            "n_excluded_frames": int((~grp["use_embryo_flag"]).sum()),
            "n_target_flag_frames": int(grp[target_flag].sum()),
            "target_frame_fraction": float(grp[target_flag].mean()),
            "n_other_canonical_frames": int(sum(int(grp[flag].sum()) for flag in other_flags)),
            "n_other_canonical_families_active": int(len(active_other_flags)),
            "active_canonical_flags": "|".join(active_canonical_flags),
            "active_other_canonical_flags": "|".join(active_other_flags),
            "included_in_bin": included_in_bin,
            "dead_like": dead_like,
            "only_target_canonical": bool(representative[target_flag] and not any(bool(representative.get(flag, False)) for flag in other_flags)),
            "target_family_only_in_bin": bool(not active_other_flags),
            "representative_snip_id": str(representative["snip_id"]),
            "representative_frame_index": int(representative["frame_index"]),
            "representative_stage_hpf": float(representative[time_col]),
            "representative_use_embryo_flag": bool(representative["use_embryo_flag"]),
            "representative_active_flags": "|".join(active_flags) if active_flags else "",
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["genotype", "time_bin_center", "representative_stage_hpf", "representative_frame_index"]
    ).reset_index(drop=True)


def build_flag_example_frame_summary(
    frame_df: pd.DataFrame,
    *,
    target_flag: str,
    bin_width: float = DEFAULT_BIN_WIDTH,
    time_col: str = DEFAULT_TIME_COL,
) -> pd.DataFrame:
    if target_flag not in DISPLAY_FLAG_ORDER:
        raise ValueError(f"Unsupported target_flag: {target_flag}")
    work = frame_df.copy()
    work["dead_like"] = work["dead_flag"] | work["dead_flag2"]
    work["time_bin_start"] = (np.floor(pd.to_numeric(work[time_col], errors="coerce") / float(bin_width)) * float(bin_width)).astype(int)
    work["time_bin_center"] = work["time_bin_start"].astype(float) + float(bin_width) / 2.0
    other_flags = [flag for flag in CANONICAL_QC_FLAGS if flag != target_flag]
    rows: list[dict[str, object]] = []
    flagged = work[work[target_flag]].copy()
    for _, row in flagged.sort_values(["genotype", time_col, "frame_index"]).iterrows():
        active_flags = _active_flags(row)
        active_other_flags = [flag for flag in other_flags if bool(row.get(flag, False))]
        rows.append(
            {
                "experiment_date": str(row["experiment_date"]),
                "source_experiment_id": str(row["source_experiment_id"]),
                "experiment_id": str(row["experiment_id"]),
                "genotype": str(row["genotype"]),
                "embryo_id": str(row["embryo_id"]),
                "snip_id": str(row["snip_id"]),
                "frame_index": int(row["frame_index"]),
                "frame_stage_hpf": float(row[time_col]),
                "time_bin_start": int(row["time_bin_start"]),
                "time_bin_center": float(row["time_bin_center"]),
                "included_in_frame": bool(row["use_embryo_flag"]),
                "dead_like": bool(row["dead_like"]),
                "n_other_canonical_families_active": int(len(active_other_flags)),
                "active_canonical_flags": "|".join([flag for flag in CANONICAL_QC_FLAGS if bool(row.get(flag, False))]),
                "active_other_canonical_flags": "|".join(active_other_flags),
                "target_family_only_in_frame": bool(not active_other_flags),
                "representative_snip_id": str(row["snip_id"]),
                "representative_frame_index": int(row["frame_index"]),
                "representative_stage_hpf": float(row[time_col]),
                "representative_use_embryo_flag": bool(row["use_embryo_flag"]),
                "representative_active_flags": "|".join(active_flags) if active_flags else "",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def select_flag_examples(
    summary_df: pd.DataFrame,
    *,
    max_examples_per_genotype: int = DEFAULT_EXAMPLES_PER_GENOTYPE,
    require_target_family_only_in_bin: bool = False,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    candidates = summary_df[(~summary_df["dead_like"]) & (~summary_df["included_in_bin"])].copy()
    if require_target_family_only_in_bin:
        candidates = candidates[candidates["target_family_only_in_bin"]].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        [
            "genotype",
            "target_family_only_in_bin",
            "only_target_canonical",
            "target_frame_fraction",
            "n_target_flag_frames",
            "n_excluded_frames",
            "representative_stage_hpf",
        ],
        ascending=[True, False, False, False, False, False, True],
    ).copy()
    selected = (
        candidates.groupby("genotype", group_keys=False)
        .head(int(max_examples_per_genotype))
        .reset_index(drop=True)
    )
    selected["example_rank_within_genotype"] = selected.groupby("genotype").cumcount() + 1
    return selected


def select_flag_example_frames(
    summary_df: pd.DataFrame,
    *,
    max_examples_per_genotype: int = DEFAULT_EXAMPLES_PER_GENOTYPE,
    require_target_family_only_in_frame: bool = True,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    candidates = summary_df[(~summary_df["dead_like"]) & (~summary_df["included_in_frame"])].copy()
    if require_target_family_only_in_frame:
        candidates = candidates[candidates["target_family_only_in_frame"]].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        [
            "genotype",
            "target_family_only_in_frame",
            "frame_stage_hpf",
            "frame_index",
        ],
        ascending=[True, False, True, True],
    ).copy()
    candidates = candidates.groupby(["genotype", "embryo_id"], group_keys=False).head(1).reset_index(drop=True)
    selected = (
        candidates.groupby("genotype", group_keys=False)
        .head(int(max_examples_per_genotype))
        .reset_index(drop=True)
    )
    selected["example_rank_within_genotype"] = selected.groupby("genotype").cumcount() + 1
    return selected


def summarize_flag_examples(
    frame_df: pd.DataFrame,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
    time_col: str = DEFAULT_TIME_COL,
    target_flags: list[str] | None = None,
) -> pd.DataFrame:
    target_flags = target_flags or list(DEFAULT_TARGET_FLAGS)
    rows: list[dict[str, object]] = []
    for target_flag in target_flags:
        work = frame_df.copy()
        work["time_bin_start"] = (np.floor(pd.to_numeric(work[time_col], errors="coerce") / float(bin_width)) * float(bin_width)).astype(int)
        work["time_bin_center"] = work["time_bin_start"].astype(float) + float(bin_width) / 2.0
        work["dead_like"] = work["dead_flag"] | work["dead_flag2"]
        work["bin_uid"] = (
            work["experiment_date"].astype(str)
            + "::"
            + work["embryo_id"].astype(str)
            + "::"
            + work["time_bin_start"].astype(str)
        )
        frame_grouped = (
            work.groupby(["genotype", "time_bin_center"], as_index=False)
            .agg(
                total_rows=("snip_id", "size"),
                target_flagged_rows=(target_flag, "sum"),
                excluded_rows=("use_embryo_flag", lambda s: int((~s).sum())),
            )
        )
        bin_summary = build_flag_example_summary(
            frame_df,
            target_flag=target_flag,
            bin_width=bin_width,
            time_col=time_col,
        )
        if bin_summary.empty:
            continue
        bin_grouped = (
            bin_summary.groupby(["genotype", "time_bin_center"], as_index=False)
            .agg(
                target_flagged_bins=("embryo_id", "size"),
                alive_target_flagged_bins=("dead_like", lambda s: int((~s).sum())),
                alive_excluded_target_flagged_bins=("included_in_bin", lambda s: int((~s).sum())),
                alive_excluded_only_target_bins=(
                    "only_target_canonical",
                    lambda s: int(
                        (
                            s.astype(bool)
                            & (~bin_summary.loc[s.index, "dead_like"].astype(bool))
                            & (~bin_summary.loc[s.index, "included_in_bin"].astype(bool))
                        ).sum()
                    ),
                ),
                alive_excluded_target_family_only_bins=(
                    "target_family_only_in_bin",
                    lambda s: int(
                        (
                            s.astype(bool)
                            & (~bin_summary.loc[s.index, "dead_like"].astype(bool))
                            & (~bin_summary.loc[s.index, "included_in_bin"].astype(bool))
                        ).sum()
                    ),
                ),
            )
        )
        grouped = frame_grouped.merge(bin_grouped, on=["genotype", "time_bin_center"], how="left")
        grouped["target_flag"] = target_flag
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _choose_display_row(group: pd.DataFrame, frame_index: int) -> pd.Series:
    exact = group[group["frame_index"] == int(frame_index)]
    if not exact.empty:
        return exact.iloc[0]
    distances = (group["frame_index"].astype(int) - int(frame_index)).abs()
    return group.iloc[int(distances.argmin())]


def _annotate_frame(
    frame: np.ndarray,
    *,
    display_t_hpf: float,
    source_row: pd.Series,
    target_flag: str,
    exemplar_row: pd.Series,
) -> np.ndarray:
    import cv2

    frame_canvas = frame.copy()
    h, w = frame_canvas.shape[:2]
    panel_w = 250
    canvas = np.full((h, w + panel_w, 3), 248, dtype=np.uint8)
    canvas[:, :w] = frame_canvas

    target_color = (0, 153, 255) if target_flag == "no_yolk_flag" else (255, 170, 0)
    flags = _active_flags(source_row)
    sam2_details = _sam2_qc_details(source_row)

    overlay = canvas[:, :w].copy()
    cv2.rectangle(overlay, (0, 0), (150, 42), (235, 235, 235), thickness=-1)
    cv2.addWeighted(overlay, 0.72, canvas[:, :w], 0.28, 0, canvas[:, :w])
    put_text_with_outline_colors(
        canvas[:, :w],
        f"{display_t_hpf:.1f} hpf",
        (12, 29),
        font_scale=0.70,
        fg_bgr=(20, 20, 20),
        outline_bgr=(255, 255, 255),
    )

    panel_x0 = w
    cv2.rectangle(canvas, (panel_x0, 0), (panel_x0 + panel_w - 1, h - 1), (248, 248, 248), thickness=-1)
    cv2.line(canvas, (panel_x0, 0), (panel_x0, h - 1), (205, 205, 205), thickness=2)

    draw_rounded_rect(canvas, (panel_x0 + 16, 14), (panel_x0 + panel_w - 16, 52), target_color, corner_radius=10, thickness=-1)
    put_text_with_outline_colors(
        canvas,
        "QC menu",
        (panel_x0 + 28, 40),
        font_scale=0.72,
        fg_bgr=(255, 255, 255),
        outline_bgr=(0, 0, 0),
    )

    menu_lines: list[str] = []
    if bool(source_row.get(target_flag, False)):
        menu_lines.append(f"target: {target_flag.replace('_flag', '')}")
    if flags:
        menu_lines.extend(flag.replace("_flag", "") for flag in flags)
    else:
        menu_lines.append("none")
    if sam2_details:
        menu_lines.append("sam2 details:")
        menu_lines.extend(f"  - {detail}" for detail in sam2_details)

    y = 86
    line_step = 28
    for line in menu_lines:
        put_text_with_outline_colors(
            canvas,
            line,
            (panel_x0 + 18, y),
            font_scale=0.50,
            fg_bgr=(35, 35, 35),
            outline_bgr=(255, 255, 255),
        )
        y += line_step
    return canvas


def render_flag_example_media(
    example_row: pd.Series,
    frame_df: pd.DataFrame,
    *,
    target_flag: str,
    output_dir: Path,
    snip_root: Path = SNIP_ROOT,
    render_config: ExampleRenderConfig | None = None,
) -> dict[str, object]:
    import cv2

    render_config = render_config or ExampleRenderConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    embryo_rows = frame_df[
        (frame_df["experiment_date"] == example_row["experiment_date"])
        & (frame_df["embryo_id"] == example_row["embryo_id"])
    ].copy()
    embryo_rows = embryo_rows.sort_values(["predicted_stage_hpf", "frame_index"]).reset_index(drop=True)
    track = build_embryo_track(embryo_rows)

    center = float(example_row["representative_stage_hpf"])
    start_hpf = max(float(embryo_rows["predicted_stage_hpf"].min()), center - float(render_config.window_hpf) / 2.0)
    end_hpf = min(float(embryo_rows["predicted_stage_hpf"].max()), center + float(render_config.window_hpf) / 2.0)
    if end_hpf <= start_hpf:
        start_hpf = float(embryo_rows["predicted_stage_hpf"].min())
        end_hpf = float(embryo_rows["predicted_stage_hpf"].max())

    sequence = render_embryo_sequence(
        track,
        snip_root=snip_root,
        start_hpf=float(start_hpf),
        end_hpf=float(end_hpf),
        fps=int(render_config.fps),
        n_frames_out=int(render_config.n_frames_out),
        missing_frame_bgr=(245, 245, 245),
        pad_to_even=True,
    )

    stem = (
        f"{target_flag}__{example_row['experiment_date']}__{example_row['embryo_id']}__"
        f"{int(example_row['time_bin_start']):03d}hpf"
    )
    video_path = output_dir / f"{stem}.mp4"
    poster_path = output_dir / f"{stem}.png"

    first = sequence.frames[0]
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(sequence.fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {video_path}")

    poster_frames: list[np.ndarray] = []
    poster_indices = np.linspace(0, max(0, len(sequence.frames) - 1), 6, dtype=int).tolist()
    try:
        for idx, (frame, display_t, fi_lo, fi_hi, alpha) in enumerate(
            zip(
                sequence.frames,
                sequence.times_hpf.tolist(),
                sequence.frame_indices_lo.tolist(),
                sequence.frame_indices_hi.tolist(),
                sequence.alpha.tolist(),
                strict=False,
            )
        ):
            frame_index = int(fi_hi) if float(alpha) >= 0.5 else int(fi_lo)
            source_row = _choose_display_row(embryo_rows, frame_index)
            annotated = _annotate_frame(
                frame,
                display_t_hpf=float(display_t),
                source_row=source_row,
                target_flag=target_flag,
                exemplar_row=example_row,
            )
            writer.write(annotated)
            if idx in poster_indices:
                poster_frames.append(annotated)
    finally:
        writer.release()

    if poster_frames:
        poster = np.concatenate(poster_frames, axis=1)
        cv2.imwrite(str(poster_path), poster)

    return {
        "video_path": str(video_path),
        "poster_path": str(poster_path),
        "video_width": int(width),
        "video_height": int(height),
        "video_start_hpf": float(start_hpf),
        "video_end_hpf": float(end_hpf),
    }


__all__ = [
    "ExampleRenderConfig",
    "build_flag_example_frame_summary",
    "build_flag_example_summary",
    "load_build04_frame_qc_dataframe",
    "render_flag_example_media",
    "select_flag_example_frames",
    "select_flag_examples",
    "summarize_flag_examples",
]
