"""Reusable embryo-centric frame lookup and rendering utilities.

This module is intentionally focused on embryo IDs and snip-backed rendering,
not on any one presentation format. The same core helpers can be consumed by:

- paired feature-trace plus embryo videos
- full 96-well plate timelapse movies
- future embryo-centric still exports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbryoTrack:
    """Time-aligned frame indices for one embryo."""

    embryo_id: str
    experiment_date: str
    times_hpf: np.ndarray
    frame_indices: np.ndarray
    well: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbryoSelectionPolicy:
    """Selection policy for choosing one embryo track from a pool."""

    prefer_e01: bool = True
    verify_endpoint_snips: bool = True
    alive_threshold: float = 0.5


@dataclass(frozen=True)
class PlateTrackSelection:
    """Selected per-well embryo tracks plus timing metadata."""

    tracks: list[EmbryoTrack]
    start_age_by_well: dict[str, float]
    last_alive_by_well: dict[str, tuple[float, int] | None]
    default_t_min: float
    robust_t_max: float


@dataclass(frozen=True)
class EmbryoTileStyle:
    """Rendering configuration for a single embryo tile."""

    radius: int
    show_label: bool = False
    label: str = ""
    outside_bgr: tuple[int, int, int] = (0, 0, 0)
    well_fill_bgr: tuple[int, int, int] = (0, 0, 0)
    well_rim_bgr: tuple[int, int, int] = (255, 255, 255)
    label_fg_bgr: tuple[int, int, int] = (255, 255, 255)
    label_outline_bgr: tuple[int, int, int] = (0, 0, 0)
    well_rim_thickness: int = 1


EmbryoRenderStyle = EmbryoTileStyle


@dataclass(frozen=True)
class EmbryoSequenceRender:
    """Rendered embryo frames plus the playback schedule that produced them."""

    frames: list[np.ndarray]
    times_hpf: np.ndarray
    fps: int
    frame_indices_lo: np.ndarray
    frame_indices_hi: np.ndarray
    alpha: np.ndarray


def build_snip_path(snip_root: Path, experiment_date: str, embryo_id: str, frame_index: int) -> Path:
    """Return the canonical snip JPG path for one embryo frame."""
    return snip_root / str(experiment_date) / f"{embryo_id}_t{int(frame_index):04d}.jpg"


def read_snip_bgr(snip_path: Path) -> Optional[np.ndarray]:
    """Load one snip as a BGR image, returning ``None`` when missing."""
    import cv2

    img = cv2.imread(str(snip_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def build_embryo_track(
    df: pd.DataFrame,
    *,
    embryo_id_col: str = "embryo_id",
    experiment_col: str = "experiment_date",
    time_col: str = "predicted_stage_hpf",
    frame_col: str = "frame_index",
    well_col: str | None = None,
) -> EmbryoTrack:
    """Build one deduplicated embryo track from a frame-level dataframe."""
    if df.empty:
        raise ValueError("Cannot build an embryo track from an empty dataframe.")

    work = df.copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work[frame_col] = pd.to_numeric(work[frame_col], errors="coerce")
    work = work.dropna(subset=[embryo_id_col, experiment_col, time_col, frame_col]).copy()
    if work.empty:
        raise ValueError("Embryo track dataframe has no valid time/frame rows.")

    work[embryo_id_col] = work[embryo_id_col].astype(str)
    work[experiment_col] = work[experiment_col].astype(str)
    work = work.sort_values([time_col, frame_col]).copy()

    embryo_ids = work[embryo_id_col].drop_duplicates().tolist()
    if len(embryo_ids) != 1:
        raise ValueError(f"Expected exactly one embryo_id, found {embryo_ids[:5]}")

    exp_dates = work[experiment_col].drop_duplicates().tolist()
    if len(exp_dates) != 1:
        raise ValueError(f"Expected exactly one experiment_date for {embryo_ids[0]!r}, found {exp_dates[:5]}")

    times = work[time_col].to_numpy(dtype=float)
    frame_indices = work[frame_col].to_numpy(dtype=int)
    if times.size > 1:
        keep = np.concatenate([[True], times[1:] != times[:-1]])
        times = times[keep]
        frame_indices = frame_indices[keep]

    if times.size == 0:
        raise ValueError(f"Embryo {embryo_ids[0]!r} has no usable timepoints.")

    well = None
    if well_col is not None and well_col in work.columns:
        well_vals = work[well_col].dropna().astype(str).drop_duplicates().tolist()
        well = well_vals[0] if well_vals else None

    return EmbryoTrack(
        embryo_id=str(embryo_ids[0]),
        experiment_date=str(exp_dates[0]),
        times_hpf=times,
        frame_indices=frame_indices,
        well=well,
    )


def resolve_frame_schedule(
    track: EmbryoTrack,
    t_out: np.ndarray,
    *,
    clamp_time: float | None = None,
    clamp_frame_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve frame interpolation schedule for a track over requested output times."""
    times = np.asarray(track.times_hpf, dtype=float)
    fis = np.asarray(track.frame_indices, dtype=int)
    t_out = np.asarray(t_out, dtype=float)

    n = int(t_out.size)
    fi0 = np.zeros(n, dtype=np.int32)
    fi1 = np.zeros(n, dtype=np.int32)
    alpha = np.zeros(n, dtype=np.float32)

    for k, t in enumerate(t_out.tolist()):
        if clamp_time is not None and clamp_frame_index is not None and t >= float(clamp_time):
            fi0[k] = int(clamp_frame_index)
            fi1[k] = int(clamp_frame_index)
            alpha[k] = 0.0
            continue
        if t <= float(times[0]):
            fi0[k] = int(fis[0])
            fi1[k] = int(fis[0])
            alpha[k] = 0.0
            continue
        if t >= float(times[-1]):
            fi0[k] = int(fis[-1])
            fi1[k] = int(fis[-1])
            alpha[k] = 0.0
            continue

        j = int(np.searchsorted(times, t, side="left"))
        i0 = max(0, j - 1)
        i1 = min(int(times.size - 1), j)
        t0 = float(times[i0])
        t1 = float(times[i1])
        fi0[k] = int(fis[i0])
        fi1[k] = int(fis[i1])
        if i0 == i1 or t1 <= t0 + 1e-9:
            alpha[k] = 0.0
        else:
            alpha[k] = float(min(1.0, max(0.0, (t - t0) / (t1 - t0))))

    return fi0, fi1, alpha


def blend_frame_pair(
    frame_lo: Optional[np.ndarray],
    frame_hi: Optional[np.ndarray],
    alpha: float,
) -> Optional[np.ndarray]:
    """Blend two embryo frames, falling back to whichever exists."""
    import cv2

    if frame_lo is None and frame_hi is None:
        return None
    if frame_lo is None:
        return frame_hi.copy() if frame_hi is not None else None
    if frame_hi is None:
        return frame_lo.copy()
    if alpha <= 0.0:
        return frame_lo.copy()
    if alpha >= 1.0:
        return frame_hi.copy()
    return cv2.addWeighted(
        frame_lo.astype(np.float32),
        float(1.0 - alpha),
        frame_hi.astype(np.float32),
        float(alpha),
        0,
    ).astype(np.uint8)


def resolve_playback_frame_count(
    *,
    fps: int,
    video_duration_s: float | None = None,
    n_frames_out: int | None = None,
) -> int:
    """Resolve playback length, requiring exactly one duration/frame-count mode."""
    has_duration = video_duration_s is not None
    has_n_frames = n_frames_out is not None
    if has_duration == has_n_frames:
        raise ValueError("Provide exactly one of video_duration_s or n_frames_out.")

    if int(fps) <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")

    if has_duration:
        duration = float(video_duration_s)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"video_duration_s must be positive and finite, got {video_duration_s!r}")
        return max(2, int(round(duration * float(fps))))

    n_frames = int(n_frames_out)
    if n_frames < 2:
        raise ValueError(f"n_frames_out must be >= 2, got {n_frames!r}")
    return n_frames


def render_embryo_sequence(
    track: EmbryoTrack,
    *,
    snip_root: Path,
    start_hpf: float,
    end_hpf: float,
    fps: int = 30,
    video_duration_s: float | None = None,
    n_frames_out: int | None = None,
    clamp_time: float | None = None,
    clamp_frame_index: int | None = None,
    missing_frame_bgr: tuple[int, int, int] = (0, 0, 0),
    pad_to_even: bool = True,
    frame_reader: Callable[[Path], Optional[np.ndarray]] | None = None,
) -> EmbryoSequenceRender:
    """Render a time-aligned embryo frame sequence from biological time inputs.

    The caller specifies the biological window (`start_hpf`, `end_hpf`) and
    either the desired playback duration or the exact output frame count.
    """
    if not math.isfinite(float(start_hpf)) or not math.isfinite(float(end_hpf)):
        raise ValueError("start_hpf and end_hpf must be finite.")
    if float(end_hpf) <= float(start_hpf):
        raise ValueError(
            f"end_hpf must be greater than start_hpf, got {start_hpf!r}, {end_hpf!r}"
        )

    n_frames = resolve_playback_frame_count(
        fps=int(fps),
        video_duration_s=video_duration_s,
        n_frames_out=n_frames_out,
    )
    out_times = np.linspace(float(start_hpf), float(end_hpf), int(n_frames))
    fi0_schedule, fi1_schedule, alpha_schedule = resolve_frame_schedule(
        track,
        out_times,
        clamp_time=clamp_time,
        clamp_frame_index=clamp_frame_index,
    )

    reader = frame_reader or read_snip_bgr
    unique_frame_indices = sorted(
        {int(fi) for fi in fi0_schedule.tolist()} | {int(fi) for fi in fi1_schedule.tolist()}
    )
    snip_cache: dict[int, np.ndarray] = {}
    for frame_index in unique_frame_indices:
        snip_path = build_snip_path(snip_root, track.experiment_date, track.embryo_id, frame_index)
        img = reader(snip_path)
        if img is not None:
            snip_cache[int(frame_index)] = img

    if not snip_cache:
        raise RuntimeError(
            f"No snip frames could be loaded for embryo {track.embryo_id!r} in "
            f"{track.experiment_date!r} under {snip_root}."
        )

    first_snip = next(iter(snip_cache.values()))
    snip_h, snip_w = first_snip.shape[:2]
    frame_w = int(snip_w)
    frame_h = int(snip_h)
    if pad_to_even:
        if frame_w % 2 != 0:
            frame_w += 1
        if frame_h % 2 != 0:
            frame_h += 1

    frames: list[np.ndarray] = []
    for fi_lo, fi_hi, alpha in zip(
        fi0_schedule.tolist(),
        fi1_schedule.tolist(),
        alpha_schedule.tolist(),
        strict=False,
    ):
        snip = blend_frame_pair(
            snip_cache.get(int(fi_lo)),
            snip_cache.get(int(fi_hi)),
            float(alpha),
        )
        canvas = np.full((frame_h, frame_w, 3), np.array(missing_frame_bgr, dtype=np.uint8), dtype=np.uint8)
        if snip is not None:
            h, w = snip.shape[:2]
            canvas[: min(h, frame_h), : min(w, frame_w)] = snip[: min(h, frame_h), : min(w, frame_w)]
        frames.append(canvas)

    return EmbryoSequenceRender(
        frames=frames,
        times_hpf=out_times,
        fps=int(fps),
        frame_indices_lo=fi0_schedule,
        frame_indices_hi=fi1_schedule,
        alpha=alpha_schedule,
    )


def export_embryo_video(
    sequence: EmbryoSequenceRender,
    output_path: Path,
    *,
    codec: str = "mp4v",
    frame_transform: Callable[[np.ndarray, float], np.ndarray] | None = None,
) -> tuple[int, int]:
    """Write a rendered embryo sequence to MP4 and return `(width, height)`."""
    import cv2

    if not sequence.frames:
        raise ValueError("Cannot export an empty embryo sequence.")

    first = sequence.frames[0]
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*str(codec))
    writer = cv2.VideoWriter(str(output_path), fourcc, float(sequence.fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    try:
        for frame, t_hpf in zip(sequence.frames, sequence.times_hpf.tolist(), strict=False):
            out_frame = frame.copy()
            if frame_transform is not None:
                out_frame = frame_transform(out_frame, float(t_hpf))
            if out_frame.shape[:2] != (height, width):
                raise ValueError(
                    "frame_transform must preserve frame shape; "
                    f"expected {(height, width)}, got {out_frame.shape[:2]}"
                )
            writer.write(out_frame)
    finally:
        writer.release()

    return int(width), int(height)


def select_plate_tracks_from_metadata(
    embryo_metadata_csv: Path,
    *,
    wells_subset: Optional[set[str]] = None,
    snip_root: Optional[Path] = None,
    experiment_date: Optional[str] = None,
    policy: EmbryoSelectionPolicy | None = None,
) -> PlateTrackSelection:
    """Select one representative embryo track per well from metadata CSV."""
    policy = policy or EmbryoSelectionPolicy()

    header = pd.read_csv(embryo_metadata_csv, nrows=0)
    want = [
        "embryo_id",
        "well_id",
        "frame_index",
        "predicted_stage_hpf",
        "start_age_hpf",
        "fraction_alive",
        "dead_flag",
    ]
    cols = [c for c in want if c in header.columns]
    missing_critical = [c for c in ["embryo_id", "well_id", "frame_index", "predicted_stage_hpf"] if c not in cols]
    if missing_critical:
        raise ValueError(f"Missing required columns in {embryo_metadata_csv.name}: {missing_critical}")

    df = pd.read_csv(embryo_metadata_csv, usecols=cols, low_memory=False)
    df["well_id"] = df["well_id"].astype(str).str.strip().str.upper()
    if wells_subset is not None:
        df = df[df["well_id"].isin(wells_subset)].copy()
    if df.empty:
        raise ValueError("No rows remaining after applying well filters.")

    df["embryo_id"] = df["embryo_id"].astype(str)
    df["frame_index"] = pd.to_numeric(df["frame_index"], errors="coerce").astype("Int64")
    df["predicted_stage_hpf"] = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    df = df[df["frame_index"].notna() & df["predicted_stage_hpf"].notna()].copy()
    if df.empty:
        raise ValueError("No valid frame/time rows in metadata.")

    if "start_age_hpf" in df.columns:
        df["start_age_hpf"] = pd.to_numeric(df["start_age_hpf"], errors="coerce")
    else:
        df["start_age_hpf"] = np.nan

    if "fraction_alive" in df.columns:
        df["fraction_alive"] = pd.to_numeric(df["fraction_alive"], errors="coerce")
    else:
        df["fraction_alive"] = np.nan

    if "dead_flag" in df.columns:
        df["dead_flag"] = pd.to_numeric(df["dead_flag"], errors="coerce")
    else:
        df["dead_flag"] = np.nan

    df["well"] = df["well_id"]
    df["is_e01"] = df["embryo_id"].astype(str).str.endswith("_e01").astype(int)

    per_emb = (
        df.groupby(["well", "embryo_id"], observed=True)
        .agg(
            n_rows=("frame_index", "size"),
            max_hpf=("predicted_stage_hpf", "max"),
            min_hpf=("predicted_stage_hpf", "min"),
            alive_mean=("fraction_alive", "mean"),
            start_age=("start_age_hpf", "median"),
            min_fi=("frame_index", lambda s: int(pd.to_numeric(s, errors="coerce").min())),
            max_fi=("frame_index", lambda s: int(pd.to_numeric(s, errors="coerce").max())),
        )
        .reset_index()
    )
    per_emb["alive_mean"] = per_emb["alive_mean"].fillna(1.0)
    per_emb["start_age"] = per_emb["start_age"].fillna(per_emb["min_hpf"])
    per_emb = per_emb.merge(
        df.groupby(["well", "embryo_id"], observed=True)["is_e01"].max().reset_index(),
        on=["well", "embryo_id"],
        how="left",
    )
    per_emb["is_e01"] = per_emb["is_e01"].fillna(0).astype(int)

    sort_columns = ["well", "alive_mean", "max_hpf", "n_rows", "max_fi"]
    ascending = [True, False, False, False, False]
    if policy.prefer_e01:
        sort_columns.insert(1, "is_e01")
        ascending.insert(1, False)
    per_emb = per_emb.sort_values(sort_columns, ascending=ascending)

    def endpoint_snips_ok(embryo_id: str, min_fi: int, max_fi: int) -> bool:
        if not policy.verify_endpoint_snips:
            return True
        if snip_root is None or experiment_date is None:
            return True
        p0 = build_snip_path(snip_root, experiment_date, embryo_id, int(min_fi))
        p1 = build_snip_path(snip_root, experiment_date, embryo_id, int(max_fi))
        return p0.exists() and p1.exists()

    best_rows = []
    for well, group in per_emb.groupby("well", sort=False):
        picked = None
        for row in group.itertuples(index=False):
            if endpoint_snips_ok(str(row.embryo_id), int(row.min_fi), int(row.max_fi)):
                picked = row
                break
        if picked is None:
            picked = group.iloc[0]
        best_rows.append(picked._asdict() if hasattr(picked, "_asdict") else dict(picked))

    best = pd.DataFrame(best_rows)
    robust_t_max = float(best["max_hpf"].quantile(0.95))
    if not math.isfinite(robust_t_max):
        robust_t_max = float(best["max_hpf"].max())

    default_t_min = float(best["start_age"].median())
    if not math.isfinite(default_t_min):
        default_t_min = float(best["min_hpf"].min())

    tracks: list[EmbryoTrack] = []
    start_age_by_well: dict[str, float] = {}
    last_alive_by_well: dict[str, tuple[float, int] | None] = {}

    best_by_well = dict(
        zip(best["well"].astype(str).tolist(), best["embryo_id"].astype(str).tolist(), strict=False)
    )
    for well, embryo_id in best_by_well.items():
        group = df[(df["well"] == well) & (df["embryo_id"] == embryo_id)].copy()
        if group.empty:
            continue
        track = build_embryo_track(group, well_col="well")
        tracks.append(track)

        start_age = float(best.loc[best["well"] == well, "start_age"].iloc[0])
        if math.isfinite(start_age):
            start_age_by_well[str(well)] = start_age

        last_alive_by_well[str(well)] = None
        use_dead_flag = "dead_flag" in group.columns and group["dead_flag"].notna().any()
        if use_dead_flag:
            alive = group[pd.to_numeric(group["dead_flag"], errors="coerce").fillna(0.0) <= 0.5].copy()
        else:
            alive = group[pd.to_numeric(group["fraction_alive"], errors="coerce") > float(policy.alive_threshold)].copy()

        if not alive.empty:
            alive = alive.sort_values(["predicted_stage_hpf", "frame_index"])
            for _, last in alive.iloc[::-1].iterrows():
                try:
                    lh = float(last["predicted_stage_hpf"])
                    lfi = int(last["frame_index"])
                    if not math.isfinite(lh):
                        continue
                    if policy.verify_endpoint_snips and snip_root is not None and experiment_date is not None:
                        if not build_snip_path(snip_root, experiment_date, str(embryo_id), int(lfi)).exists():
                            continue
                    last_alive_by_well[str(well)] = (lh, lfi)
                    break
                except Exception:
                    continue

    return PlateTrackSelection(
        tracks=tracks,
        start_age_by_well=start_age_by_well,
        last_alive_by_well=last_alive_by_well,
        default_t_min=default_t_min,
        robust_t_max=robust_t_max,
    )


def put_text_with_outline_colors(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    fg_bgr: tuple[int, int, int],
    outline_bgr: tuple[int, int, int],
    font: Optional[int] = None,
) -> None:
    """Draw outlined text onto an image using OpenCV."""
    import cv2

    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(2.0 * font_scale)))
    outline = thickness + 3
    cv2.putText(img, text, org, font, float(font_scale), outline_bgr, int(outline), cv2.LINE_AA)
    cv2.putText(img, text, org, font, float(font_scale), fg_bgr, int(thickness), cv2.LINE_AA)


def draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    corner_radius: int,
    thickness: int = -1,
) -> None:
    """Draw a rounded rectangle onto an image."""
    import cv2

    x0, y0 = int(pt1[0]), int(pt1[1])
    x1, y1 = int(pt2[0]), int(pt2[1])
    r = min(int(corner_radius), (x1 - x0) // 2, (y1 - y0) // 2)
    r = max(0, r)
    if r == 0:
        cv2.rectangle(img, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_AA)
        return

    if thickness == -1:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x0 + r, y0 + r), r, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x1 - r, y0 + r), r, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x0 + r, y1 - r), r, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x1 - r, y1 - r), r, color, thickness=-1, lineType=cv2.LINE_AA)
        return

    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)


def render_embryo_circle_tile(
    snip_bgr: Optional[np.ndarray],
    style: EmbryoTileStyle,
) -> np.ndarray:
    """Render one embryo snip into a circular tile suitable for plate layouts."""
    import cv2

    radius = int(style.radius)
    diameter = int(2 * radius)
    tile = np.zeros((diameter, diameter, 3), dtype=np.uint8)
    tile[:, :] = np.array(style.outside_bgr, dtype=np.uint8)

    mask = np.zeros((diameter, diameter), dtype=np.uint8)
    cv2.circle(mask, (radius, radius), int(radius - 1), 255, thickness=-1, lineType=cv2.LINE_AA)
    tile[mask > 0] = np.array(style.well_fill_bgr, dtype=np.uint8)

    if snip_bgr is not None:
        h, w = snip_bgr.shape[:2]
        if h > 0 and w > 0:
            scale = min(diameter / float(w), diameter / float(h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(snip_bgr, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
            x0 = int((diameter - new_w) // 2)
            y0 = int((diameter - new_h) // 2)
            snip_mask = mask[y0 : y0 + new_h, x0 : x0 + new_w]
            tile_region = tile[y0 : y0 + new_h, x0 : x0 + new_w]
            tile_region[snip_mask > 0] = resized[snip_mask > 0]

    if radius > 12:
        inner_rim_bgr = tuple(max(0, min(255, c + 15)) for c in style.well_rim_bgr)
        cv2.circle(
            tile,
            (radius, radius),
            max(1, int(radius - 3)),
            tuple(map(int, inner_rim_bgr)),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    cv2.circle(
        tile,
        (radius, radius),
        int(radius - 1),
        tuple(map(int, style.well_rim_bgr)),
        thickness=int(style.well_rim_thickness),
        lineType=cv2.LINE_AA,
    )

    if style.show_label and style.label:
        put_text_with_outline_colors(
            tile,
            style.label,
            (8, diameter - 10),
            font_scale=max(0.35, radius / 80.0),
            fg_bgr=style.label_fg_bgr,
            outline_bgr=style.label_outline_bgr,
        )

    return tile
