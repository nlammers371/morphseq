"""
Render a full 96-well plate timelapse as a single talk-ready MP4.

This composes per-well embryo "snip" JPEGs into a horizontal 8x12 plate movie.
Each well is rendered as a black circular well with the embryo snip fitted
inside (corners clipped to the well circle).

Primary inputs (read-only):
  - Embryo metadata CSV: morphseq_playground/metadata/embryo_metadata_files/{exp}_embryo_metadata.csv
  - Snip JPEGs:         morphseq_playground/training_data/bf_embryo_snips/{exp}/{embryo_id}_t{frame_index:04d}.jpg
  - YX1 plate XY coords: morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv

Defaults tuned for a first pass talk asset:
  - experiment_date=20260206 (starts ~10.75 HPF, 96 wells, good coverage)
  - output=1080p MP4, 20 fps, 300 frames
  - death_policy=freeze_at_death (freeze at last alive frame based on fraction_alive)

Usage:
  PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
  "$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/make_96well_plate_timelapse.py --experiment-date 20260206
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from analyze.viz.embryo_renderer import (
    EmbryoRenderStyle,
    EmbryoSelectionPolicy,
    EmbryoTrack,
    build_snip_path,
    draw_rounded_rect as _draw_rounded_rect_shared,
    put_text_with_outline_colors as _put_text_with_outline_colors_shared,
    read_snip_bgr,
    render_embryo_circle_tile,
    resolve_frame_schedule,
    select_plate_tracks_from_metadata,
)


_WELL_RE = re.compile(r"^[A-H]\d{2}$")


@dataclass
class FrameCache:
    fi0: Optional[int] = None
    img0: Optional[np.ndarray] = None
    fi1: Optional[int] = None
    img1: Optional[np.ndarray] = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a 96-well plate timelapse mosaic MP4 from snip JPEGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment-date", default="20260206", help="Experiment date / id (snip folder name).")
    p.add_argument(
        "--embryo-metadata-dir",
        default="morphseq_playground/metadata/embryo_metadata_files",
        help="Directory containing *_embryo_metadata.csv files.",
    )
    p.add_argument(
        "--embryo-metadata-csv",
        default=None,
        help="Optional explicit embryo metadata CSV path. Overrides --embryo-metadata-dir.",
    )
    p.add_argument(
        "--snip-root",
        default="morphseq_playground/training_data/bf_embryo_snips",
        help="Root directory containing per-experiment snip JPEG folders.",
    )
    p.add_argument(
        "--yx1-coords-csv",
        default="morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv",
        help="CSV with columns well,x_um,y_um for the YX1 plate layout.",
    )
    p.add_argument("--layout", choices=["yx1_um", "grid"], default="yx1_um", help="Well layout method.")
    p.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="Rendering theme for talk slides.",
    )
    p.add_argument("--width", type=int, default=1920, help="Output video width (px).")
    p.add_argument("--height", type=int, default=1080, help="Output video height (px).")
    p.add_argument("--margin-px", type=int, default=60, help="Canvas margin around the plate (px).")
    p.add_argument(
        "--header-px",
        type=int,
        default=140,
        help="(Legacy, ignored by new layout solver.) Reserve pixels at the top.",
    )
    p.add_argument("--well-radius-px", type=int, default=None, help="Override computed well radius (px).")
    p.add_argument(
        "--draw-plate-border",
        action="store_true",
        default=True,
        help="Draw a rectangular plate outline around the wells.",
    )
    p.add_argument(
        "--no-plate-border",
        dest="draw_plate_border",
        action="store_false",
        help="Disable plate border.",
    )
    p.add_argument(
        "--label-rows-cols",
        action="store_true",
        default=True,
        help="Label plate rows (A-H) and columns (1-12).",
    )
    p.add_argument(
        "--no-label-rows-cols",
        dest="label_rows_cols",
        action="store_false",
        help="Disable row/column labels.",
    )
    p.add_argument("--fps", type=int, default=20, help="Frames per second.")
    p.add_argument("--n-frames-out", type=int, default=300, help="Number of output frames.")
    p.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional output duration in seconds. If provided, overrides --n-frames-out as round(fps*duration_s).",
    )
    p.add_argument("--t-min", type=float, default=None, help="Min HPF for playback. Default: start_age_hpf.")
    p.add_argument("--t-max", type=float, default=None, help="Max HPF for playback. Default: robust max over embryos.")
    p.add_argument(
        "--missing",
        choices=["blank", "hold_last"],
        default="hold_last",
        help="Behavior when a well runs out of frames or a JPEG is missing.",
    )
    p.add_argument(
        "--death-policy",
        choices=["ignore", "freeze_at_death", "blank_after_death"],
        default="freeze_at_death",
        help="How to handle embryo 'death' based on fraction_alive. "
        "freeze_at_death clamps to the last frame with fraction_alive > --alive-threshold.",
    )
    p.add_argument(
        "--alive-threshold",
        type=float,
        default=0.5,
        help="fraction_alive threshold considered alive (used by --death-policy).",
    )
    p.add_argument("--show-hpf", action="store_true", default=True, help="Overlay HPF label.")
    p.add_argument("--hide-hpf", dest="show_hpf", action="store_false", help="Disable HPF label overlay.")
    p.add_argument(
        "--time-label-position",
        choices=["top_left", "top_center"],
        default="top_center",
        help="Placement of the time label in the reserved header band.",
    )
    p.add_argument(
        "--time-label-decimals",
        type=int,
        default=0,
        help="Number of decimals to show for HPF in the time label (0 disables sub-hour display).",
    )
    p.add_argument("--show-well-labels", action="store_true", default=False, help="Overlay well labels in tiles (debug).")
    p.add_argument(
        "--plate-corner-radius",
        type=int,
        default=None,
        help="Rounded rectangle corner radius for plate border (default: auto = radius//2).",
    )
    p.add_argument(
        "--no-plate-shadow",
        action="store_true",
        default=False,
        help="Disable the subtle drop shadow behind the plate.",
    )
    p.add_argument(
        "--well-rim-thickness",
        type=int,
        default=1,
        help="Well rim stroke width in pixels.",
    )
    p.add_argument(
        "--wells",
        default=None,
        help="Comma-separated wells to render (subset for quick iteration). Example: A01,A02,B01",
    )
    p.add_argument(
        "--out-mp4",
        default=None,
        help="Output MP4 path. Default: results/.../figures/{experiment}_plate_timelapse_1080p.mp4",
    )
    p.add_argument(
        "--debug-layout",
        action="store_true",
        default=False,
        help="Draw debug overlay: usable zone (cyan), plate bbox (yellow), label bboxes (green), well footprints (red).",
    )
    p.add_argument(
        "--single-frame-png",
        default=None,
        help="If set, render a single frame at t_min to this PNG path (for layout verification) and exit.",
    )
    return p.parse_args()


def _ensure_even(x: int) -> int:
    x = int(x)
    return x if x % 2 == 0 else x + 1


def _all_wells_96() -> list[str]:
    return [f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)]


def _parse_wells_arg(wells: Optional[str]) -> Optional[list[str]]:
    if wells is None:
        return None
    out: list[str] = []
    for tok in str(wells).split(","):
        w = tok.strip().upper()
        if not w:
            continue
        if not _WELL_RE.match(w):
            raise ValueError(f"Invalid well in --wells: {w!r} (expected like A01)")
        out.append(w)
    if not out:
        return None
    # preserve canonical 96-well ordering
    order = {w: i for i, w in enumerate(_all_wells_96())}
    out.sort(key=lambda w: order.get(w, 10**9))
    return out


def _infer_well_from_embryo_id(embryo_id: str) -> Optional[str]:
    parts = str(embryo_id).split("_")
    for p in parts:
        p = p.strip().upper()
        if _WELL_RE.match(p):
            return p
    return None


def _snip_path(snip_root: Path, experiment_date: str, embryo_id: str, frame_index: int) -> Path:
    return build_snip_path(snip_root, experiment_date, embryo_id, frame_index)


def _load_metadata_tracks(
    embryo_metadata_csv: Path,
    *,
    wells_subset: Optional[set[str]],
    alive_threshold: float = 0.5,
    snip_root: Optional[Path] = None,
    experiment_date: Optional[str] = None,
    verify_snips: bool = True,
) -> tuple[list[EmbryoTrack], dict[str, float], dict[str, tuple[float, int] | None], float, float]:
    selection = select_plate_tracks_from_metadata(
        embryo_metadata_csv,
        wells_subset=wells_subset,
        snip_root=snip_root,
        experiment_date=experiment_date,
        policy=EmbryoSelectionPolicy(
            prefer_e01=True,
            verify_endpoint_snips=bool(verify_snips),
            alive_threshold=float(alive_threshold),
        ),
    )
    return (
        selection.tracks,
        selection.start_age_by_well,
        selection.last_alive_by_well,
        selection.default_t_min,
        selection.robust_t_max,
    )


def _load_yx1_coords(yx1_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(yx1_csv)
    if not {"well", "x_um", "y_um"}.issubset(set(df.columns)):
        raise ValueError(f"{yx1_csv} must have columns: well,x_um,y_um")
    df["well"] = df["well"].astype(str).str.strip().str.upper()
    df["x_um"] = pd.to_numeric(df["x_um"], errors="coerce")
    df["y_um"] = pd.to_numeric(df["y_um"], errors="coerce")
    df = df[df["well"].notna() & df["x_um"].notna() & df["y_um"].notna()].copy()
    return df


def _compute_centers_px(
    *,
    layout: str,
    center_x_range: tuple[int, int],
    center_y_range: tuple[int, int],
    yx1_coords: Optional[pd.DataFrame] = None,
) -> dict[str, tuple[int, int]]:
    """Place 96 well centers within the given pixel ranges.

    center_x_range: (min_cx, max_cx) — leftmost and rightmost well center x.
    center_y_range: (min_cy, max_cy) — topmost and bottommost well center y.
    """
    wells = _all_wells_96()
    centers: dict[str, tuple[int, int]] = {}
    cx_min, cx_max = int(center_x_range[0]), int(center_x_range[1])
    cy_min, cy_max = int(center_y_range[0]), int(center_y_range[1])

    if layout == "grid":
        pitch_x = (cx_max - cx_min) / 11.0 if cx_max > cx_min else 1.0
        pitch_y = (cy_max - cy_min) / 7.0 if cy_max > cy_min else 1.0
        for ri, row in enumerate("ABCDEFGH"):
            for ci in range(1, 13):
                well = f"{row}{ci:02d}"
                x = int(round(cx_min + (ci - 1) * pitch_x))
                y = int(round(cy_min + ri * pitch_y))
                centers[well] = (x, y)
        return centers

    if layout != "yx1_um":
        raise ValueError(f"Unknown layout: {layout}")
    if yx1_coords is None:
        raise ValueError("yx1_coords required for layout=yx1_um")

    coords = yx1_coords.set_index("well", drop=True)
    missing = [well for well in wells if well not in coords.index]
    if missing:
        raise ValueError(f"YX1 coords missing wells: {missing[:10]} (and {max(0, len(missing) - 10)} more)")

    # Flip x so columns increase left->right (A01 left, A12 right).
    x_um = (-coords.loc[wells, "x_um"].to_numpy(dtype=float)).astype(float)
    y_um = (coords.loc[wells, "y_um"].to_numpy(dtype=float)).astype(float)

    x0_um, x1_um = float(np.min(x_um)), float(np.max(x_um))
    y0_um, y1_um = float(np.min(y_um)), float(np.max(y_um))
    dx_um = max(1e-9, x1_um - x0_um)
    dy_um = max(1e-9, y1_um - y0_um)
    avail_x = max(1, cx_max - cx_min)
    avail_y = max(1, cy_max - cy_min)
    scale = min(avail_x / dx_um, avail_y / dy_um)

    # Center the layout within the available range
    used_x = dx_um * scale
    used_y = dy_um * scale
    offset_x = cx_min + (avail_x - used_x) / 2.0
    offset_y = cy_min + (avail_y - used_y) / 2.0

    for well, xu, yu in zip(wells, x_um.tolist(), y_um.tolist(), strict=False):
        x = int(round(offset_x + (float(xu) - x0_um) * scale))
        y = int(round(offset_y + (float(yu) - y0_um) * scale))
        centers[well] = (x, y)
    return centers


def _estimate_radius_px(centers: dict[str, tuple[int, int]], *, override: Optional[int]) -> int:
    if override is not None:
        return max(6, int(override))
    # Estimate pitch from adjacent wells
    dxs = []
    dys = []
    for row in "ABCDEFGH":
        row_wells = [f"{row}{c:02d}" for c in range(1, 13)]
        for a, b in zip(row_wells[:-1], row_wells[1:], strict=False):
            xa, ya = centers[a]
            xb, yb = centers[b]
            dxs.append(abs(xb - xa))
    for col in range(1, 13):
        col_wells = [f"{r}{col:02d}" for r in "ABCDEFGH"]
        for a, b in zip(col_wells[:-1], col_wells[1:], strict=False):
            xa, ya = centers[a]
            xb, yb = centers[b]
            dys.append(abs(yb - ya))
    pitch = min(float(np.median(dxs)) if dxs else 0.0, float(np.median(dys)) if dys else 0.0)
    if not math.isfinite(pitch) or pitch <= 0:
        pitch = 120.0
    return max(10, int(round(0.42 * pitch)))


def _schedule_for_track(
    track: EmbryoTrack,
    t_out: np.ndarray,
    *,
    clamp_time: Optional[float] = None,
    clamp_fi: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return resolve_frame_schedule(
        track,
        t_out,
        clamp_time=clamp_time,
        clamp_frame_index=clamp_fi,
    )


def _put_text_with_outline(img: np.ndarray, text: str, org: tuple[int, int], *, font_scale: float = 1.0) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(2.0 * font_scale)))
    outline = thickness + 3
    cv2.putText(img, text, org, font, float(font_scale), (0, 0, 0), int(outline), cv2.LINE_AA)
    cv2.putText(img, text, org, font, float(font_scale), (255, 255, 255), int(thickness), cv2.LINE_AA)


def _put_text_with_outline_colors(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    fg_bgr: tuple[int, int, int],
    outline_bgr: tuple[int, int, int],
    font: Optional[int] = None,
) -> None:
    _put_text_with_outline_colors_shared(
        img,
        text,
        org,
        font_scale=font_scale,
        fg_bgr=fg_bgr,
        outline_bgr=outline_bgr,
        font=font,
    )


def _draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    corner_radius: int,
    thickness: int = -1,
) -> None:
    _draw_rounded_rect_shared(img, pt1, pt2, color, corner_radius, thickness=thickness)


def _fit_into_circle_tile(
    snip_bgr: Optional[np.ndarray],
    *,
    radius: int,
    show_label: bool,
    label: str,
    outside_bgr: tuple[int, int, int],
    well_fill_bgr: tuple[int, int, int],
    well_rim_bgr: tuple[int, int, int],
    label_fg_bgr: tuple[int, int, int],
    label_outline_bgr: tuple[int, int, int],
    well_rim_thickness: int = 1,
) -> np.ndarray:
    return render_embryo_circle_tile(
        snip_bgr,
        EmbryoRenderStyle(
            radius=radius,
            show_label=show_label,
            label=label,
            outside_bgr=outside_bgr,
            well_fill_bgr=well_fill_bgr,
            well_rim_bgr=well_rim_bgr,
            label_fg_bgr=label_fg_bgr,
            label_outline_bgr=label_outline_bgr,
            well_rim_thickness=well_rim_thickness,
        ),
    )


def _plate_bbox(centers: dict[str, tuple[int, int]], wells: list[str], radius: int, pad: int) -> tuple[int, int, int, int]:
    xs = [centers[w][0] for w in wells]
    ys = [centers[w][1] for w in wells]
    x0 = int(min(xs) - radius - pad)
    x1 = int(max(xs) + radius + pad)
    y0 = int(min(ys) - radius - pad)
    y1 = int(max(ys) + radius + pad)
    return x0, y0, x1, y1


def _read_snp(snip_path: Path) -> Optional[np.ndarray]:
    return read_snip_bgr(snip_path)


def main() -> None:
    args = _parse_args()

    import cv2

    width = _ensure_even(int(args.width))
    height = _ensure_even(int(args.height))
    fps = int(args.fps)
    n_frames_out = int(args.n_frames_out)
    if args.duration_s is not None:
        n_frames_out = max(2, int(round(float(args.duration_s) * float(fps))))
    margin_px = int(args.margin_px)
    header_px = int(args.header_px)

    wells_list = _parse_wells_arg(args.wells)
    wells_subset = set(wells_list) if wells_list is not None else None

    snip_root = Path(args.snip_root)
    exp = str(args.experiment_date)

    theme = str(args.theme)
    if theme == "dark":
        canvas_bg_bgr = (8, 8, 10)
        plate_fill_bgr = (18, 18, 20)
        plate_shadow_bgr = (20, 20, 22)
        tile_outside_bgr = (18, 18, 20)
        plate_border_bgr = (180, 175, 170)
        label_fg_bgr = (220, 215, 210)
        label_outline_bgr = (15, 15, 15)
        well_fill_bgr = (0, 0, 0)
        well_rim_bgr = (90, 85, 80)
        time_fg_bgr = (225, 228, 230)
        time_unit_fg_bgr = (180, 175, 170)
    else:
        canvas_bg_bgr = (245, 245, 248)
        plate_fill_bgr = (235, 235, 238)
        plate_shadow_bgr = (210, 210, 215)
        tile_outside_bgr = (235, 235, 238)
        plate_border_bgr = (80, 80, 85)
        label_fg_bgr = (40, 40, 45)
        label_outline_bgr = (240, 240, 245)
        well_fill_bgr = (0, 0, 0)
        well_rim_bgr = (170, 170, 175)
        time_fg_bgr = (30, 30, 35)
        time_unit_fg_bgr = (100, 100, 105)

    if args.embryo_metadata_csv:
        embryo_csv = Path(args.embryo_metadata_csv)
    else:
        embryo_csv = Path(args.embryo_metadata_dir) / f"{exp}_embryo_metadata.csv"
    if not embryo_csv.exists():
        raise FileNotFoundError(f"Missing embryo metadata CSV: {embryo_csv}")

    tracks, start_age_by_well, last_alive_by_well, default_t_min, robust_t_max = _load_metadata_tracks(
        embryo_csv,
        wells_subset=wells_subset,
        alive_threshold=float(args.alive_threshold),
        snip_root=snip_root,
        experiment_date=exp,
        verify_snips=True,
    )
    if not tracks:
        raise RuntimeError("No embryo tracks selected.")

    t_min = float(args.t_min) if args.t_min is not None else float(default_t_min)
    t_max = float(args.t_max) if args.t_max is not None else float(robust_t_max)
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError(f"Invalid (t_min,t_max)=({t_min},{t_max})")

    t_out = np.linspace(t_min, t_max, num=int(n_frames_out), dtype=float)

    # ── Declarative measure-then-solve layout ──
    # Font sizes are determined from canvas height (input), breaking the
    # feedback loop.  Everything is measured, then the usable center range
    # is solved, wells are placed, and finally validated.

    yx1_coords = None
    if args.layout == "yx1_um":
        yx1_coords = _load_yx1_coords(Path(args.yx1_coords_csv))

    label_font = cv2.FONT_HERSHEY_DUPLEX

    # Step 1: Font sizes from canvas dimensions — deterministic, no feedback loop
    axis_font_scale = max(1.1, height / 850.0)
    axis_font_scale = min(axis_font_scale, 4.0)
    time_digit_scale = 1.5 * (axis_font_scale / 1.1)
    time_unit_scale = 0.85 * (axis_font_scale / 1.1)

    # Step 2: Measure all text extents with the actual font backend
    label_thickness = max(1, int(round(2.0 * axis_font_scale)))
    gap = int(round(8 * axis_font_scale))  # gap between labels and plate edge

    # Row labels (A-H) — measure widest
    row_label_widths = []
    row_label_height = 0
    for row in "ABCDEFGH":
        (tw, th), _ = cv2.getTextSize(row, label_font, float(axis_font_scale), label_thickness)
        row_label_widths.append(tw)
        row_label_height = max(row_label_height, th)
    max_row_label_w = max(row_label_widths) if row_label_widths else 0
    row_label_gutter = (max_row_label_w + gap) if args.label_rows_cols else 0

    # Column labels (1-12) — measure tallest
    col_label_widths = {}
    col_label_height = 0
    for col in range(1, 13):
        (tw, th), _ = cv2.getTextSize(str(col), label_font, float(axis_font_scale), label_thickness)
        col_label_widths[col] = tw
        col_label_height = max(col_label_height, th)
    col_label_gutter = (col_label_height + gap) if args.label_rows_cols else 0

    # Time label zone
    time_digit_thickness = max(1, int(round(2.0 * time_digit_scale)))
    time_unit_thickness = max(1, int(round(2.0 * time_unit_scale)))
    if args.show_hpf:
        (tdw, tdh), _ = cv2.getTextSize("48", cv2.FONT_HERSHEY_DUPLEX, float(time_digit_scale), time_digit_thickness)
        (tuw, tuh), _ = cv2.getTextSize(" HPF", cv2.FONT_HERSHEY_DUPLEX, float(time_unit_scale), time_unit_thickness)
        time_text_h = max(tdh, tuh)
        time_zone_h = time_text_h + gap
    else:
        tdw = tdh = tuw = tuh = time_text_h = 0
        time_zone_h = 0

    # Step 3: Preliminary layout to estimate radius, then compute exact reservations
    # Use generous initial bounds to get a radius estimate
    prelim_cx = (row_label_gutter + margin_px, width - margin_px)
    prelim_cy = (time_zone_h + col_label_gutter + margin_px, height - margin_px)
    centers_prelim = _compute_centers_px(
        layout=str(args.layout),
        center_x_range=prelim_cx,
        center_y_range=prelim_cy,
        yx1_coords=yx1_coords,
    )
    radius = _estimate_radius_px(centers_prelim, override=args.well_radius_px)
    bbox_pad = int(round(0.75 * radius))
    border_thickness = max(2, int(round(radius / 22.0)))
    eps = 4  # safety margin in pixels
    well_footprint = radius + bbox_pad + border_thickness // 2 + eps

    # Step 4: Compute required outer reservations (from canvas edge to well center)
    top_reserve = time_zone_h + col_label_gutter + well_footprint
    bottom_reserve = well_footprint
    left_reserve = row_label_gutter + well_footprint
    right_reserve = well_footprint

    # Usable rect for well centers
    cx_min = int(round(left_reserve))
    cx_max = int(round(width - right_reserve))
    cy_min = int(round(top_reserve))
    cy_max = int(round(height - bottom_reserve))

    if cx_max <= cx_min or cy_max <= cy_min:
        raise RuntimeError(
            f"Canvas {width}x{height} too small for layout. "
            f"Need cx=[{cx_min},{cx_max}], cy=[{cy_min},{cy_max}]"
        )

    # Step 5: Place wells and recompute radius from final centers
    centers = _compute_centers_px(
        layout=str(args.layout),
        center_x_range=(cx_min, cx_max),
        center_y_range=(cy_min, cy_max),
        yx1_coords=yx1_coords,
    )
    radius = _estimate_radius_px(centers, override=args.well_radius_px)
    bbox_pad = int(round(0.75 * radius))

    # Snap all well centers to integer pixels
    for w in list(centers.keys()):
        centers[w] = (int(round(centers[w][0])), int(round(centers[w][1])))

    # Track lookup per well
    track_by_well: dict[str, EmbryoTrack] = {t.well: t for t in tracks}
    cache_by_well: dict[str, FrameCache] = {w: FrameCache() for w in track_by_well.keys()}

    # Precompute schedules for selected wells
    fi0_by_well: dict[str, np.ndarray] = {}
    fi1_by_well: dict[str, np.ndarray] = {}
    alpha_by_well: dict[str, np.ndarray] = {}
    for w, tr in track_by_well.items():
        clamp_time = None
        clamp_fi = None
        if str(args.death_policy) in {"freeze_at_death", "blank_after_death"}:
            last = last_alive_by_well.get(w)
            if last is not None:
                clamp_time, clamp_fi = float(last[0]), int(last[1])
        fi0, fi1, alpha = _schedule_for_track(tr, t_out, clamp_time=clamp_time, clamp_fi=clamp_fi)
        fi0_by_well[w] = fi0
        fi1_by_well[w] = fi1
        alpha_by_well[w] = alpha

    # Output path (used for MP4 mode; single-frame PNG uses --single-frame-png)
    if args.out_mp4:
        out_mp4 = Path(args.out_mp4)
    else:
        out_mp4 = Path(__file__).resolve().parent / "figures" / "plate_timelapse" / f"{exp}_plate_timelapse_1080p.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    wells_to_draw = wells_list if wells_list is not None else _all_wells_96()
    all_96 = _all_wells_96()  # always use full plate for layout/centering
    ever_loaded = {w: False for w in wells_to_draw}
    well_rim_thickness = max(1, int(args.well_rim_thickness))

    plate_corner_radius = int(args.plate_corner_radius) if args.plate_corner_radius is not None else max(4, radius // 2)

    # Step 6: Compute label anchors from plate bbox
    plate_x0, plate_y0, plate_x1, plate_y1 = _plate_bbox(centers, all_96, radius=radius, pad=bbox_pad)

    # Row labels: right-aligned just left of plate bbox
    row_y = {row: centers[f"{row}01"][1] for row in "ABCDEFGH"}
    col_x = {col: centers[f"A{col:02d}"][0] for col in range(1, 13)}
    row_label_right_edge = int(plate_x0 - gap // 2)
    # Column labels: baseline just above plate bbox
    col_label_baseline_y = int(plate_y0 - gap // 2)

    # Pre-compute snapped label positions for validation and drawing
    # Row label bboxes: (x0, y0, x1, y1) in canvas coords
    row_label_bboxes: dict[str, tuple[int, int, int, int]] = {}
    for row in "ABCDEFGH":
        (tw, th), _ = cv2.getTextSize(row, label_font, float(axis_font_scale), label_thickness)
        lx = int(row_label_right_edge - tw)
        ly = int(row_y[row] + th // 2)
        # cv2 text origin is bottom-left; bbox is (left, top, right, bottom)
        row_label_bboxes[row] = (lx, ly - th, lx + tw, ly)

    # Column label bboxes
    col_label_bboxes: dict[int, tuple[int, int, int, int]] = {}
    for col in range(1, 13):
        col_str = str(col)
        (tw, th), _ = cv2.getTextSize(col_str, label_font, float(axis_font_scale), label_thickness)
        lx = int(col_x[col] - tw // 2)
        ly = int(col_label_baseline_y)
        col_label_bboxes[col] = (lx, ly - th, lx + tw, ly)

    # Time label bbox (use widest possible text "48 HPF")
    time_label_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    if args.show_hpf:
        total_time_w = tdw + tuw
        time_zone_top = plate_y0 - col_label_gutter - time_zone_h
        time_label_y = int(time_zone_top + time_zone_h * 0.7)
        time_label_y = max(int(time_text_h + eps), time_label_y)
        # Snap to int
        time_label_y = int(round(time_label_y))

        if str(args.time_label_position) == "top_left":
            digit_x_base = 40
        else:
            plate_cx = (plate_x0 + plate_x1) // 2
            digit_x_base = int(round(plate_cx - total_time_w / 2))
            digit_x_base = max(eps, digit_x_base)
        digit_x_base = int(round(digit_x_base))
        time_label_bbox = (digit_x_base, time_label_y - time_text_h, digit_x_base + total_time_w, time_label_y)

    # Step 7: Validation assertions
    outer_r = radius + bbox_pad
    # Every well circle inside canvas
    for w in all_96:
        cx_w, cy_w = centers[w]
        assert cy_w + outer_r + eps <= height, (
            f"Well {w} clips bottom: center_y={cy_w} + outer_r={outer_r} + eps={eps} = {cy_w + outer_r + eps} > height={height}"
        )
        assert cy_w - outer_r - eps >= 0, (
            f"Well {w} clips top: center_y={cy_w} - outer_r={outer_r} - eps={eps} = {cy_w - outer_r - eps} < 0"
        )
        assert cx_w + outer_r + eps <= width, (
            f"Well {w} clips right: center_x={cx_w} + outer_r={outer_r} + eps={eps} = {cx_w + outer_r + eps} > width={width}"
        )
        assert cx_w - outer_r - eps >= 0, (
            f"Well {w} clips left: center_x={cx_w} - outer_r={outer_r} - eps={eps} = {cx_w - outer_r - eps} < 0"
        )
    # Plate bbox inside canvas
    assert plate_y1 <= height - eps, f"Plate bottom {plate_y1} exceeds canvas {height}"
    assert plate_x1 <= width - eps, f"Plate right {plate_x1} exceeds canvas {width}"
    assert plate_y0 >= eps, f"Plate top {plate_y0} < eps={eps}"
    assert plate_x0 >= eps, f"Plate left {plate_x0} < eps={eps}"

    # Label bboxes inside canvas
    if args.label_rows_cols:
        for row, bb in row_label_bboxes.items():
            assert bb[0] >= 0, f"Row label {row} clips left: x0={bb[0]}"
            assert bb[2] <= width, f"Row label {row} clips right: x1={bb[2]}"
            assert bb[1] >= 0, f"Row label {row} clips top: y0={bb[1]}"
            assert bb[3] <= height, f"Row label {row} clips bottom: y1={bb[3]}"
        for col, bb in col_label_bboxes.items():
            assert bb[0] >= 0, f"Col label {col} clips left: x0={bb[0]}"
            assert bb[2] <= width, f"Col label {col} clips right: x1={bb[2]}"
            assert bb[1] >= 0, f"Col label {col} clips top: y0={bb[1]}"
            assert bb[3] <= height, f"Col label {col} clips bottom: y1={bb[3]}"
    if args.show_hpf:
        tb = time_label_bbox
        assert tb[0] >= 0, f"Time label clips left: x0={tb[0]}"
        assert tb[2] <= width, f"Time label clips right: x1={tb[2]}"
        assert tb[1] >= 0, f"Time label clips top: y0={tb[1]}"
        assert tb[3] <= height, f"Time label clips bottom: y1={tb[3]}"
        # No overlap between time label and any column label
        if args.label_rows_cols:
            for col, cb in col_label_bboxes.items():
                overlap = (
                    tb[0] < cb[2] and tb[2] > cb[0] and  # x overlap
                    tb[1] < cb[3] and tb[3] > cb[1]       # y overlap
                )
                assert not overlap, (
                    f"Time label bbox {tb} overlaps column {col} label bbox {cb}"
                )

    print(
        f"Layout: {width}x{height}, radius={radius}, bbox_pad={bbox_pad}, "
        f"plate=[{plate_x0},{plate_y0}]-[{plate_x1},{plate_y1}], "
        f"font_scale={axis_font_scale:.2f}, time_digit_scale={time_digit_scale:.2f}"
    )

    def _render_frame(canvas: np.ndarray, k: int, t: float) -> None:
        """Render one complete frame onto *canvas* (mutated in-place)."""
        canvas[:, :] = np.array(canvas_bg_bgr, dtype=np.uint8)

        if args.draw_plate_border:
            bx0, by0, bx1, by1 = plate_x0, plate_y0, plate_x1, plate_y1
            # Drop shadow (offset behind plate)
            if not args.no_plate_shadow:
                _draw_rounded_rect(canvas, (bx0 + 4, by0 + 4), (bx1 + 4, by1 + 4), plate_shadow_bgr, plate_corner_radius, thickness=-1)
            # Plate interior fill
            _draw_rounded_rect(canvas, (bx0, by0), (bx1, by1), plate_fill_bgr, plate_corner_radius, thickness=-1)
            # Plate border outline
            _draw_rounded_rect(canvas, (bx0, by0), (bx1, by1), plate_border_bgr, plate_corner_radius, thickness=int(border_thickness))

        if args.label_rows_cols:
            for row in "ABCDEFGH":
                bb = row_label_bboxes[row]
                lx, ly = bb[0], bb[3]  # bottom-left origin for cv2
                _put_text_with_outline_colors(
                    canvas, row, (lx, ly),
                    font_scale=axis_font_scale,
                    fg_bgr=label_fg_bgr,
                    outline_bgr=label_outline_bgr,
                    font=label_font,
                )
            for col in range(1, 13):
                bb = col_label_bboxes[col]
                lx, ly = bb[0], bb[3]
                _put_text_with_outline_colors(
                    canvas, str(col), (lx, ly),
                    font_scale=axis_font_scale,
                    fg_bgr=label_fg_bgr,
                    outline_bgr=label_outline_bgr,
                    font=label_font,
                )

        for well in wells_to_draw:
            cx_w, cy_w = centers[well]
            tr = track_by_well.get(well)
            if tr is None:
                tile = _fit_into_circle_tile(
                    None,
                    radius=radius,
                    show_label=bool(args.show_well_labels),
                    label=str(well),
                    outside_bgr=tile_outside_bgr,
                    well_fill_bgr=well_fill_bgr,
                    well_rim_bgr=well_rim_bgr,
                    label_fg_bgr=label_fg_bgr,
                    label_outline_bgr=label_outline_bgr,
                    well_rim_thickness=well_rim_thickness,
                )
            else:
                fi0 = int(fi0_by_well[well][k])
                fi1 = int(fi1_by_well[well][k])
                a = float(alpha_by_well[well][k])

                fc = cache_by_well[well]

                def load_cached(fi: int, _fc=fc, _tr=tr, _well=well) -> Optional[np.ndarray]:
                    if _fc.fi0 == fi and _fc.img0 is not None:
                        return _fc.img0
                    if _fc.fi1 == fi and _fc.img1 is not None:
                        return _fc.img1
                    p = _snip_path(snip_root, exp, _tr.embryo_id, int(fi))
                    img = _read_snp(p)
                    if img is not None:
                        ever_loaded[_well] = True
                    if _fc.fi0 is None or _fc.fi0 == fi or (_fc.fi1 is not None and _fc.fi0 == _fc.fi1):
                        _fc.fi0, _fc.img0 = int(fi), img
                    elif _fc.fi1 is None or _fc.fi1 == fi:
                        _fc.fi1, _fc.img1 = int(fi), img
                    else:
                        if abs(int(fi) - int(_fc.fi0)) >= abs(int(fi) - int(_fc.fi1)):
                            _fc.fi0, _fc.img0 = int(fi), img
                        else:
                            _fc.fi1, _fc.img1 = int(fi), img
                    return img

                img0 = load_cached(fi0)
                img1 = load_cached(fi1) if fi1 != fi0 else img0

                snip = None
                if img0 is None and img1 is None:
                    snip = None
                elif img0 is None:
                    snip = img1
                elif img1 is None:
                    snip = img0
                else:
                    if fi0 == fi1 or a <= 0.0:
                        snip = img0
                    elif a >= 1.0:
                        snip = img1
                    else:
                        snip = cv2.addWeighted(img0, float(1.0 - a), img1, float(a), 0.0)

                if args.missing == "blank" and tr.times_hpf.size > 0:
                    if t > float(tr.times_hpf[-1]) + 1e-6:
                        snip = None

                if str(args.death_policy) == "blank_after_death":
                    last = last_alive_by_well.get(well)
                    if last is None:
                        snip = None
                    else:
                        if t > float(last[0]) + 1e-6:
                            snip = None

                tile = _fit_into_circle_tile(
                    snip,
                    radius=radius,
                    show_label=bool(args.show_well_labels),
                    label=str(well),
                    outside_bgr=tile_outside_bgr,
                    well_fill_bgr=well_fill_bgr,
                    well_rim_bgr=well_rim_bgr,
                    label_fg_bgr=label_fg_bgr,
                    label_outline_bgr=label_outline_bgr,
                    well_rim_thickness=well_rim_thickness,
                )

            d = tile.shape[0]
            x0 = int(cx_w - d // 2)
            y0 = int(cy_w - d // 2)
            x1 = x0 + d
            y1 = y0 + d

            # Clip to canvas
            cl_x0 = max(0, x0)
            cl_y0 = max(0, y0)
            cl_x1 = min(int(width), x1)
            cl_y1 = min(int(height), y1)
            if cl_x1 <= cl_x0 or cl_y1 <= cl_y0:
                continue
            tx0 = cl_x0 - x0
            ty0 = cl_y0 - y0
            tx1 = tx0 + (cl_x1 - cl_x0)
            ty1 = ty0 + (cl_y1 - cl_y0)
            canvas[cl_y0:cl_y1, cl_x0:cl_x1] = tile[ty0:ty1, tx0:tx1]

        if args.show_hpf:
            dec = max(0, int(args.time_label_decimals))
            hpf_txt = f"{t:.0f}" if dec == 0 else f"{t:.{dec}f}"
            time_font = cv2.FONT_HERSHEY_DUPLEX

            # Measure this frame's digit text (width varies with digit count)
            (dw_cur, _dh_cur), _ = cv2.getTextSize(hpf_txt, time_font, float(time_digit_scale), time_digit_thickness)
            (uw_cur, _), _ = cv2.getTextSize(" HPF", time_font, float(time_unit_scale), time_unit_thickness)
            total_w_cur = dw_cur + uw_cur

            # Use pre-computed y; recompute x for this frame's text width
            tl_y = time_label_bbox[3]  # baseline y
            if str(args.time_label_position) == "top_left":
                dx = 40
            else:
                plate_cx = (plate_x0 + plate_x1) // 2
                dx = int(round(plate_cx - total_w_cur / 2))
                dx = max(eps, dx)
            dx = int(round(dx))

            _put_text_with_outline_colors(
                canvas, hpf_txt, (dx, tl_y),
                font_scale=time_digit_scale,
                fg_bgr=time_fg_bgr,
                outline_bgr=label_outline_bgr,
                font=time_font,
            )
            _put_text_with_outline_colors(
                canvas, " HPF", (dx + dw_cur, tl_y),
                font_scale=time_unit_scale,
                fg_bgr=time_unit_fg_bgr,
                outline_bgr=label_outline_bgr,
                font=time_font,
            )

        # Debug layout overlay
        if args.debug_layout:
            # Usable zone rectangle (cyan)
            cv2.rectangle(canvas, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 1, cv2.LINE_AA)
            # Plate bbox (yellow)
            cv2.rectangle(canvas, (plate_x0, plate_y0), (plate_x1, plate_y1), (0, 255, 255), 1, cv2.LINE_AA)
            # Label bounding boxes (green)
            if args.label_rows_cols:
                for bb in row_label_bboxes.values():
                    cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1, cv2.LINE_AA)
                for bb in col_label_bboxes.values():
                    cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1, cv2.LINE_AA)
            if args.show_hpf:
                tb = time_label_bbox
                cv2.rectangle(canvas, (tb[0], tb[1]), (tb[2], tb[3]), (0, 255, 0), 1, cv2.LINE_AA)
            # Well outer footprints (red, thin)
            for w in all_96:
                wcx, wcy = centers[w]
                cv2.circle(canvas, (wcx, wcy), outer_r, (0, 0, 255), 1, cv2.LINE_AA)

    # Single-frame PNG mode: render one frame and exit
    if args.single_frame_png:
        canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        t_single = float(t_out[0])
        _render_frame(canvas, 0, t_single)
        png_path = Path(args.single_frame_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png_path), canvas)
        print(f"Wrote single-frame PNG: {png_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

    try:
        for k, t in enumerate(t_out.tolist()):
            canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
            _render_frame(canvas, k, t)
            writer.write(canvas)

            if (k + 1) % 25 == 0 or (k + 1) == n_frames_out:
                print(f"[{k+1:>4d}/{n_frames_out}] t={t:.2f} HPF  radius={radius}px  out={out_mp4.name}")
    finally:
        writer.release()

    print(f"Wrote: {out_mp4}")
    never = [w for w, ok in ever_loaded.items() if not ok]
    if never:
        never_s = ",".join(never[:24])
        extra = f" (+{len(never)-24} more)" if len(never) > 24 else ""
        print(f"WARNING: {len(never)} wells never loaded any snip JPEG: {never_s}{extra}")


if __name__ == "__main__":
    main()
