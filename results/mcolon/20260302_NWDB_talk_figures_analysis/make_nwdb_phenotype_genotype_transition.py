"""
NWDB talk: curvature-only transition videos between genotype and phenotype overlays.

This script renders homozygous-background curvature animations with fixed phenotype
overlay embryos. It is intended to bridge the genotype-only and phenotype-only
animations in the NWDB talk figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _nwdb_transition_plot_style import (
    PLOT_DPI,
    apply_transition_rcparams,
    style_transition_figure,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _default_plot_px() -> tuple[int, int]:
    from _nwdb_transition_plot_style import PLOT_FIGSIZE_IN

    return int(round(PLOT_FIGSIZE_IN[0] * PLOT_DPI)), int(round(PLOT_FIGSIZE_IN[1] * PLOT_DPI))


def _fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    canvas = fig.canvas
    canvas.draw()

    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[..., :3])

    w, h = canvas.get_width_height()
    if hasattr(canvas, "tostring_rgb"):
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((int(h), int(w), 3))
        return np.ascontiguousarray(rgb)

    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((int(h), int(w), 4))
    return np.ascontiguousarray(argb[..., 1:])


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _coalesce_columns(df: pd.DataFrame, dst: str, src: str) -> None:
    if dst not in df.columns or src not in df.columns:
        return
    if df[dst].isna().all() and (~df[src].isna()).any():
        df[dst] = df[src]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class OverlaySpec:
    key: str
    label: str
    embryo_id: str
    color_hex: str


@dataclass(frozen=True)
class VideoSpec:
    order: int
    output_stem: str
    static_overlays: tuple[OverlaySpec, ...]
    animated_overlay: OverlaySpec


PHENOTYPE_COLORS = {
    "High_to_Low": "#E76FA2",
    "Low_to_High": "#2FB7B0",
    "Not Penetrant": "#3A3A3A",
}


OVERLAYS = {
    "high_to_low": OverlaySpec(
        key="high_to_low",
        label="High_to_Low",
        embryo_id="20251113_A02_e01",
        color_hex=PHENOTYPE_COLORS["High_to_Low"],
    ),
    "low_to_high": OverlaySpec(
        key="low_to_high",
        label="Low_to_High",
        embryo_id="20251106_H04_e01",
        color_hex=PHENOTYPE_COLORS["Low_to_High"],
    ),
}


VIDEO_SPECS = (
    VideoSpec(
        order=1,
        output_stem="homozygous_background__low_to_high_only__20251106_H04_e01",
        static_overlays=(),
        animated_overlay=OVERLAYS["low_to_high"],
    ),
    VideoSpec(
        order=2,
        output_stem="homozygous_background__high_to_low_only__20251113_A02_e01",
        static_overlays=(),
        animated_overlay=OVERLAYS["high_to_low"],
    ),
    VideoSpec(
        order=3,
        output_stem="homozygous_background__low_to_high_then_high_to_low__20251106_H04_e01__20251113_A02_e01",
        static_overlays=(OVERLAYS["low_to_high"],),
        animated_overlay=OVERLAYS["high_to_low"],
    ),
    VideoSpec(
        order=4,
        output_stem="homozygous_background__high_to_low_then_low_to_high__20251113_A02_e01__20251106_H04_e01",
        static_overlays=(OVERLAYS["high_to_low"],),
        animated_overlay=OVERLAYS["low_to_high"],
    ),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create phenotype/genotype transition curvature videos for NWDB talk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output folder (figures/ will be created inside).",
    )
    p.add_argument(
        "--figures-subdir",
        default="phenotype_transition_homozygous/20s",
        help="Subfolder inside figures/ for outputs.",
    )
    p.add_argument(
        "--data-dir",
        default="results/mcolon/20251229_cep290_phenotype_extraction/final_data",
        help="Directory containing embryo_data_with_labels.csv and embryo_cluster_labels.csv.",
    )
    p.add_argument("--feature-col", default="curvature", help="Feature column to animate.")
    p.add_argument("--t-min", type=float, default=24.0, help="Background plot x-axis min HPF.")
    p.add_argument("--t-max", type=float, default=120.0, help="Background plot x-axis max HPF.")
    p.add_argument("--cursor-min", type=float, default=24.0, help="Animated cursor min HPF.")
    p.add_argument("--cursor-max", type=float, default=120.0, help="Animated cursor max HPF.")
    p.add_argument("--fps", type=int, default=20, help="Frames per second.")
    p.add_argument("--n-frames-out", type=int, default=400, help="Number of output frames.")
    p.add_argument("--plot-width", type=int, default=None, help="Optional plot width override.")
    p.add_argument("--plot-height", type=int, default=None, help="Optional plot height override.")
    p.add_argument("--smooth-sigma", type=float, default=2.0, help="Trace smoothing sigma.")
    p.add_argument("--trend-linestyle", default="dotted", choices=["solid", "dashed", "dotted", "-", "--", ":"], help="Background trend linestyle.")
    p.add_argument("--trace-outline", action="store_true", default=False, help="Draw black outline around overlay traces.")
    p.add_argument("--seed", type=int, default=0, help="Random seed placeholder for reproducibility.")
    p.add_argument("--ylim", default=None, help="Optional y limits as 'low,high'.")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> Paths:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = (project_root / args.data_dir).resolve()
    figures_dir = Path(args.out_dir).resolve() / "figures" / str(args.figures_subdir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        project_root=project_root,
        data_dir=data_dir,
        figures_dir=figures_dir,
    )


def _load_embryo_frames(data_dir: Path, raw_feature_col: str) -> pd.DataFrame:
    embryo_frames_path = data_dir / "embryo_data_with_labels.csv"
    embryo_labels_path = data_dir / "embryo_cluster_labels.csv"
    if not embryo_frames_path.exists():
        raise FileNotFoundError(f"Missing: {embryo_frames_path}")
    if not embryo_labels_path.exists():
        raise FileNotFoundError(f"Missing: {embryo_labels_path}")

    labels_raw = pd.read_csv(embryo_labels_path, usecols=["embryo_id", "cluster_categories"])
    labels_raw["embryo_id"] = labels_raw["embryo_id"].astype(str)
    labels_raw["cluster_categories"] = labels_raw["cluster_categories"].astype(str).str.strip()
    labels = labels_raw.drop_duplicates(subset=["embryo_id"], keep="first").copy()

    usecols = [
        "embryo_id",
        "experiment_date",
        "frame_index",
        "predicted_stage_hpf",
        "genotype",
        raw_feature_col,
        "use_embryo_flag",
    ]
    usecols += [
        "Height (um)",
        "Height (px)",
        "Width (um)",
        "Width (px)",
        "BF Channel",
        "Objective",
        "Time (s)",
        "Time Rel (s)",
    ]
    usecols += [
        "height_um",
        "height_px",
        "width_um",
        "width_px",
        "bf_channel",
        "objective",
        "raw_time_s",
        "relative_time_s",
    ]

    header = pd.read_csv(embryo_frames_path, nrows=0)
    existing = [c for c in usecols if c in header.columns]
    df = pd.read_csv(embryo_frames_path, usecols=existing, low_memory=False)
    df["embryo_id"] = df["embryo_id"].astype(str)

    if "use_embryo_flag" in df.columns:
        use_flag = df["use_embryo_flag"]
        if use_flag.dtype == bool:
            df = df[use_flag].copy()
        else:
            df = df[use_flag.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])].copy()

    df["predicted_stage_hpf"] = _safe_float_series(df["predicted_stage_hpf"])
    df[raw_feature_col] = _safe_float_series(df[raw_feature_col])

    _coalesce_columns(df, "height_um", "Height (um)")
    _coalesce_columns(df, "height_px", "Height (px)")
    _coalesce_columns(df, "width_um", "Width (um)")
    _coalesce_columns(df, "width_px", "Width (px)")
    _coalesce_columns(df, "bf_channel", "BF Channel")
    _coalesce_columns(df, "objective", "Objective")
    _coalesce_columns(df, "raw_time_s", "Time (s)")
    _coalesce_columns(df, "relative_time_s", "Time Rel (s)")

    return df.merge(labels, on="embryo_id", how="left", validate="many_to_one")


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    c = color_hex.strip()
    if c.startswith("#"):
        c = c[1:]
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return (b, g, r)


def _gaussian_smooth(vals: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or vals.size < 3:
        return vals
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(vals.astype(float), sigma=sigma, mode="reflect")


def _ax_data_to_pixel(ax, x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy_disp = ax.transData.transform(np.column_stack([x_data, y_data]))
    fig = ax.figure
    fig_h = fig.get_figheight() * fig.dpi
    px = xy_disp[:, 0]
    py = fig_h - xy_disp[:, 1]
    return px.astype(np.float32), py.astype(np.float32)


def _put_text_with_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font,
    font_scale: float,
    fg_bgr: tuple[int, int, int],
    thickness: int,
    outline_bgr: tuple[int, int, int] = (255, 255, 255),
    outline_thickness: Optional[int] = None,
) -> None:
    import cv2

    if outline_thickness is None:
        outline_thickness = int(thickness) + 3
    cv2.putText(img, text, org, font, float(font_scale), outline_bgr, int(outline_thickness), cv2.LINE_AA)
    cv2.putText(img, text, org, font, float(font_scale), fg_bgr, int(thickness), cv2.LINE_AA)


def _build_trace_pixels(ax, trace_df: pd.DataFrame, feature_col: str, smooth_sigma: float, *, extend_to: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = trace_df["predicted_stage_hpf"].to_numpy(dtype=float)
    vals = trace_df[feature_col].to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(vals)
    times = times[finite]
    vals = vals[finite]
    order = np.argsort(times)
    times = times[order]
    vals = _gaussian_smooth(vals[order], smooth_sigma)
    if extend_to is not None and times.size > 0 and times[-1] < extend_to:
        times = np.concatenate([times, [float(extend_to)]])
        vals = np.concatenate([vals, [vals[-1]]])
    if times.size >= 2:
        px, py = _ax_data_to_pixel(ax, times, vals)
    else:
        px = np.array([], dtype=np.float32)
        py = np.array([], dtype=np.float32)
    return times, px, py


def _draw_trace(frame: np.ndarray, px: np.ndarray, py: np.ndarray, color_bgr: tuple[int, int, int], *, upto_mask: Optional[np.ndarray] = None, tip: bool = True, trace_outline: bool = False) -> None:
    import cv2

    if px.size == 0:
        return

    if upto_mask is None:
        px_vis = px.astype(np.int32)
        py_vis = py.astype(np.int32)
    else:
        if not upto_mask.any():
            return
        px_vis = px[upto_mask].astype(np.int32)
        py_vis = py[upto_mask].astype(np.int32)

    if px_vis.size >= 2:
        pts = np.column_stack([px_vis, py_vis]).reshape(-1, 1, 2)
        halo_frame = frame.copy()
        cv2.polylines(halo_frame, [pts], isClosed=False, color=color_bgr, thickness=7, lineType=cv2.LINE_AA)
        cv2.addWeighted(halo_frame, 0.18, frame, 0.82, 0, frame)
        if trace_outline:
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts], isClosed=False, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

    if tip and px_vis.size >= 1:
        tip_x, tip_y = int(px_vis[-1]), int(py_vis[-1])
        outer = (0, 0, 0) if trace_outline else (255, 255, 255)
        cv2.circle(frame, (tip_x, tip_y), 6, outer, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (tip_x, tip_y), 5, color_bgr, -1, lineType=cv2.LINE_AA)


def _make_transition_video(
    out_mp4: Path,
    background_df: pd.DataFrame,
    static_traces: tuple[pd.DataFrame, ...],
    animated_trace: pd.DataFrame,
    feature_col: str,
    t_min: float,
    t_max: float,
    cursor_min: float,
    cursor_max: float,
    fps: int,
    n_frames_out: int,
    plot_width: int,
    plot_height: int,
    smooth_sigma: float,
    trace_outline: bool,
    trend_linestyle: str,
    ylim: Optional[tuple[float, float]],
) -> None:
    import cv2
    from analyze.viz.plotting.feature_over_time import plot_feature_over_time
    from analyze.viz.styling.color_mapping_config import GENOTYPE_COLORS

    apply_transition_rcparams()
    dpi = PLOT_DPI

    bg = background_df[background_df["predicted_stage_hpf"].between(t_min, t_max, inclusive="both")].copy()
    bg = bg[bg[feature_col].notna()].copy()

    bg_fig = plot_feature_over_time(
        bg,
        features=feature_col,
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="genotype",
        color_lookup=GENOTYPE_COLORS,
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        backend="matplotlib",
        ylim=ylim,
        trend_linestyle=trend_linestyle,
    )
    style_transition_figure(bg_fig)

    bg_ax = bg_fig.axes[0]
    leg = bg_ax.get_legend()
    if leg is not None:
        leg.remove()

    unfaded_png = out_mp4.with_name(out_mp4.stem.replace("curvature_animation", "background_unfaded") + ".png")
    bg_fig.savefig(str(unfaded_png), dpi=dpi)

    bg_fig.canvas.draw()
    canvas_w, canvas_h = bg_fig.canvas.get_width_height()
    plot_width, plot_height = int(canvas_w), int(canvas_h)

    x_lim = bg_ax.get_xlim()
    y_lim = bg_ax.get_ylim()
    y0 = float(y_lim[0])

    bg_rgb = _fig_to_rgb_array(bg_fig).copy()
    bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
    fig_h_px = bg_fig.get_figheight() * bg_fig.dpi
    bbox = bg_ax.get_window_extent()
    ax_px_x0 = int(round(bbox.x0))
    ax_px_x1 = int(round(bbox.x1))
    ax_px_y0 = int(round(fig_h_px - bbox.y1))
    ax_px_y1 = int(round(fig_h_px - bbox.y0))
    plt.close(bg_fig)

    white = np.full_like(bg_bgr, 255, dtype=np.uint8)
    cv2.addWeighted(bg_bgr, 0.35, white, 0.65, 0, bg_bgr)

    def _t_to_px(t_val: float) -> int:
        xy = bg_ax.transData.transform([[float(t_val), y0]])
        return int(round(xy[0, 0]))

    static_pixels: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
    for trace_df in static_traces:
        overlay = str(trace_df["cluster_categories"].iloc[0])
        color_bgr = _hex_to_bgr(PHENOTYPE_COLORS[overlay])
        _, px, py = _build_trace_pixels(bg_ax, trace_df, feature_col, smooth_sigma)
        static_pixels.append((px, py, color_bgr))

    animated_name = str(animated_trace["cluster_categories"].iloc[0])
    animated_color_bgr = _hex_to_bgr(PHENOTYPE_COLORS[animated_name])
    animated_times, animated_px, animated_py = _build_trace_pixels(bg_ax, animated_trace, feature_col, smooth_sigma)

    out_times = np.linspace(float(cursor_min), float(cursor_max), int(n_frames_out))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (int(plot_width), int(plot_height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

    try:
        for t in out_times:
            frame = bg_bgr.copy()

            for px, py, color_bgr in static_pixels:
                _draw_trace(frame, px, py, color_bgr, tip=False, trace_outline=trace_outline)

            if animated_times.size >= 1:
                mask = animated_times <= float(t)
                if not mask.any():
                    mask[0] = True
                _draw_trace(
                    frame,
                    animated_px,
                    animated_py,
                    animated_color_bgr,
                    upto_mask=mask,
                    tip=True,
                    trace_outline=trace_outline,
                )

                feat_t_max = float(animated_times[-1])
                if float(t) <= feat_t_max:
                    cx = _t_to_px(float(t))
                    cx = max(ax_px_x0, min(ax_px_x1, cx))
                    cv2.line(frame, (cx, ax_px_y0), (cx, ax_px_y1), animated_color_bgr, 1, lineType=cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = _parse_args()
    paths = _resolve_paths(args)

    sys.path.insert(0, str(paths.project_root / "src"))
    from analyze.utils.stats import normalize_arbitrary_feature
    from analyze.viz.styling.color_mapping_config import GENOTYPE_COLORS

    raw_feature_col = "baseline_deviation_normalized"
    feature_col = str(args.feature_col)
    use_derived_curvature = feature_col == "curvature"

    df = _load_embryo_frames(paths.data_dir, raw_feature_col=raw_feature_col)
    if use_derived_curvature:
        df["curvature"] = normalize_arbitrary_feature(
            df[raw_feature_col],
            low=0,
            high_percentile=100,
            clip=False,
        )
        print(
            "Derived 'curvature' column from 'baseline_deviation_normalized' via "
            "normalize_arbitrary_feature(low=0, high_percentile=100, clip=False)"
        )

    df = df[df["predicted_stage_hpf"].notna()].copy()
    df_homo = df[df["genotype"].astype(str) == "cep290_homozygous"].copy()
    if df_homo.empty:
        raise RuntimeError("No homozygous embryos found for background.")

    overlay_frames: dict[str, pd.DataFrame] = {}
    for overlay in OVERLAYS.values():
        feat_df = df[df["embryo_id"].astype(str) == overlay.embryo_id].copy()
        feat_df = feat_df.sort_values("predicted_stage_hpf")
        feat_df = feat_df[feat_df["predicted_stage_hpf"].between(float(args.t_min), float(args.t_max), inclusive="both")].copy()
        if feat_df.empty:
            raise RuntimeError(f"No rows found in {args.t_min}-{args.t_max} HPF for {overlay.embryo_id}")
        overlay_frames[overlay.key] = feat_df

    ylim = None
    if args.ylim is not None:
        parts = [float(x.strip()) for x in str(args.ylim).split(",")]
        if len(parts) != 2:
            raise ValueError(f"--ylim must be two comma-separated floats, got {args.ylim!r}")
        ylim = (parts[0], parts[1])

    default_w, default_h = _default_plot_px()
    plot_width = int(args.plot_width) if args.plot_width is not None else default_w
    plot_height = int(args.plot_height) if args.plot_height is not None else default_h

    print(f"Plot dimensions: {plot_width}x{plot_height} px")
    print(f"Plot x-axis: {args.t_min}--{args.t_max} HPF, cursor scans {args.cursor_min}--{args.cursor_max} HPF")
    print(f"Rendering {args.n_frames_out} frames at {args.fps} fps ({args.n_frames_out / args.fps:.1f}s)")
    print(f"Homozygous background embryos: {df_homo['embryo_id'].nunique()}")
    print(f"Homozygous background color: {GENOTYPE_COLORS['cep290_homozygous']}")

    saved = []
    for spec in VIDEO_SPECS:
        numbered_stem = f"{spec.order:02d}_{spec.output_stem}"
        print(f"=== {numbered_stem} ===")
        static_traces = tuple(overlay_frames[o.key] for o in spec.static_overlays)
        animated_trace = overlay_frames[spec.animated_overlay.key]
        out_mp4 = paths.figures_dir / f"curvature_animation_{numbered_stem}.mp4"
        _make_transition_video(
            out_mp4=out_mp4,
            background_df=df_homo,
            static_traces=static_traces,
            animated_trace=animated_trace,
            feature_col=feature_col,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            cursor_min=float(args.cursor_min),
            cursor_max=float(args.cursor_max),
            fps=int(args.fps),
            n_frames_out=int(args.n_frames_out),
            plot_width=int(plot_width),
            plot_height=int(plot_height),
            smooth_sigma=float(args.smooth_sigma),
            trace_outline=bool(args.trace_outline),
            trend_linestyle=str(args.trend_linestyle),
            ylim=ylim,
        )
        print(f"  Saved: {out_mp4}")
        saved.append(out_mp4)

    print("Saved files:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
