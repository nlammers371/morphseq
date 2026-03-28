"""
NWDB talk: Per-genotype curvature animations + synchronized embryo snip movies.

Produces two MP4s per genotype (wildtype, heterozygous, homozygous) with identical
FPS and frame count so they can be played side-by-side in slides:
1) curvature trace plot that "draws" left-to-right over HPF with background
   population traces faded out — one genotype per panel, matching the faceted plot.
2) embryo snip frames advancing in sync with HPF (mask outlines disabled).

Data sources (read-only):
- results/mcolon/20251229_cep290_phenotype_extraction/final_data/
  - embryo_data_with_labels.csv
  - embryo_cluster_labels.csv
- morphseq_playground/training_data/bf_embryo_snips/{experiment_date}/...jpg
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve default figure pixel dimensions from the faceting engine renderer
# so this script automatically tracks any changes to the config.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from analyze.viz.embryo_renderer import (
    build_embryo_track,
    build_snip_path,
    export_embryo_video,
    render_embryo_sequence,
)

def _default_plot_px() -> tuple[int, int]:
    """Return (width_px, height_px) matching plot_feature_over_time's default single-panel size."""
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    import inspect
    try:
        from analyze.viz.plotting.faceting_engine.renderers.matplotlib import render_matplotlib
        src = inspect.getsource(render_matplotlib)
        # Parse the figsize line: figsize = (W_PER_COL * n_cols, H_PER_ROW * n_rows)
        # For a single panel: n_rows=1, n_cols=1 → figsize = (W, H)
        import re
        m = re.search(r"figsize\s*=\s*\((\d+(?:\.\d+)?)\s*\*\s*n_cols\s*,\s*(\d+(?:\.\d+)?)\s*\*\s*n_rows\)", src)
        if m:
            w_in = float(m.group(1))
            h_in = float(m.group(2))
        else:
            w_in, h_in = 5.0, 4.5  # fallback to known values
    except Exception:
        w_in, h_in = 5.0, 4.5
    dpi = int(plt.rcParams.get("figure.dpi", 100))
    return int(round(w_in * dpi)), int(round(h_in * dpi))


def _fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """
    Convert a matplotlib Agg figure to an RGB uint8 image array of shape (H, W, 3).

    Matplotlib's canvas APIs have changed across versions; prefer buffer_rgba when available.
    """

    canvas = fig.canvas
    canvas.draw()

    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba())
        rgb = rgba[..., :3]
        return np.ascontiguousarray(rgb)

    w, h = canvas.get_width_height()
    if hasattr(canvas, "tostring_rgb"):
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((int(h), int(w), 3))
        return np.ascontiguousarray(rgb)

    # Fallback: ARGB bytes
    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((int(h), int(w), 4))
    rgb = argb[..., 1:]
    return np.ascontiguousarray(rgb)


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    snip_root: Path
    out_dir: Path
    figures_dir: Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create NWDB CEP290 per-genotype curvature + embryo animations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output folder (figures/ will be created inside).",
    )
    p.add_argument(
        "--data-dir",
        default="results/mcolon/20251229_cep290_phenotype_extraction/final_data",
        help="Directory containing embryo_data_with_labels.csv and embryo_cluster_labels.csv.",
    )
    p.add_argument(
        "--cluster-category",
        default="Not Penetrant",
        help="cluster_categories value to select (ignored when --no-cluster-filter is set).",
    )
    p.add_argument(
        "--no-cluster-filter",
        action="store_true",
        default=True,
        help="Disable cluster_categories filter; use all embryos per genotype as background.",
    )
    p.add_argument(
        "--cluster-filter",
        dest="no_cluster_filter",
        action="store_false",
        help="Enable cluster_categories filter (use --cluster-category value).",
    )
    p.add_argument(
        "--feature-col",
        default="curvature",
        help="Feature column to plot over time (derived 'curvature' or raw column name).",
    )
    p.add_argument(
        "--genotypes",
        default="wildtype,heterozygous,homozygous",
        help="Comma-separated genotype suffixes to animate (one animation pair per suffix).",
    )
    p.add_argument(
        "--cluster-categories",
        default="High_to_Low,Low_to_High,Not Penetrant",
        help="Comma-separated cluster_categories values to animate (one animation pair per category). "
             "Used when --panel-by=cluster_categories.",
    )
    p.add_argument(
        "--panel-by",
        choices=["genotype", "cluster_categories"],
        default="genotype",
        help="Which column defines the per-panel animations. "
             "'genotype' renders one panel per genotype suffix; "
             "'cluster_categories' renders one panel per cluster category.",
    )
    p.add_argument(
        "--featured-embryo-ids",
        default=None,
        help="Comma-separated embryo_ids per panel (same order as --genotypes or --cluster-categories). "
             "Use 'auto' for auto-selection. E.g. '20250512_D02_e01,auto,20250512_E06_e01'.",
    )
    p.add_argument(
        "--ylim",
        default=None,
        help="Comma-separated y-axis limits, e.g. '0,1.0'. Matches notebook ylim.",
    )
    p.add_argument(
        "--t-min",
        type=float,
        default=24.0,
        help="Minimum HPF for animation.",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=120.0,
        help="Maximum HPF for the plot x-axis range.",
    )
    p.add_argument(
        "--cursor-min",
        type=float,
        default=None,
        help="HPF where the cursor starts scanning (default: same as --t-min). "
             "Use to start later, e.g. --t-min 24 --cursor-min 40.",
    )
    p.add_argument(
        "--cursor-max",
        type=float,
        default=None,
        help="HPF where the cursor stops scanning (default: same as --t-max). "
             "Use to decouple animation duration from plot range, e.g. --t-max 120 --cursor-max 80.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for both videos.",
    )
    p.add_argument(
        "--n-frames-out",
        type=int,
        default=300,
        help="Number of output frames for both videos.",
    )
    p.add_argument(
        "--background-max-embryos",
        type=int,
        default=250,
        help="Max number of background embryos to draw (sampled if more).",
    )
    p.add_argument(
        "--min-featured-snips",
        type=int,
        default=50,
        help="Minimum number of snip frames required for the featured embryo.",
    )
    p.add_argument(
        "--n-featured-per-panel",
        type=int,
        default=1,
        help="How many featured embryos to render per panel (top-ranked candidates).",
    )
    p.add_argument(
        "--candidate-end-n",
        type=int,
        default=5,
        help="Number of final timepoints used to compute each embryo's end value for candidate ranking.",
    )
    p.add_argument(
        "--min-featured-end-value",
        type=float,
        default=None,
        help="Minimum end value (median of last --candidate-end-n points) required for a featured embryo. "
             "Useful for Low_to_High to require it ends high (e.g. 0.6).",
    )
    p.add_argument(
        "--min-featured-min-hpf",
        type=float,
        default=None,
        help="Minimum start HPF (min predicted_stage_hpf) required for the featured embryo within the time window. "
             "Useful to prefer embryos that start later (e.g. 40). If omitted, no constraint is applied.",
    )
    p.add_argument(
        "--min-featured-max-hpf",
        type=float,
        default=60.0,
        help="Minimum max HPF required for the featured embryo (within [t-min,t-max]).",
    )
    p.add_argument(
        "--featured-embryo-id",
        default=None,
        help="Explicit embryo_id to feature (single-genotype mode; ignored if --featured-embryo-ids is set).",
    )
    p.add_argument(
        "--prefer-genotype-suffix",
        default="wildtype",
        help="When auto-selecting, prefer this genotype suffix if available (e.g. wildtype).",
    )
    p.add_argument(
        "--hold-last-frame",
        action="store_true",
        default=True,
        help="Hold last available snip frame after the embryo runs out of data.",
    )
    p.add_argument(
        "--no-hold-last-frame",
        dest="hold_last_frame",
        action="store_false",
        help="Do not hold last snip frame (repeats last valid anyway for writer).",
    )
    p.add_argument(
        "--extend-featured-trace",
        action="store_true",
        default=False,
        help="Extend the featured trace flat after its last timepoint to t-max (visualize 'flat').",
    )
    p.add_argument(
        "--no-extend-featured-trace",
        dest="extend_featured_trace",
        action="store_false",
        help="Do not extend the featured trace beyond its last timepoint.",
    )
    p.add_argument(
        "--figures-subdir",
        default=None,
        help="Subfolder inside figures/ for output (e.g. 'genotype_overlays').",
    )
    p.add_argument(
        "--plot-width",
        type=int,
        default=None,
        help="Plot video width in pixels (default: auto-derived from faceting engine config).",
    )
    p.add_argument(
        "--plot-height",
        type=int,
        default=None,
        help="Plot video height in pixels (default: auto-derived from faceting engine config).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for background sampling.",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier: >1 is faster (fewer output frames), <1 is slower.",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for smoothing the featured trace (in data points; 0 = no smoothing).",
    )
    p.add_argument(
        "--trace-outline",
        action="store_true",
        default=False,
        help="Draw a black outline stroke around the featured trace (helps on busy backgrounds).",
    )
    p.add_argument(
        "--filter-genotype",
        default=None,
        help="If set, restrict background embryos to this genotype suffix (e.g. 'homozygous'). "
             "Useful when panel-by=cluster_categories but you only want one genotype shown.",
    )
    p.add_argument(
        "--trend-linestyle",
        default="dotted",
        choices=["solid", "dashed", "dotted", "-", "--", ":"],
        help="Linestyle for trend line (default: dotted)",
    )
    p.add_argument(
        "--bin-width",
        type=float,
        default=0.5,
        help="HPF bin width used for the trend line in the background plot.",
    )
    p.add_argument(
        "--plot-style",
        default="background",
        choices=["background", "trace_only"],
        help="Render the curvature animation either with faded population background traces "
             "or as a clean single-trace plot.",
    )
    p.add_argument(
        "--skip-embryo-video",
        action="store_true",
        default=False,
        help="Only render the curvature plot animation; skip the synchronized embryo snip MP4.",
    )
    return p.parse_args()


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    c = color_hex.strip()
    if c.startswith("#"):
        c = c[1:]
    if len(c) != 6:
        return (128, 128, 128)
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return (b, g, r)


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _coalesce_columns(df: pd.DataFrame, dst: str, src: str) -> None:
    """Fill dst with src when dst is entirely missing but src exists."""
    if dst not in df.columns or src not in df.columns:
        return
    if df[dst].isna().all() and (~df[src].isna()).any():
        df[dst] = df[src]


def _resolve_paths(args: argparse.Namespace) -> Paths:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = (project_root / args.data_dir).resolve()
    snip_root = project_root / "morphseq_playground" / "training_data" / "bf_embryo_snips"
    out_dir = Path(args.out_dir).resolve()
    figures_dir = out_dir / "figures"
    if args.figures_subdir:
        figures_dir = figures_dir / args.figures_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)
    return Paths(
        project_root=project_root,
        data_dir=data_dir,
        snip_root=snip_root,
        out_dir=out_dir,
        figures_dir=figures_dir,
    )


def _load_embryo_frames(
    data_dir: Path,
    raw_feature_col: str,
    cluster_category: Optional[str] = None,
) -> pd.DataFrame:
    """Load embryo frame data, optionally filtering by cluster category.

    When cluster_category is None, all embryos are loaded (no cluster filter).
    """
    embryo_frames_path = data_dir / "embryo_data_with_labels.csv"
    embryo_labels_path = data_dir / "embryo_cluster_labels.csv"
    if not embryo_frames_path.exists():
        raise FileNotFoundError(f"Missing: {embryo_frames_path}")
    if not embryo_labels_path.exists():
        raise FileNotFoundError(f"Missing: {embryo_labels_path}")

    labels_raw = pd.read_csv(embryo_labels_path, usecols=["embryo_id", "cluster_categories"])
    labels_raw["embryo_id"] = labels_raw["embryo_id"].astype(str)
    labels_raw["cluster_categories"] = labels_raw["cluster_categories"].astype(str).str.strip()
    # embryo_cluster_labels.csv is often frame-level (many rows per embryo). Deduplicate to one row per embryo.
    n_unique = labels_raw.groupby("embryo_id")["cluster_categories"].nunique(dropna=False)
    multi = n_unique[n_unique > 1]
    if not multi.empty:
        examples = multi.head(10).index.tolist()
        print(
            f"WARNING: {len(multi)} embryo_id values have multiple cluster_categories in {embryo_labels_path.name}; "
            f"using first. Example embryo_id: {examples}"
        )
    labels = labels_raw.drop_duplicates(subset=["embryo_id"], keep="first").copy()

    # Determine which embryo IDs to keep
    if cluster_category is not None:
        target = str(cluster_category).strip()
        keep_ids = set(labels.loc[labels["cluster_categories"] == target, "embryo_id"].astype(str).tolist())
        if not keep_ids:
            got = sorted(labels["cluster_categories"].dropna().astype(str).unique().tolist())[:30]
            raise ValueError(f"No embryos found for cluster_categories=={target!r}. Example categories: {got}")
    else:
        keep_ids = set(labels["embryo_id"].astype(str).tolist())

    usecols = [
        "embryo_id",
        "experiment_date",
        "frame_index",
        "predicted_stage_hpf",
        "genotype",
        raw_feature_col,
        "exported_mask_path",
        "region_label",
        "use_embryo_flag",
    ]
    # Some legacy exports use these pretty column names only; keep for later coalesce.
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
    # Also include snakecase if present.
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
    missing_critical = [c for c in ["embryo_id", "predicted_stage_hpf", "genotype", raw_feature_col] if c not in existing]
    if missing_critical:
        raise ValueError(f"Missing required columns in embryo_data_with_labels.csv: {missing_critical}")

    df = pd.read_csv(embryo_frames_path, usecols=existing, low_memory=False)
    df["embryo_id"] = df["embryo_id"].astype(str)
    df = df[df["embryo_id"].isin(keep_ids)].copy()
    if "use_embryo_flag" in df.columns:
        # handle both bool and string-like
        use_flag = df["use_embryo_flag"]
        if use_flag.dtype == bool:
            df = df[use_flag].copy()
        else:
            df = df[use_flag.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])].copy()

    df["predicted_stage_hpf"] = _safe_float_series(df["predicted_stage_hpf"])
    df[raw_feature_col] = _safe_float_series(df[raw_feature_col])

    # Coalesce pretty columns into snakecase for downstream convenience.
    _coalesce_columns(df, "height_um", "Height (um)")
    _coalesce_columns(df, "height_px", "Height (px)")
    _coalesce_columns(df, "width_um", "Width (um)")
    _coalesce_columns(df, "width_px", "Width (px)")
    _coalesce_columns(df, "bf_channel", "BF Channel")
    _coalesce_columns(df, "objective", "Objective")
    _coalesce_columns(df, "raw_time_s", "Time (s)")
    _coalesce_columns(df, "relative_time_s", "Time Rel (s)")

    # Add cluster label column for convenience
    df = df.merge(labels, on="embryo_id", how="left", validate="many_to_one")

    return df


def _genotype_suffix(genotype: str) -> str:
    s = str(genotype).strip().lower()
    if "_" in s:
        return s.split("_")[-1]
    return s


def _safe_name(s: str) -> str:
    return str(s).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


def _snip_path(snip_root: Path, experiment_date: str, embryo_id: str, frame_index: int) -> Path:
    return build_snip_path(snip_root, experiment_date, embryo_id, frame_index)


def _pick_featured_embryo(
    df: pd.DataFrame,
    feature_col: str,
    t_min: float,
    t_max: float,
    snip_root: Path,
    min_snips: int,
    min_min_hpf: Optional[float],
    min_max_hpf: float,
    prefer_suffix: Optional[str],
    explicit_embryo_id: Optional[str],
) -> str:
    if explicit_embryo_id is not None:
        if explicit_embryo_id not in set(df["embryo_id"].astype(str)):
            raise ValueError(f"--featured-embryo-id {explicit_embryo_id!r} not present after filtering.")
        return explicit_embryo_id

    g = df.copy()
    g = g[g["predicted_stage_hpf"].between(t_min, t_max, inclusive="both")].copy()
    g = g[g[feature_col].notna()].copy()

    summary = (
        g.groupby("embryo_id", observed=True)
        .agg(
            experiment_date=("experiment_date", "first"),
            genotype=("genotype", "first"),
            n_rows=("frame_index", "size"),
            min_hpf=("predicted_stage_hpf", "min"),
            max_hpf=("predicted_stage_hpf", "max"),
        )
        .reset_index()
    )
    summary["suffix"] = summary["genotype"].map(_genotype_suffix)

    # filter criteria
    summary = summary[summary["n_rows"] >= int(min_snips)].copy()
    if min_min_hpf is not None:
        summary = summary[summary["min_hpf"] >= float(min_min_hpf)].copy()
    summary = summary[summary["max_hpf"] >= float(min_max_hpf)].copy()
    if summary.empty:
        raise ValueError(
            "No candidate featured embryo found after applying filters. "
            f"Try lowering --min-featured-snips/--min-featured-max-hpf or specify --featured-embryo-id."
        )

    # prefer suffix (e.g. wildtype)
    if prefer_suffix is None:
        summary["prefer"] = 0
    else:
        prefer_suffix = str(prefer_suffix).strip().lower()
        summary["prefer"] = (summary["suffix"] == prefer_suffix).astype(int)

    # verify snip existence for a small sample of frames (avoid picking missing snips)
    ok_rows = []
    for _, row in summary.sort_values(
        ["prefer", "min_hpf", "max_hpf", "n_rows"], ascending=[False, True, False, False]
    ).iterrows():
        embryo_id = str(row["embryo_id"])
        exp_date = str(row["experiment_date"])
        # Check first/last frame indices available in g
        sub = g[g["embryo_id"] == embryo_id].sort_values("frame_index")
        fi0 = int(sub["frame_index"].iloc[0])
        fi1 = int(sub["frame_index"].iloc[-1])
        if _snip_path(snip_root, exp_date, embryo_id, fi0).exists() and _snip_path(
            snip_root, exp_date, embryo_id, fi1
        ).exists():
            ok_rows.append(embryo_id)
            break

    if not ok_rows:
        raise ValueError("Could not find a featured embryo with resolvable snip JPG paths.")
    return ok_rows[0]


def _rank_featured_candidates(
    df: pd.DataFrame,
    feature_col: str,
    t_min: float,
    t_max: float,
    snip_root: Path,
    min_snips: int,
    min_min_hpf: Optional[float],
    min_max_hpf: float,
    prefer_suffix: Optional[str],
    candidate_end_n: int,
    min_end_value: Optional[float],
    exclude_embryo_ids: set[str],
    top_k: int,
) -> pd.DataFrame:
    """Return ranked candidate embryos (best-first) with end_value and coverage stats."""
    g = df.copy()
    g = g[g["predicted_stage_hpf"].between(t_min, t_max, inclusive="both")].copy()
    g = g[g[feature_col].notna()].copy()
    if exclude_embryo_ids:
        g = g[~g["embryo_id"].astype(str).isin({str(x) for x in exclude_embryo_ids})].copy()

    if g.empty:
        return pd.DataFrame()

    g = g.sort_values(["embryo_id", "predicted_stage_hpf", "frame_index"])

    # End value: median of last N values per embryo within the window
    n_end = max(1, int(candidate_end_n))
    end_vals = (
        g.groupby("embryo_id", observed=True)[feature_col]
        .tail(n_end)
        .groupby(g["embryo_id"], observed=True)
        .median()
        .rename("end_value")
        .reset_index()
    )

    summary = (
        g.groupby("embryo_id", observed=True)
        .agg(
            experiment_date=("experiment_date", "first"),
            genotype=("genotype", "first"),
            n_rows=("frame_index", "size"),
            min_hpf=("predicted_stage_hpf", "min"),
            max_hpf=("predicted_stage_hpf", "max"),
            fi0=("frame_index", "first"),
            fi1=("frame_index", "last"),
        )
        .reset_index()
    )
    summary["suffix"] = summary["genotype"].map(_genotype_suffix)
    summary = summary.merge(end_vals, on="embryo_id", how="left", validate="one_to_one")

    # Filters
    summary = summary[summary["n_rows"] >= int(min_snips)].copy()
    if min_min_hpf is not None:
        summary = summary[summary["min_hpf"] >= float(min_min_hpf)].copy()
    summary = summary[summary["max_hpf"] >= float(min_max_hpf)].copy()
    if min_end_value is not None:
        summary = summary[summary["end_value"] >= float(min_end_value)].copy()
    if summary.empty:
        return summary

    # Prefer suffix (if provided)
    if prefer_suffix is None:
        summary["prefer"] = 0
    else:
        prefer_suffix = str(prefer_suffix).strip().lower()
        summary["prefer"] = (summary["suffix"] == prefer_suffix).astype(int)

    summary = summary.sort_values(
        ["prefer", "end_value", "max_hpf", "n_rows", "min_hpf"],
        ascending=[False, False, False, False, True],
    )

    # Verify snip existence for first/last frame_index (avoid missing snips)
    keep_rows = []
    for r in summary.itertuples(index=False):
        embryo_id = str(r.embryo_id)
        exp_date = str(r.experiment_date)
        p0 = _snip_path(snip_root, exp_date, embryo_id, int(r.fi0))
        p1 = _snip_path(snip_root, exp_date, embryo_id, int(r.fi1))
        if p0.exists() and p1.exists():
            keep_rows.append(embryo_id)
        if len(keep_rows) >= int(top_k):
            break

    if not keep_rows:
        return pd.DataFrame()

    return summary[summary["embryo_id"].astype(str).isin(set(keep_rows))].copy()


def _put_text_with_outline(
    img: "np.ndarray",
    text: str,
    org: tuple[int, int],
    font,
    font_scale: float,
    fg_bgr: tuple[int, int, int],
    thickness: int,
    outline_bgr: tuple[int, int, int] = (0, 0, 0),
    outline_thickness: Optional[int] = None,
) -> None:
    """Draw text with an outline for visibility (OpenCV BGR)."""
    import cv2

    if outline_thickness is None:
        outline_thickness = int(thickness) + 3
    cv2.putText(img, text, org, font, float(font_scale), outline_bgr, int(outline_thickness), cv2.LINE_AA)
    cv2.putText(img, text, org, font, float(font_scale), fg_bgr, int(thickness), cv2.LINE_AA)


def _nearest_index(sorted_times: np.ndarray, t: float) -> int:
    """Return index of nearest value in sorted_times to t (clipped)."""
    if sorted_times.size == 0:
        return 0
    if t <= sorted_times[0]:
        return 0
    if t >= sorted_times[-1]:
        return int(sorted_times.size - 1)
    j = int(np.searchsorted(sorted_times, t))
    i = j - 1
    if abs(sorted_times[i] - t) <= abs(sorted_times[j] - t):
        return i
    return j


def _gaussian_smooth(vals: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1-D array with a Gaussian kernel (reflect padding)."""
    if sigma <= 0 or vals.size < 3:
        return vals
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(vals.astype(float), sigma=sigma, mode="reflect")


def _ax_data_to_pixel(ax, x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert data coordinates to pixel coordinates on the Agg canvas."""
    xy_disp = ax.transData.transform(np.column_stack([x_data, y_data]))
    # matplotlib display coords: (0,0) = bottom-left; image coords: (0,0) = top-left
    fig = ax.figure
    fig_h = fig.get_figheight() * fig.dpi
    px = xy_disp[:, 0]
    py = fig_h - xy_disp[:, 1]
    return px.astype(np.float32), py.astype(np.float32)


def _prepare_featured_trace(
    featured_df: pd.DataFrame,
    feature_col: str,
    smooth_sigma: float,
    *,
    extend_to: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    times = featured_df["predicted_stage_hpf"].to_numpy(dtype=float)
    vals = featured_df[feature_col].to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(vals)
    times = times[finite]
    vals = vals[finite]
    if times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(times)
    times = times[order]
    vals = _gaussian_smooth(vals[order], smooth_sigma)

    if extend_to is not None and times[-1] < float(extend_to):
        times = np.concatenate([times, [float(extend_to)]])
        vals = np.concatenate([vals, [vals[-1]]])

    return times, vals


def _build_reference_plot_figure(
    df: pd.DataFrame,
    feature_col: str,
    color_by: str,
    color_lookup: dict,
    t_min: float,
    t_max: float,
    background_max_embryos: int,
    seed: int,
    trend_linestyle: str,
    bin_width: float,
    ylim: Optional[tuple[float, float]],
    force_xlim: bool,
):
    from analyze.viz.plotting.feature_over_time import plot_feature_over_time

    rng = np.random.default_rng(int(seed))

    bg = df[df["predicted_stage_hpf"].between(t_min, t_max, inclusive="both")].copy()
    bg = bg[bg[feature_col].notna()].copy()
    bg_ids = bg["embryo_id"].astype(str).unique().tolist()
    if len(bg_ids) > int(background_max_embryos):
        bg_ids = rng.choice(bg_ids, size=int(background_max_embryos), replace=False).tolist()
    bg = bg[bg["embryo_id"].isin(bg_ids)].copy()

    fig = plot_feature_over_time(
        bg,
        features=feature_col,
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by=color_by,
        color_lookup=color_lookup,
        show_individual=True,
        show_trend=True,
        show_error_band=False,
        backend="matplotlib",
        xlim=(float(t_min), float(t_max)) if force_xlim else None,
        ylim=ylim,
        trend_linestyle=trend_linestyle,
        bin_width=float(bin_width),
    )
    ax = fig.axes[0]
    if force_xlim:
        ax.set_xlim(float(t_min), float(t_max))
        ax.set_xticks(np.linspace(float(t_min), float(t_max), 5))
    return fig, ax


def _blank_trace_figure_from_reference(ref_fig: plt.Figure, ref_ax) -> tuple[plt.Figure, any]:
    fig = plt.figure(figsize=ref_fig.get_size_inches(), dpi=ref_fig.dpi)
    ax = fig.add_axes(ref_ax.get_position())
    # Preserve the rendered limits from the reference plot. Copying the tick
    # locations here can expand the axis (e.g. from ~20-124 out to 0-140),
    # so let Matplotlib keep the reference locator behavior.
    ax.set_xlim(ref_ax.get_xlim())
    ax.set_ylim(ref_ax.get_ylim())
    ax.set_xlabel(ref_ax.get_xlabel())
    ax.set_ylabel(ref_ax.get_ylabel())
    if ref_ax.get_title():
        ax.set_title(ref_ax.get_title())
    ax.grid(alpha=0.15, linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _save_trace_only_still(
    out_png: Path,
    ref_fig: plt.Figure,
    ref_ax,
    featured_df: pd.DataFrame,
    feature_col: str,
    featured_color_hex: str,
    smooth_sigma: float,
    extend_featured_trace: bool,
    t_max: float,
    trace_outline: bool,
) -> None:
    still_fig, still_ax = _blank_trace_figure_from_reference(ref_fig, ref_ax)
    extend_to = float(t_max) if extend_featured_trace else None
    xs, ys = _prepare_featured_trace(
        featured_df,
        feature_col,
        smooth_sigma,
        extend_to=extend_to,
    )
    if xs.size >= 1:
        line = still_ax.plot(xs, ys, color=featured_color_hex, linewidth=3.0, solid_capstyle="round")[0]
        if bool(trace_outline):
            line.set_path_effects([pe.Stroke(linewidth=4.8, foreground="black"), pe.Normal()])
    still_fig.savefig(str(out_png), dpi=ref_fig.dpi)
    plt.close(still_fig)


def _make_plot_video(
    out_mp4: Path,
    df: pd.DataFrame,
    featured_df: pd.DataFrame,
    feature_col: str,
    featured_color_hex: str,
    color_by: str,
    color_lookup: dict,
    t_min: float,
    t_max: float,
    cursor_min: float,
    cursor_max: float,
    fps: int,
    n_frames_out: int,
    plot_width: int,
    plot_height: int,
    background_max_embryos: int,
    seed: int,
    extend_featured_trace: bool,
    smooth_sigma: float,
    trace_outline: bool,
    trend_linestyle: str = "dotted",
    bin_width: float = 0.5,
    plot_style: str = "background",
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    import cv2

    plt.rcParams.update({
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelsize": 17,
    })
    dpi = 100

    print("  Rendering styled reference figure...")
    ref_fig, ref_ax = _build_reference_plot_figure(
        df=df,
        feature_col=feature_col,
        color_by=color_by,
        color_lookup=color_lookup,
        t_min=t_min,
        t_max=t_max,
        background_max_embryos=background_max_embryos,
        seed=seed,
        trend_linestyle=trend_linestyle,
        bin_width=bin_width,
        ylim=ylim,
        force_xlim=(plot_style == "background"),
    )

    # Save unfaded PNGs before removing legend for video
    bg_ax_leg = ref_fig.axes[0]
    leg = bg_ax_leg.get_legend()

    if plot_style == "background":
        unfaded_png = out_mp4.with_name(out_mp4.stem.replace("curvature_animation", "background_unfaded") + ".png")
        ref_fig.savefig(str(unfaded_png), bbox_inches="tight", dpi=dpi)
        print(f"  Saved unfaded background: {unfaded_png.name}")

        if leg is not None:
            leg.remove()
            bg_ax_leg.legend(
                handles=leg.legend_handles,
                labels=[t.get_text() for t in leg.get_texts()],
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                bbox_transform=bg_ax_leg.transAxes,
                fontsize=leg.get_texts()[0].get_fontsize() if leg.get_texts() else 12,
                frameon=True, framealpha=0.9,
            )
        unfaded_outside_png = out_mp4.with_name(out_mp4.stem.replace("curvature_animation", "background_unfaded_legend_outside") + ".png")
        ref_fig.savefig(str(unfaded_outside_png), bbox_inches="tight", dpi=dpi)
        print(f"  Saved unfaded background (legend outside): {unfaded_outside_png.name}")

    trace_only_png = out_mp4.with_name(out_mp4.stem.replace("curvature_animation", "trace_only_static") + ".png")
    _save_trace_only_still(
        out_png=trace_only_png,
        ref_fig=ref_fig,
        ref_ax=ref_ax,
        featured_df=featured_df,
        feature_col=feature_col,
        featured_color_hex=featured_color_hex,
        smooth_sigma=smooth_sigma,
        extend_featured_trace=extend_featured_trace,
        t_max=t_max,
        trace_outline=trace_outline,
    )
    print(f"  Saved trace-only still: {trace_only_png.name}")

    # Remove legend entirely for the video — cleaner look
    leg_out = bg_ax_leg.get_legend()
    if leg_out is not None:
        leg_out.remove()

    if plot_style == "trace_only":
        plot_fig, bg_ax = _blank_trace_figure_from_reference(ref_fig, ref_ax)
        plt.close(ref_fig)
    else:
        plot_fig = ref_fig
        bg_ax = ref_ax

    # Auto-detect video size from the rendered canvas — no forced resize
    plot_fig.canvas.draw()
    canvas_w, canvas_h = plot_fig.canvas.get_width_height()
    plot_width, plot_height = int(canvas_w), int(canvas_h)
    print(f"  Video frame size: {plot_width}x{plot_height} px (auto-detected from canvas)")

    # Read y/x limits from the rendered figure (trust what the function set)
    x_lim = bg_ax.get_xlim()
    y_lim = bg_ax.get_ylim()
    y0, y1 = float(y_lim[0]), float(y_lim[1])

    bg_rgb = _fig_to_rgb_array(plot_fig).copy()
    bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
    plt.close(plot_fig)

    if plot_style == "background":
        white = np.full_like(bg_bgr, 255, dtype=np.uint8)
        cv2.addWeighted(bg_bgr, 0.35, white, 0.65, 0, bg_bgr)

    # --- Precompute featured trace in pixel space using the baked axes transform ---
    extend_to = float(t_max) if extend_featured_trace else None
    featured_times, featured_vals = _prepare_featured_trace(
        featured_df,
        feature_col,
        smooth_sigma,
        extend_to=extend_to,
    )

    if featured_times.size >= 2:
        px_all, py_all = _ax_data_to_pixel(bg_ax, featured_times, featured_vals)
    else:
        px_all = py_all = np.array([], dtype=np.float32)

    def _t_to_px(t_val: float) -> int:
        xy = bg_ax.transData.transform([[float(t_val), y0]])
        return int(round(xy[0, 0]))

    # Axes bounding box in pixel coords (image origin = top-left)
    fig_h_px = plot_height
    bbox = bg_ax.get_window_extent()  # display coords, origin bottom-left
    ax_px_x0 = int(round(bbox.x0))
    ax_px_x1 = int(round(bbox.x1))
    ax_px_y0 = int(round(fig_h_px - bbox.y1))  # flip to image coords (top)
    ax_px_y1 = int(round(fig_h_px - bbox.y0))  # flip to image coords (bottom)

    feat_bgr = _hex_to_bgr(featured_color_hex)

    out_times = np.linspace(float(cursor_min), float(cursor_max), int(n_frames_out))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, float(fps), (int(plot_width), int(plot_height)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_mp4}")

    try:
        for t in out_times:
            frame = bg_bgr.copy()
            t_f = float(t)

            # Draw the trace only where there's actual data
            if px_all.size >= 2:
                m = featured_times <= t_f
                if not m.any():
                    m[0] = True
                px_vis = px_all[m].astype(np.int32)
                py_vis = py_all[m].astype(np.int32)

                if px_vis.size >= 2:
                    pts = np.column_stack([px_vis, py_vis]).reshape(-1, 1, 2)
                    # Halo
                    halo_frame = frame.copy()
                    cv2.polylines(halo_frame, [pts], isClosed=False, color=feat_bgr,
                                  thickness=7, lineType=cv2.LINE_AA)
                    cv2.addWeighted(halo_frame, 0.18, frame, 0.82, 0, frame)
                    # Optional black outline stroke, then colored line on top
                    if bool(trace_outline):
                        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 0),
                                      thickness=3, lineType=cv2.LINE_AA)
                    cv2.polylines(frame, [pts], isClosed=False, color=feat_bgr,
                                  thickness=2, lineType=cv2.LINE_AA)

                if px_vis.size >= 1:
                    if plot_style != "trace_only":
                        tip_x, tip_y = int(px_vis[-1]), int(py_vis[-1])
                        outer = (0, 0, 0) if bool(trace_outline) else (255, 255, 255)
                        cv2.circle(frame, (tip_x, tip_y), 6, outer, -1, lineType=cv2.LINE_AA)
                        cv2.circle(frame, (tip_x, tip_y), 5, feat_bgr, -1, lineType=cv2.LINE_AA)

            # Cursor only visible while within the embryo's data range
            feat_t_max = float(featured_times[-1]) if featured_times.size > 0 else t_max
            if t_f <= feat_t_max:
                cx = _t_to_px(t_f)
                cx = max(ax_px_x0, min(ax_px_x1, cx))
                cv2.line(frame, (cx, ax_px_y0), (cx, ax_px_y1), feat_bgr, 1, lineType=cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()


def _load_snip_frame(
    snip_root: Path,
    experiment_date: str,
    embryo_id: str,
    row: "pd.Series",
    last_img: Optional["np.ndarray"],
) -> Optional["np.ndarray"]:
    """Load one snip frame (raw size). Returns None only if missing and no fallback."""
    import cv2

    frame_index = int(row["frame_index"])
    p = _snip_path(snip_root, experiment_date, embryo_id, frame_index)
    if not p.exists():
        return last_img

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return last_img

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr


def _make_embryo_video(
    out_mp4: Path,
    featured_df: pd.DataFrame,
    snip_root: Path,
    genotype_color_hex: str,
    t_min: float,
    t_max: float,
    cursor_min: float,
    cursor_max: float,
    fps: int,
    n_frames_out: int,
    hold_last_frame: bool,
) -> tuple[int, int]:
    """Render embryo snip video using raw JPEG dimensions with HPF label overlaid.

    The video dimensions match the raw snip JPEGs (e.g. 256x576). The HPF label
    is drawn directly on top of each frame with a semi-transparent background strip.

    Returns (video_width, video_height) so the caller can report dimensions.
    """
    import cv2

    featured_df = featured_df.sort_values("predicted_stage_hpf").copy()
    track = build_embryo_track(featured_df)
    times = track.times_hpf
    color_bgr = _hex_to_bgr(genotype_color_hex)

    # HPF label style
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    print("  Rendering embryo sequence from shared embryo renderer...")
    sequence = render_embryo_sequence(
        track,
        snip_root=snip_root,
        start_hpf=float(cursor_min),
        end_hpf=float(cursor_max),
        fps=int(fps),
        n_frames_out=int(n_frames_out),
        missing_frame_bgr=(250, 250, 250),
        pad_to_even=True,
    )

    def _overlay_hpf_label(frame: np.ndarray, t: float) -> np.ndarray:
        canvas = frame.copy()
        vid_h, vid_w = canvas.shape[:2]
        display_t = min(float(t), float(times[-1])) if times.size > 0 else float(t)
        label = f"{display_t:.1f} hpf"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        pad_y = 10
        strip_h = th + baseline + (2 * pad_y)
        box_alpha = 0.55
        box_color = (235, 235, 235)

        text_x = (vid_w - tw) // 2
        text_y = (strip_h + th) // 2 - baseline // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (int(vid_w - 1), int(min(vid_h - 1, strip_h))), box_color, -1)
        cv2.addWeighted(overlay, float(box_alpha), canvas, float(1.0 - box_alpha), 0, canvas)

        _put_text_with_outline(
            canvas,
            label,
            (int(text_x), int(text_y)),
            font=font,
            font_scale=float(font_scale),
            fg_bgr=color_bgr,
            thickness=int(font_thickness),
            outline_bgr=(0, 0, 0),
            outline_thickness=int(font_thickness) + 3,
        )
        return canvas

    return export_embryo_video(
        sequence,
        out_mp4,
        frame_transform=_overlay_hpf_label,
    )


def main() -> None:
    args = _parse_args()
    paths = _resolve_paths(args)

    sys.path.insert(0, str(paths.project_root / "src"))
    from analyze.trajectory_analysis.viz.styling import get_color_for_genotype
    from analyze.utils.stats import normalize_arbitrary_feature
    from analyze.viz.styling.color_mapping_config import GENOTYPE_COLORS

    # Panel keys
    if args.panel_by == "genotype":
        panel_keys = [s.strip() for s in args.genotypes.split(",") if s.strip()]
    else:
        panel_keys = [s.strip() for s in args.cluster_categories.split(",") if s.strip()]

    # Parse per-panel featured embryo IDs (if provided)
    featured_ids_map: dict[str, Optional[str]] = {}
    if args.featured_embryo_ids is not None:
        ids_list = [s.strip() for s in args.featured_embryo_ids.split(",")]
        if len(ids_list) != len(panel_keys):
            raise ValueError(
                f"--featured-embryo-ids has {len(ids_list)} entries but panel list has {len(panel_keys)}. "
                "They must match in length."
            )
        for key, eid in zip(panel_keys, ids_list):
            featured_ids_map[key] = None if eid.lower() == "auto" else eid
    elif args.featured_embryo_id is not None:
        # Legacy single-embryo override: apply to matching genotype only
        for key in panel_keys:
            featured_ids_map[key] = None
        # We'll check which genotype this embryo belongs to after loading data
        legacy_featured_id = args.featured_embryo_id
    else:
        for key in panel_keys:
            featured_ids_map[key] = None
        legacy_featured_id = None

    # Parse ylim
    ylim: Optional[tuple[float, float]] = None
    if args.ylim is not None:
        parts = [float(x.strip()) for x in args.ylim.split(",")]
        if len(parts) != 2:
            raise ValueError(f"--ylim must be two comma-separated floats, got: {args.ylim!r}")
        ylim = (parts[0], parts[1])

    # Determine the raw feature column needed from the CSV
    raw_feature_col = "baseline_deviation_normalized"
    use_derived_curvature = args.feature_col == "curvature"
    feature_col = args.feature_col

    # Load data
    cluster_cat = None if args.no_cluster_filter else args.cluster_category
    df = _load_embryo_frames(
        data_dir=paths.data_dir,
        raw_feature_col=raw_feature_col,
        cluster_category=cluster_cat,
    )

    # Derive the "curvature" column if needed
    if use_derived_curvature:
        df["curvature"] = normalize_arbitrary_feature(
            df[raw_feature_col],
            low=0,
            high_percentile=100,
            clip=False,
        )
        print(f"Derived 'curvature' column from '{raw_feature_col}' via normalize_arbitrary_feature(low=0, high_percentile=100, clip=False)")
        if ylim is None:
            ylim = (0.0, 1.0)

    # Basic range filter
    df = df[df["predicted_stage_hpf"].notna()].copy()

    # Resolve legacy --featured-embryo-id to the correct genotype
    if args.featured_embryo_ids is None and args.featured_embryo_id is not None:
        legacy_featured_id = args.featured_embryo_id
        match = df[df["embryo_id"].astype(str) == str(legacy_featured_id)]
        if not match.empty:
            suffix = _genotype_suffix(str(match["genotype"].iloc[0]))
            if args.panel_by == "genotype":
                if suffix in featured_ids_map:
                    featured_ids_map[suffix] = legacy_featured_id
            else:
                cat = str(match["cluster_categories"].iloc[0])
                if cat in featured_ids_map:
                    featured_ids_map[cat] = legacy_featured_id

    # Shared dimensions
    n_frames = max(2, int(round(int(args.n_frames_out) / float(args.speed))))
    default_w, default_h = _default_plot_px()
    out_w = int(args.plot_width) if args.plot_width is not None else default_w
    out_h = int(args.plot_height) if args.plot_height is not None else default_h
    cursor_min = float(args.cursor_min) if args.cursor_min is not None else float(args.t_min)
    cursor_max = float(args.cursor_max) if args.cursor_max is not None else float(args.t_max)
    print(f"Plot dimensions: {out_w}x{out_h} px"
          + (" (from faceting engine config)" if args.plot_width is None else " (user override)"))
    print(f"Data window: {args.t_min}--{args.t_max} HPF, cursor scans {cursor_min}--{cursor_max} HPF")
    print(f"Rendering {n_frames} frames at {args.fps} fps "
          f"({n_frames / args.fps:.1f}s) -- speed x{args.speed}")
    print(f"Plot style: {args.plot_style}")
    if args.plot_style == "trace_only":
        print("Trace-only x-axis: auto-resolved from the rendered reference figure")
    print(f"Trend bin width: {args.bin_width} HPF")
    if ylim is not None:
        print(f"ylim: {ylim}")
    print(f"Panel by: {args.panel_by}")
    print(f"Panels: {panel_keys}")
    print()

    saved_files: list[str] = []

    PHENOTYPE_COLORS = {
        "High_to_Low": "#E76FA2",     # rose
        "Low_to_High": "#2FB7B0",     # teal
        "Not Penetrant": "#3A3A3A",   # charcoal
    }

    for panel_key in panel_keys:
        if args.panel_by == "genotype":
            genotype_suffix = panel_key
            full_genotype = f"cep290_{genotype_suffix}"
            panel_label = f"{genotype_suffix} ({full_genotype})"
            df_panel = df[df["genotype"] == full_genotype].copy()
            if df_panel.empty:
                print(f"  WARNING: No data for genotype '{full_genotype}', skipping.")
                continue
            bg_color_by = "genotype"
            bg_color_lookup = GENOTYPE_COLORS
            featured_color_hex = get_color_for_genotype(full_genotype)
            prefer_suffix = genotype_suffix
            out_prefix = genotype_suffix
        else:
            cluster_category = panel_key
            panel_label = f"{cluster_category}"
            df_panel = df[df["cluster_categories"] == cluster_category].copy()
            if args.filter_genotype:
                full_fg = f"cep290_{args.filter_genotype}"
                df_panel = df_panel[df_panel["genotype"].isin([args.filter_genotype, full_fg])].copy()
            if df_panel.empty:
                print(f"  WARNING: No data for cluster_categories '{cluster_category}', skipping.")
                continue
            bg_color_by = "cluster_categories"
            bg_color_lookup = PHENOTYPE_COLORS
            featured_color_hex = PHENOTYPE_COLORS.get(cluster_category, "#808080")
            prefer_suffix = None
            out_prefix = _safe_name(cluster_category)

        print(f"=== Panel: {panel_label} ===")

        # Pick featured embryo for this panel. Use the cursor window for selection so
        # you can keep a wide plot x-axis (e.g. 24–120) while selecting embryos that
        # start near --cursor-min (e.g. ~40) and have good coverage afterwards.
        pick_t_min = max(float(args.t_min), float(cursor_min))
        pick_t_max = min(float(args.t_max), float(cursor_max))

        explicit_id = featured_ids_map.get(panel_key, None)
        n_want = max(1, int(args.n_featured_per_panel))
        selected_ids: list[str] = []
        exclude_ids: set[str] = set()
        if explicit_id is not None:
            selected_ids.append(str(explicit_id))
            exclude_ids.add(str(explicit_id))

        rank_df = _rank_featured_candidates(
            df=df_panel,
            feature_col=feature_col,
            t_min=pick_t_min,
            t_max=pick_t_max,
            snip_root=paths.snip_root,
            min_snips=int(args.min_featured_snips),
            min_min_hpf=float(args.min_featured_min_hpf) if args.min_featured_min_hpf is not None else None,
            min_max_hpf=float(args.min_featured_max_hpf),
            prefer_suffix=prefer_suffix,
            candidate_end_n=int(args.candidate_end_n),
            min_end_value=float(args.min_featured_end_value) if args.min_featured_end_value is not None else None,
            exclude_embryo_ids=exclude_ids,
            top_k=max(30, n_want * 10),
        )

        if len(selected_ids) < n_want:
            ranked_ids = rank_df["embryo_id"].astype(str).tolist() if not rank_df.empty else []
            for eid in ranked_ids:
                if eid not in exclude_ids and eid not in selected_ids:
                    selected_ids.append(eid)
                if len(selected_ids) >= n_want:
                    break

        if not selected_ids:
            # Fallback to legacy picker if ranking produced nothing
            selected_ids = [
                _pick_featured_embryo(
                    df=df_panel,
                    feature_col=feature_col,
                    t_min=pick_t_min,
                    t_max=pick_t_max,
                    snip_root=paths.snip_root,
                    min_snips=int(args.min_featured_snips),
                    min_min_hpf=float(args.min_featured_min_hpf) if args.min_featured_min_hpf is not None else None,
                    min_max_hpf=float(args.min_featured_max_hpf),
                    prefer_suffix=prefer_suffix,
                    explicit_embryo_id=None,
                )
            ]

        if len(selected_ids) > 1:
            print(f"  Selected {len(selected_ids)} featured embryos:")
            if not rank_df.empty:
                show = rank_df[rank_df["embryo_id"].astype(str).isin(set(selected_ids))].copy()
                show = show.sort_values(["end_value", "max_hpf", "n_rows"], ascending=[False, False, False])
                for r in show.itertuples(index=False):
                    print(
                        f"   - {r.embryo_id}  end={float(r.end_value):.3f}  "
                        f"hpf={float(r.min_hpf):.1f}-{float(r.max_hpf):.1f}  rows={int(r.n_rows)}  {r.genotype}"
                    )
            else:
                for eid in selected_ids:
                    print(f"   - {eid}")

        for featured_id in selected_ids:
            featured_df = df_panel[df_panel["embryo_id"].astype(str) == str(featured_id)].copy()
            featured_df = featured_df.sort_values("predicted_stage_hpf")
            in_window = featured_df["predicted_stage_hpf"].between(pick_t_min, float(args.t_max), inclusive="both")
            if in_window.any():
                featured_df = featured_df.loc[in_window].copy()

            if featured_df.empty:
                print(f"  WARNING: Featured embryo {featured_id} has no rows in time window, skipping.")
                continue

            plot_mp4 = paths.figures_dir / f"curvature_animation_{out_prefix}_{featured_id}.mp4"
            embryo_mp4 = paths.figures_dir / f"embryo_animation_{out_prefix}_{featured_id}.mp4"

            print(f"  Featured embryo: {featured_id}")
            print(f"  Overlay color: {featured_color_hex}")
            print(f"  Rows in window: {len(featured_df)}")
            print(f"  HPF range: {featured_df['predicted_stage_hpf'].min():.2f} .. {featured_df['predicted_stage_hpf'].max():.2f}")
            print(f"  Background embryos: {df_panel['embryo_id'].nunique()}")

            _make_plot_video(
                out_mp4=plot_mp4,
                df=df_panel,
                featured_df=featured_df,
                feature_col=feature_col,
                featured_color_hex=featured_color_hex,
                color_by=bg_color_by,
                color_lookup=bg_color_lookup,
                t_min=float(args.t_min),
                t_max=float(args.t_max),
                cursor_min=cursor_min,
                cursor_max=cursor_max,
                fps=int(args.fps),
                n_frames_out=n_frames,
                plot_width=out_w,
                plot_height=out_h,
                background_max_embryos=int(args.background_max_embryos),
                seed=int(args.seed),
                extend_featured_trace=bool(args.extend_featured_trace),
                smooth_sigma=float(args.smooth_sigma),
                trace_outline=bool(args.trace_outline),
                trend_linestyle=args.trend_linestyle,
                bin_width=float(args.bin_width),
                plot_style=args.plot_style,
                ylim=ylim,
            )

            saved_files.append(str(plot_mp4))
            trace_png = plot_mp4.with_name(plot_mp4.stem.replace("curvature_animation", "trace_only_static") + ".png")
            if trace_png.exists():
                saved_files.append(str(trace_png))

            if bool(args.skip_embryo_video):
                print("  Skipped embryo video (--skip-embryo-video)")
            else:
                emb_w, emb_h = _make_embryo_video(
                    out_mp4=embryo_mp4,
                    featured_df=featured_df,
                    snip_root=paths.snip_root,
                    genotype_color_hex=featured_color_hex,
                    t_min=float(args.t_min),
                    t_max=float(args.t_max),
                    cursor_min=cursor_min,
                    cursor_max=cursor_max,
                    fps=int(args.fps),
                    n_frames_out=n_frames,
                    hold_last_frame=bool(args.hold_last_frame),
                )
                print(f"  Embryo video dimensions: {emb_w}x{emb_h} (raw JPEG size)")
                saved_files.append(str(embryo_mp4))
            print()

    print("Saved:")
    for f in saved_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
