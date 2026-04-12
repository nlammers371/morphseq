"""
Generate static "complete track" images for candidate CEP290 Not-Penetrant embryos.

This is intended as a quick visual triage tool: produce one PNG per embryo showing
its full feature-over-time trace in a shared HPF window (default 24–120).

Outputs:
- figures/candidate_tracks/curvature_track_{embryo_id}.png
- figures/candidate_tracks/candidates_metrics.csv

Data sources (read-only):
- results/mcolon/20251229_cep290_phenotype_extraction/final_data/
  - embryo_data_with_labels.csv
  - embryo_cluster_labels.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateMetrics:
    embryo_id: str
    experiment_date: str
    genotype: str
    n_points: int
    t_min: float
    t_max: float
    span: float
    coverage_frac: float
    smooth_step_median: float
    smooth_residual_median: float


GENOTYPE_FIXES = {
    "cep290_unkown": "cep290_unknown",
    "cep290_homozyous": "cep290_homozygous",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate candidate complete-track images for Not Penetrant embryos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        default="results/mcolon/20251229_cep290_phenotype_extraction/final_data",
        help="Directory containing embryo_data_with_labels.csv and embryo_cluster_labels.csv.",
    )
    p.add_argument("--cluster-category", default="Not Penetrant", help="cluster_categories value to filter.")
    p.add_argument(
        "--feature-col",
        default="baseline_deviation_normalized",
        help="Feature column to plot (e.g. baseline_deviation_normalized, curvature, embedding_distance).",
    )
    p.add_argument("--t-min", type=float, default=24.0, help="Minimum HPF for window.")
    p.add_argument("--t-max", type=float, default=120.0, help="Maximum HPF for window.")
    p.add_argument("--bin-width", type=float, default=0.5, help="HPF bin width for smoothing/coverage metrics.")
    p.add_argument("--smooth-sigma-bins", type=float, default=2.0, help="Gaussian sigma in bins for smoothing.")
    p.add_argument("--min-points", type=int, default=60, help="Minimum points in window to consider candidate.")
    p.add_argument("--top-n", type=int, default=40, help="How many candidate images to write.")
    p.add_argument(
        "--prefer-genotype-suffix",
        default="wildtype",
        help="If set, rank embryos with this genotype suffix higher (e.g. wildtype/heterozygous/homozygous).",
    )
    p.add_argument(
        "--require-snips",
        action="store_true",
        help="Only keep embryos whose first/last snip JPG exists (useful if you plan to animate later).",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "figures" / "candidate_tracks"),
        help="Output directory for PNGs + metrics CSV.",
    )
    return p.parse_args()


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _genotype_suffix(genotype: str) -> str:
    s = str(genotype).strip().lower()
    if "_" in s:
        return s.split("_")[-1]
    return s


def _snip_path(snip_root: Path, experiment_date: str, embryo_id: str, frame_index: int) -> Path:
    return snip_root / str(experiment_date) / f"{embryo_id}_t{int(frame_index):04d}.jpg"


def _gaussian_smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if sigma is None or float(sigma) <= 0:
        return y.copy()
    sigma = float(sigma)
    radius = int(max(1, round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= float(w.sum())

    good = np.isfinite(y)
    y0 = np.where(good, y, 0.0)
    num = np.convolve(y0, w, mode="same")
    den = np.convolve(good.astype(float), w, mode="same")
    out = np.full_like(num, np.nan, dtype=float)
    m = den > 1e-8
    out[m] = num[m] / den[m]
    return out


def _bin_median(times: np.ndarray, values: np.ndarray, t_min: float, t_max: float, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(float(t_min), float(t_max) + float(bin_width), float(bin_width))
    centers = edges[:-1] + 0.5 * float(bin_width)
    binned = np.full((centers.size,), np.nan, dtype=float)

    if times.size == 0:
        return centers, binned

    idx = np.digitize(times, edges) - 1
    for i in range(centers.size):
        m = idx == i
        if not m.any():
            continue
        binned[i] = float(np.nanmedian(values[m]))
    return centers, binned


def _load_not_penetrant_frames(data_dir: Path, cluster_category: str, feature_col: str) -> pd.DataFrame:
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

    target = str(cluster_category).strip()
    keep_ids = set(labels.loc[labels["cluster_categories"] == target, "embryo_id"].astype(str).tolist())
    if not keep_ids:
        got = sorted(labels["cluster_categories"].dropna().astype(str).unique().tolist())[:30]
        raise ValueError(f"No embryos found for cluster_categories=={target!r}. Example categories: {got}")

    # Read only needed cols (header-driven).
    usecols = ["embryo_id", "experiment_date", "frame_index", "predicted_stage_hpf", "genotype", feature_col, "use_embryo_flag"]
    header = pd.read_csv(embryo_frames_path, nrows=0)
    existing = [c for c in usecols if c in header.columns]
    missing_critical = [c for c in ["embryo_id", "predicted_stage_hpf", "genotype", feature_col] if c not in existing]
    if missing_critical:
        raise ValueError(f"Missing required columns in embryo_data_with_labels.csv: {missing_critical}")

    df = pd.read_csv(embryo_frames_path, usecols=existing, low_memory=False)
    df["embryo_id"] = df["embryo_id"].astype(str)
    df = df[df["embryo_id"].isin(keep_ids)].copy()

    if "use_embryo_flag" in df.columns:
        use_flag = df["use_embryo_flag"]
        if use_flag.dtype == bool:
            df = df[use_flag].copy()
        else:
            df = df[use_flag.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])].copy()

    df["predicted_stage_hpf"] = _safe_float_series(df["predicted_stage_hpf"])
    df[feature_col] = _safe_float_series(df[feature_col])

    df["genotype"] = df["genotype"].astype("string").str.strip().str.lower().replace(GENOTYPE_FIXES)
    return df


def _compute_metrics(
    sub: pd.DataFrame,
    feature_col: str,
    t_min: float,
    t_max: float,
    bin_width: float,
    smooth_sigma_bins: float,
) -> CandidateMetrics | None:
    sub = sub[sub["predicted_stage_hpf"].between(t_min, t_max, inclusive="both")].copy()
    sub = sub[sub[feature_col].notna()].copy()
    if sub.empty:
        return None

    embryo_id = str(sub["embryo_id"].iloc[0])
    experiment_date = str(sub["experiment_date"].iloc[0]) if "experiment_date" in sub.columns else "unknown"
    genotype = str(sub["genotype"].iloc[0])

    sub = sub.sort_values("predicted_stage_hpf")
    times = sub["predicted_stage_hpf"].to_numpy(dtype=float)
    vals = sub[feature_col].to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(vals)
    times = times[finite]
    vals = vals[finite]
    if times.size < 2:
        return None

    t0, t1 = float(times.min()), float(times.max())
    span = float(t1 - t0)

    grid_t, binned = _bin_median(times, vals, t_min, t_max, bin_width)
    covered = np.isfinite(binned)
    coverage_frac = float(covered.mean()) if covered.size else 0.0
    smooth = _gaussian_smooth_1d(binned, smooth_sigma_bins)

    diffs = np.diff(smooth[np.isfinite(smooth)])
    smooth_step_median = float(np.nanmedian(np.abs(diffs))) if diffs.size else float("inf")

    res = binned - smooth
    smooth_residual_median = float(np.nanmedian(np.abs(res[np.isfinite(res)]))) if np.isfinite(res).any() else float("inf")

    return CandidateMetrics(
        embryo_id=embryo_id,
        experiment_date=experiment_date,
        genotype=genotype,
        n_points=int(times.size),
        t_min=t0,
        t_max=t1,
        span=span,
        coverage_frac=coverage_frac,
        smooth_step_median=smooth_step_median,
        smooth_residual_median=smooth_residual_median,
    )


def _plot_candidate(
    out_png: Path,
    sub: pd.DataFrame,
    feature_col: str,
    t_min: float,
    t_max: float,
    bin_width: float,
    smooth_sigma_bins: float,
    genotype_color: str,
    metrics: CandidateMetrics,
) -> None:
    sub = sub[sub["predicted_stage_hpf"].between(t_min, t_max, inclusive="both")].copy()
    sub = sub[sub[feature_col].notna()].copy()
    sub = sub.sort_values("predicted_stage_hpf")

    times = sub["predicted_stage_hpf"].to_numpy(dtype=float)
    vals = sub[feature_col].to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(vals)
    times = times[finite]
    vals = vals[finite]

    grid_t, binned = _bin_median(times, vals, t_min, t_max, bin_width)
    smooth = _gaussian_smooth_1d(binned, smooth_sigma_bins)

    yvals = vals[np.isfinite(vals)]
    if yvals.size:
        y0 = float(np.nanpercentile(yvals, 1))
        y1 = float(np.nanpercentile(yvals, 99))
        if not math.isfinite(y0) or not math.isfinite(y1) or y0 == y1:
            y0, y1 = float(np.nanmin(yvals)), float(np.nanmax(yvals))
        pad = 0.12 * (y1 - y0 if y1 > y0 else 1.0)
        y0 -= pad
        y1 += pad
    else:
        y0, y1 = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=160)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.6)

    ax.plot(times, vals, color=genotype_color, alpha=0.35, linewidth=1.2, zorder=2)
    ax.scatter(times, vals, color=genotype_color, alpha=0.25, s=10, linewidths=0, zorder=3)
    ax.plot(grid_t, smooth, color=genotype_color, alpha=0.95, linewidth=2.4, zorder=5)

    ax.set_xlim(float(t_min), float(t_max))
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Hours post fertilization (hpf)")
    ax.set_ylabel(str(feature_col))

    title = (
        f"{metrics.embryo_id}  |  {metrics.genotype}  |  "
        f"points={metrics.n_points}  |  range={metrics.t_min:.1f}-{metrics.t_max:.1f} hpf  |  "
        f"coverage={metrics.coverage_frac*100:.1f}%"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root / "src"))
    from analyze.viz.styling.color_mapping_config import GENOTYPE_SUFFIX_COLORS  # noqa: E402

    snip_root = project_root / "morphseq_playground" / "training_data" / "bf_embryo_snips"

    df = _load_not_penetrant_frames(data_dir=data_dir, cluster_category=args.cluster_category, feature_col=args.feature_col)
    df = df[df["predicted_stage_hpf"].between(args.t_min, args.t_max, inclusive="both")].copy()
    if df.empty:
        raise ValueError("No rows left after filtering to time window; check --t-min/--t-max and data.")

    prefer_suffix = str(args.prefer_genotype_suffix).strip().lower()

    metrics_list: list[CandidateMetrics] = []
    for embryo_id, sub in df.groupby("embryo_id", observed=True):
        m = _compute_metrics(
            sub=sub,
            feature_col=args.feature_col,
            t_min=args.t_min,
            t_max=args.t_max,
            bin_width=args.bin_width,
            smooth_sigma_bins=args.smooth_sigma_bins,
        )
        if m is None:
            continue
        if m.n_points < int(args.min_points):
            continue
        if args.require_snips and "frame_index" in sub.columns and "experiment_date" in sub.columns:
            sub2 = sub.sort_values("frame_index")
            exp_date = str(sub2["experiment_date"].iloc[0])
            fi0 = int(sub2["frame_index"].iloc[0])
            fi1 = int(sub2["frame_index"].iloc[-1])
            if not (_snip_path(snip_root, exp_date, str(embryo_id), fi0).exists() and _snip_path(snip_root, exp_date, str(embryo_id), fi1).exists()):
                continue
        metrics_list.append(m)

    if not metrics_list:
        raise ValueError("No candidates found after applying filters; try lowering --min-points or disabling --require-snips.")

    metrics_df = pd.DataFrame([m.__dict__ for m in metrics_list])
    metrics_df["suffix"] = metrics_df["genotype"].map(_genotype_suffix)
    metrics_df["prefer"] = (metrics_df["suffix"] == prefer_suffix).astype(int)

    # Rank: prefer suffix, maximize coverage + span + points, then smoothness (lower is better)
    metrics_df["coverage_score"] = metrics_df["coverage_frac"] * np.clip(metrics_df["span"], 0, None)
    metrics_df = metrics_df.sort_values(
        ["prefer", "coverage_score", "coverage_frac", "span", "n_points", "smooth_residual_median", "smooth_step_median"],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)

    metrics_csv = out_dir / "candidates_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    n_write = int(min(int(args.top_n), len(metrics_df)))
    print(f"Writing {n_write} candidate images to: {out_dir}")
    for _, row in metrics_df.head(n_write).iterrows():
        embryo_id = str(row["embryo_id"])
        sub = df[df["embryo_id"].astype(str) == embryo_id].copy()
        genotype = str(row["genotype"])
        suffix = _genotype_suffix(genotype)
        genotype_color = GENOTYPE_SUFFIX_COLORS.get(suffix, "#808080")

        m = CandidateMetrics(
            embryo_id=embryo_id,
            experiment_date=str(row["experiment_date"]),
            genotype=genotype,
            n_points=int(row["n_points"]),
            t_min=float(row["t_min"]),
            t_max=float(row["t_max"]),
            span=float(row["span"]),
            coverage_frac=float(row["coverage_frac"]),
            smooth_step_median=float(row["smooth_step_median"]),
            smooth_residual_median=float(row["smooth_residual_median"]),
        )
        out_png = out_dir / f"curvature_track_{embryo_id}.png"
        _plot_candidate(
            out_png=out_png,
            sub=sub,
            feature_col=args.feature_col,
            t_min=args.t_min,
            t_max=args.t_max,
            bin_width=args.bin_width,
            smooth_sigma_bins=args.smooth_sigma_bins,
            genotype_color=genotype_color,
            metrics=m,
        )

    print(f"Saved metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
