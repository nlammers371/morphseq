"""
07_focus_analysis.py
====================
Per-embryo focus analysis using saved NCC/entropy grids + SAM2 masks.

For each (t, p) with an existing grid and mask, this script:
  - loads the entropy grid saved during the motion QC scan
  - estimates a background entropy reference from tiles outside the mask
  - computes mask-aware entropy + relative entropy summaries
  - optionally merges the motion QC summaries for context
  - writes 07_focus_output/focus_summaries.csv
  - renders quick diagnostic plots

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/07_focus_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

from src.data_pipeline.quality_control.zstack_motion_qc import (
    load_grids,
    embryo_entropy_summary,
    tile_origin_coords,
)


BASE_DIR = Path(__file__).resolve().parent
MOTION_DIR = BASE_DIR.parent / "20260421_motion_artifact_detection"
GRIDS_DIR = MOTION_DIR / "06_scan_output/grids"
SERIES_MAP = MOTION_DIR / "06_scan_output/series_well_map.csv"
MOTION_SUMMARY_CSV = MOTION_DIR / "07_embryo_ncc_output/embryo_ncc_summaries.csv"
LOOKUP_CSV = MORPHSEQ_ROOT / "docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv"
MASKS_DIR = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
OUT_DIR = BASE_DIR / "07_focus_output"
CSV_PATH = OUT_DIR / "focus_summaries.csv"

TILE_COVERAGE_MIN = 0.25
BG_COVERAGE_MAX = 0.05
MIN_BG_TILES = 4

LABEL_COLORS = {
    "Bad Images": "#d62728",
    "Great Images": "#2ca02c",
    "Okay Images": "#ff7f0e",
}


def load_series_map() -> dict[int, str]:
    df = pd.read_csv(SERIES_MAP)
    return dict(zip(df["series_number"], df["well_index"]))


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array(Image.open(path)) > 0


def mask_tile_weights(mask: np.ndarray, tile_size: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_origins, x_origins = tile_origin_coords(mask.shape, tile_size, stride)
    weights = np.zeros((len(y_origins), len(x_origins)), dtype=np.float32)
    for iy, y0 in enumerate(y_origins):
        y1 = min(y0 + tile_size, mask.shape[0])
        for ix, x0 in enumerate(x_origins):
            x1 = min(x0 + tile_size, mask.shape[1])
            weights[iy, ix] = float(mask[y0:y1, x0:x1].mean())
    return weights, y_origins, x_origins


def estimate_background_entropy_grid(
    entropy_grid: np.ndarray,
    mask: np.ndarray,
    tile_size: int,
    stride: int,
) -> tuple[np.ndarray, dict]:
    """
    Estimate a per-slice background reference from tiles outside the embryo mask.

    The returned grid has the same shape as `entropy_grid` and contains the
    same per-slice background value broadcast across the tile plane.
    """
    weights, _, _ = mask_tile_weights(mask, tile_size, stride)

    bg_valid = weights <= BG_COVERAGE_MAX
    if int(bg_valid.sum()) < MIN_BG_TILES:
        bg_valid = weights < TILE_COVERAGE_MIN
    if int(bg_valid.sum()) < MIN_BG_TILES:
        flat = weights.ravel()
        n_pick = min(max(MIN_BG_TILES, flat.size // 5), flat.size)
        if n_pick > 0:
            pick = np.argsort(flat)[:n_pick]
            bg_valid = np.zeros_like(weights, dtype=bool)
            bg_valid.ravel()[pick] = True

    if not bg_valid.any():
        bg_valid = np.ones_like(weights, dtype=bool)

    bg_slice_vals = np.nanmean(entropy_grid[:, bg_valid], axis=1)
    bg_grid = np.broadcast_to(bg_slice_vals[:, None, None], entropy_grid.shape).astype(np.float32).copy()

    stats = {
        "bg_n_tiles": int(bg_valid.sum()),
        "bg_tile_frac": float(bg_valid.mean()),
        "bg_entropy_mean": float(np.nanmean(bg_slice_vals)),
        "bg_entropy_min": float(np.nanmin(bg_slice_vals)),
        "bg_entropy_std": float(np.nanstd(bg_slice_vals)),
    }
    return bg_grid, stats


def load_labeled_lookup() -> pd.DataFrame | None:
    if not LOOKUP_CSV.exists():
        return None
    lk = pd.read_csv(LOOKUP_CSV)
    lk["p"] = lk["nd2_series_num"] - 1
    lk["t"] = lk["time_int"]
    return lk[["t", "p", "category"]].copy()


def load_motion_summary() -> pd.DataFrame | None:
    if not MOTION_SUMMARY_CSV.exists():
        return None
    df = pd.read_csv(MOTION_SUMMARY_CSV)
    keep = [
        "t",
        "p",
        "well",
        "ncc_min",
        "ncc_p05",
        "ncc_mean",
        "bad_pair_frac",
        "ncc_bad_tile_frac",
        "local_ncc_std_mean",
        "longest_bad_run",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()


def iter_stack_paths() -> list[Path]:
    return sorted(GRIDS_DIR.glob("t*.npz"))


def run_analysis() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series_map = load_series_map()
    p_to_well = {sn - 1: well for sn, well in series_map.items()}
    motion_df = load_motion_summary()
    label_df = load_labeled_lookup()

    rows: list[dict] = []
    n_no_mask = 0
    n_no_grid = 0

    npz_files = iter_stack_paths()
    print(f"Found {len(npz_files)} grid files")

    for npz_path in npz_files:
        stem = npz_path.stem  # t003_p007
        t = int(stem[1:4])
        p = int(stem[6:9])
        well = p_to_well.get(p)
        if well is None:
            continue

        mask_path = MASKS_DIR / f"20250912_{well}_ch00_t{t:04d}_masks_emnum_1.png"
        mask = load_mask(mask_path)
        if mask is None:
            n_no_mask += 1
            continue

        try:
            data = load_grids(npz_path)
        except Exception as exc:
            print(f"  SKIP corrupt grid {npz_path.name}: {exc}")
            n_no_grid += 1
            continue

        entropy_grid = data["entropy_grid"]
        tile_size = int(data["tile_size"])
        stride = int(data["stride"])

        bg_grid, bg_stats = estimate_background_entropy_grid(
            entropy_grid=entropy_grid,
            mask=mask,
            tile_size=tile_size,
            stride=stride,
        )

        summary = embryo_entropy_summary(
            entropy_grid=entropy_grid,
            bg_entropy_grid=bg_grid,
            mask=mask,
            tile_size=tile_size,
            stride=stride,
            min_tile_coverage=TILE_COVERAGE_MIN,
        )

        row = {
            "t": t,
            "p": p,
            "well": well,
            "tile_size": tile_size,
            "stride": stride,
            **bg_stats,
            **summary,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["t", "p"]).reset_index(drop=True)
    if motion_df is not None and not motion_df.empty:
        df = df.merge(motion_df, on=["t", "p", "well"], how="left")
    if label_df is not None and not label_df.empty:
        df = df.merge(label_df, on=["t", "p"], how="left")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} focus summaries -> {CSV_PATH}")
    if n_no_mask:
        print(f"  ({n_no_mask} stacks skipped - no mask file)")
    if n_no_grid:
        print(f"  ({n_no_grid} stacks skipped - corrupt grid)")

    return df


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, title in [
        (axes[0], "rel_entropy_mean", "Relative entropy mean (mask-aware)"),
        (axes[1], "entropy_mean", "Entropy mean (mask-aware)"),
    ]:
        vals = df[col].dropna()
        ax.hist(vals, bins=60, color="#4c72b0", edgecolor="none", alpha=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(title)

        if "category" in df.columns:
            for cat, grp in df.groupby("category"):
                color = LABEL_COLORS.get(cat, "black")
                for val in grp[col].dropna():
                    ax.axvline(val, color=color, alpha=0.35, linewidth=0.8)

    fig.suptitle("Focus QC distributions", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))

    scatter_df = df.dropna(subset=["rel_entropy_mean"]).copy()
    if "ncc_min" in scatter_df.columns and scatter_df["ncc_min"].notna().any():
        scatter_df = scatter_df.dropna(subset=["ncc_min"])
        ax.hexbin(scatter_df["ncc_min"], scatter_df["rel_entropy_mean"], gridsize=55, cmap="Blues", mincnt=1)
    else:
        ax.scatter(
            np.arange(len(scatter_df)),
            scatter_df["rel_entropy_mean"],
            s=18,
            alpha=0.6,
            color="#4c72b0",
        )

    if "category" in df.columns:
        for cat, grp in df.groupby("category"):
            color = LABEL_COLORS.get(cat, "black")
            sub = grp.dropna(subset=["rel_entropy_mean"])
            if "ncc_min" in sub.columns:
                ax.scatter(sub["ncc_min"], sub["rel_entropy_mean"], s=55, edgecolors="k",
                           linewidths=0.5, color=color, label=cat, zorder=5)
            else:
                ax.scatter(np.arange(len(sub)), sub["rel_entropy_mean"], s=55, edgecolors="k",
                           linewidths=0.5, color=color, label=cat, zorder=5)

    if "ncc_min" in df.columns and df["ncc_min"].notna().any():
        ax.set_xlabel("ncc_min")
        ax.axvline(0.85, color="red", linestyle="--", linewidth=1, label="NCC threshold (0.85)")
    else:
        ax.set_xlabel("stack index")

    ax.set_ylabel("rel_entropy_mean")
    ax.set_title("Focus QC: NCC vs relative entropy")
    if "category" in df.columns:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "scatter_rel_entropy_vs_ncc_min.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def print_examples(df: pd.DataFrame) -> None:
    cols = ["t", "p", "well", "rel_entropy_mean", "rel_entropy_min", "entropy_mean"]
    if "ncc_min" in df.columns:
        cols.insert(3, "ncc_min")

    cohorts = [
        ("High focus (best rel_entropy)", df.nlargest(5, "rel_entropy_mean")),
        ("Low focus (worst rel_entropy)", df.nsmallest(5, "rel_entropy_mean")),
    ]
    print("\n=== Example focus cohorts ===")
    for label, sub in cohorts:
        print(f"\n--- {label} ---")
        print(sub[cols].to_string(index=False))


def main() -> None:
    df = run_analysis()
    plot_distributions(df)
    plot_scatter(df)
    print_examples(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
