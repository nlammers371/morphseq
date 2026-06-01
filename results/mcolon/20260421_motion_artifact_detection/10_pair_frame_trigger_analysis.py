"""
10_pair_frame_trigger_analysis.py
=================================
Inspect per-Z-pair NCC severity for selected (well, t) examples to explain why
bad_pair_frac can be noisy/coarse.

Outputs:
  figures/threshold_bins/v3_bad_pair_frac_analysis/pair_trigger_examples/
    - pair_severity_all_examples.csv
    - pair_trigger_summary.md
    - <well>_t<tttt>_pair_profile.png
    - <well>_t<tttt>_pair_values.csv

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260421_motion_artifact_detection/10_pair_frame_trigger_analysis.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

from src.data_pipeline.quality_control.zstack_motion_qc import load_grids, tile_origin_coords


BASE_DIR = Path(__file__).parent
SERIES_MAP_CSV = BASE_DIR / "06_scan_output/series_well_map.csv"
GRIDS_DIR = BASE_DIR / "06_scan_output/grids"
SUMMARY_CSV = BASE_DIR / "07_embryo_ncc_output/embryo_ncc_summaries.csv"
MASKS_DIR = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
OUT_DIR = BASE_DIR / "figures/threshold_bins/v3_bad_pair_frac_analysis/pair_trigger_examples"

BAD_THRESH = 0.90
MIN_TILE_COVERAGE = 0.25

EXAMPLES = [
    ("E11", 230),
    ("A10", 98),
    ("E11", 79),
    ("C04", 11),
    ("D05", 17),
]


def load_summary_lookup() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_CSV)


def load_well_to_p() -> dict[str, int]:
    df = pd.read_csv(SERIES_MAP_CSV)
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        out[str(row["well_index"])] = int(row["series_number"]) - 1
    return out


def load_mask(well: str, t: int) -> np.ndarray | None:
    path = MASKS_DIR / f"20250912_{well}_ch00_t{t:04d}_masks_emnum_1.png"
    if not path.exists():
        return None
    arr = np.array(Image.open(path))
    return arr > 0


def mask_valid_tiles(mask: np.ndarray, tile_size: int, stride: int, min_coverage: float) -> np.ndarray:
    y_origins, x_origins = tile_origin_coords(mask.shape, tile_size, stride)
    valid = np.zeros((len(y_origins), len(x_origins)), dtype=bool)
    for iy, y0 in enumerate(y_origins):
        y1 = min(y0 + tile_size, mask.shape[0])
        for ix, x0 in enumerate(x_origins):
            x1 = min(x0 + tile_size, mask.shape[1])
            frac = float(mask[y0:y1, x0:x1].mean())
            valid[iy, ix] = frac >= min_coverage
    return valid


def summarize_pairs(ncc_grid: np.ndarray, valid_tiles: np.ndarray) -> pd.DataFrame:
    masked_grid = ncc_grid[:, valid_tiles]
    rows = []
    for pair_idx in range(masked_grid.shape[0]):
        vals = masked_grid[pair_idx]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            rows.append(
                {
                    "pair_idx": pair_idx,
                    "pair_label": f"z{pair_idx}-z{pair_idx+1}",
                    "pair_mean_ncc": np.nan,
                    "pair_p05_ncc": np.nan,
                    "pair_min_ncc": np.nan,
                    "pair_bad_tile_frac": np.nan,
                    "pair_n_valid_tiles": 0,
                    "pair_is_bad": True,
                }
            )
            continue

        pair_mean = float(np.mean(vals))
        rows.append(
            {
                "pair_idx": pair_idx,
                "pair_label": f"z{pair_idx}-z{pair_idx+1}",
                "pair_mean_ncc": pair_mean,
                "pair_p05_ncc": float(np.percentile(vals, 5)),
                "pair_min_ncc": float(np.min(vals)),
                "pair_bad_tile_frac": float(np.mean(vals < BAD_THRESH)),
                "pair_n_valid_tiles": int(vals.size),
                "pair_is_bad": bool(pair_mean < BAD_THRESH),
            }
        )
    return pd.DataFrame(rows)


def plot_pair_profile(df_pair: pd.DataFrame, well: str, t: int, summary_row: pd.Series | None) -> Path:
    out = OUT_DIR / f"{well}_t{t:04d}_pair_profile.png"
    x = df_pair["pair_idx"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(x, df_pair["pair_mean_ncc"], marker="o", label="pair_mean_ncc", color="#1f77b4")
    axes[0].plot(x, df_pair["pair_p05_ncc"], marker="s", label="pair_p05_ncc", color="#ff7f0e", alpha=0.9)
    axes[0].axhline(BAD_THRESH, color="red", ls="--", lw=1, label=f"bad_thresh={BAD_THRESH:.2f}")
    axes[0].set_ylabel("NCC")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_ylim(0, 1.01)

    colors = ["#d62728" if b else "#2ca02c" for b in df_pair["pair_is_bad"].tolist()]
    axes[1].bar(x, df_pair["pair_bad_tile_frac"], color=colors, alpha=0.8)
    axes[1].set_ylabel("bad_tile_frac")
    axes[1].set_xlabel("adjacent z-pair index")
    axes[1].set_ylim(0, 1.0)

    if summary_row is None:
        title = f"{well} t={t}"
    else:
        title = (
            f"{well} t={t} | ncc_p05={summary_row['ncc_p05']:.3f} | "
            f"bad_pair_frac={summary_row['bad_pair_frac']:.3f} | n_tiles={int(summary_row['n_tiles'])}"
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def find_summary_row(df_summary: pd.DataFrame, well: str, t: int) -> pd.Series | None:
    if df_summary.empty:
        return None
    sub = df_summary[(df_summary["well"] == well) & (df_summary["t"] == t)]
    if sub.empty:
        return None
    return sub.iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    well_to_p = load_well_to_p()
    df_summary = load_summary_lookup()

    all_rows: list[pd.DataFrame] = []
    notes: list[str] = []

    for well, t in EXAMPLES:
        p = well_to_p.get(well)
        if p is None:
            notes.append(f"- {well} t={t}: missing well in series map")
            continue

        npz_path = GRIDS_DIR / f"t{t:03d}_p{p:03d}.npz"
        if not npz_path.exists():
            notes.append(f"- {well} t={t}: missing grid ({npz_path.name})")
            continue

        mask = load_mask(well, t)
        if mask is None:
            notes.append(f"- {well} t={t}: missing mask PNG")
            continue

        data = load_grids(npz_path)
        ncc_grid = data["ncc_grid"]
        valid_tiles = mask_valid_tiles(
            mask=mask,
            tile_size=int(data["tile_size"]),
            stride=int(data["stride"]),
            min_coverage=MIN_TILE_COVERAGE,
        )

        if not valid_tiles.any():
            notes.append(f"- {well} t={t}: no valid mask-overlapping tiles at min_coverage={MIN_TILE_COVERAGE}")
            continue

        df_pair = summarize_pairs(ncc_grid=ncc_grid, valid_tiles=valid_tiles)
        summary_row = find_summary_row(df_summary, well, t)

        df_pair.insert(0, "well", well)
        df_pair.insert(1, "t", t)
        df_pair.insert(2, "p", p)
        df_pair.to_csv(OUT_DIR / f"{well}_t{t:04d}_pair_values.csv", index=False)
        plot_path = plot_pair_profile(df_pair, well, t, summary_row)

        bad_pairs = df_pair.loc[df_pair["pair_is_bad"], "pair_label"].tolist()
        bad_pairs_s = ", ".join(bad_pairs) if bad_pairs else "none"
        notes.append(f"- {well} t={t}: bad_pairs={len(bad_pairs)}/14 [{bad_pairs_s}] | plot={plot_path.name}")
        all_rows.append(df_pair)

    if all_rows:
        out_csv = OUT_DIR / "pair_severity_all_examples.csv"
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(out_csv, index=False)
        notes.insert(0, f"- saved pair severity table: {out_csv.name}")

    md_path = OUT_DIR / "pair_trigger_summary.md"
    md_text = "# Pair-trigger summary\n\n" + "\n".join(notes) + "\n"
    md_path.write_text(md_text)

    print(f"Saved outputs in: {OUT_DIR}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
