"""
07_embryo_ncc_analysis.py
=========================
Per-embryo NCC analysis using saved .npz grids + SAM2 masks.

For each (t, p) with an existing grid, loads the embryo mask and calls
embryo_ncc_summary — the mask-aware version that only scores tiles
overlapping the embryo. This is the first meaningful QC signal;
whole-frame metrics from 06 are not used for QC decisions.

Outputs:
  07_embryo_ncc_output/embryo_ncc_summaries.csv
  07_embryo_ncc_output/ncc_min_distribution.png
  07_embryo_ncc_output/bad_pair_frac_distribution.png
  07_embryo_ncc_output/scatter_ncc_min_vs_bad_frac.png

Usage:
  python 07_embryo_ncc_analysis.py
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
    embryo_ncc_summary,
)

GRIDS_DIR   = Path(__file__).parent / "06_scan_output/grids"
SERIES_MAP  = Path(__file__).parent / "06_scan_output/series_well_map.csv"
MASKS_DIR   = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
LOOKUP_CSV  = MORPHSEQ_ROOT / "docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv"
OUT_DIR     = Path(__file__).parent / "07_embryo_ncc_output"

BAD_THRESH  = 0.90
NCC_MIN_FAIL_THRESH      = 0.85
BAD_PAIR_FRAC_FAIL_THRESH = 0.10

LABEL_COLORS = {
    "Bad Images":   "#d62728",
    "Great Images": "#2ca02c",
    "Okay Images":  "#ff7f0e",
}


def load_series_map() -> dict[int, str]:
    """Returns {series_number (1-based) -> well_index} e.g. {1: 'A01', ...}"""
    df = pd.read_csv(SERIES_MAP)
    return dict(zip(df["series_number"], df["well_index"]))


def mask_path_for(well: str, t: int) -> Path:
    """e.g. well='B10', t=15 -> ..._B10_ch00_t0015_masks_emnum_1.png"""
    return MASKS_DIR / f"20250912_{well}_ch00_t{t:04d}_masks_emnum_1.png"


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = np.array(Image.open(path))
    # masks are stored as uint8; nonzero = embryo
    return img > 0


def run_analysis() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series_map = load_series_map()     # series_num (1-based) -> well
    p_to_well  = {sn - 1: well for sn, well in series_map.items()}  # 0-based p -> well

    npz_files = sorted(GRIDS_DIR.glob("t*.npz"))
    print(f"Found {len(npz_files)} grid files")

    rows = []
    n_no_mask = 0

    for npz_path in npz_files:
        stem = npz_path.stem   # e.g. t003_p007
        t = int(stem[1:4])
        p = int(stem[6:9])
        well = p_to_well.get(p)
        if well is None:
            continue

        mask_p = mask_path_for(well, t)
        mask = load_mask(mask_p)
        if mask is None:
            n_no_mask += 1
            continue

        try:
            data = load_grids(npz_path)
        except Exception as exc:
            print(f"  SKIP corrupt grid {npz_path.name}: {exc}")
            continue
        summary = embryo_ncc_summary(
            ncc_grid=data["ncc_grid"],
            mask=mask,
            tile_size=data["tile_size"],
            stride=data["stride"],
            bad_thresh=BAD_THRESH,
        )
        flag = "FAIL" if (
            (not np.isnan(summary["ncc_min"]) and summary["ncc_min"] < NCC_MIN_FAIL_THRESH) or
            (not np.isnan(summary["bad_pair_frac"]) and summary["bad_pair_frac"] > BAD_PAIR_FRAC_FAIL_THRESH)
        ) else "PASS"

        rows.append({"t": t, "p": p, "well": well, "qc_flag": flag, **summary})

    df = pd.DataFrame(rows).sort_values(["t", "p"]).reset_index(drop=True)
    csv_out = OUT_DIR / "embryo_ncc_summaries.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} embryo summaries → {csv_out}")
    if n_no_mask:
        print(f"  ({n_no_mask} stacks skipped — no mask file)")

    # flag counts
    counts = df["qc_flag"].value_counts()
    print(f"\nQC flag counts:")
    for flag, n in counts.items():
        print(f"  {flag}: {n} / {len(df)} ({100*n/len(df):.1f}%)")

    return df


def load_labeled(df: pd.DataFrame) -> pd.DataFrame:
    lk = pd.read_csv(LOOKUP_CSV)
    lk["p"] = lk["nd2_series_num"] - 1
    lk["t"] = lk["time_int"]
    return df.merge(lk[["t", "p", "category"]], on=["t", "p"], how="inner")


def plot_distributions(df: pd.DataFrame, labeled: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, thresh, title in [
        (axes[0], "ncc_min",       NCC_MIN_FAIL_THRESH,       "Embryo ncc_min (mask-aware)"),
        (axes[1], "bad_pair_frac", BAD_PAIR_FRAC_FAIL_THRESH, "Embryo bad_pair_frac (mask-aware)"),
    ]:
        vals = df[col].dropna()
        ax.hist(vals, bins=60, color="#4c72b0", edgecolor="none", alpha=0.8)
        ax.axvline(thresh, color="red", lw=1.5, ls="--", label=f"threshold={thresh}")

        for cat, grp in labeled.groupby("category"):
            color = LABEL_COLORS.get(cat, "black")
            for val in grp[col].dropna():
                ax.axvline(val, color=color, lw=1.5, alpha=0.8, zorder=5)

        # legend: threshold + label categories
        handles = [plt.Line2D([0], [0], color="red", ls="--", lw=1.5, label=f"threshold={thresh}")]
        for cat, color in LABEL_COLORS.items():
            if cat in labeled["category"].values:
                handles.append(plt.Line2D([0], [0], color=color, lw=1.5, label=cat))
        ax.legend(handles=handles, fontsize=8)

        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(title)

    fig.suptitle(f"Embryo-level NCC QC — 20250912 WT tricane — T=0–{df.t.max()} ({len(df)} embryo-timepoint images)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_scatter(df: pd.DataFrame, labeled: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    # split by flag for color
    ax.scatter(df.loc[df.qc_flag == "PASS", "ncc_min"],
               df.loc[df.qc_flag == "PASS", "bad_pair_frac"],
               color="#4c72b0", alpha=0.3, s=15, label="PASS")
    ax.scatter(df.loc[df.qc_flag == "FAIL", "ncc_min"],
               df.loc[df.qc_flag == "FAIL", "bad_pair_frac"],
               color="#d62728", alpha=0.6, s=20, label="FAIL")

    # overlay labeled examples
    for cat, grp in labeled.groupby("category"):
        color = LABEL_COLORS.get(cat, "black")
        ax.scatter(grp["ncc_min"], grp["bad_pair_frac"],
                   color=color, edgecolors="k", lw=0.8, s=80, zorder=6, label=cat)

    ax.axvline(NCC_MIN_FAIL_THRESH,       color="red",    lw=1, ls="--")
    ax.axhline(BAD_PAIR_FRAC_FAIL_THRESH, color="orange", lw=1, ls="--")
    ax.set_xlabel("ncc_min (embryo tiles only)")
    ax.set_ylabel("bad_pair_frac (embryo tiles only)")
    ax.set_title("Embryo-level motion QC scatter")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "scatter_ncc_min_vs_bad_frac.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def print_labeled_breakdown(df: pd.DataFrame, labeled: pd.DataFrame) -> None:
    print("\n=== Labeled example breakdown ===")
    cols = ["t", "p", "well", "ncc_min", "bad_pair_frac", "n_tiles", "qc_flag"]
    for cat, grp in labeled.groupby("category"):
        print(f"\n--- {cat} ---")
        print(grp[cols].to_string(index=False))


def main() -> None:
    df = run_analysis()
    labeled = load_labeled(df)
    plot_distributions(df, labeled)
    plot_scatter(df, labeled)
    print_labeled_breakdown(df, labeled)
    print("\nDone.")


if __name__ == "__main__":
    main()
