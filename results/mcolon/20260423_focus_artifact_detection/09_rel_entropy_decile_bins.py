"""
09_rel_entropy_decile_bins.py
==============================
Produce one threshold-bin figure per decile of rel_entropy_mean,
mirroring the ncc_p05 threshold_bins figures in:
  results/mcolon/20260421_motion_artifact_detection/figures/threshold_bins/v2_ncc_p05_coverage25/

Each figure shows a grid of embryos (focus-stacked JPEG + 15 Z slices + metric
bars) that fall in that decile of rel_entropy_mean.

Ranking: ascending = most negative = worst focus (D1 worst, D10 best).

Inputs:
  - results/mcolon/20260421_motion_artifact_detection/07_embryo_ncc_output/embryo_ncc_summaries.csv
  - results/mcolon/20260421_motion_artifact_detection/slice_metrics_relative.csv

Output:
  - figures/threshold_bins/rel_entropy_deciles/D01_<lo>_to_<hi>.png  ... D10_...png
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

HERE           = Path(__file__).resolve().parent
MOTION_DIR     = HERE.parent / "20260421_motion_artifact_detection"
SUMMARIES_CSV  = MOTION_DIR / "07_embryo_ncc_output/embryo_ncc_summaries.csv"
SLICE_REL_CSV  = MOTION_DIR / "slice_metrics_relative.csv"
ND2_PATH       = MORPHSEQ_ROOT / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR      = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
IMAGES_DIR     = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images"
GRIDS_DIR      = MOTION_DIR / "06_scan_output/grids"
SERIES_WELL_MAP = MOTION_DIR / "06_scan_output/series_well_map.csv"
OUT_DIR        = HERE / "figures/threshold_bins/rel_entropy_deciles"

DATE        = "20250912"
N_DECILES   = 10
SAMPLES_PER = 6   # columns per figure

DECILE_COLORS = [
    "#d62728", "#e05000", "#e07800", "#e0a000", "#c8c000",
    "#a0b800", "#78b000", "#50a800", "#30a030", "#1a8820",
]

METRICS = [
    ("rel_entropy_mean", "REL ENTROPY mean",  True),   # good_high=True: closer to 0 is better
    ("ncc_p05",          "NCC p05",            True),
    ("bad_pair_frac",    "Bad-pair frac",      False),
    ("ncc_min",          "NCC min",            True),
]


def build_merged() -> pd.DataFrame:
    summaries = pd.read_csv(SUMMARIES_CSV)

    rel = pd.read_csv(SLICE_REL_CSV)
    rel_mean = (
        rel.groupby(["well", "time_int", "embryo"], observed=True)["rel_entropy"]
        .mean()
        .reset_index()
        .rename(columns={"rel_entropy": "rel_entropy_mean", "time_int": "t"})
    )
    # summaries uses column "t" for time_int; embryo is always 1 in this dataset
    merged = summaries.merge(rel_mean[["well", "t", "rel_entropy_mean"]],
                             on=["well", "t"], how="left")
    return merged


def sample_decile(df: pd.DataFrame, dec: int) -> list[dict]:
    """Pick up to SAMPLES_PER embryos evenly spaced within the decile."""
    grp = df[df["decile"] == dec].reset_index(drop=True)
    if len(grp) == 0:
        return []
    n = len(grp)
    if n <= SAMPLES_PER:
        idxs = list(range(n))
    else:
        idxs = [int(n * k / (SAMPLES_PER - 1)) for k in range(SAMPLES_PER)]
        idxs = sorted(set(min(n - 1, i) for i in idxs))

    color = DECILE_COLORS[dec % len(DECILE_COLORS)]
    examples = []
    for rank_i, row_i in enumerate(idxs):
        row = grp.iloc[row_i]
        ex = {
            "well":  row["well"],
            "t":     int(row["t"]),
            "p":     int(row["p"]),
            "label": f"D{dec+1}-{rank_i+1}",
            "color": color,
        }
        for key, _, _ in METRICS:
            ex[key] = row.get(key, np.nan)
        examples.append(ex)
    return examples


def main() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ranked_stack_figure", MOTION_DIR / "ranked_stack_figure.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_ranked_figure = mod.make_ranked_figure

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_merged()
    valid = df["rel_entropy_mean"].notna()
    ranked = df[valid].sort_values("rel_entropy_mean").reset_index(drop=True)
    ranked["decile"] = pd.qcut(ranked["rel_entropy_mean"], N_DECILES,
                               labels=False, duplicates="drop")

    lo_vals = ranked.groupby("decile")["rel_entropy_mean"].min()
    hi_vals = ranked.groupby("decile")["rel_entropy_mean"].max()
    counts  = ranked.groupby("decile").size()

    print(f"Total embryos with rel_entropy_mean: {valid.sum()}")
    for d in range(N_DECILES):
        lo = lo_vals.get(d, float("nan"))
        hi = hi_vals.get(d, float("nan"))
        n  = counts.get(d, 0)
        print(f"  D{d+1:02d}  [{lo:.3f}, {hi:.3f}]  n={n}")

    for dec in range(N_DECILES):
        lo = lo_vals.get(dec, float("nan"))
        hi = hi_vals.get(dec, float("nan"))
        examples = sample_decile(ranked, dec)
        if not examples:
            print(f"D{dec+1}: no examples, skipping")
            continue

        out_path = OUT_DIR / f"D{dec+1:02d}_{lo:.3f}_to_{hi:.3f}.png"
        print(f"\nRendering D{dec+1} [{lo:.3f}, {hi:.3f}]  ({len(examples)} columns) → {out_path.name}")
        make_ranked_figure(
            examples=examples,
            metrics=METRICS,
            nd2_path=ND2_PATH,
            masks_dir=MASKS_DIR,
            images_dir=IMAGES_DIR,
            ncc_grids_dir=GRIDS_DIR,
            out_path=out_path,
            series_well_map_csv=SERIES_WELL_MAP,
            date=DATE,
            col_width=3.6,
            fig_height=14.0,
            dpi=160,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
