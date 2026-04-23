"""
08_decile_ranked_figure.py
==========================
Sample 2 embryos from each of 10 deciles (ranked by a chosen metric) and
produce a ranked Z-stack quality figure using make_ranked_figure().

Ranking metric: ncc_p05 (ascending = worst motion first).
Each decile contributes 2 columns → 20 columns total.

Output: figures/decile_ranked_ncc_p05.png
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

HERE          = Path(__file__).resolve().parent
SUMMARIES_CSV = HERE / "07_embryo_ncc_output/embryo_ncc_summaries.csv"
ND2_PATH      = MORPHSEQ_ROOT / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR     = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
IMAGES_DIR    = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images"
GRIDS_DIR         = HERE / "06_scan_output/grids"
SERIES_WELL_MAP   = HERE / "06_scan_output/series_well_map.csv"
FIG_DIR           = HERE / "figures"

# Ranking metric: sort ascending (low = bad motion), decile 1 = worst
RANK_METRIC    = "ncc_p05"
N_DECILES      = 10
SAMPLES_PER    = 2
DATE           = "20250912"

# Metrics shown in the bar panel — (key, display_label, good_high)
METRICS = [
    ("ncc_p05",           "NCC p05",          True),
    ("ncc_min",           "NCC min",          True),
    ("bad_pair_frac",     "Bad-pair frac",    False),
    ("ncc_bad_tile_frac", "Bad-tile frac",    False),
    ("longest_bad_run",   "Longest bad run",  False),
    ("ncc_mean",          "NCC mean",         True),
    ("local_ncc_std_mean","NCC spatial std",  False),
]

# Decile edge colors: red → orange → yellow → green
DECILE_COLORS = [
    "#d62728", "#e05000", "#e07800", "#e0a000", "#c8c000",
    "#a0b800", "#78b000", "#50a800", "#30a030", "#1a8820",
]


def sample_deciles(df: pd.DataFrame) -> list[dict]:
    """
    Rank df by RANK_METRIC ascending, assign 10 decile bins,
    sample SAMPLES_PER embryos from each decile.

    Sampling strategy: pick the embryo closest to the 25th and 75th
    percentile within each decile (spreads within-decile coverage).
    """
    valid = df[RANK_METRIC].notna()
    ranked = df[valid].sort_values(RANK_METRIC).reset_index(drop=True)
    ranked["decile"] = pd.qcut(ranked[RANK_METRIC], N_DECILES,
                                labels=False, duplicates="drop")

    examples = []
    for dec in range(N_DECILES):
        grp = ranked[ranked["decile"] == dec].reset_index(drop=True)
        if len(grp) == 0:
            continue
        n = len(grp)
        # pick indices near Q1 and Q3 within the decile
        i_q1 = max(0, int(n * 0.25))
        i_q3 = min(n - 1, int(n * 0.75))
        picks = sorted({i_q1, i_q3})
        # if decile has only 1 member, just take it
        if len(picks) < SAMPLES_PER and n >= SAMPLES_PER:
            picks = [int(n * k / (SAMPLES_PER - 1)) for k in range(SAMPLES_PER)]
            picks = sorted(set(min(n - 1, i) for i in picks))

        color = DECILE_COLORS[dec % len(DECILE_COLORS)]
        for rank_i, row_i in enumerate(picks[:SAMPLES_PER]):
            row = grp.iloc[row_i]
            label = f"D{dec+1}-{rank_i+1}"
            ex = {
                "well":  row["well"],
                "t":     int(row["t"]),
                "p":     int(row["p"]),
                "label": label,
                "color": color,
            }
            for key, _, _ in METRICS:
                ex[key] = row.get(key, np.nan)
            examples.append(ex)

    return examples


def main() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ranked_stack_figure", HERE / "ranked_stack_figure.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_ranked_figure = mod.make_ranked_figure

    df = pd.read_csv(SUMMARIES_CSV)
    print(f"Loaded {len(df)} embryo summaries")

    if RANK_METRIC not in df.columns:
        raise ValueError(f"Ranking metric '{RANK_METRIC}' not in CSV. "
                         f"Available: {list(df.columns)}")

    examples = sample_deciles(df)
    print(f"Sampled {len(examples)} embryos across {N_DECILES} deciles")
    for ex in examples:
        print(f"  {ex['label']:6s}  well={ex['well']:3s}  t={ex['t']:3d}  "
              f"p={ex['p']:2d}  {RANK_METRIC}={ex.get(RANK_METRIC, float('nan')):.4f}")

    out_path = FIG_DIR / f"decile_ranked_{RANK_METRIC}.png"
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
    )
    print(f"\nFigure saved → {out_path}")


if __name__ == "__main__":
    main()
