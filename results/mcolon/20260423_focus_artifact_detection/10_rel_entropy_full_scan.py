"""
10_rel_entropy_full_scan.py
============================
Compute rel_entropy_mean (embryo Shannon entropy − background entropy) for
every (T, P) stack in the nd2 that has a SAM2 mask.

Designed to run as a qsub array job, one chunk per T range.
Opens the nd2 once and loops sequentially — multiprocessing avoided because
nd2 dask handles do not survive fork.

Usage:
  # smoke test — first 2 timepoints, first 5 positions
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/10_rel_entropy_full_scan.py \
    --t-limit 2 --p-limit 5

  # full range (for qsub, specify --t-start / --t-end per chunk)
  conda run -n segmentation_grounded_sam --no-capture-output python \
    results/mcolon/20260423_focus_artifact_detection/10_rel_entropy_full_scan.py \
    --t-start 0 --t-end 20

Output:
  10_scan_output/rel_entropy_summaries.csv    (merged full run)
  10_scan_output/chunk_t000_t020.csv          (per-chunk intermediate)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import nd2
from PIL import Image

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

ND2_PATH   = MORPHSEQ_ROOT / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR  = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
SERIES_MAP = MORPHSEQ_ROOT / "results/mcolon/20260421_motion_artifact_detection/06_scan_output/series_well_map.csv"
OUT_DIR    = Path(__file__).resolve().parent / "10_scan_output"
CSV_PATH   = OUT_DIR / "rel_entropy_summaries.csv"
DATE       = "20250912"
CHECKPOINT_EVERY = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--t-start",  type=int, default=0)
    p.add_argument("--t-end",    type=int, default=None)
    p.add_argument("--t-limit",  type=int, default=None, help="smoke-test: first N timepoints from t-start")
    p.add_argument("--p-limit",  type=int, default=None, help="smoke-test: first N positions")
    return p.parse_args()


def _entropy(pixels: np.ndarray) -> float:
    hist, _ = np.histogram(pixels, bins=256, range=(0, 65535))
    h = hist[hist > 0].astype(np.float64)
    h /= h.sum()
    return float(-np.sum(h * np.log2(h + 1e-12)))


def process_stack(stack: np.ndarray, mask: np.ndarray) -> dict:
    background = ndi.binary_erosion(~mask, iterations=10)
    emb_ys, emb_xs = np.where(mask)
    bg_ys,  bg_xs  = np.where(background)

    if len(emb_ys) == 0 or len(bg_ys) == 0:
        return {"rel_entropy_mean": float("nan"), "rel_entropy_min": float("nan"),
                "rel_entropy_std": float("nan"), "n_z": stack.shape[0]}

    rels = []
    for z in range(stack.shape[0]):
        sl = stack[z]
        rels.append(_entropy(sl[emb_ys, emb_xs]) - _entropy(sl[bg_ys, bg_xs]))

    rel = np.array(rels)
    return {
        "rel_entropy_mean": float(rel.mean()),
        "rel_entropy_min":  float(rel.min()),
        "rel_entropy_std":  float(rel.std()),
        "n_z":              int(stack.shape[0]),
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sm = pd.read_csv(SERIES_MAP)
    p_to_well = {int(r.series_number) - 1: r.well_index for _, r in sm.iterrows()}

    with nd2.ND2File(str(ND2_PATH)) as f:
        dask_arr = f.to_dask()
        T_total, P_total = dask_arr.shape[0], dask_arr.shape[1]

        t_end = T_total if args.t_end is None else min(args.t_end, T_total)
        if args.t_limit:
            t_end = min(args.t_start + args.t_limit, t_end)
        P = P_total if args.p_limit is None else min(args.p_limit, P_total)

        chunk_csv = None
        if args.t_start > 0 or args.t_end is not None:
            label = f"t{args.t_start:03d}_t{t_end:03d}"
            chunk_csv = OUT_DIR / f"chunk_{label}.csv"
        out_csv = chunk_csv if chunk_csv is not None else CSV_PATH

        import socket
        total = (t_end - args.t_start) * P
        print(f"[{socket.gethostname()}] T={args.t_start}–{t_end-1}, P=0–{P-1} → {total} stacks")

        rows: list[dict] = []
        done = 0

        for t in range(args.t_start, t_end):
            for p in range(P):
                well = p_to_well.get(p, f"p{p:03d}")
                mask_path = MASKS_DIR / f"{DATE}_{well}_ch00_t{t:04d}_masks_emnum_1.png"

                row: dict = {"t": t, "p": p, "well": well}

                if not mask_path.exists():
                    row.update({"has_mask": False, "rel_entropy_mean": float("nan"),
                                "rel_entropy_min": float("nan"), "rel_entropy_std": float("nan"), "n_z": 0})
                else:
                    mask = np.array(Image.open(mask_path)).astype(bool)
                    if not mask.any():
                        row.update({"has_mask": False, "rel_entropy_mean": float("nan"),
                                    "rel_entropy_min": float("nan"), "rel_entropy_std": float("nan"), "n_z": 0})
                    else:
                        stack = dask_arr[t, p].compute().astype(np.float32)
                        metrics = process_stack(stack, mask)
                        row.update({"has_mask": True, **metrics})

                rows.append(row)
                done += 1

                if done % 50 == 0:
                    print(f"  {done}/{total}  t={t} p={p} well={well}  "
                          f"rel_entropy_mean={row.get('rel_entropy_mean', float('nan')):.3f}")

                if done % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(rows).sort_values(["t", "p"]).to_csv(out_csv, index=False)
                    print(f"  checkpoint → {out_csv}")

    df = pd.DataFrame(rows).sort_values(["t", "p"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    n_mask = int(df["has_mask"].sum()) if "has_mask" in df.columns else 0
    print(f"\nSaved → {out_csv}  ({len(df)} rows, {n_mask} with mask)")


if __name__ == "__main__":
    main()
