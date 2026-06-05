"""
11_focus_stacked_metrics.py
===========================
Compute focus-quality metrics on the final LoG focus-stacked image.

This uses the same `_focus_stack()` helper as the YX1 stitched image builder,
then measures embryo-vs-background texture on the final 2D image. This is
intended to test whether post-stack metrics track visual usability better than
per-Z rel_entropy_mean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import nd2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

from src.data_pipeline.image_building.yx1.stitched_ff_builder import _focus_stack


HERE = Path(__file__).resolve().parent
ND2_PATH = MORPHSEQ_ROOT / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR = MORPHSEQ_ROOT / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
SERIES_MAP = HERE.parent / "20260421_motion_artifact_detection/06_scan_output/series_well_map.csv"
INPUT_CSV = HERE / "10_scan_output/rel_entropy_summaries.csv"
OUT_CSV = HERE / "11_focus_stacked_metrics/focus_stacked_metric_examples.csv"
DATE = "20250912"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--range-min", type=float, default=-0.70)
    ap.add_argument("--range-max", type=float, default=-0.40)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--include", action="append", default=[], help="Explicit WELL:T example, e.g. D05:97")
    return ap.parse_args()


def load_well_to_p() -> dict[str, int]:
    df = pd.read_csv(SERIES_MAP)
    return {str(r["well_index"]): int(r["series_number"]) - 1 for _, r in df.iterrows()}


def load_mask(well: str, t: int) -> np.ndarray | None:
    path = MASKS_DIR / f"{DATE}_{well}_ch00_t{t:04d}_masks_emnum_1.png"
    if not path.exists():
        return None
    mask = np.array(Image.open(path)).astype(bool)
    return mask if mask.any() else None


def entropy_u8(values: np.ndarray) -> float:
    hist, _ = np.histogram(values, bins=256, range=(0, 255))
    p = hist[hist > 0].astype(np.float64)
    if p.size == 0:
        return float("nan")
    p /= p.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def metric_row(ff_u8: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    bg = ndi.binary_erosion(~mask, iterations=10)
    if not bg.any():
        bg = ~mask

    emb_px = ff_u8[mask]
    bg_px = ff_u8[bg]

    lap = cv2.Laplacian(ff_u8.astype(np.float32), cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    emb_lap = lap_abs[mask]
    bg_lap = lap_abs[bg]

    emb_ent = entropy_u8(emb_px)
    bg_ent = entropy_u8(bg_px)
    emb_iqr = float(np.percentile(emb_px, 95) - np.percentile(emb_px, 5))
    bg_iqr = float(np.percentile(bg_px, 95) - np.percentile(bg_px, 5))
    emb_lap_mean = float(np.mean(emb_lap))
    bg_lap_mean = float(np.mean(bg_lap))

    return {
        "ff_entropy_emb": emb_ent,
        "ff_entropy_bg": bg_ent,
        "ff_rel_entropy": emb_ent - bg_ent,
        "ff_mean_emb": float(np.mean(emb_px)),
        "ff_mean_bg": float(np.mean(bg_px)),
        "ff_rel_mean": float(np.mean(emb_px) - np.mean(bg_px)),
        "ff_iqr_emb": emb_iqr,
        "ff_iqr_bg": bg_iqr,
        "ff_rel_iqr": emb_iqr - bg_iqr,
        "ff_lap_abs_mean_emb": emb_lap_mean,
        "ff_lap_abs_mean_bg": bg_lap_mean,
        "ff_rel_lap_abs_mean": emb_lap_mean - bg_lap_mean,
        "ff_lap_abs_ratio": emb_lap_mean / (bg_lap_mean + 1e-9),
    }


def choose_examples(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for item in args.include:
        well, t_str = item.split(":", 1)
        sub = df[(df["well"].astype(str) == well) & (df["t"].astype(int) == int(t_str))]
        rows.append(sub)

    band = df[
        df["rel_entropy_mean"].between(args.range_min, args.range_max, inclusive="both")
        & df["has_mask"].astype(bool)
    ].sort_values("rel_entropy_mean")
    if len(band) > 0:
        idx = np.linspace(0, len(band) - 1, min(args.samples, len(band))).astype(int)
        rows.append(band.iloc[idx])

    if not rows:
        return pd.DataFrame(columns=df.columns)

    return (
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(["well", "t"])
        .sort_values(["rel_entropy_mean", "well", "t"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    required = {"t", "well", "rel_entropy_mean", "has_mask"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{args.input_csv} missing required columns: {missing}")

    examples = choose_examples(df, args)
    if examples.empty:
        raise ValueError("No examples selected")

    well_to_p = load_well_to_p()
    rows: list[dict] = []

    with nd2.ND2File(str(ND2_PATH)) as nd:
        dask_arr = nd.to_dask()
        for i, row in examples.iterrows():
            well = str(row["well"])
            t = int(row["t"])
            p = well_to_p[well]
            print(
                f"[{i + 1}/{len(examples)}] {well} t={t} p_nd2={p} "
                f"rel_entropy_mean={row['rel_entropy_mean']:.3f}",
                flush=True,
            )

            mask = load_mask(well, t)
            if mask is None:
                continue

            stack = dask_arr[t, p, :, :, :].compute()
            ff = _focus_stack(stack, device=args.device, filter_size=3)
            metrics = metric_row(ff, mask)
            rows.append({**row.to_dict(), "p_nd2": p, **metrics})

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved -> {args.out_csv} ({len(out)} rows)", flush=True)
    print(
        out[["well", "t", "rel_entropy_mean", "ff_rel_entropy", "ff_rel_lap_abs_mean", "ff_lap_abs_ratio"]]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
