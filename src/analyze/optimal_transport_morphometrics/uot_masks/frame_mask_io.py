"""Mask I/O helpers for UOT mask transport."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import os

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from data_pipeline.segmentation.grounded_sam2.mask_export import (
    load_labeled_mask,
    extract_individual_masks,
)

from analyze.utils.optimal_transport import UOTFrame, UOTFramePair


DEFAULT_USECOLS = [
    "experiment_date",
    "experiment_id",
    "well",
    "time_int",
    "embryo_id",
    "frame_index",
    "mask_rle",
    "mask_height_px",
    "mask_width_px",
    "image_id",
    "snip_id",
    "relative_time_s",
    "raw_time_s",
    "Height (um)",
    "Height (px)",
    "Width (um)",
    "Width (px)",
]


def _ensure_2d(mask: np.ndarray) -> np.ndarray:
    if mask.ndim > 2:
        mask = mask.squeeze()
        if mask.ndim > 2:
            mask = mask[..., 0]
    return mask


def load_mask_from_rle_counts(rle_counts: str, height_px: int, width_px: int) -> np.ndarray:
    rle_data = {"counts": rle_counts, "size": [int(height_px), int(width_px)]}
    mask = decode_mask_rle(rle_data)
    mask = _ensure_2d(mask)
    return mask.astype(np.uint8)


def _extract_time_stub(row: pd.Series) -> Optional[str]:
    """Extract time stub as 4-digit string from row metadata."""
    for key in ("time_int", "frame_index"):
        if key in row and pd.notnull(row[key]):
            try:
                return f"{int(row[key]):04d}"
            except (TypeError, ValueError):
                continue
    return None


def _load_build02_aux_mask(
    data_root: Path,
    row: pd.Series,
    mask_shape: Tuple[int, int],
    keyword: str = "yolk",
) -> Optional[np.ndarray]:
    """Load Build02 auxiliary mask (e.g., yolk) by row metadata."""
    import skimage.io as io
    from skimage.transform import resize

    seg_root = data_root / "segmentation"
    if not seg_root.exists():
        return None

    date = str(row.get("experiment_date", ""))
    well = row.get("well", None)
    time_stub = _extract_time_stub(row)
    if not date or well is None or time_stub is None:
        return None

    stub = f"{well}_t{time_stub}"

    for p in seg_root.iterdir():
        if p.is_dir() and keyword in p.name:
            date_dir = p / date
            if not date_dir.exists():
                continue
            candidates = sorted(date_dir.glob(f"*{stub}*"))
            if not candidates:
                continue
            arr_raw = io.imread(candidates[0])
            if arr_raw.max() >= 255:
                arr = (arr_raw > 127).astype(np.uint8)
            else:
                arr = (arr_raw > 0).astype(np.uint8)
            if arr.shape != mask_shape:
                arr = resize(
                    arr.astype(float),
                    mask_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.uint8)
            return arr
    return None


def _compute_um_per_pixel(row: pd.Series) -> float:
    """Compute um_per_pixel from CSV metadata."""
    if "Height (um)" in row and "Height (px)" in row:
        height_um = float(row["Height (um)"])
        height_px = float(row["Height (px)"])
        if height_px > 0:
            return height_um / height_px
    # Fallback: try width
    if "Width (um)" in row and "Width (px)" in row:
        width_um = float(row["Width (um)"])
        width_px = float(row["Width (px)"])
        if width_px > 0:
            return width_um / width_px
    # If no metadata available, return NaN
    return np.nan


def load_mask_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_index: int,
    usecols: Optional[List[str]] = None,
    data_root: Optional[Path] = None,
) -> UOTFrame:
    if usecols is None:
        usecols = DEFAULT_USECOLS
    df = pd.read_csv(csv_path, usecols=usecols)
    subset = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if subset.empty:
        raise ValueError(f"No mask found for embryo_id={embryo_id} frame_index={frame_index}")
    row = subset.iloc[0]
    mask = load_mask_from_rle_counts(row["mask_rle"], row["mask_height_px"], row["mask_width_px"])

    # Add um_per_pixel to metadata
    meta = row.to_dict()
    meta["um_per_pixel"] = _compute_um_per_pixel(row)
    if data_root is None:
        env_root = os.environ.get("MORPHSEQ_DATA_ROOT")
        if env_root:
            data_root = Path(env_root)
    if data_root is not None:
        yolk_mask = _load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
        if yolk_mask is not None:
            meta["yolk_mask"] = yolk_mask

    return UOTFrame(embryo_mask=mask, meta=meta)


def load_mask_pair_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_index_src: int,
    frame_index_tgt: int,
    usecols: Optional[List[str]] = None,
    data_root: Optional[Path] = None,
) -> UOTFramePair:
    if usecols is None:
        usecols = DEFAULT_USECOLS
    df = pd.read_csv(csv_path, usecols=usecols)
    subset = df[(df["embryo_id"] == embryo_id) & (df["frame_index"].isin([frame_index_src, frame_index_tgt]))]
    if len(subset) < 2:
        raise ValueError(
            f"Expected two frames for embryo_id={embryo_id} at {frame_index_src},{frame_index_tgt}"
        )
    src_row = subset[subset["frame_index"] == frame_index_src].iloc[0]
    tgt_row = subset[subset["frame_index"] == frame_index_tgt].iloc[0]

    src_meta = src_row.to_dict()
    src_meta["um_per_pixel"] = _compute_um_per_pixel(src_row)

    tgt_meta = tgt_row.to_dict()
    tgt_meta["um_per_pixel"] = _compute_um_per_pixel(tgt_row)

    if data_root is None:
        env_root = os.environ.get("MORPHSEQ_DATA_ROOT")
        if env_root:
            data_root = Path(env_root)

    src = UOTFrame(
        embryo_mask=load_mask_from_rle_counts(
            src_row["mask_rle"], src_row["mask_height_px"], src_row["mask_width_px"]
        ),
        meta=src_meta,
    )
    tgt = UOTFrame(
        embryo_mask=load_mask_from_rle_counts(
            tgt_row["mask_rle"], tgt_row["mask_height_px"], tgt_row["mask_width_px"]
        ),
        meta=tgt_meta,
    )
    if data_root is not None:
        src_yolk = _load_build02_aux_mask(data_root, src_row, src.embryo_mask.shape, keyword="yolk")
        if src_yolk is not None:
            src.meta["yolk_mask"] = src_yolk
        tgt_yolk = _load_build02_aux_mask(data_root, tgt_row, tgt.embryo_mask.shape, keyword="yolk")
        if tgt_yolk is not None:
            tgt.meta["yolk_mask"] = tgt_yolk
    return UOTFramePair(src=src, tgt=tgt, pair_meta={"csv_path": str(csv_path)})


def load_mask_series_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_indices: Optional[Iterable[int]] = None,
    usecols: Optional[List[str]] = None,
    data_root: Optional[Path] = None,
) -> List[UOTFrame]:
    if usecols is None:
        usecols = DEFAULT_USECOLS
    df = pd.read_csv(csv_path, usecols=usecols)
    subset = df[df["embryo_id"] == embryo_id]
    if frame_indices is not None:
        frame_set = set(frame_indices)
        subset = subset[subset["frame_index"].isin(frame_set)]
    subset = subset.sort_values("frame_index")
    if subset.empty:
        raise ValueError(f"No masks found for embryo_id={embryo_id}")

    if data_root is None:
        env_root = os.environ.get("MORPHSEQ_DATA_ROOT")
        if env_root:
            data_root = Path(env_root)

    frames: List[UOTFrame] = []
    for _, row in subset.iterrows():
        mask = load_mask_from_rle_counts(row["mask_rle"], row["mask_height_px"], row["mask_width_px"])
        meta = row.to_dict()
        meta["um_per_pixel"] = _compute_um_per_pixel(row)
        if data_root is not None:
            yolk_mask = _load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
            if yolk_mask is not None:
                meta["yolk_mask"] = yolk_mask
        frames.append(UOTFrame(embryo_mask=mask, meta=meta))
    return frames


def load_mask_from_png(
    mask_path: Path,
    embryo_id: Optional[str] = None,
    label: Optional[int] = None,
) -> np.ndarray:
    labeled = load_labeled_mask(mask_path)
    if label is None and embryo_id is None:
        label = 1
    if embryo_id is not None:
        masks = extract_individual_masks(labeled)
        if embryo_id not in masks:
            raise ValueError(f"embryo_id={embryo_id} not found in {mask_path}")
        return masks[embryo_id]
    return (labeled == label).astype(np.uint8)
