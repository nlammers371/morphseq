"""
Fraction alive computation from UNet viability masks.

Computes continuous viability metric (0-1) from UNet via masks.
Extracted from build03A_process_images.py and qc_utils.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import skimage.io as io

from data_pipeline.segmentation_and_tracking.utils.mask_processing import clean_embryo_mask


def compute_fraction_alive(
    embryo_mask: np.ndarray,
    via_mask: Optional[np.ndarray],
) -> float:
    """
    Compute fraction of embryo that is alive (not dead tissue).

    Args:
        embryo_mask: Binary embryo mask
        via_mask: Binary viability mask (1 = dead tissue), can be None

    Returns:
        Fraction alive (0.0 to 1.0), or 1.0 if via_mask unavailable
    """
    # Ensure binary masks
    embryo_binary = clean_embryo_mask(embryo_mask).astype(np.uint8)
    embryo_area = np.sum(embryo_binary)

    if embryo_area == 0:
        return np.nan

    # If no via mask, assume fully alive
    if via_mask is None:
        return 1.0

    via_binary = clean_embryo_mask(via_mask).astype(np.uint8)

    # Compute overlap between embryo and dead tissue
    dead_tissue = np.logical_and(embryo_binary, via_binary).astype(np.uint8)
    dead_area = np.sum(dead_tissue)

    # Fraction alive = 1 - (dead_area / embryo_area)
    fraction_alive = 1.0 - (dead_area / embryo_area)

    # Clamp to [0, 1]
    fraction_alive = np.clip(fraction_alive, 0.0, 1.0)

    return float(fraction_alive)


def extract_fraction_alive_batch(
    tracking_df: pd.DataFrame,
    mask_dir: Path | None = None,
    via_mask_dir: Optional[Path] = None,
    via_mask_lookup: Optional[dict[str, Path]] = None,
    mask_path_col: str = 'embryo_mask_path',
) -> pd.DataFrame:
    """
    Extract fraction_alive for batch of snips.

    Args:
        tracking_df: Segmentation tracking DataFrame
        mask_dir: Directory containing embryo masks, used as fallback when
            explicit per-row mask paths are missing.
        via_mask_dir: Directory containing UNet viability masks (optional)
        mask_path_col: Column name containing explicit per-row mask paths

    Returns:
        DataFrame with snip_id and fraction_alive
    """
    def _resolve_mask_path(row: pd.Series) -> Path | None:
        mask_path_value = row.get(mask_path_col)
        if pd.notna(mask_path_value):
            candidate = Path(str(mask_path_value))
            if candidate.exists():
                return candidate

        if mask_dir is None:
            return None

        snip_id = row['snip_id']
        candidate = mask_dir / f"{snip_id}_mask.png"
        if candidate.exists():
            return candidate

        image_id = row.get('image_id')
        if pd.notna(image_id):
            candidate = mask_dir / f"{image_id}_masks.png"
            if candidate.exists():
                return candidate

        candidate = mask_dir / f"{snip_id}.png"
        if candidate.exists():
            return candidate

        return None

    results = []

    for idx, row in tracking_df.iterrows():
        snip_id = row['snip_id']
        image_id = row.get('image_id', snip_id.rsplit('_', 1)[0])

        mask_path = _resolve_mask_path(row)
        if mask_path is None or not mask_path.exists():
            results.append({
                'snip_id': snip_id,
                'fraction_alive': np.nan,
            })
            continue

        try:
            embryo_mask = io.imread(mask_path)

            # Load viability mask if available
            via_mask = None
            if via_mask_lookup and image_id in via_mask_lookup:
                via_path = Path(via_mask_lookup[image_id])
                if via_path.exists():
                    via_mask = io.imread(via_path)
            elif via_mask_dir:
                via_path = via_mask_dir / f"{image_id}_via.png"
                if via_path.exists():
                    via_mask = io.imread(via_path)

            # Compute fraction alive
            frac = compute_fraction_alive(embryo_mask, via_mask)

            results.append({
                'snip_id': snip_id,
                'fraction_alive': frac,
            })

        except Exception as e:
            print(f"Warning: Failed to process {snip_id}: {e}")
            results.append({
                'snip_id': snip_id,
                'fraction_alive': np.nan,
            })

    return pd.DataFrame(results)
