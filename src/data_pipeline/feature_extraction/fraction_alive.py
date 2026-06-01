"""
Fraction alive computation from UNet viability masks.

Computes continuous viability metric (0-1) from UNet via masks.
Extracted from build03A_process_images.py and qc_utils.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import skimage.io as io

from data_pipeline.segmentation_and_tracking.utils.mask_processing import clean_embryo_mask
from data_pipeline.shared.path_contracts import require_existing_path


def compute_fraction_alive(
    embryo_mask: np.ndarray,
    via_mask: Optional[np.ndarray],
) -> float:
    """Compute fraction of embryo that is alive (not dead tissue)."""
    embryo_binary = clean_embryo_mask(embryo_mask).astype(np.uint8)
    embryo_area = np.sum(embryo_binary)
    if embryo_area == 0:
        return np.nan
    if via_mask is None:
        raise ValueError('fraction_alive: via mask is required by contract but was missing')
    via_binary = clean_embryo_mask(via_mask).astype(np.uint8)
    dead_tissue = np.logical_and(embryo_binary, via_binary).astype(np.uint8)
    dead_area = np.sum(dead_tissue)
    fraction_alive = 1.0 - (dead_area / embryo_area)
    return float(np.clip(fraction_alive, 0.0, 1.0))


def extract_fraction_alive_batch(
    tracking_df: pd.DataFrame,
    mask_dir: Path | None = None,
    via_mask_dir: Optional[Path] = None,
    via_mask_lookup: Optional[dict[str, Path]] = None,
    mask_path_col: str = 'exported_mask_path',
) -> pd.DataFrame:
    """Extract fraction_alive for batch of snips."""
    results = []
    for _, row in tracking_df.iterrows():
        snip_id = row['snip_id']
        image_id = row.get('image_id', snip_id.rsplit('_', 1)[0])
        mask_path = require_existing_path(
            row.get(mask_path_col),
            context='fraction_alive',
            field_name=mask_path_col,
            row_id=str(snip_id),
        )
        embryo_mask = io.imread(mask_path)

        via_path = None
        if via_mask_lookup and image_id in via_mask_lookup:
            via_path = require_existing_path(
                via_mask_lookup[image_id],
                context='fraction_alive',
                field_name='via_mask_path',
                row_id=str(snip_id),
            )
        elif via_mask_dir:
            candidate = via_mask_dir / f"{image_id}_via.png"
            via_path = require_existing_path(
                candidate,
                context='fraction_alive',
                field_name='via_mask_path',
                row_id=str(snip_id),
            )
        else:
            raise ValueError(f'fraction_alive: no via mask lookup or directory provided for snip_id={snip_id}')

        via_mask = io.imread(via_path) if via_path is not None else None
        frac = compute_fraction_alive(embryo_mask, via_mask)
        results.append({'snip_id': snip_id, 'fraction_alive': frac})

    return pd.DataFrame(results)
