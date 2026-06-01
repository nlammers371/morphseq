"""
Pose and kinematics feature extraction from SAM2 tracking.

Computes centroid position, bounding box, orientation, displacement, and speed.
Extracted from build03A_process_images.py get_embryo_stats function (lines 820-833).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import skimage.io as io
from skimage.measure import regionprops

from data_pipeline.segmentation_and_tracking.utils.mask_processing import clean_embryo_mask
from data_pipeline.shared.path_contracts import require_existing_path

from .mask_geometry_metrics import compute_mask_geometry


def compute_pose_features(
    mask: np.ndarray,
    pixel_size_um: float,
) -> Dict:
    """Compute pose features from binary mask."""
    mask_binary = clean_embryo_mask(mask).astype(np.uint8)
    props = regionprops(mask_binary)

    if not props:
        return {
            'orientation_angle': np.nan,
            'bbox_width_um': np.nan,
            'bbox_height_um': np.nan,
            'bbox_x_min': np.nan,
            'bbox_y_min': np.nan,
            'bbox_x_max': np.nan,
            'bbox_y_max': np.nan,
        }

    prop = props[0]
    orientation = prop.orientation
    min_row, min_col, max_row, max_col = prop.bbox
    bbox_width_um = (max_col - min_col) * pixel_size_um
    bbox_height_um = (max_row - min_row) * pixel_size_um

    return {
        'orientation_angle': float(orientation),
        'bbox_width_um': float(bbox_width_um),
        'bbox_height_um': float(bbox_height_um),
        'bbox_x_min': float(min_col * pixel_size_um),
        'bbox_y_min': float(min_row * pixel_size_um),
        'bbox_x_max': float(max_col * pixel_size_um),
        'bbox_y_max': float(max_row * pixel_size_um),
    }


def compute_kinematics(
    current_centroid_um: tuple,
    previous_centroid_um: Optional[tuple],
    current_time_s: float,
    previous_time_s: Optional[float],
) -> Dict:
    """Compute kinematic features between consecutive frames."""
    if previous_centroid_um is None or previous_time_s is None:
        return {
            'displacement_um': np.nan,
            'speed_um_per_s': np.nan,
            'delta_x_um': np.nan,
            'delta_y_um': np.nan,
            'delta_time_s': np.nan,
        }

    delta_x = current_centroid_um[0] - previous_centroid_um[0]
    delta_y = current_centroid_um[1] - previous_centroid_um[1]
    displacement = np.sqrt(delta_x**2 + delta_y**2)
    delta_time = current_time_s - previous_time_s
    speed = displacement / delta_time if delta_time > 0 else np.nan

    return {
        'displacement_um': float(displacement),
        'speed_um_per_s': float(speed),
        'delta_x_um': float(delta_x),
        'delta_y_um': float(delta_y),
        'delta_time_s': float(delta_time),
    }


def extract_pose_kinematics_batch(
    tracking_df: pd.DataFrame,
    mask_dir: Path | None = None,
    sort_by: list = ['embryo_id', 'time_int'],
    pixel_size_col: str = 'micrometers_per_pixel',
    mask_path_col: str = 'exported_mask_path',
) -> pd.DataFrame:
    """Extract pose and kinematics metrics for batch of snips."""
    def _resolve_time(row: pd.Series) -> float:
        for col in ('experiment_time_s', 'time_s'):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        raise ValueError(f"pose_kinematics: missing required time column for snip_id={row.get('snip_id')}")

    sort_cols = list(sort_by)
    if 'experiment_time_s' in tracking_df.columns and 'experiment_time_s' not in sort_cols:
        sort_cols = ['embryo_id', 'experiment_time_s']
    elif 'time_s' in tracking_df.columns and 'time_s' not in sort_cols:
        sort_cols = ['embryo_id', 'time_s']

    df_sorted = tracking_df.sort_values(by=sort_cols).reset_index(drop=True)
    results = []

    for idx, row in df_sorted.iterrows():
        snip_id = row['snip_id']
        embryo_id = row['embryo_id']
        mask_path = require_existing_path(
            row.get(mask_path_col),
            context='pose_kinematics',
            field_name=mask_path_col,
            row_id=str(snip_id),
        )
        mask = io.imread(mask_path)

        if pixel_size_col not in row.index or pd.isna(row[pixel_size_col]):
            raise ValueError(f"pose_kinematics: missing required pixel size column '{pixel_size_col}' for snip_id={snip_id}")
        pixel_size = float(row[pixel_size_col])
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(f"pose_kinematics: invalid pixel size {pixel_size!r} for snip_id={snip_id}")

        pose_stats = compute_mask_geometry(mask, float(pixel_size))
        orientation_stats = compute_pose_features(mask, float(pixel_size))
        current_centroid = (pose_stats.get('centroid_x_um', np.nan), pose_stats.get('centroid_y_um', np.nan))
        pose_features = {
            'snip_id': snip_id,
            'orientation_angle': orientation_stats.get('orientation_angle', np.nan),
            'bbox_width_um': pose_stats.get('width_um', np.nan),
            'bbox_height_um': pose_stats.get('length_um', np.nan),
        }

        current_time = _resolve_time(row)
        prev_idx = idx - 1 if idx > 0 and df_sorted.loc[idx - 1, 'embryo_id'] == embryo_id else None
        if prev_idx is not None:
            prev_row = df_sorted.loc[prev_idx]
            prev_mask_path = require_existing_path(
                prev_row.get(mask_path_col),
                context='pose_kinematics',
                field_name=mask_path_col,
                row_id=str(prev_row.get('snip_id')),
            )
            prev_mask = io.imread(prev_mask_path)
            if pixel_size_col not in prev_row.index or pd.isna(prev_row[pixel_size_col]):
                raise ValueError(f"pose_kinematics: missing required pixel size column '{pixel_size_col}' for snip_id={prev_row.get('snip_id')}")
            prev_pixel_size = float(prev_row[pixel_size_col])
            if not np.isfinite(prev_pixel_size) or prev_pixel_size <= 0:
                raise ValueError(f"pose_kinematics: invalid pixel size {prev_pixel_size!r} for snip_id={prev_row.get('snip_id')}")
            prev_pose_stats = compute_mask_geometry(prev_mask, float(prev_pixel_size))
            prev_centroid = (
                prev_pose_stats.get('centroid_x_um', np.nan),
                prev_pose_stats.get('centroid_y_um', np.nan),
            )
            prev_time = _resolve_time(prev_row)
            kinematics = compute_kinematics(current_centroid, prev_centroid, current_time, prev_time)
        else:
            kinematics = {
                'displacement_um': np.nan,
                'speed_um_per_s': np.nan,
                'delta_x_um': np.nan,
                'delta_y_um': np.nan,
                'delta_time_s': np.nan,
            }

        results.append({**pose_features, **kinematics})

    return pd.DataFrame(results)
