"""
Pose and kinematics feature extraction from SAM2 tracking.

Computes centroid position, bounding box, orientation, displacement, and speed.
Extracted from build03A_process_images.py get_embryo_stats function (lines 820-833).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import skimage.io as io
from skimage.measure import regionprops

from .mask_geometry_metrics import compute_mask_geometry


def compute_pose_features(
    mask: np.ndarray,
    pixel_size_um: float,
) -> Dict:
    """
    Compute pose features from binary mask.

    Args:
        mask: Binary embryo mask (H x W)
        pixel_size_um: Pixel size in micrometers

    Returns:
        Dictionary with pose metrics
    """
    mask_binary = (mask > 0).astype(np.uint8)

    # Use regionprops for robust measurements
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

    # Orientation angle (radians)
    orientation = prop.orientation

    # Bounding box in pixels
    min_row, min_col, max_row, max_col = prop.bbox

    # Convert to micrometers
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
    """
    Compute kinematic features between consecutive frames.

    Args:
        current_centroid_um: (x, y) centroid in micrometers
        previous_centroid_um: Previous (x, y) centroid in micrometers
        current_time_s: Current time in seconds
        previous_time_s: Previous time in seconds

    Returns:
        Dictionary with kinematics metrics
    """
    if previous_centroid_um is None or previous_time_s is None:
        return {
            'displacement_um': np.nan,
            'speed_um_per_s': np.nan,
            'delta_x_um': np.nan,
            'delta_y_um': np.nan,
            'delta_time_s': np.nan,
        }

    # Compute displacement
    delta_x = current_centroid_um[0] - previous_centroid_um[0]
    delta_y = current_centroid_um[1] - previous_centroid_um[1]
    displacement = np.sqrt(delta_x**2 + delta_y**2)

    # Compute time delta
    delta_time = current_time_s - previous_time_s

    # Compute speed (avoid division by zero)
    if delta_time > 0:
        speed = displacement / delta_time
    else:
        speed = np.nan

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
    sort_by: list = ['embryo_id', 'frame_index'],
    pixel_size_col: str = 'micrometers_per_pixel',
    mask_path_col: str = 'embryo_mask_path',
) -> pd.DataFrame:
    """
    Extract pose and kinematics metrics for batch of snips.

    Args:
        tracking_df: Tracking DataFrame with mask paths and metadata
        mask_dir: Directory containing mask images, used as fallback when
            explicit per-row mask paths are missing.
        sort_by: Columns to sort by for temporal ordering
        pixel_size_col: Column name for pixel size
        mask_path_col: Column name containing explicit per-row mask paths

    Returns:
        DataFrame with pose and kinematics metrics per snip_id
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

    def _resolve_time(row: pd.Series) -> float:
        for col in ('experiment_time_s', 'time_s', 'time_int'):
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        return float('nan')

    sort_cols = list(sort_by)
    if 'experiment_time_s' in tracking_df.columns and 'experiment_time_s' not in sort_cols:
        sort_cols = ['embryo_id', 'experiment_time_s']
    elif 'time_s' in tracking_df.columns and 'time_s' not in sort_cols:
        sort_cols = ['embryo_id', 'time_s']

    # Sort by embryo and time
    df_sorted = tracking_df.sort_values(by=sort_cols).reset_index(drop=True)

    results = []

    for idx, row in df_sorted.iterrows():
        snip_id = row['snip_id']
        embryo_id = row['embryo_id']

        # Resolve mask and compute current pose from the raw mask geometry.
        mask_path = _resolve_mask_path(row)
        if mask_path is not None and mask_path.exists():
            mask = io.imread(mask_path)
            pixel_size = row.get(pixel_size_col, row.get('source_micrometers_per_pixel', np.nan))
            pixel_size = float(pixel_size)
            if not np.isfinite(pixel_size) or pixel_size <= 0:
                raise ValueError(f"Invalid pixel size: {pixel_size}")
            pose_stats = compute_mask_geometry(mask, float(pixel_size))
            orientation_stats = compute_pose_features(mask, float(pixel_size))
            current_centroid = (pose_stats.get('centroid_x_um', np.nan), pose_stats.get('centroid_y_um', np.nan))
            pose_features = {
                'snip_id': snip_id,
                'orientation_angle': orientation_stats.get('orientation_angle', np.nan),
                'bbox_width_um': pose_stats.get('width_um', np.nan),
                'bbox_height_um': pose_stats.get('length_um', np.nan),
            }
        else:
            current_centroid = (np.nan, np.nan)
            pose_features = {
                'snip_id': snip_id,
                'orientation_angle': np.nan,
                'bbox_width_um': np.nan,
                'bbox_height_um': np.nan,
            }

        current_time = _resolve_time(row)

        # Find previous frame for same embryo
        prev_idx = None
        if idx > 0 and df_sorted.loc[idx - 1, 'embryo_id'] == embryo_id:
            prev_idx = idx - 1

        # Compute kinematics
        if prev_idx is not None:
            prev_row = df_sorted.loc[prev_idx]
            prev_mask_path = _resolve_mask_path(prev_row)
            if prev_mask_path is not None and prev_mask_path.exists():
                prev_mask = io.imread(prev_mask_path)
                prev_pixel_size = prev_row.get(pixel_size_col, prev_row.get('source_micrometers_per_pixel', np.nan))
                prev_pixel_size = float(prev_pixel_size)
                if not np.isfinite(prev_pixel_size) or prev_pixel_size <= 0:
                    raise ValueError(f"Invalid pixel size: {prev_pixel_size}")
                prev_pose_stats = compute_mask_geometry(prev_mask, float(prev_pixel_size))
                prev_centroid = (
                    prev_pose_stats.get('centroid_x_um', np.nan),
                    prev_pose_stats.get('centroid_y_um', np.nan),
                )
            else:
                prev_centroid = (np.nan, np.nan)
            prev_time = _resolve_time(prev_row)

            kinematics = compute_kinematics(
                current_centroid,
                prev_centroid,
                current_time,
                prev_time,
            )
        else:
            kinematics = {
                'displacement_um': np.nan,
                'speed_um_per_s': np.nan,
                'delta_x_um': np.nan,
                'delta_y_um': np.nan,
                'delta_time_s': np.nan,
            }

        # Combine features
        result = {**pose_features, **kinematics}
        results.append(result)

    return pd.DataFrame(results)
