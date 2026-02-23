"""
Pose and kinematics feature extraction from SAM2 tracking.

Computes centroid position, bounding box, orientation, displacement, and speed.
Extracted from build03A_process_images.py get_embryo_stats function (lines 820-833).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from skimage.measure import regionprops


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
    sort_by: list = ['embryo_id', 'frame_index'],
) -> pd.DataFrame:
    """
    Extract pose and kinematics metrics for batch of snips.

    Args:
        tracking_df: Tracking DataFrame with centroid, time, and metadata
        sort_by: Columns to sort by for temporal ordering

    Returns:
        DataFrame with pose and kinematics metrics per snip_id
    """
    # Sort by embryo and time
    df_sorted = tracking_df.sort_values(by=sort_by).reset_index(drop=True)

    results = []

    for idx, row in df_sorted.iterrows():
        snip_id = row['snip_id']
        embryo_id = row['embryo_id']

        # Current frame data
        current_centroid = (row.get('centroid_x_um', np.nan), row.get('centroid_y_um', np.nan))
        current_time = row.get('time_s', np.nan)

        # Extract pose features from current frame
        pose_features = {
            'snip_id': snip_id,
            'orientation_angle': row.get('orientation_angle', np.nan),
            'bbox_width_um': row.get('bbox_width_um', np.nan),
            'bbox_height_um': row.get('bbox_height_um', np.nan),
        }

        # Find previous frame for same embryo
        prev_idx = None
        if idx > 0 and df_sorted.loc[idx - 1, 'embryo_id'] == embryo_id:
            prev_idx = idx - 1

        # Compute kinematics
        if prev_idx is not None:
            prev_row = df_sorted.loc[prev_idx]
            prev_centroid = (prev_row.get('centroid_x_um', np.nan), prev_row.get('centroid_y_um', np.nan))
            prev_time = prev_row.get('time_s', np.nan)

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
