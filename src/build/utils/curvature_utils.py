"""
Curvature Analysis Utilities for Build04 Pipeline

Provides functions to compute curvature metrics for cleaned embryo masks.
Extracted from segmentation_sandbox/scripts/body_axis_analysis/process_curvature_batch.py
for integration into the Build04 stage inference pipeline.

Usage:
    from src.build.utils.curvature_utils import compute_embryo_curvature

    # Compute curvature metrics for a cleaned mask
    curvature_metrics = compute_embryo_curvature(
        cleaned_mask,
        um_per_pixel=0.5,
        verbose=False
    )

    # Metrics include:
    # - total_length_um, mean_curvature_per_um, std_curvature_per_um, max_curvature_per_um
    # - n_centerline_points
    # - baseline_deviation_um, max_baseline_deviation_um, arc_length_ratio, etc.
"""

import numpy as np
import time
from typing import Dict, Tuple
import traceback
import sys
import json
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline
from segmentation_sandbox.scripts.body_axis_analysis.curvature_metrics import compute_all_simple_metrics


def get_nan_metrics_dict(include_arrays: bool = False) -> Dict:
    """
    Returns a dictionary with all curvature metrics set to NaN.
    Used when centerline extraction fails.

    Args:
        include_arrays: If True, include array columns (as None/empty)

    Returns:
        Dict with keys for all curvature metrics, all set to NaN
    """
    metrics = {
        'total_length_um': np.nan,
        'mean_curvature_per_um': np.nan,
        'std_curvature_per_um': np.nan,
        'max_curvature_per_um': np.nan,
        'n_centerline_points': 0,
        'baseline_deviation_um': np.nan,
        'max_baseline_deviation_um': np.nan,
        'baseline_deviation_std_um': np.nan,
        'arc_length_ratio': np.nan,
        'arc_length_um': np.nan,
        'chord_length_um': np.nan,
        'keypoint_deviation_q1_um': np.nan,
        'keypoint_deviation_mid_um': np.nan,
        'keypoint_deviation_q3_um': np.nan,
    }

    if include_arrays:
        metrics.update({
            'centerline_x_json': None,
            'centerline_y_json': None,
            'curvature_values_json': None,
            'arc_length_values_json': None,
        })

    return metrics


def compute_embryo_curvature(
    cleaned_mask: np.ndarray,
    um_per_pixel: float,
    bspline_smoothing: float = 5.0,
    verbose: bool = False,
    include_arrays: bool = True
) -> Dict:
    """
    Compute curvature metrics for a single cleaned embryo mask.

    Pipeline:
    1. Extract centerline using geodesic method
    2. Calculate B-spline curvature along centerline
    3. Compute summary statistics (mean, std, max curvature)
    4. Compute simple metrics (baseline deviation, arc-length ratio, keypoint deviations)

    Args:
        cleaned_mask: Binary mask array (H, W) with embryo segmentation
                     Expected to be cleaned by clean_embryo_mask()
        um_per_pixel: Conversion factor to convert pixels to micrometers
        bspline_smoothing: B-spline smoothing parameter (default 5.0)
        verbose: If True, print processing information
        include_arrays: If True, include JSON-serialized centerline and curvature arrays

    Returns:
        Dict with curvature metrics:
            - total_length_um: Total centerline length in micrometers
            - mean_curvature_per_um: Mean curvature per micrometer
            - std_curvature_per_um: Standard deviation of curvature
            - max_curvature_per_um: Maximum curvature per micrometer
            - n_centerline_points: Number of points on centerline
            - baseline_deviation_um: Average perpendicular deviation from baseline
            - max_baseline_deviation_um: Maximum perpendicular deviation
            - baseline_deviation_std_um: Standard deviation of baseline deviations
            - arc_length_ratio: Ratio of arc length to chord length
            - arc_length_um: Total arc length in micrometers
            - chord_length_um: Straight-line distance between endpoints
            - keypoint_deviation_q1_um: Deviation at 25th percentile
            - keypoint_deviation_mid_um: Deviation at midpoint (50th percentile)
            - keypoint_deviation_q3_um: Deviation at 75th percentile
            - curvature_success: True if extraction succeeded, False otherwise
            - processing_time_s: Time taken to compute metrics
    """
    start_time = time.time()

    try:
        # Extract centerline using geodesic method
        spline_x, spline_y, curvature, arc_length = extract_centerline(
            cleaned_mask,
            method='geodesic',
            um_per_pixel=um_per_pixel,
            bspline_smoothing=bspline_smoothing
        )

        # Check if extraction succeeded
        if len(curvature) == 0:
            if verbose:
                print("  ⚠️  Centerline extraction returned empty curvature")
            metrics = get_nan_metrics_dict(include_arrays=include_arrays)
            metrics['curvature_success'] = False
        else:
            # Calculate summary statistics
            metrics = {
                'total_length_um': float(arc_length[-1]) if len(arc_length) > 0 else np.nan,
                'mean_curvature_per_um': float(np.mean(curvature)),
                'std_curvature_per_um': float(np.std(curvature)),
                'max_curvature_per_um': float(np.max(curvature)),
                'n_centerline_points': len(spline_x),
            }

            # Compute simple metrics (baseline deviation, arc-length ratio, keypoint deviations)
            simple_metrics = compute_all_simple_metrics(
                spline_x, spline_y, um_per_pixel=um_per_pixel
            )
            metrics.update(simple_metrics)
            metrics['curvature_success'] = True

            # Add JSON-serialized arrays if requested
            if include_arrays:
                metrics['centerline_x_json'] = json.dumps(spline_x.tolist() if isinstance(spline_x, np.ndarray) else spline_x)
                metrics['centerline_y_json'] = json.dumps(spline_y.tolist() if isinstance(spline_y, np.ndarray) else spline_y)
                metrics['curvature_values_json'] = json.dumps(curvature.tolist() if isinstance(curvature, np.ndarray) else curvature)
                metrics['arc_length_values_json'] = json.dumps(arc_length.tolist() if isinstance(arc_length, np.ndarray) else arc_length)

            if verbose:
                print(f"  ✓ Curvature computed: length={metrics['total_length_um']:.2f}um, "
                      f"mean_curve={metrics['mean_curvature_per_um']:.3f}/um")

    except Exception as e:
        if verbose:
            print(f"  ❌ Curvature computation failed: {str(e)}")
            traceback.print_exc()

        metrics = get_nan_metrics_dict(include_arrays=include_arrays)
        metrics['curvature_success'] = False
        metrics['error'] = str(e)

    metrics['processing_time_s'] = time.time() - start_time
    return metrics
