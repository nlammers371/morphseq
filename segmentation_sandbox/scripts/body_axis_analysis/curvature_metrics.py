"""
Simple Curvature Metrics for Embryo Morphology Analysis

This module provides intuitive, easy-to-interpret curvature measurements that can be
computed from spline coordinates after centerline extraction. These metrics are designed
to capture gross morphological differences without requiring complex mathematical operations.

The three main metrics:
1. Baseline Deviation: How far the midline deviates from a straight head-to-tail line
2. Arc-Length Ratio: Ratio of actual midline length to straight-line distance
3. Key Point Deviation: Distance of specific points (quarter, mid, 3-quarter) from baseline

Usage:
    from segmentation_sandbox.scripts.body_axis_analysis.curvature_metrics import (
        compute_all_simple_metrics
    )

    # spline_x, spline_y are 200-point arrays from centerline extraction
    metrics = compute_all_simple_metrics(spline_x, spline_y, um_per_pixel=3.23)

    print(f"Baseline deviation: {metrics['baseline_deviation_um']:.2f} μm")
    print(f"Arc-length ratio: {metrics['arc_length_ratio']:.3f}")
"""

import numpy as np
from typing import Dict, List, Tuple


def _point_to_line_distance(px: float, py: float,
                            x1: float, y1: float,
                            x2: float, y2: float) -> float:
    """
    Compute perpendicular distance from point (px, py) to line defined by (x1, y1) to (x2, y2).

    Uses the formula:
        distance = |ax + by + c| / sqrt(a^2 + b^2)

    Where line equation is: ax + by + c = 0
    Derived from two points as: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0

    Args:
        px, py: Coordinates of point
        x1, y1: First point defining line
        x2, y2: Second point defining line

    Returns:
        Perpendicular distance from point to line
    """
    # Coefficients of line equation ax + by + c = 0
    a = y2 - y1
    b = -(x2 - x1)
    c = (x2 - x1) * y1 - (y2 - y1) * x1

    # Distance formula
    numerator = np.abs(a * px + b * py + c)
    denominator = np.sqrt(a**2 + b**2)

    # Handle edge case where line is a point
    if denominator == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    return numerator / denominator


def compute_baseline_deviation(spline_x: np.ndarray, spline_y: np.ndarray,
                               um_per_pixel: float = 1.0) -> Dict[str, float]:
    """
    Compute average distance of midline from straight head-to-tail baseline.

    This metric measures how much the embryo's midline "bows out" from a straight line
    connecting its head and tail. Higher values indicate more curved embryos.

    Process:
    1. Draw straight line (baseline) from head (first point) to tail (last point)
    2. Compute perpendicular distance from each midline point to this baseline
    3. Calculate mean, max, and std of these distances

    Args:
        spline_x: Array of x coordinates (typically 200 points)
        spline_y: Array of y coordinates (typically 200 points)
        um_per_pixel: Conversion factor from pixels to microns

    Returns:
        dict with:
            - baseline_deviation_um: Mean perpendicular distance (microns)
            - max_baseline_deviation_um: Maximum perpendicular distance (microns)
            - baseline_deviation_std_um: Standard deviation of distances (microns)

    Example:
        >>> metrics = compute_baseline_deviation(spline_x, spline_y, um_per_pixel=3.23)
        >>> print(f"Mean deviation: {metrics['baseline_deviation_um']:.2f} μm")
    """
    # Head and tail coordinates (first and last points)
    head_x, head_y = spline_x[0], spline_y[0]
    tail_x, tail_y = spline_x[-1], spline_y[-1]

    # Compute perpendicular distance for each point along spline
    distances = np.array([
        _point_to_line_distance(spline_x[i], spline_y[i], head_x, head_y, tail_x, tail_y)
        for i in range(len(spline_x))
    ])

    # Convert to microns
    distances_um = distances * um_per_pixel

    return {
        'baseline_deviation_um': float(np.mean(distances_um)),
        'max_baseline_deviation_um': float(np.max(distances_um)),
        'baseline_deviation_std_um': float(np.std(distances_um))
    }


def compute_arc_length_ratio(spline_x: np.ndarray, spline_y: np.ndarray,
                             um_per_pixel: float = 1.0) -> Dict[str, float]:
    """
    Compute ratio of arc length (total midline length) to chord length (straight-line distance).

    This metric provides a simple normalized measure of overall curvature:
    - Ratio close to 1.0: Embryo is nearly straight
    - Ratio > 1.0: Embryo is curved (higher = more curved)

    The ratio is always >= 1.0 because the arc length is always at least as long as
    the chord (shortest distance between two points is a straight line).

    Args:
        spline_x: Array of x coordinates (typically 200 points)
        spline_y: Array of y coordinates (typically 200 points)
        um_per_pixel: Conversion factor from pixels to microns

    Returns:
        dict with:
            - arc_length_ratio: Total arc / chord length (unitless, >= 1.0)
            - arc_length_um: Total arc length (microns)
            - chord_length_um: Straight-line head-to-tail distance (microns)

    Example:
        >>> metrics = compute_arc_length_ratio(spline_x, spline_y, um_per_pixel=3.23)
        >>> print(f"Arc/chord ratio: {metrics['arc_length_ratio']:.3f}")
        >>> # ratio of 1.05 means embryo is 5% longer than straight line
    """
    # Compute arc length by summing distances between consecutive points
    dx = np.diff(spline_x)
    dy = np.diff(spline_y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    arc_length_px = np.sum(segment_lengths)

    # Compute chord length (straight line from head to tail)
    chord_length_px = np.sqrt(
        (spline_x[-1] - spline_x[0])**2 +
        (spline_y[-1] - spline_y[0])**2
    )

    # Convert to microns
    arc_length_um = arc_length_px * um_per_pixel
    chord_length_um = chord_length_px * um_per_pixel

    # Compute ratio (handle edge case of zero chord length)
    if chord_length_um > 0:
        arc_length_ratio = arc_length_um / chord_length_um
    else:
        arc_length_ratio = 1.0  # Degenerate case

    return {
        'arc_length_ratio': float(arc_length_ratio),
        'arc_length_um': float(arc_length_um),
        'chord_length_um': float(chord_length_um)
    }


def compute_key_point_deviations(spline_x: np.ndarray, spline_y: np.ndarray,
                                 um_per_pixel: float = 1.0,
                                 key_fractions: List[float] = [0.25, 0.5, 0.75]) -> Dict[str, float]:
    """
    Compute deviation at specific landmark points along the spline.

    This metric identifies where along the body the embryo is bending the most.
    By default, measures deviation at quarter (25%), mid (50%), and three-quarter (75%) points.

    Process:
    1. Identify key points along spline by fraction (e.g., 0.5 = midpoint)
    2. For each key point, compute perpendicular distance to head-tail baseline
    3. Report deviations at each landmark

    Args:
        spline_x: Array of x coordinates (typically 200 points)
        spline_y: Array of y coordinates (typically 200 points)
        um_per_pixel: Conversion factor from pixels to microns
        key_fractions: List of positions along spline (0.0 = head, 1.0 = tail)

    Returns:
        dict with deviation at each key fraction:
            - keypoint_deviation_q1_um: Distance at 25% point (microns)
            - keypoint_deviation_mid_um: Distance at 50% point (microns)
            - keypoint_deviation_q3_um: Distance at 75% point (microns)

    Example:
        >>> metrics = compute_key_point_deviations(spline_x, spline_y, um_per_pixel=3.23)
        >>> print(f"Mid-body deviation: {metrics['keypoint_deviation_mid_um']:.2f} μm")
    """
    # Head and tail coordinates
    head_x, head_y = spline_x[0], spline_y[0]
    tail_x, tail_y = spline_x[-1], spline_y[-1]

    n_points = len(spline_x)
    results = {}

    # Map fraction names for standard quartile points
    fraction_names = {
        0.25: 'q1',
        0.5: 'mid',
        0.75: 'q3'
    }

    for fraction in key_fractions:
        # Convert fraction to index (0.0 = 0, 1.0 = n_points-1)
        idx = int(np.round(fraction * (n_points - 1)))
        idx = np.clip(idx, 0, n_points - 1)  # Ensure valid index

        # Get point coordinates
        px, py = spline_x[idx], spline_y[idx]

        # Compute distance to baseline
        distance_px = _point_to_line_distance(px, py, head_x, head_y, tail_x, tail_y)
        distance_um = distance_px * um_per_pixel

        # Use descriptive name if available, otherwise use fraction
        name = fraction_names.get(fraction, f'f{fraction:.2f}')
        results[f'keypoint_deviation_{name}_um'] = float(distance_um)

    return results


def compute_all_simple_metrics(spline_x: np.ndarray, spline_y: np.ndarray,
                               um_per_pixel: float = 1.0,
                               key_fractions: List[float] = [0.25, 0.5, 0.75]) -> Dict[str, float]:
    """
    Convenience function to compute all simple curvature metrics at once.

    Combines results from:
    - compute_baseline_deviation()
    - compute_arc_length_ratio()
    - compute_key_point_deviations()

    Args:
        spline_x: Array of x coordinates (typically 200 points)
        spline_y: Array of y coordinates (typically 200 points)
        um_per_pixel: Conversion factor from pixels to microns
        key_fractions: List of positions for key point deviations

    Returns:
        dict with all metrics combined (10 total metrics by default):
            From baseline deviation:
                - baseline_deviation_um
                - max_baseline_deviation_um
                - baseline_deviation_std_um
            From arc-length ratio:
                - arc_length_ratio
                - arc_length_um
                - chord_length_um
            From key point deviations:
                - keypoint_deviation_q1_um
                - keypoint_deviation_mid_um
                - keypoint_deviation_q3_um

    Example:
        >>> metrics = compute_all_simple_metrics(spline_x, spline_y, um_per_pixel=3.23)
        >>> for key, value in metrics.items():
        ...     print(f"{key}: {value:.3f}")
    """
    results = {}

    # Compute all three metric groups
    results.update(compute_baseline_deviation(spline_x, spline_y, um_per_pixel))
    results.update(compute_arc_length_ratio(spline_x, spline_y, um_per_pixel))
    results.update(compute_key_point_deviations(spline_x, spline_y, um_per_pixel, key_fractions))

    return results


# Helper function for visualizations
def get_baseline_coordinates(spline_x: np.ndarray, spline_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get coordinates for plotting the head-to-tail baseline.

    Args:
        spline_x: Array of x coordinates
        spline_y: Array of y coordinates

    Returns:
        baseline_x: Array of 2 x-coordinates [head_x, tail_x]
        baseline_y: Array of 2 y-coordinates [head_y, tail_y]

    Example:
        >>> baseline_x, baseline_y = get_baseline_coordinates(spline_x, spline_y)
        >>> plt.plot(baseline_x, baseline_y, 'b--', label='Baseline')
    """
    return np.array([spline_x[0], spline_x[-1]]), np.array([spline_y[0], spline_y[-1]])
