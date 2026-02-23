"""
Body Axis Spline Utilities

Functions for head/tail identification and body axis spline alignment.
These are helper functions used by the geodesic and PCA centerline extraction methods.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def identify_head_by_taper(mask: np.ndarray, body_axis_spline: np.ndarray,
                           n_samples: int = 20, window_size: int = 10) -> int:
    """
    Identify which end of body axis spline is the head based on width tapering.
    Width decreases from head to tail.

    Args:
        mask: Binary mask
        body_axis_spline: (N, 2) array of body axis points (x, y)
        n_samples: Number of points along body axis to sample
        window_size: Window for width measurement

    Returns:
        0 if body_axis_spline[0] is head, 1 if body_axis_spline[-1] is head
    """
    if len(body_axis_spline) < n_samples:
        n_samples = len(body_axis_spline)

    # Sample points along body axis spline
    indices = np.linspace(0, len(body_axis_spline)-1, n_samples, dtype=int)
    sample_points = body_axis_spline[indices]

    # Measure width at each point
    widths = []
    for i, point in enumerate(sample_points):
        if i == 0 or i == len(sample_points) - 1:
            # For endpoints, just measure local radius
            widths.append(_measure_local_width(mask, point, window_size))
        else:
            # For middle points, measure perpendicular width
            # Get tangent direction
            if i < len(sample_points) - 1:
                tangent = sample_points[i+1] - sample_points[i-1]
            else:
                tangent = sample_points[i] - sample_points[i-1]

            # Perpendicular direction
            perp = np.array([-tangent[1], tangent[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-10)

            # Measure width in perpendicular direction
            width = _measure_width_along_direction(mask, point, perp, window_size)
            widths.append(width)

    widths = np.array(widths)

    # Compute width gradient (positive = increasing toward end)
    # Use linear regression to get overall trend
    x = np.arange(len(widths))
    slope = np.polyfit(x, widths, 1)[0]

    # If slope is negative, width decreases from start to end
    # So start is head (return 0)
    # If slope is positive, width increases from start to end
    # So end is head (return 1)
    return 1 if slope > 0 else 0


def _measure_local_width(mask: np.ndarray, point: np.ndarray, radius: int) -> float:
    """Measure local width as diameter of circle around point."""
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    circle_mask = (x - point[0])**2 + (y - point[1])**2 <= radius**2
    area = np.sum(mask & circle_mask)
    # Approximate width as 2 * sqrt(area/pi)
    if area > 0:
        return 2 * np.sqrt(area / np.pi)
    return 0.0


def _measure_width_along_direction(mask: np.ndarray, point: np.ndarray,
                                    direction: np.ndarray, max_dist: int) -> float:
    """Measure width by scanning in perpendicular direction."""
    # Scan in both directions perpendicular to centerline
    dist_pos = 0
    dist_neg = 0

    # Positive direction
    for d in range(1, max_dist):
        p = point + d * direction
        if 0 <= p[1] < mask.shape[0] and 0 <= p[0] < mask.shape[1]:
            if mask[int(p[1]), int(p[0])]:
                dist_pos = d
            else:
                break
        else:
            break

    # Negative direction
    for d in range(1, max_dist):
        p = point - d * direction
        if 0 <= p[1] < mask.shape[0] and 0 <= p[0] < mask.shape[1]:
            if mask[int(p[1]), int(p[0])]:
                dist_neg = d
            else:
                break
        else:
            break

    return dist_pos + dist_neg


def align_spline_orientation(spline1_x: np.ndarray, spline1_y: np.ndarray,
                             spline2_x: np.ndarray, spline2_y: np.ndarray):
    """
    Align two splines to have the same head-to-tail orientation.

    Compares distance between endpoints to determine if splines
    are oriented the same way. If not, flips spline2.

    Args:
        spline1_x, spline1_y: First spline coordinates
        spline2_x, spline2_y: Second spline coordinates

    Returns:
        aligned_spline2_x, aligned_spline2_y: Possibly flipped spline2
        was_flipped: Boolean indicating if spline2 was flipped
    """
    if len(spline1_x) == 0 or len(spline2_x) == 0:
        return spline2_x, spline2_y, False

    # Get endpoints
    s1_start = np.array([spline1_x[0], spline1_y[0]])
    s1_end = np.array([spline1_x[-1], spline1_y[-1]])
    s2_start = np.array([spline2_x[0], spline2_y[0]])
    s2_end = np.array([spline2_x[-1], spline2_y[-1]])

    # Distance if aligned (start-to-start + end-to-end)
    dist_aligned = np.linalg.norm(s1_start - s2_start) + np.linalg.norm(s1_end - s2_end)

    # Distance if flipped (start-to-end + end-to-start)
    dist_flipped = np.linalg.norm(s1_start - s2_end) + np.linalg.norm(s1_end - s2_start)

    # If flipped distance is smaller, flip spline2
    if dist_flipped < dist_aligned:
        return spline2_x[::-1], spline2_y[::-1], True
    else:
        return spline2_x, spline2_y, False


def orient_spline_head_to_tail(spline_x: np.ndarray, spline_y: np.ndarray,
                               mask: np.ndarray) -> tuple:
    """
    Orient body axis spline from head to tail based on width tapering.

    Args:
        spline_x, spline_y: Body axis spline coordinates
        mask: Binary mask for width measurement

    Returns:
        oriented_x, oriented_y: Spline oriented head-to-tail
        was_flipped: Boolean indicating if spline was flipped
    """
    if len(spline_x) == 0:
        return spline_x, spline_y, False

    # Stack into (N, 2) array
    body_axis_spline = np.column_stack([spline_x, spline_y])

    # Determine which end is head
    head_idx = identify_head_by_taper(mask, body_axis_spline)

    # If head is at end (idx=1), flip the spline
    if head_idx == 1:
        return spline_x[::-1], spline_y[::-1], True
    else:
        return spline_x, spline_y, False
