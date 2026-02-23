"""
Body Axis Spline Utilities

Functions for head/tail identification, body axis spline alignment,
and skeleton preprocessing for geodesic analysis.
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


def prune_skeleton_for_geodesic(skeleton: np.ndarray, mask: np.ndarray,
                                 min_branch_length_fraction: float = 0.10,
                                 min_width_fraction: float = 0.50,
                                 min_aspect_ratio: float = 3.0,
                                 max_branch_angle: float = 60.0) -> tuple:
    """
    Remove SHORT branches in thin regions and SHARP-ANGLED branches using adaptive pruning.

    This function segments the skeleton into branches and removes those that are either:
    1. Simultaneously short, thin, AND stubby (low aspect ratio), OR
    2. Branch off at sharp angles from the main body axis (perpendicular fins)

    This preserves long thin structures like tails and body continuations while removing
    short thin protrusions and perpendicular fins.

    Strategy:
    1. Compute adaptive thresholds normalized to embryo size
    2. Segment skeleton into individual branches from endpoints
    3. Identify longest branch (main body axis)
    4. For each branch, analyze: length, width, aspect ratio, and angle
    5. Remove branches that meet removal criteria
    6. Preserve branches that continue body direction

    Args:
        skeleton: Binary skeleton image (2D numpy array)
        mask: Binary mask image (2D numpy array, same shape as skeleton)
        min_branch_length_fraction: Fraction of embryo major axis length.
            Branches shorter than this are candidates for removal. Default 0.10 (10%).
        min_width_fraction: Fraction of median skeleton width.
            Branches thinner than this are candidates for removal. Default 0.50 (50%).
        min_aspect_ratio: Minimum length/width ratio to preserve.
            Branches with lower aspect ratio are stubby (fins). Default 3.0.
        max_branch_angle: Maximum angle (degrees) from main trunk.
            Branches at sharper angles are perpendicular fins. Default 60.0°.

    Returns:
        pruned_skeleton: Skeleton with short thin branches removed (2D numpy array)
        pruning_stats: Dictionary with diagnostic information:
            - 'min_branch_length': Computed minimum branch length in pixels
            - 'width_threshold': Computed width threshold in pixels
            - 'embryo_length': Embryo major axis length in pixels
            - 'median_width': Median skeleton width in pixels
            - 'n_branches_analyzed': Number of branches detected
            - 'n_branches_removed': Number of branches removed
            - 'original_skeleton_pixels': Number of pixels in original skeleton
            - 'pruned_skeleton_pixels': Number of pixels after pruning
            - 'removed_pixels': Number of pixels removed
            - 'removed_fraction': Fraction of skeleton removed (0-1)

    Example:
        >>> from skimage.morphology import skeletonize
        >>> skeleton = skeletonize(mask)
        >>> pruned_skel, stats = prune_skeleton_for_geodesic(skeleton, mask)
        >>> print(f"Removed {stats['n_branches_removed']} branches ({stats['removed_fraction']*100:.1f}% of pixels)")
    """
    from scipy.ndimage import label, convolve
    from skimage.measure import regionprops

    # Edge case: empty skeleton
    if skeleton.sum() == 0:
        return skeleton.copy(), {
            'min_branch_length': 0,
            'width_threshold': 0,
            'embryo_length': 0,
            'median_width': 0,
            'n_branches_analyzed': 0,
            'n_branches_removed': 0,
            'original_skeleton_pixels': 0,
            'pruned_skeleton_pixels': 0,
            'removed_pixels': 0,
            'removed_fraction': 0
        }

    # 1. Compute adaptive thresholds normalized to embryo size
    props = regionprops(mask.astype(int))[0]
    embryo_length = props.major_axis_length

    distance_map = distance_transform_edt(mask)
    skeleton_widths = 2 * distance_map[skeleton]
    median_width = np.median(skeleton_widths)

    min_branch_length = embryo_length * min_branch_length_fraction
    width_threshold = median_width * min_width_fraction

    # 2. Find endpoints and branch points
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')

    endpoints = (neighbor_count == 1) & skeleton  # Endpoints (degree 1)
    branch_points = (neighbor_count > 2) & skeleton  # Junctions (degree 3+)

    # 3. Trace branches from each endpoint to first junction
    # A "branch" is a path from an endpoint to a junction (or to another endpoint if no junctions)
    branches = _trace_branches_from_endpoints(skeleton, endpoints, branch_points)
    n_branches = len(branches)

    # 3b. Find longest branch (main body axis/trunk)
    longest_branch_idx = 0
    max_length = 0
    for i, branch in enumerate(branches):
        branch_array = np.array(branch)
        length = _compute_branch_length(branch_array)
        if length > max_length:
            max_length = length
            longest_branch_idx = i

    main_trunk = branches[longest_branch_idx] if n_branches > 0 else []

    # 4. Analyze each branch and decide whether to keep or remove
    pruned_skeleton = skeleton.copy()
    n_branches_removed = 0
    n_branches_removed_by_angle = 0
    n_branches_removed_by_size = 0

    for branch_coords in branches:
        if len(branch_coords) < 2:
            # Too small to analyze meaningfully, keep it
            continue

        # Convert branch coords from list of tuples to numpy array
        branch_coords_array = np.array(branch_coords)

        # Measure branch properties
        branch_length = _compute_branch_length(branch_coords_array)

        # Get mask for this branch
        branch_mask = np.zeros_like(skeleton, dtype=bool)
        for coord in branch_coords_array:
            branch_mask[coord[0], coord[1]] = True

        branch_widths = 2 * distance_map[branch_mask]
        mean_branch_width = np.mean(branch_widths)

        # Compute aspect ratio (length / width)
        aspect_ratio = branch_length / (mean_branch_width + 1e-6)

        # Compute angle from main trunk (if this isn't the main trunk)
        branch_idx = branches.index(branch_coords)
        angle_from_trunk = 0.0
        if branch_idx != longest_branch_idx and len(main_trunk) > 0:
            angle_from_trunk = _compute_branch_angle(branch_coords, main_trunk, branch_points)

        # Decision logic: Remove if EITHER condition is met
        is_short = branch_length < min_branch_length
        is_thin = mean_branch_width < width_threshold
        is_stubby = aspect_ratio < min_aspect_ratio
        is_sharp_angle = angle_from_trunk > max_branch_angle

        # Criteria 1: Short AND thin AND stubby
        remove_by_size = is_short and is_thin and is_stubby

        # Criteria 2: Sharp angle from main trunk
        remove_by_angle = is_sharp_angle and branch_idx != longest_branch_idx

        if remove_by_size or remove_by_angle:
            # This is likely a fin → remove
            pruned_skeleton[branch_mask] = 0
            n_branches_removed += 1
            if remove_by_angle:
                n_branches_removed_by_angle += 1
            if remove_by_size:
                n_branches_removed_by_size += 1
        # Otherwise keep (long OR thick OR elongated OR continues body direction)

    # 5. Compute statistics
    original_pixels = int(skeleton.sum())
    pruned_pixels = int(pruned_skeleton.sum())
    removed_pixels = original_pixels - pruned_pixels
    removed_fraction = removed_pixels / original_pixels if original_pixels > 0 else 0

    pruning_stats = {
        'min_branch_length': float(min_branch_length),
        'width_threshold': float(width_threshold),
        'embryo_length': float(embryo_length),
        'median_width': float(median_width),
        'max_branch_angle': float(max_branch_angle),
        'n_branches_analyzed': int(n_branches),
        'n_branches_removed': int(n_branches_removed),
        'n_branches_removed_by_angle': int(n_branches_removed_by_angle),
        'n_branches_removed_by_size': int(n_branches_removed_by_size),
        'original_skeleton_pixels': original_pixels,
        'pruned_skeleton_pixels': pruned_pixels,
        'removed_pixels': removed_pixels,
        'removed_fraction': float(removed_fraction)
    }

    return pruned_skeleton, pruning_stats


def _compute_branch_angle(branch, main_trunk, branch_points):
    """
    Compute angle (in degrees) at which a branch deviates from the main trunk.

    Args:
        branch: List of (y, x) coordinates for the branch
        main_trunk: List of (y, x) coordinates for the main trunk
        branch_points: Binary mask of junction points

    Returns:
        Angle in degrees (0-180). Higher = sharper deviation.
    """
    if len(branch) < 2 or len(main_trunk) < 2:
        return 0.0

    # Find junction point (last point of branch)
    junction = branch[-1]

    # Get branch direction (last few points before junction)
    branch_direction = _get_branch_direction_at_junction(branch, junction)

    # Get trunk direction (continuing past junction)
    trunk_direction = _get_trunk_direction_at_junction(main_trunk, junction)

    if branch_direction is None or trunk_direction is None:
        return 0.0

    # Compute angle between vectors
    angle = _angle_between_vectors(branch_direction, trunk_direction)

    return angle


def _get_branch_direction_at_junction(branch, junction):
    """Get direction vector of branch approaching the junction."""
    n_points = min(5, len(branch))
    if n_points < 2:
        return None

    start_idx = max(0, len(branch) - n_points)
    start_point = np.array(branch[start_idx])
    end_point = np.array(branch[-1])

    direction = end_point - start_point
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None

    return direction / norm


def _get_trunk_direction_at_junction(main_trunk, junction):
    """Get direction vector of trunk continuing past the junction."""
    # Find junction in trunk
    junction_idx = None
    for i, coord in enumerate(main_trunk):
        if coord == junction or (isinstance(coord, (tuple, list, np.ndarray)) and
                                 isinstance(junction, (tuple, list, np.ndarray)) and
                                 len(coord) == 2 and len(junction) == 2 and
                                 coord[0] == junction[0] and coord[1] == junction[1]):
            junction_idx = i
            break

    if junction_idx is None:
        return None

    # Get direction continuing PAST junction
    n_points = min(5, len(main_trunk) - junction_idx - 1)
    if n_points < 1:
        # Junction is at end of trunk, use direction approaching junction
        n_points = min(5, junction_idx)
        if n_points < 1:
            return None
        start_idx = max(0, junction_idx - n_points)
        start_point = np.array(main_trunk[start_idx])
        end_point = np.array(main_trunk[junction_idx])
    else:
        # Normal case: direction continuing past junction
        start_point = np.array(main_trunk[junction_idx])
        end_point = np.array(main_trunk[junction_idx + n_points])

    direction = end_point - start_point
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None

    return direction / norm


def _angle_between_vectors(v1, v2):
    """Compute angle in degrees between two direction vectors."""
    if v1 is None or v2 is None:
        return 0.0

    # Dot product gives cos(angle)
    cos_angle = np.dot(v1, v2)

    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def _trace_branches_from_endpoints(skeleton: np.ndarray,
                                    endpoints: np.ndarray,
                                    branch_points: np.ndarray) -> list:
    """
    Trace branches from each endpoint to the first junction.

    A "branch" is defined as a path from an endpoint (degree-1 node) to
    the first junction point (degree-3+ node) encountered.

    Args:
        skeleton: Binary skeleton image
        endpoints: Binary mask of endpoint pixels (degree 1)
        branch_points: Binary mask of junction pixels (degree 3+)

    Returns:
        List of branches, where each branch is a list of (y, x) coordinates
    """
    branches = []
    visited = np.zeros_like(skeleton, dtype=bool)

    endpoint_coords = np.argwhere(endpoints)

    for endpoint in endpoint_coords:
        if visited[endpoint[0], endpoint[1]]:
            continue

        # Trace from this endpoint until we hit a junction or another visited pixel
        # Note: _trace_single_branch marks pixels as visited internally
        branch = _trace_single_branch(skeleton, endpoint, branch_points, visited)

        if len(branch) > 0:
            branches.append(branch)

    return branches


def _trace_single_branch(skeleton: np.ndarray,
                         start_point: np.ndarray,
                         branch_points: np.ndarray,
                         visited: np.ndarray) -> list:
    """
    Trace a single branch from a starting point until hitting a junction.

    Args:
        skeleton: Binary skeleton image
        start_point: (y, x) coordinate to start from
        branch_points: Binary mask of junction points
        visited: Binary mask of already-visited pixels

    Returns:
        List of (y, x) coordinates along the branch
    """
    branch = [tuple(start_point)]
    current = start_point

    # Mark start as visited immediately
    visited[current[0], current[1]] = True

    max_iterations = 10000  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Find unvisited neighbors on skeleton
        neighbors = _get_skeleton_neighbors(skeleton, current, visited)

        if len(neighbors) == 0:
            # Dead end - no more unvisited neighbors
            break

        if len(neighbors) > 1:
            # Multiple unvisited neighbors - shouldn't happen if we trace properly
            # Just pick the first one
            pass

        # Move to next neighbor
        current = neighbors[0]
        branch.append(tuple(current))

        # Mark as visited IMMEDIATELY to prevent loops
        visited[current[0], current[1]] = True

        # Stop if we hit a junction point (but include it in the branch)
        if branch_points[current[0], current[1]]:
            break

    if iteration >= max_iterations:
        print(f"Warning: Branch tracing hit max iterations ({max_iterations})")

    return branch


def _get_skeleton_neighbors(skeleton: np.ndarray,
                            point: np.ndarray,
                            visited: np.ndarray) -> list:
    """
    Get 8-connected skeleton neighbors of a point that haven't been visited.

    Args:
        skeleton: Binary skeleton image
        point: (y, x) coordinate
        visited: Binary mask of visited pixels

    Returns:
        List of (y, x) neighbor coordinates
    """
    neighbors = []
    y, x = point

    # Check 8-connected neighbors
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue

            ny, nx = y + dy, x + dx

            # Check bounds
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                # Check if it's on skeleton and not visited
                if skeleton[ny, nx] and not visited[ny, nx]:
                    neighbors.append(np.array([ny, nx]))

    return neighbors


def _compute_branch_length(branch_coords: np.ndarray) -> float:
    """
    Compute geodesic length along a skeleton branch.

    Approximates length by sorting coordinates along the branch path
    and summing Euclidean distances between consecutive points.

    Args:
        branch_coords: (N, 2) array of skeleton coordinates

    Returns:
        Branch length in pixels
    """
    if len(branch_coords) <= 1:
        return 0.0

    # Sort coordinates to trace along branch path
    sorted_coords = _sort_coords_along_path(branch_coords)

    # Sum distances between consecutive points
    length = 0.0
    for i in range(len(sorted_coords) - 1):
        dist = np.linalg.norm(sorted_coords[i+1] - sorted_coords[i])
        length += dist

    return length


def _sort_coords_along_path(coords: np.ndarray) -> np.ndarray:
    """
    Sort skeleton coordinates to trace along branch path.

    Uses greedy nearest-neighbor chaining to order coordinates.

    Args:
        coords: (N, 2) array of skeleton coordinates

    Returns:
        Sorted coordinates tracing the branch path
    """
    if len(coords) <= 1:
        return coords

    # Start from first coordinate
    sorted_coords = [coords[0]]
    remaining = list(coords[1:])

    # Greedily chain nearest neighbors
    while remaining:
        current = sorted_coords[-1]
        # Find nearest remaining point
        distances = [np.linalg.norm(current - r) for r in remaining]
        nearest_idx = np.argmin(distances)
        sorted_coords.append(remaining[nearest_idx])
        remaining.pop(nearest_idx)

    return np.array(sorted_coords)
