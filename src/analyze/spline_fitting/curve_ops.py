"""Curve geometry operations: discretization, segmentation, and projection.

This module provides all operations on discretized curves including:
- Splitting curves into segments by arc-length
- Assigning data points to segments
- Projecting points onto curves and planes
- Computing curve-related geometric quantities

These operations are tightly coupled because they all work with the discretized
curve representation and share common geometric primitives.

Example:
    >>> from src.analyze.spline_fitting import create_spline_segments_for_df
    >>>
    >>> # Segment a trajectory into developmental stages
    >>> segmented_df = create_spline_segments_for_df(
    ...     df, n_segments=5,
    ...     segment_labels=['early', 'mid1', 'mid2', 'mid3', 'late']
    ... )
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from tqdm import tqdm


# =============================================================================
# Core Geometric Primitives
# =============================================================================

def compute_spline_distances(spline_pts):
    """Compute cumulative arc-length along discretized curve.

    Parameters
    ----------
    spline_pts : ndarray, shape (n_points, n_dims)
        Curve coordinates.

    Returns
    -------
    cumdist : ndarray, shape (n_points,)
        Cumulative distance at each point (starts at 0).
    total_dist : float
        Total curve length.
    """
    diffs = np.diff(spline_pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumdist = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    return cumdist, cumdist[-1]


def point_to_segment_distance(point, seg_start, seg_end):
    """Compute minimum distance from 3D point to line segment.

    Parameters
    ----------
    point : ndarray, shape (3,)
        Query point.
    seg_start, seg_end : ndarray, shape (3,)
        Segment endpoints.

    Returns
    -------
    distance : float
        Minimum distance from point to segment.
    """
    seg_vec = seg_end - seg_start
    pt_vec = point - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq == 0.0:
        return np.linalg.norm(point - seg_start)
    t = np.dot(pt_vec, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    projection = seg_start + t * seg_vec
    return np.linalg.norm(point - projection)


def point_to_segment_projection(p, seg_start, seg_end):
    """Project point onto line segment and compute distance.

    Parameters
    ----------
    p : ndarray, shape (3,)
        Query point.
    seg_start, seg_end : ndarray, shape (3,)
        Segment endpoints.

    Returns
    -------
    closest : ndarray, shape (3,)
        Closest point on segment to p.
    dist : float
        Distance from p to closest point.
    """
    seg_vec = seg_end - seg_start
    pt_vec = p - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq == 0.0:
        return seg_start, np.linalg.norm(p - seg_start)

    t = np.dot(pt_vec, seg_vec) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    closest = seg_start + t_clamped * seg_vec
    dist = np.linalg.norm(p - closest)
    return closest, dist


# =============================================================================
# Curve Segmentation
# =============================================================================

def split_spline(spline_pts, k):
    """Split curve into k segments of equal arc-length.

    Parameters
    ----------
    spline_pts : ndarray, shape (n_points, n_dims)
        Discretized curve coordinates.
    k : int
        Number of segments.

    Returns
    -------
    segment_list : list of (ndarray, ndarray)
        List of (start_pt, end_pt) tuples, length k.
    """
    cumdist, total_dist = compute_spline_distances(spline_pts)
    segment_distances = np.linspace(0, total_dist, k+1)
    segment_indices = np.searchsorted(cumdist, segment_distances)
    segment_list = []
    for i in range(k):
        i0 = min(segment_indices[i], len(spline_pts) - 1)
        i1 = min(segment_indices[i+1], len(spline_pts) - 1)
        start_pt = spline_pts[i0]
        end_pt = spline_pts[i1]
        segment_list.append((start_pt, end_pt))
    return segment_list


def assign_points_to_segments(pert_df, segment_list, coord_cols=None):
    """Assign data points to nearest spline segment.

    Parameters
    ----------
    pert_df : pd.DataFrame
        Data points with coordinate columns.
    segment_list : list of (ndarray, ndarray)
        List of (start_pt, end_pt) from split_spline().
    coord_cols : list of str, optional
        Column names for coordinates. Defaults to ['PCA_1', 'PCA_2', 'PCA_3'].

    Returns
    -------
    pert_df : pd.DataFrame
        Input DataFrame with added 'segment_id' column.
    """
    if coord_cols is None:
        coord_cols = ["PCA_1", "PCA_2", "PCA_3"]

    points = pert_df[coord_cols].values
    segment_midpoints = np.array([(s + e) / 2 for s, e in segment_list])

    # Compute all distances at once
    distances = cdist(points, segment_midpoints)
    segment_ids = distances.argmin(axis=1)

    pert_df["segment_id"] = segment_ids
    return pert_df


def perform_robust_pca(seg_points):
    """Compute principal axis using robust covariance estimation.

    Uses Minimum Covariance Determinant (MCD) for robustness to outliers.
    Falls back to standard PCA if MCD fails.

    Parameters
    ----------
    seg_points : ndarray, shape (n_points, n_dims)
        Points in segment.

    Returns
    -------
    principal_axis : ndarray, shape (n_dims,) or None
        Unit vector along principal axis, or None if insufficient data.
    """
    if len(seg_points) < 2:
        return None
    try:
        mcd = MinCovDet().fit(seg_points)
        cov = mcd.covariance_
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal_axis = eig_vecs[:, np.argmax(eig_vals)]
        principal_axis /= np.linalg.norm(principal_axis)
        return principal_axis
    except:
        # Fallback to standard PCA if MCD fails
        pca = PCA(n_components=min(3, len(seg_points)))
        pca.fit(seg_points)
        principal_axis = pca.components_[0]
        principal_axis /= np.linalg.norm(principal_axis)
        return principal_axis


def create_spline_segments_for_df(
    df,
    pert_splines=None,
    spline_col=None,
    k=50,
    n_segments=None,
    segment_labels=None,
    group_by_col="phenotype",
    coord_cols=None,
    stage_col="predicted_stage_hpf"
):
    """Segment trajectory data using fitted splines.

    For each group in df, splits its spline into k segments, assigns data points
    to segments, and computes per-segment statistics (principal axis, midpoint, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory data with coordinates and grouping column.
    pert_splines : pd.DataFrame, optional
        Pre-fitted splines. Must have group_by_col and coord_cols.
        If None, must provide spline_col pointing to spline data.
    spline_col : str, optional
        Column in df containing spline data (alternative to pert_splines).
    k : int, default=50
        Number of segments per spline.
    n_segments : int, optional
        Alias for k (for API consistency).
    segment_labels : list of str, optional
        Labels for segments (length must equal k/n_segments).
    group_by_col : str, default='phenotype'
        Column to group by.
    coord_cols : list of str, optional
        Coordinate column names. Defaults to ['PCA_1', 'PCA_2', 'PCA_3'].
    stage_col : str, default='predicted_stage_hpf'
        Developmental stage column for segment statistics.

    Returns
    -------
    df_augmented : pd.DataFrame
        Input df with added 'segment_id' column (and 'segment_label' if provided).
    segment_info_df : pd.DataFrame
        Per-segment statistics with columns:
        - group_by_col, seg_id, segment_avg_time, segment_var_time
        - principal_axis_{x,y,z}, segment_midpoint_{x,y,z}
        - segment_start_{x,y,z}, segment_end_{x,y,z}
    pert_splines : pd.DataFrame
        Input pert_splines (returned for convenience).
    """
    if coord_cols is None:
        coord_cols = ["PCA_1", "PCA_2", "PCA_3"]

    if n_segments is not None:
        k = n_segments

    if segment_labels is not None and len(segment_labels) != k:
        raise ValueError(f"segment_labels length ({len(segment_labels)}) must match k ({k})")

    if pert_splines is None:
        raise ValueError("pert_splines must be provided (spline_col not yet implemented)")

    # Copy df to avoid mutation
    df_augmented = df.copy()

    # Store per-segment info
    segment_info_records = []

    # Process each group
    unique_groups = df_augmented[group_by_col].unique()
    for pert in tqdm(unique_groups, desc=f"Processing {group_by_col}"):
        # Extract group data
        pert_df = df_augmented[df_augmented[group_by_col] == pert].copy()
        if pert_df.empty:
            continue

        # Get spline for this group
        spline_data = pert_splines[pert_splines[group_by_col] == pert]
        if spline_data.empty:
            continue

        spline_points = spline_data[coord_cols].values

        # Split spline into segments
        segment_list = split_spline(spline_points, k)

        # Assign points to segments
        pert_df = assign_points_to_segments(pert_df, segment_list, coord_cols=coord_cols)

        # Update df_augmented
        df_augmented.loc[pert_df.index, "segment_id"] = pert_df["segment_id"]

        # Per-segment analysis
        for seg_id in range(k):
            seg_points_df = pert_df[pert_df["segment_id"] == seg_id]
            seg_points = seg_points_df[coord_cols].values

            if len(seg_points) < 2:
                principal_axis = None
            else:
                principal_axis = perform_robust_pca(seg_points)

            seg_start, seg_end = segment_list[seg_id]
            midpoint = 0.5 * (seg_start + seg_end)

            # Compute statistics
            segment_avg_time = seg_points_df[stage_col].mean() if stage_col in seg_points_df.columns else np.nan
            segment_var_time = seg_points_df[stage_col].var() if stage_col in seg_points_df.columns else np.nan

            # Build record
            if principal_axis is None:
                px, py, pz = np.nan, np.nan, np.nan
            else:
                px, py, pz = principal_axis

            mx, my, mz = midpoint
            sx, sy, sz = seg_start
            ex, ey, ez = seg_end

            record = {
                group_by_col: pert,
                "seg_id": seg_id,
                "segment_avg_time": segment_avg_time,
                "segment_var_time": segment_var_time,
                "principal_axis_x": px,
                "principal_axis_y": py,
                "principal_axis_z": pz,
                "segment_midpoint_x": mx,
                "segment_midpoint_y": my,
                "segment_midpoint_z": mz,
                "segment_start_x": sx,
                "segment_start_y": sy,
                "segment_start_z": sz,
                "segment_end_x": ex,
                "segment_end_y": ey,
                "segment_end_z": ez
            }
            segment_info_records.append(record)

    # Build segment_info_df
    segment_info_df = pd.DataFrame(segment_info_records)

    # Add segment labels if provided
    if segment_labels is not None:
        label_map = {i: label for i, label in enumerate(segment_labels)}
        df_augmented["segment_label"] = df_augmented["segment_id"].map(label_map)
        segment_info_df["segment_label"] = segment_info_df["seg_id"].map(label_map)

    return df_augmented, segment_info_df, pert_splines


# =============================================================================
# Projection Operations
# =============================================================================

def project_onto_plane(p, midpoint, normal):
    """Project point onto plane defined by midpoint and normal.

    Parameters
    ----------
    p : ndarray, shape (3,)
        Query point.
    midpoint : ndarray, shape (3,)
        Point on plane.
    normal : ndarray, shape (3,)
        Plane normal (assumed unit length).

    Returns
    -------
    plane_point : ndarray, shape (3,)
        Orthogonal projection of p onto plane.
    distance_to_plane : float
        Absolute distance from p to plane.
    distance_to_axis : float
        Distance from p to principal axis (line through midpoint in direction normal).
    hypotenuse : float
        Euclidean distance from p to closest point on axis.
    """
    alpha = np.dot((p - midpoint), normal)
    plane_point = p - alpha * normal

    distance_to_plane = abs(alpha)
    closest_on_axis = midpoint + alpha * normal
    distance_to_axis = np.linalg.norm(p - closest_on_axis)
    hypotenuse = np.sqrt(distance_to_plane**2 + distance_to_axis**2)

    return plane_point, distance_to_plane, distance_to_axis, hypotenuse


def project_points_onto_reference_spline(
    df_points,
    reference_spline_info,
    coord_cols=None,
    metadata_cols=None
):
    """Project data points onto reference spline and segment planes.

    For each point in df_points:
    1. Find closest segment in reference spline
    2. Project onto segment's plane (defined by principal axis)
    3. Compute closest point on segment in 3D
    4. Record distances and projections

    Parameters
    ----------
    df_points : pd.DataFrame
        Points to project. Must have coord_cols.
    reference_spline_info : pd.DataFrame
        Segment info from create_spline_segments_for_df().
        Must have columns: seg_id, principal_axis_{x,y,z},
        segment_midpoint_{x,y,z}, segment_{start,end}_{x,y,z}.
    coord_cols : list of str, optional
        Coordinate columns. Defaults to ['PCA_1', 'PCA_2', 'PCA_3'].
    metadata_cols : list of str, optional
        Additional columns to preserve. Defaults to common metadata columns.

    Returns
    -------
    projection_df : pd.DataFrame
        Projected points with columns:
        - metadata_cols (preserved from input)
        - coord_cols (original coordinates)
        - ref_seg_id (assigned segment)
        - closest_on_spline_{x,y,z} (3D projection onto segment)
        - plane_point_{x,y,z} (projection onto segment plane)
        - distance_to_plane, distance_to_axis, hypotenuse
    """
    if coord_cols is None:
        coord_cols = ["PCA_1", "PCA_2", "PCA_3"]

    if metadata_cols is None:
        metadata_cols = ["snip_id", "embryo_id", "phenotype", "predicted_stage_hpf"]

    # Build segment lookup structure
    segment_dicts = []
    for _, row in reference_spline_info.iterrows():
        seg_id = row["seg_id"]

        principal_axis = np.array([
            row["principal_axis_x"],
            row["principal_axis_y"],
            row["principal_axis_z"]
        ], dtype=float)

        midpoint = np.array([
            row["segment_midpoint_x"],
            row["segment_midpoint_y"],
            row["segment_midpoint_z"]
        ], dtype=float)

        seg_start = np.array([
            row["segment_start_x"],
            row["segment_start_y"],
            row["segment_start_z"]
        ], dtype=float)

        seg_end = np.array([
            row["segment_end_x"],
            row["segment_end_y"],
            row["segment_end_z"]
        ], dtype=float)

        # Normalize principal axis
        norm = np.linalg.norm(principal_axis)
        if norm > 1e-12:
            principal_axis = principal_axis / norm

        segment_dicts.append({
            "seg_id": seg_id,
            "principal_axis": principal_axis,
            "midpoint": midpoint,
            "seg_start": seg_start,
            "seg_end": seg_end
        })

    # Project each point
    records = []
    for idx, row in df_points.iterrows():
        # Extract coordinate
        p = np.array([row[col] for col in coord_cols], dtype=float)

        # Find closest segment
        min_dist = np.inf
        best_segment = None

        for seg_info in segment_dicts:
            closest_pt, dist = point_to_segment_projection(p, seg_info["seg_start"], seg_info["seg_end"])
            if dist < min_dist:
                min_dist = dist
                best_segment = seg_info

        if best_segment is None:
            continue

        # Project onto best segment
        principal_axis = best_segment["principal_axis"]
        midpoint = best_segment["midpoint"]
        seg_id = best_segment["seg_id"]
        seg_start = best_segment["seg_start"]
        seg_end = best_segment["seg_end"]

        # 3D projection onto segment
        closest_on_spline, _ = point_to_segment_projection(p, seg_start, seg_end)

        # Plane projection
        plane_point, distance_to_plane, distance_to_axis, hypotenuse = project_onto_plane(
            p, midpoint, principal_axis
        )

        # Build output record
        record = {}

        # Preserve metadata
        for col in metadata_cols:
            if col in row:
                record[col] = row[col]

        # Add coordinates
        for i, col in enumerate(coord_cols):
            record[col] = p[i]

        # Add projection results
        record["ref_seg_id"] = seg_id
        record["closest_on_spline_x"] = closest_on_spline[0]
        record["closest_on_spline_y"] = closest_on_spline[1]
        record["closest_on_spline_z"] = closest_on_spline[2]
        record["plane_point_x"] = plane_point[0]
        record["plane_point_y"] = plane_point[1]
        record["plane_point_z"] = plane_point[2]
        record["distance_to_plane"] = distance_to_plane
        record["distance_to_axis"] = distance_to_axis
        record["hypotenuse"] = hypotenuse

        records.append(record)

    projection_df = pd.DataFrame(records)
    return projection_df
