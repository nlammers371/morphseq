import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.colors as mcolors
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy.interpolate import CubicSpline, interp1d


# ============================
# Utility Functions
# ============================

def add_pca_components(
    df: pd.DataFrame,
    prefix: str = "z_mu_b",
    n_components: int = 3,
) -> pd.DataFrame:
    """
    Detect columns starting with `prefix` (default: 'z_mu_b'), run PCA on them,
    and append PCA_1..PCA_k columns to the input DataFrame.

    Returns:
        The same DataFrame object with PCA_* columns added."""
    # 1) Detect biological columns
    bio_cols = [c for c in df.columns if c.startswith(prefix)]
    if not bio_cols:
        raise ValueError(f"No columns start with prefix '{prefix}'.")

    # 2) Extract matrix and validate
    X = df[bio_cols].to_numpy()
    if np.isnan(X).any():
        raise ValueError("NaNs detected in biological columns. Please impute or drop before PCA.")

    # 3) Fit PCA (clip components to available features)
    k = min(n_components, X.shape[1])
    if k < 1:
        raise ValueError("Not enough features for PCA.")
    pca = PCA(n_components=k)
    pcs = pca.fit_transform(X)

    # 4) Append PCA columns
    pca_cols = [f"PCA_{i+1}" for i in range(k)]
    df.loc[:, pca_cols] = pcs

    return df


def compute_spline_distances(spline_pts):
    """Compute cumulative distances along the spline."""
    diffs = np.diff(spline_pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumdist = np.insert(np.cumsum(seg_lengths), 0, 0.0)
    return cumdist, cumdist[-1]

def split_spline(spline_pts, k):
    """
    Split the spline into k segments based on cumulative distance.
    Returns a list of (start_pt, end_pt) tuples.
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

def point_to_segment_distance(point, seg_start, seg_end):
    """
    Minimum distance from a 3D point to a line segment defined by seg_start, seg_end.
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

from scipy.spatial.distance import cdist

def assign_points_to_segments(pert_df, segment_list):
    points = pert_df[["PCA_1", "PCA_2", "PCA_3"]].values
    segment_midpoints = np.array([(s + e) / 2 for s, e in segment_list])
    
    # Compute all distances at once
    distances = cdist(points, segment_midpoints)
    segment_ids = distances.argmin(axis=1)
    
    pert_df["segment_id"] = segment_ids
    return pert_df

def perform_robust_pca(seg_points):
    """
    Perform robust PCA using Minimum Covariance Determinant (MCD).
    Returns the principal axis (first eigenvector) or None if not enough points.
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
        pca = PCA(n_components=3)
        pca.fit(seg_points)
        principal_axis = pca.components_[0]
        principal_axis /= np.linalg.norm(principal_axis)
        return principal_axis

def create_spline_segments_for_df(df, pert_splines, k=50, group_by_col="phenotype"):
    """
    For each group in df, finds its spline in pert_splines, splits it into k segments,
    assigns each point to a segment, performs robust PCA for each segment, and returns:

    1) df_augmented: original df augmented with 'segment_id'
    2) segment_info_df: columns = [
        group_by_col, seg_id, segment_avg_time, segment_var_time,
        principal_axis_x, principal_axis_y, principal_axis_z,
        segment_midpoint_x, segment_midpoint_y, segment_midpoint_z,
        segment_start_x, segment_start_y, segment_start_z,
        segment_end_x, segment_end_y, segment_end_z
    ]
    3) (optionally) return pert_splines, which is often unchanged
    """
    # 1. Copy df so we don't mutate the original
    df_augmented = df.copy()
    
    # 2. We'll store per-segment info in this list of dicts
    segment_info_records = []
    
    # 3. Loop over each group
    unique_groups = df_augmented[group_by_col].unique()
    for pert in tqdm(unique_groups, desc=f"Processing {group_by_col}"):
        # A) Extract all points for this group
        pert_df = df_augmented[df_augmented[group_by_col] == pert].copy()
        if pert_df.empty:
            continue
        
        # B) Retrieve the spline points for this group
        spline_data = pert_splines[pert_splines[group_by_col] == pert]
        if spline_data.empty:
            continue
        
        # Convert to NumPy
        spline_points = spline_data[["PCA_1", "PCA_2", "PCA_3"]].values
        
        # C) Split the spline
        segment_list = split_spline(spline_points, k)
        
        # D) Assign points to segments
        pert_df = assign_points_to_segments(pert_df, segment_list)
        
        # E) Update df_augmented with new segment IDs
        df_augmented.loc[pert_df.index, "segment_id"] = pert_df["segment_id"]
        
        # F) For each segment, perform robust PCA
        seg_data_dict = {}
        for seg_id in range(k):
            seg_points_df = pert_df[pert_df["segment_id"] == seg_id]
            seg_points = seg_points_df[["PCA_1", "PCA_2", "PCA_3"]].values
            
            if len(seg_points) < 2:
                principal_axis = None
            else:
                principal_axis = perform_robust_pca(seg_points)
            
            seg_start, seg_end = segment_list[seg_id]
            midpoint = 0.5 * (seg_start + seg_end)
            
            seg_data_dict[seg_id] = {
                "principal_axis": principal_axis,
                "segment_midpoint": midpoint,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "segment_avg_time": seg_points_df["predicted_stage_hpf"].mean(),
                "segment_var_time": seg_points_df["predicted_stage_hpf"].var()
            }
        
        # G) Build segment_info_records
        for seg_id, info in seg_data_dict.items():
            principal_axis = info["principal_axis"]
            midpoint = info["segment_midpoint"]
            seg_start = info["segment_start"]
            seg_end = info["segment_end"]
            segment_avg_time = info["segment_avg_time"]
            segment_var_time = info["segment_var_time"]
            
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
    
    # 4. Convert records to DataFrame
    segment_info_df = pd.DataFrame(segment_info_records)
    return df_augmented, segment_info_df, pert_splines

# ============================
# Main Wrapper Function
# ============================
def build_splines_and_segments(
    df,
    model_index,
    LocalPrincipalCurveClass,
    save_dir = None,
    comparisons=None,
    group_by_col="genotype",
    z_mu_biological_columns=None,
    n_components=3,
    bandwidth=0.5,
    max_iter=250,
    tol=1e-3,
    angle_penalty_exp=2,
    early_stage_offset=1.0,
    late_stage_offset=3.0,
    k=50
):
    """
    1) Builds splines for each group in `comparisons` using LocalPrincipalCurve
    2) Creates `df_augmented` by assigning segment IDs for each group
    3) Returns `pert_splines`, `df_augmented`, and `segment_info_df`
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least [group_by_col, "predicted_stage_hpf"] and either
        ["PCA_1", "PCA_2", "PCA_3"] OR z_mu_biological_columns for PCA computation.
    comparisons : list
        List of group values to process.
    group_by_col : str
        Column name to group by (default: "genotype"). Can be any column including integer columns.
    z_mu_biological_columns : list, optional
        List of column names to use for PCA if PCA columns don't exist. If None and PCA columns
        are missing, will attempt to auto-detect biological feature columns.
    n_components : int, optional
        Number of PCA components to compute (default: 3).
    save_dir : str
        Directory to save the spline CSV if desired.
    model_index : int
        Model index used in naming output files.
    LocalPrincipalCurveClass : class
        Reference to your LocalPrincipalCurve class (or a similar spline-fitting class).
    bandwidth : float
        Bandwidth parameter for LocalPrincipalCurve.
    max_iter : int
        Max iterations for LocalPrincipalCurve fitting.
    tol : float
        Tolerance for LocalPrincipalCurve convergence.
    angle_penalty_exp : int
        Angle penalty exponent for LocalPrincipalCurve.
    early_stage_offset : float
        Window (in hours) for selecting "early" timepoints to compute the average start point.
    late_stage_offset : float
        Window (in hours) for selecting "late" timepoints to compute the average end point.
    k : int
        Number of segments to split each spline into.

    Returns
    -------
    pert_splines : pd.DataFrame
        DataFrame containing the spline points for each group.
    df_augmented : pd.DataFrame
        Original DataFrame plus a `segment_id` column.
    segment_info_df : pd.DataFrame
        Per-segment PCA info (principal_axis, midpoint, etc.).
    """

    # ----------------------------
    # 2. Build Spline Data
    # ----------------------------
    # If comparisons is None, use all unique values from the group_by_col
    if comparisons is None:
        comparisons = df[group_by_col].dropna().unique().tolist()

    print(f"Building spline data for each {group_by_col}...")
    splines_records = []

    df = df[df[group_by_col].isin(comparisons)]
    
    for idx, pert in enumerate(tqdm(comparisons, desc=f"Creating splines for each {group_by_col}")):
        # Filter the DataFrame for the given group
        pert_df = df[df[group_by_col] == pert].copy()
        if pert_df.empty:
            # If no data points for this phenotype, skip
            continue

        # Extract PCA coordinates
        pert_3d = pert_df[["PCA_1", "PCA_2", "PCA_3"]].values
        
        # Compute average early stage point
        min_time = pert_df["predicted_stage_hpf"].min()
        early_mask = (pert_df["predicted_stage_hpf"] >= min_time) & \
                     (pert_df["predicted_stage_hpf"] < min_time + early_stage_offset)
        avg_early_timepoint = pert_df.loc[early_mask, ["PCA_1", "PCA_2", "PCA_3"]].mean().values
        
        # Compute average late stage point
        max_time = pert_df["predicted_stage_hpf"].max()
        late_mask = (pert_df["predicted_stage_hpf"] >= (max_time - late_stage_offset))
        avg_late_timepoint = pert_df.loc[late_mask, ["PCA_1", "PCA_2", "PCA_3"]].mean().values

        # Downsample for curve fitting (example: 5% for wt, 10% for others)
        if len(pert_3d) == 0:
            continue

        # Fit LocalPrincipalCurve
        lpc = LocalPrincipalCurveClass(
            bandwidth=bandwidth,
            max_iter=max_iter,
            tol=tol,
            angle_penalty_exp=angle_penalty_exp
        )
        
        # Fit with the optional start_points/end_point to anchor the spline
        lpc.fit(
            pert_3d_subset,
            start_points=avg_early_timepoint,
        )
        
        spline_points = None
        if len(lpc.cubic_splines) > 0:
            # If your local principal curve class stores the final spline
            spline_points = lpc.cubic_splines[0]
        else:
            # If no spline was built, skip
            continue
        
        # Create a temporary DataFrame for the current spline
        spline_df = pd.DataFrame(spline_points, columns=["PCA_1", "PCA_2", "PCA_3"])
        spline_df[group_by_col] = pert
        
        # Collect for later concatenation
        splines_records.append(spline_df)

    # Concatenate all spline DataFrames
    if splines_records:
        pert_splines = pd.concat(splines_records, ignore_index=True)
    else:
        # Fallback to an empty DataFrame if no splines
        pert_splines = pd.DataFrame(columns=["PCA_1", "PCA_2", "PCA_3", group_by_col])

    # Optionally, save the spline data
    if save_dir:
        spline_csv_path = os.path.join(save_dir, f"pert_splines_{model_index}_unique.csv")
        pert_splines.to_csv(spline_csv_path, index=False)
        print(f"Spline DataFrame 'pert_splines' saved to: {spline_csv_path}")

    # ----------------------------
    # 3. Create segments for each group using the function above
    # ----------------------------
    print("Assigning segments and building segment_info_df...")
    df_augmented, segment_info_df, pert_splines_out = create_spline_segments_for_df(
        df=df,
        pert_splines=pert_splines,
        k=k,
        group_by_col=group_by_col
    )

    # Return all three final structures
    return pert_splines_out, df_augmented, segment_info_df


import numpy as np
import pandas as pd

def point_to_segment_projection(p, seg_start, seg_end):
    """
    Return the closest point on the line segment [seg_start, seg_end] to p
    and the distance from p to that point.
    
    p, seg_start, seg_end are all 3D numpy arrays.
    """
    seg_vec = seg_end - seg_start
    pt_vec = p - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq == 0.0:
        # Degenerate segment; closest point is seg_start
        return seg_start, np.linalg.norm(p - seg_start)
    
    t = np.dot(pt_vec, seg_vec) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    closest = seg_start + t_clamped * seg_vec
    dist = np.linalg.norm(p - closest)
    return closest, dist

def project_onto_plane(p, midpoint, normal):
    """
    Given:
      - p: a 3D point (numpy array)
      - midpoint: a 3D point on the plane
      - normal: the plane's normal (3D)
    Returns:
      - plane_point: orthogonal projection of p onto the plane
      - distance_to_plane: absolute distance from p to the plane
      - distance_to_axis: distance from p to the principal axis (line through midpoint in direction normal)
      - hypotenuse: sqrt(distance_to_plane^2 + distance_to_axis^2)
    """
    # (Assume normal is unit-length. If not, we should normalize it.)
    # alpha = (p - m) · n
    alpha = np.dot((p - midpoint), normal)
    
    # plane_point = p - alpha * n
    plane_point = p - alpha * normal
    
    distance_to_plane = abs(alpha)  # since normal is unit
    # The "axis" is the line { midpoint + t * normal }, so the closest point on the axis is midpoint + alpha * normal
    closest_on_axis = midpoint + alpha * normal
    distance_to_axis = np.linalg.norm(p - closest_on_axis)
    hypotenuse = np.sqrt(distance_to_plane**2 + distance_to_axis**2)
    
    return plane_point, distance_to_plane, distance_to_axis, hypotenuse


def project_points_onto_reference_spline(
    df_points,
    reference_spline_info,
    k_segments=None
):
    """
    Projects the rows in df_points onto a reference spline (and planes) given by reference_spline_info.
    
    Steps:
      1. For each point in df_points:
         - Find the closest segment in reference_spline_info by line-segment distance.
         - Let 'seg_id' = that segment's ID.
      2. Use the principal axis (plane normal) and midpoint of that segment to project the point onto the plane.
      3. Also compute the closest point on the line segment in 3D.
      4. Record all relevant distances (distance_to_plane, distance_to_axis, hypotenuse, etc.).
    
    Returns a new DataFrame 'projection_df' with columns:
      [ snip_id, embryo_id, phenotype, predicted_stage_hpf,
        PCA_1, PCA_2, PCA_3,
        ref_seg_id, 
        closest_on_spline_x, closest_on_spline_y, closest_on_spline_z,
        plane_point_x, plane_point_y, plane_point_z,
        distance_to_plane, distance_to_axis, hypotenuse
      ]
    """
    # 1) Convert reference_spline_info into a structure for quick iteration
    #    Each segment row: seg_id, principal_axis, midpoint, seg_start, seg_end
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
        
        # Optional: if principal_axis is not guaranteed unit length, normalize it
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
        
    # 2) For each point in df_points, find the closest segment
    records = []
    
    # We'll iterate over df_points rows
    for idx, row in df_points.iterrows():
        # Extract the 3D point
        p = np.array([row["PCA_1"], row["PCA_2"], row["PCA_3"]], dtype=float)
        
        # Find the segment that yields the smallest distance
        min_dist = np.inf
        best_segment = None
        
        for seg_info in segment_dicts:
            closest_pt, dist = point_to_segment_projection(p, seg_info["seg_start"], seg_info["seg_end"])
            if dist < min_dist:
                min_dist = dist
                best_segment = seg_info
        
        if best_segment is None:
            # In an extreme edge case, if we can't find anything, skip
            continue
        
        # Now project onto the plane for this best segment
        principal_axis = best_segment["principal_axis"]
        midpoint = best_segment["midpoint"]
        seg_id = best_segment["seg_id"]
        seg_start = best_segment["seg_start"]
        seg_end = best_segment["seg_end"]
        
        # 3) Closest point on the segment itself
        closest_on_spline, _ = point_to_segment_projection(p, seg_start, seg_end)
        
        # 4) Plane projection
        plane_point, distance_to_plane, distance_to_axis, hypotenuse = project_onto_plane(p, midpoint, principal_axis)
        
        record = {
            "snip_id": row.get("snip_id", None),
            "embryo_id": row.get("embryo_id", None),
            "phenotype": row.get("phenotype", None),
            "predicted_stage_hpf": row.get("predicted_stage_hpf", None),
            
            "PCA_1": p[0],
            "PCA_2": p[1],
            "PCA_3": p[2],
            
            "ref_seg_id": seg_id,  # The segment this point ended up belonging to
            "closest_on_spline_x": closest_on_spline[0],
            "closest_on_spline_y": closest_on_spline[1],
            "closest_on_spline_z": closest_on_spline[2],
            
            "plane_point_x": plane_point[0],
            "plane_point_y": plane_point[1],
            "plane_point_z": plane_point[2],
            
            "distance_to_plane": distance_to_plane,
            "distance_to_axis": distance_to_axis,
            "hypotenuse": hypotenuse
        }
        records.append(record)
    
    projection_df = pd.DataFrame(records)

    return projection_df


import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_3d_spline_and_projections(
    projection_dfs,
    segment_info_df,
    pert_splines=None,
    save_dir=None,
    filename="splines_with_plane.html",
    k=None,
    title="3D Visualization of Spline and Projections",
    plane_opacity=0.2,
    axis_length=0.5,
    show_legend=True,
    plane_size=1.0,
    plane_grid_steps=10
):
    """
    Plots a 3D visualization of:
      1) The reference spline (if `pert_splines` is provided).
      2) Multiple sets of projected points (`projection_dfs`), each with columns:
         ["PCA_1", "PCA_2", "PCA_3", "ref_seg_id", "phenotype", ...]
         Points are colored by their 'ref_seg_id' to the reference spline.
      3) Principal axes for each segment (from `segment_info_df`).
      4) Plane surfaces (also from `segment_info_df`), each as a separate toggleable trace.

    Parameters
    ----------
    projection_dfs : list or pd.DataFrame
        One or more DataFrames of projected points.  
        Each DataFrame should have at least columns:
            ["PCA_1", "PCA_2", "PCA_3", "ref_seg_id", "phenotype", ...]
        where 'ref_seg_id' is the segment on the reference spline to which the point was assigned.
    segment_info_df : pd.DataFrame
        The reference segment info (e.g., from WT) with columns like:
            [
              "seg_id",
              "principal_axis_x", "principal_axis_y", "principal_axis_z",
              "segment_midpoint_x", "segment_midpoint_y", "segment_midpoint_z",
              "segment_start_x",   "segment_start_y",   "segment_start_z",
              "segment_end_x",     "segment_end_y",     "segment_end_z",
              "phenotype" (optional, if relevant)
            ]
        This defines the planes, principal axes, etc.
    pert_splines : pd.DataFrame or None
        Optional. If provided, plots the reference spline line.
        Columns expected: ["phenotype", "PCA_1", "PCA_2", "PCA_3"].
    save_dir : str or None
        If provided, the figure is saved as an HTML file in this directory.
    filename : str
        File name for the output HTML file.
    k : int or None
        Number of segments. If None, we infer it from `segment_info_df`.
    title : str
        Title for the figure.
    plane_opacity : float
        Opacity (0 to 1) for the plane surfaces.
    axis_length : float
        Half-length for drawing the principal axis line from each segment's midpoint.
    show_legend : bool
        Whether to show the Plotly legend.
    plane_size : float
        Half-width/height of the plane patch to draw around the midpoint.
    plane_grid_steps : int
        Resolution of the mesh for plane surfaces.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The constructed 3D Plotly figure.
    """
    # ---------------------------------------------------------------------
    # 0. Handle the case if a single DataFrame is passed instead of a list
    # ---------------------------------------------------------------------
    if isinstance(projection_dfs, pd.DataFrame):
        projection_dfs = [projection_dfs]
    
    # ---------------------------------------------------------------------
    # 1. Initialize the figure and color palette
    # ---------------------------------------------------------------------
    fig = go.Figure()
    color_palette = px.colors.qualitative.Dark24  # Up to 24 distinct colors

    # If k not provided, try to deduce from 'segment_info_df'
    if k is None:
        if not segment_info_df.empty:
            k = int(segment_info_df["seg_id"].max()) + 1
        else:
            k = 1

    # ---------------------------------------------------------------------
    # 2. Plot the reference spline (if provided)
    # ---------------------------------------------------------------------
    if pert_splines is not None and not pert_splines.empty:
        # We assume there's only ONE reference phenotype for the spline,
        # or we can just loop if multiple. Here, let's loop for each phenotype:
        for pert in pert_splines["phenotype"].unique():
            spline_df = pert_splines[pert_splines["phenotype"] == pert]
            spline_pts = spline_df[["PCA_1", "PCA_2", "PCA_3"]].values
            if len(spline_pts) < 2:
                continue

            fig.add_trace(go.Scatter3d(
                x=spline_pts[:, 0],
                y=spline_pts[:, 1],
                z=spline_pts[:, 2],
                mode='lines+markers',
                name=f"Spline ({pert})",
                line=dict(color='black', width=4),
                marker=dict(size=3, color='black'),
                legendgroup=f"Spline_{pert}",
            ))

    # ---------------------------------------------------------------------
    # 3. Plot each projection DataFrame, coloring points by segment
    # ---------------------------------------------------------------------
    # We create separate traces for each (phenotype, seg_id) within each DataFrame,
    # so that each perturbation can be toggled individually.
    for df_idx, proj_df in enumerate(projection_dfs):
        # Group by (phenotype, ref_seg_id). 
        # If 'phenotype' isn't present, group by ref_seg_id only (fallback).
        group_cols = ["ref_seg_id"]
        if "phenotype" in proj_df.columns:
            group_cols = ["phenotype", "ref_seg_id"]
        
        grouped = proj_df.groupby(group_cols)

        for key_tuple, group_data in grouped:
            if len(group_cols) == 2:
                pert, seg_id = key_tuple
            else:
                # If we only grouped by ref_seg_id, we have no phenotype info
                seg_id = key_tuple
                pert   = f"ProjDF_{df_idx}"  # fallback label

            # Convert seg_id to int if possible
            try:
                seg_id = int(seg_id)
            except:
                pass

            color_idx = seg_id % len(color_palette) if isinstance(seg_id, int) else 0
            seg_color = color_palette[color_idx]

            fig.add_trace(go.Scatter3d(
                x=group_data["PCA_1"],
                y=group_data["PCA_2"],
                z=group_data["PCA_3"],
                mode='markers',
                name=f"{pert} - seg {seg_id}",
                marker=dict(size=3, color=seg_color, opacity=0.7),
                legendgroup=pert,  
            ))

    # ---------------------------------------------------------------------
    # 4. Plot principal axis lines for each segment in the reference
    # ---------------------------------------------------------------------
    # segment_info_df might contain multiple phenotypes if you merged them,
    # but presumably for a single reference, it's just one phenotype or none.
    # We'll group by seg_id to handle it generally.
    for seg_id, seg_row_df in segment_info_df.groupby("seg_id"):
        # There's typically just 1 row for each (phenotype, seg_id), but let's handle n rows
        for _, row in seg_row_df.iterrows():
            px_ = row.get("principal_axis_x", np.nan)
            py_ = row.get("principal_axis_y", np.nan)
            pz_ = row.get("principal_axis_z", np.nan)
            if pd.isnull(px_) or pd.isnull(py_) or pd.isnull(pz_):
                continue
            
            principal_axis = np.array([px_, py_, pz_], dtype=float)
            midpoint = np.array([
                row.get("segment_midpoint_x", 0),
                row.get("segment_midpoint_y", 0),
                row.get("segment_midpoint_z", 0)
            ], dtype=float)

            color_idx = int(seg_id) % len(color_palette)
            axis_color = color_palette[color_idx]

            start_line = midpoint - axis_length * principal_axis
            end_line   = midpoint + axis_length * principal_axis

            fig.add_trace(go.Scatter3d(
                x=[start_line[0], end_line[0]],
                y=[start_line[1], end_line[1]],
                z=[start_line[2], end_line[2]],
                mode='lines',
                line=dict(color=axis_color, width=5),
                name=f"Axis seg {seg_id}",
                legendgroup="ReferenceAxes"
            ))

    # ---------------------------------------------------------------------
    # 5. Plot planes for each segment in the reference
    # ---------------------------------------------------------------------
    # Each plane is added as a separate trace for toggling.
    for seg_id, seg_row_df in segment_info_df.groupby("seg_id"):
        for _, row in seg_row_df.iterrows():
            # Normal
            px_ = row.get("principal_axis_x", np.nan)
            py_ = row.get("principal_axis_y", np.nan)
            pz_ = row.get("principal_axis_z", np.nan)
            if pd.isnull(px_) or pd.isnull(py_) or pd.isnull(pz_):
                continue

            plane_normal = np.array([px_, py_, pz_], dtype=float)
            norm_mag = np.linalg.norm(plane_normal)
            if norm_mag < 1e-12:
                continue

            midpoint = np.array([
                row.get("segment_midpoint_x", 0),
                row.get("segment_midpoint_y", 0),
                row.get("segment_midpoint_z", 0)
            ], dtype=float)
            d = -np.dot(plane_normal, midpoint)

            color_idx = int(seg_id) % len(color_palette)
            plane_color = color_palette[color_idx]

            # Build a grid
            xx, yy = np.meshgrid(
                np.linspace(midpoint[0] - plane_size, midpoint[0] + plane_size, plane_grid_steps),
                np.linspace(midpoint[1] - plane_size, midpoint[1] + plane_size, plane_grid_steps)
            )

            if abs(plane_normal[2]) > 1e-12:
                zz = (-plane_normal[0]*xx - plane_normal[1]*yy - d) / plane_normal[2]
            else:
                # Plane is vertical in Z
                zz = np.full_like(xx, midpoint[2])

            fig.add_trace(go.Surface(
                x=xx,
                y=yy,
                z=zz,
                opacity=plane_opacity,
                colorscale=[[0, plane_color], [1, plane_color]],
                showscale=False,
                name=f"Plane seg {seg_id}",
                legendgroup="ReferencePlanes",
            ))

    # ---------------------------------------------------------------------
    # 6. Final layout updates
    # ---------------------------------------------------------------------
    fig.update_layout(
        scene=dict(
            xaxis_title="PCA_1",
            yaxis_title="PCA_2",
            zaxis_title="PCA_3",
            aspectmode='data'
        ),
        width=1200,
        height=800,
        title=title,
        showlegend=show_legend
    )

    # Optionally save to HTML
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, filename)
        fig.write_html(outpath)
        print(f"3D visualization saved to: {outpath}")
    else:
        print("3D visualization not saved (no save_dir provided).")

    return fig

import numpy as np
import pandas as pd
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Optional: Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def annotate_embryo_time_index(df_augmented):
    """
    Adds a 'within_embryo_t_idx' column to df_augmented,
    indicating the order of each snip_id within its embryo based on experiment_time.
    """
    df_augmented = df_augmented.copy()
    df_augmented["within_embryo_t_idx"] = (
        df_augmented.sort_values(by=["embryo_id", "experiment_time"])
                   .groupby("embryo_id")
                   .cumcount()
    )
    return df_augmented

def preprocess_embryo_data(df_augmented):
    """
    Returns a dictionary: { embryo_id: sorted_df }
    where sorted_df is df for that embryo sorted by experiment_time.
    """
    embryo_dict = {}
    grouped = df_augmented.groupby('embryo_id')
    for emb_id, emb_df in grouped:
        # Sort by experiment_time ascending
        emb_df_sorted = emb_df.sort_values(by='experiment_time').reset_index(drop=True)
        embryo_dict[emb_id] = emb_df_sorted
    return embryo_dict

def single_random_journey(
    embryo_dict,
    df_augmented,
    segments_sorted=None,
    start_segment=0,
    end_segment=None,
    max_hops=10_000,
    time_column = "experiment_time",
    segment_id_col= "ref_seg_id",
):
    """
    Generates a single random 'journey' from start_segment to end_segment 
    (inclusive). If end_segment is None, picks the max segment from segments_sorted.

    Parameters:
    - embryo_dict: {embryo_id: DataFrame sorted by experiment_time}
    - df_augmented: DataFrame with 'segment_id', 'embryo_id', 'snip_id', 'experiment_time', 'within_embryo_t_idx'
    - segments_sorted: list of unique segment IDs sorted in ascending order
    - start_segment: int, starting segment ID
    - end_segment: int, ending segment ID
    - max_hops: int, maximum steps to prevent infinite loops

    Returns:
    - journey: list of dicts with keys [segment_id_col, 'embryo_id', 'snip_id', 'cumulative_time']
    """
    if segments_sorted is None:
        segments_sorted = sorted(df_augmented[segment_id_col].unique())
    if end_segment is None:
        end_segment = max(segments_sorted)

    total_time = 0.0
    journey = []
    current_segment = start_segment

    # Select a random starting point in start_segment
    start_candidates = df_augmented[df_augmented[segment_id_col] == current_segment]
    if start_candidates.empty:
        # No points in start_segment; cannot start journey
        return journey

    row_start = start_candidates.sample(n=1).iloc[0]
    curr_emb_id = row_start["embryo_id"]
    curr_snip_id = row_start["snip_id"]
    # Record first step with cumulative_time = 0
    journey.append({
        "segment_id": current_segment,
        "embryo_id": curr_emb_id,
        "snip_id": curr_snip_id,
        "cumulative_time": total_time
    })

    # Access the sorted embryo DataFrame
    emb_df = embryo_dict[curr_emb_id]
    # Find the index of the starting snip_id
    try:
        row_index = emb_df.index[emb_df["snip_id"] == curr_snip_id][0]
    except IndexError:
        # snip_id not found in embryo_dict; abort journey
        return journey

    hop_count = 0

    while current_segment < end_segment and hop_count < max_hops:
        hop_count += 1
        next_index = row_index + 1
        possible_move = False

        if next_index < len(emb_df):
            # There's a next time point in the same embryo
            row_next = emb_df.iloc[next_index]
            next_seg = row_next[segment_id_col]
            delta_t = row_next[time_column] - emb_df.iloc[row_index][time_column]

            if next_seg == current_segment:
                # Remain in the same segment
                total_time += delta_t
                journey.append({
                    "segment_id": current_segment,
                    "embryo_id": curr_emb_id,
                    "snip_id": row_next["snip_id"],
                    "cumulative_time": total_time
                })
                row_index = next_index
                possible_move = True
            elif next_seg > current_segment:
                # Move to a higher segment
                total_time += delta_t
                current_segment = next_seg
                journey.append({
                    "segment_id": current_segment,
                    "embryo_id": curr_emb_id,
                    "snip_id": row_next["snip_id"],
                    "cumulative_time": total_time
                })
                row_index = next_index
                possible_move = True
            else:
                # next_seg < current_segment; treat as no move
                pass

        if not possible_move:
            # Move to embryo in the current segment by jumping to a random embryo's point in that segment
            possible_segments = [seg for seg in segments_sorted if seg >= current_segment]
            if not possible_segments:
                # Reached the final segment; end journey
                break
            next_segment = min(possible_segments)  # Immediate next segment

            # Select a random point in next_segment
            next_seg_candidates = df_augmented[df_augmented[segment_id_col] == next_segment]
            if next_seg_candidates.empty:
                # No points in next_segment; cannot proceed
                break

            row_new = next_seg_candidates.sample(n=1).iloc[0]
            new_emb_id = row_new["embryo_id"]
            new_snip_id = row_new["snip_id"]

            # Update current position
            current_segment = next_segment
            curr_emb_id = new_emb_id
            curr_snip_id = new_snip_id
            emb_df = embryo_dict[curr_emb_id]

            try:
                row_index = emb_df.index[emb_df["snip_id"] == new_snip_id][0]
            except IndexError:
                # snip_id not found; skip to next journey
                break

            # No time accumulated when jumping to a different embryo
            journey.append({
                "segment_id": current_segment,
                "embryo_id": curr_emb_id,
                "snip_id": curr_snip_id,
                "cumulative_time": total_time
            })

    return journey

def run_bootstrap_journeys(
    df_augmented,
    num_journeys=1000,
    start_segment=0,
    end_segment=None,
    random_seed=42,
    time_column = "experiment_time",
    segment_id_col = "segment_id"
):
    """
    Repeatedly runs single_random_journey and collects results in a DataFrame.

    Parameters:
    - df_augmented: DataFrame with necessary columns
    - num_journeys: int, number of bootstrap journeys to run
    - start_segment: int, starting segment ID
    - end_segment: int, ending segment ID (if None, uses max segment)
    - random_seed: int, for reproducibility

    Returns:
    - journeys_df: DataFrame with all journey steps
    """
    # Set random seeds
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. Annotate the 'within_embryo_t_idx'
    df_aug = annotate_embryo_time_index(df_augmented)

    # Make sure there is an embryo in the star segment 

    start_segment_init = start_segment
    while True:
        start_candidates = df_aug[df_aug[segment_id_col] == start_segment]
    
        if not start_candidates.empty:
            # Valid start segment found
            print(f"No start candidates in segment {start_segment_init}."
                f"The next closest start candidate is in segment {start_segment}.")
            break  # Exit the loop when a valid start segment is found
        
            # Increment start_segment and continue the search
        start_segment += 1

    # 2. Build embryo_dict
    embryo_dict = preprocess_embryo_data(df_aug)

    # 3. Determine segments_sorted
    segments_sorted = sorted(df_aug[segment_id_col].unique())
    if end_segment is None:
        end_segment = max(segments_sorted)

    all_records = []

    for j_id in range(num_journeys):
        journey_steps = single_random_journey(
            embryo_dict=embryo_dict,
            df_augmented=df_aug,
            segments_sorted=segments_sorted,
            start_segment=start_segment,
            end_segment=end_segment,
            time_column = time_column,
        )

        for step_i, step_info in enumerate(journey_steps):
            record = {
                "journey_id": j_id,
                "step_index": step_i,
                "segment_id": step_info["segment_id"],
                "embryo_id": step_info["embryo_id"],
                "snip_id": step_info["snip_id"],
                "cumulative_time": step_info["cumulative_time"]
            }
            all_records.append(record)

    journeys_df = pd.DataFrame(all_records)
    return journeys_df
def summarize_journeys(journeys_df):
    """
    For each segment_id, compute mean and std of cumulative_time across all journeys.

    Parameters:
    - journeys_df: DataFrame with journey steps

    Returns:
    - summary_df: DataFrame with ['segment_id', 'mean_time', 'std_time', 'count']
    """
    grp = journeys_df.groupby("segment_id")["cumulative_time"]
    summary = grp.agg(['mean','std','count']).reset_index()
    summary.columns = ["segment_id", "mean_time", "std_time", "count"]
    return summary

def plot_summary(summary_df, title="Average Time to Reach Each Segment"):
    """
    Plots the average cumulative time to reach each segment with error bars.

    Parameters:
    - summary_df: DataFrame with ['segment_id', 'mean_time', 'std_time', 'count']
    - title: str, plot title
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        summary_df["segment_id"],
        summary_df["mean_time"]/(60*60),
        yerr=summary_df["std_time"],
        fmt='-o',
        ecolor='r',
        capsize=5,
        markersize=4,
        label='Mean Time with Std Dev'
    )
    plt.title(title)
    plt.xlabel("Segment ID")
    plt.ylabel("Average Cumulative Time (hours)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------- Usage Example --------------------------

# Assuming you have `df_augmented` and `pert_splines` DataFrames loaded
# Example:
# df_augmented = pd.read_csv("path_to_df_augmented.csv")
# pert_splines = pd.read_csv("path_to_pert_splines.csv")


def plot_summary(summary_df, df_augmented, title="Average Time to Reach Each Segment"):
    """
    Plots the average cumulative time to reach each segment (in hours) with error bars,
    and compares it to the predicted_stage_hpf (with standard error) from df_augmented.

    Parameters:
    ----------
    summary_df : pd.DataFrame
        Must have columns: ['segment_id', 'mean_time', 'std_time', 'count']
        where 'mean_time' and 'std_time' are in seconds, aggregated over bootstrap journeys.
    df_augmented : pd.DataFrame
        Must have columns: ['segment_id', 'predicted_stage_hpf'].
    title : str
        Plot title.
    """
    # 1. Convert bootstrap time (seconds) to hours
    starting_seg =int(min(summary_df["segment_id"]))
    time_init = df_augmented[df_augmented["segment_id"] == starting_seg]["predicted_stage_hpf"].mean()
    
    summary_df["mean_time_hours"] = summary_df["mean_time"] / 3600.0 + time_init
    summary_df["std_time_hours"] = summary_df["std_time"] / 3600.0

    # 2. Compute mean predicted_stage_hpf and standard error by segment
    pred_stats = (
        df_augmented.groupby("segment_id")["predicted_stage_hpf"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    pred_stats.columns = ["segment_id", "mean_pred_hpf", "std_pred_hpf", "count_pred"]
    pred_stats["se_pred_hpf"] = pred_stats["std_pred_hpf"] / np.sqrt(pred_stats["count_pred"])

    # 3. Merge the predicted stats with the summary of bootstrap times
    plot_df = pd.merge(summary_df, pred_stats, on="segment_id", how="left")

    # 4. Plot the bootstrap times (hours) with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        plot_df["segment_id"],
        plot_df["mean_time_hours"],
        yerr=plot_df["std_time_hours"],
        fmt='-o',
        ecolor='red',
        capsize=5,
        markersize=4,
        label='Bootstrap Mean Time ± Std (hours)'
    )

    # 5. Plot predicted stage hpf (also in hours) with standard error
    plt.errorbar(
        plot_df["segment_id"],
        plot_df["mean_pred_hpf"],
        yerr=plot_df["se_pred_hpf"],
        fmt='-s',
        color='darkorange',
        capsize=5,
        markersize=4,
        label='Predicted Stage hpf ± SE'
    )

    # 6. Final formatting
    plt.title(title)
    plt.xlabel("Segment ID")
    plt.ylabel("Time (hours)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
# plot_summary(summary_df, df_augmented, title="Average Time to Reach Each Segment")


def plot_multi_bootstrap(
    data_list,
    labels,
    title="Comparison of Average Times for Different Conditions",
    use_time_init=True
):
    """
    Plot multiple bootstrap time curves (mean ± std) on a single figure.
    
    Parameters
    ----------
    data_list : list of tuples
        Each element is (summary_df, df_augmented) for a particular condition.
        - summary_df must have columns ['segment_id', 'mean_time', 'std_time'].
          'mean_time' and 'std_time' are in seconds, aggregated over bootstrap journeys.
        - df_augmented must have at least ['segment_id', 'predicted_stage_hpf'] 
          if using time_init. If 'use_time_init' is False, df_augmented can be None.
          
    labels : list of str
        Labels for each condition (used in legend). Must match length of `data_list`.
        
    title : str
        Title for the final plot.
        
    use_time_init : bool
        If True, shifts the bootstrap mean_time by the average predicted_stage_hpf
        at the first segment found in 'summary_df'. If False, no offset is added.
    
    Returns
    -------
    None (displays a matplotlib figure)
    """
    if len(data_list) != len(labels):
        raise ValueError("data_list and labels must have the same length.")

    plt.figure(figsize=(10, 6))

    for (summary_df, df_aug), cond_label in zip(data_list, labels):
        summary_df = summary_df.copy()

        # Convert from seconds to hours
        summary_df["mean_time_hours"] = summary_df["mean_time"] / 3600.0
        summary_df["std_time_hours"] = summary_df["std_time"] / 3600.0
        
        if use_time_init:
            # Shift by the average predicted_stage_hpf for the starting segment
            # found in summary_df
            start_seg = int(summary_df["segment_id"].min())
            if df_aug is not None:
                # Compute the offset time_init from the data
                seg_mask = (df_aug["segment_id"] == start_seg)
                # In case no rows match, we handle it safely
                if seg_mask.any():
                    time_init = df_aug.loc[seg_mask, "predicted_stage_hpf"].mean()
                else:
                    time_init = 0.0
            else:
                # If df_aug not provided, can't compute time_init
                time_init = 0.0
            
            summary_df["mean_time_hours"] += time_init

        # Plot the bootstrap times (hours) with error bars
        plt.errorbar(
            summary_df["segment_id"],
            summary_df["mean_time_hours"],
            yerr=summary_df["std_time_hours"],
            fmt='-o',
            capsize=5,
            markersize=4,
            label=cond_label
        )

    plt.title(title)
    plt.xlabel("Segment ID")
    plt.ylabel("Time (hours)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_developmental_shifts(df_embryos, summary_df_wt_calc, color_by="phenotype"):
    """
    For each embryo_id in df_embryos:
      1) Identify earliest and latest time points (by experiment_time).
      2) Calculate the ratio of embryo's development time to reference development time.
      3) Compute time shift per 24 hours = (time ratio - 1) * 24.
      
    Returns a DataFrame with columns:
      [
        "embryo_id", 
        "earliest_segment", 
        "latest_segment", 
        "embryo_time_hrs", 
        "ref_time_hrs", 
        "time_ratio", 
        "time_shift_per_24hrs", 
        color_by
      ]
    One row per embryo_id.
    
    Parameters
    ----------
    df_embryos : pd.DataFrame
        Must have columns:
          - embryo_id
          - experiment_time (seconds)
          - ref_seg_id
          - plus the column for coloring (e.g., phenotype)
    summary_df_wt_calc : pd.DataFrame
        Must have columns:
          - segment_id
          - mean_time_hours (average time to that segment, in hours)
    color_by : str
        Column name in df_embryos to use for coloring or grouping in the histogram.
    """
    results = []
    
    grouped = df_embryos.groupby("embryo_id")
    for emb_id, group in grouped:
        # Sort by time
        group_sorted = group.sort_values("experiment_time")
        if len(group_sorted) < 2:
            # Not enough data points to define earliest and latest
            continue
        
        # Earliest & latest rows
        first_row = group_sorted.iloc[0]
        last_row  = group_sorted.iloc[-1]
        
        # Convert time to hours
        time_early_hrs = first_row["experiment_time"] / 3600.0
        time_late_hrs  = last_row["experiment_time"]  / 3600.0
        delta_time_hrs = time_late_hrs - time_early_hrs
        
        # Earliest & latest ref_seg_id
        seg_early = first_row["ref_seg_id"]
        seg_late  = last_row["ref_seg_id"]
        
        # If the earliest segment or latest segment is NaN or there's no difference, skip
        if pd.isnull(seg_early) or pd.isnull(seg_late) or seg_early == seg_late:
            continue
        
        # Reference development time:
        # We find the reference's average time for earliest & latest segment
        ref_row_early = summary_df_wt_calc.loc[summary_df_wt_calc["segment_id"] == seg_early]
        ref_row_late  = summary_df_wt_calc.loc[summary_df_wt_calc["segment_id"] == seg_late]
        if len(ref_row_early) == 0 or len(ref_row_late) == 0:
            # The reference summary might not have these segments
            continue
        
        ref_time_early = ref_row_early["mean_time_hours"].iloc[0]
        ref_time_late  = ref_row_late["mean_time_hours"].iloc[0]
        ref_delta_time = ref_time_late - ref_time_early
        
        # Calculate time ratio (embryo time / reference time)
        time_ratio = delta_time_hrs / ref_delta_time
        
        # Calculate time shift per 24 hours
        # Flipping the sign so that:
        # If time_ratio > 1: embryo develops slower than reference -> negative value
        # If time_ratio < 1: embryo develops faster than reference -> positive value
        # The (1 - time_ratio) * 24 gives us how many fewer/more hours 
        # the embryo needs compared to reference over a 24-hour period
        time_shift_per_24hrs = (1 - time_ratio) * 24
        
        # Grab the color-by value from either the first or last row
        # (assuming it's consistent for the entire embryo)
        col_value = first_row.get(color_by, np.nan)
        
        
        results.append({
            "embryo_id": emb_id,
            "earliest_segment": seg_early,
            "latest_segment": seg_late,
            "embryo_time_hrs": delta_time_hrs,
            "ref_time_hrs": ref_delta_time,
            "time_ratio": time_ratio,
            "time_shift_per_24hrs": time_shift_per_24hrs,
            color_by: col_value
        })
    
    return pd.DataFrame(results)

def plot_developmental_shifts_violin(shift_df, color_by="phenotype", show_dots=True, remove_outliers=True):
    """
    Given a DataFrame with a 'time_shift_per_24hrs' column and a grouping column (color_by),
    plot a violin plot of the time shifts, grouped by the specified column,
    with an option to overlay individual points as a strip plot and to remove extreme outliers.
    
    Parameters
    ----------
    shift_df : pd.DataFrame
        Must have columns ['time_shift_per_24hrs', color_by].
    color_by : str
        Column name in shift_df to group by for the violin plot.
    show_dots : bool, optional
        Whether to overlay individual points on the violin plot (default: True).
    remove_outliers : bool, optional
        Whether to remove extreme outliers based on the IQR method (default: True).
    """
    # Optionally remove outliers
    if remove_outliers:
        q1 = shift_df["time_shift_per_24hrs"].quantile(0.25)
        q3 = shift_df["time_shift_per_24hrs"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        shift_df = shift_df[(shift_df["time_shift_per_24hrs"] >= lower_bound) & 
                            (shift_df["time_shift_per_24hrs"] <= upper_bound)]
    
    plt.figure(figsize=(8, 6))

    # Create violin plot
    sns.violinplot(
        data=shift_df,
        x=color_by,
        y="time_shift_per_24hrs",
        inner="box",  # Show quartiles and medians inside violins
        scale="width",  # Adjust violin width by number of observations per group
        palette="muted"
    )
    
    # Optionally overlay individual points
    if show_dots:
        sns.stripplot(
            data=shift_df,
            x=color_by,
            y="time_shift_per_24hrs",
            color="black",  # Black dots
            alpha=0.6,      # Semi-transparent for overlapping points
            jitter=True,    # Add jitter to spread points horizontally
            dodge=True      # Separate points slightly for each hue group (if applicable)
        )

    # Customize plot
    plt.title("Developmental Time Shift Distribution by " + color_by.capitalize())
    plt.xlabel(color_by.capitalize())
    plt.ylabel("Time Shift (hours per 24hrs)\nPositive = Faster, Negative = Slower")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def build_splines_and_segments(
    df,
    model_index,
    LocalPrincipalCurveClass,
    save_dir = None,
    comparisons=None,
    group_by_col="genotype",
    z_mu_biological_columns=None,
    n_components=3,
    bandwidth=0.5,
    max_iter=250,
    tol=1e-3,
    angle_penalty_exp=2,
    early_stage_offset=1.0,
    late_stage_offset=3.0,
    k=50
):
    """
    1) Builds splines for each group in `comparisons` using LocalPrincipalCurve
    2) Creates `df_augmented` by assigning segment IDs for each group
    3) Returns `pert_splines`, `df_augmented`, and `segment_info_df`
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least [group_by_col, "predicted_stage_hpf"] and either
        ["PCA_1", "PCA_2", "PCA_3"] OR z_mu_biological_columns for PCA computation.
    comparisons : list
        List of group values to process.
    group_by_col : str
        Column name to group by (default: "genotype"). Can be any column including integer columns.
    z_mu_biological_columns : list, optional
        List of column names to use for PCA if PCA columns don't exist. If None and PCA columns
        are missing, will attempt to auto-detect biological feature columns.
    n_components : int, optional
        Number of PCA components to compute (default: 3).
    save_dir : str
        Directory to save the spline CSV if desired.
    model_index : int
        Model index used in naming output files.
    LocalPrincipalCurveClass : class
        Reference to your LocalPrincipalCurve class (or a similar spline-fitting class).
    bandwidth : float
        Bandwidth parameter for LocalPrincipalCurve.
    max_iter : int
        Max iterations for LocalPrincipalCurve fitting.
    tol : float
        Tolerance for LocalPrincipalCurve convergence.
    angle_penalty_exp : int
        Angle penalty exponent for LocalPrincipalCurve.
    early_stage_offset : float
        Window (in hours) for selecting "early" timepoints to compute the average start point.
    late_stage_offset : float
        Window (in hours) for selecting "late" timepoints to compute the average end point.
    k : int
        Number of segments to split each spline into.

    Returns
    -------
    pert_splines : pd.DataFrame
        DataFrame containing the spline points for each group.
    df_augmented : pd.DataFrame
        Original DataFrame plus a `segment_id` column.
    segment_info_df : pd.DataFrame
        Per-segment PCA info (principal_axis, midpoint, etc.).
    """
    # If comparisons is None, use all unique values from the group_by_col
    if comparisons is None:
        comparisons = df[group_by_col].dropna().unique().tolist()

    # ----------------------------
    # 2. Build Spline Data
    # ----------------------------
    print(f"Building spline data for each {group_by_col}...")
    splines_records = []
    df = df[df[group_by_col].isin(comparisons)]
    
    for idx, pert in enumerate(tqdm(comparisons, desc=f"Creating splines for each {group_by_col}")):
        # Filter the DataFrame for the given group
        pert_df = df[df[group_by_col] == pert].copy()
        if pert_df.empty:
            # If no data points for this phenotype, skip
            continue

        # Extract PCA coordinates
        pert_3d = pert_df[["PCA_1", "PCA_2", "PCA_3"]].values
        
        # Compute average early stage point
        min_time = pert_df["predicted_stage_hpf"].min()
        early_mask = (pert_df["predicted_stage_hpf"] >= min_time) & \
                     (pert_df["predicted_stage_hpf"] < min_time + early_stage_offset)
        avg_early_timepoint = pert_df.loc[early_mask, ["PCA_1", "PCA_2", "PCA_3"]].mean().values
        
        # Compute average late stage point
        max_time = pert_df["predicted_stage_hpf"].max()
        late_mask = (pert_df["predicted_stage_hpf"] >= (max_time - late_stage_offset))
        avg_late_timepoint = pert_df.loc[late_mask, ["PCA_1", "PCA_2", "PCA_3"]].mean().values

        # Downsample for curve fitting (example: 5% for wt, 10% for others)
        if len(pert_3d) == 0:
            continue

        # Fit LocalPrincipalCurve
        lpc = LocalPrincipalCurveClass(
            bandwidth=bandwidth,
            max_iter=max_iter,
            tol=tol,
            angle_penalty_exp=angle_penalty_exp
        )
        
        # Fit with the optional start_points/end_point to anchor the spline
        lpc.fit(
            pert_3d,
            start_points=avg_early_timepoint,
        )
        
        spline_points = None
        if len(lpc.cubic_splines) > 0:
            # If your local principal curve class stores the final spline
            spline_points = lpc.cubic_splines[0]
        else:
            # If no spline was built, skip
            continue
        
        # Create a temporary DataFrame for the current spline
        spline_df = pd.DataFrame(spline_points, columns=["PCA_1", "PCA_2", "PCA_3"])
        spline_df[group_by_col] = pert
        
        # Collect for later concatenation
        splines_records.append(spline_df)

    # Concatenate all spline DataFrames
    if splines_records:
        pert_splines = pd.concat(splines_records, ignore_index=True)
    else:
        # Fallback to an empty DataFrame if no splines
        pert_splines = pd.DataFrame(columns=["PCA_1", "PCA_2", "PCA_3", group_by_col])

    # Optionally, save the spline data
    if save_dir:
        spline_csv_path = os.path.join(save_dir, f"pert_splines_{model_index}_unique.csv")
        pert_splines.to_csv(spline_csv_path, index=False)
        print(f"Spline DataFrame 'pert_splines' saved to: {spline_csv_path}")

    # ----------------------------
    # 3. Create segments for each group using the function above
    # ----------------------------
    print("Assigning segments and building segment_info_df...")
    df_augmented, segment_info_df, pert_splines_out = create_spline_segments_for_df(
        df=df,
        pert_splines=pert_splines,
        k=k,
        group_by_col=group_by_col
    )

    # Return all three final structures
    return pert_splines_out, df_augmented, segment_info_df



class LocalPrincipalCurve:
    def __init__(self, bandwidth=0.5, max_iter=100, tol=1e-4, angle_penalty_exp=2, h=None):
        """
        Initialize the Local Principal Curve solver.
        """
        self.bandwidth = bandwidth
        self.h = h if h is not None else self.bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.angle_penalty_exp = angle_penalty_exp

        self.initializations = []
        self.paths = []
        self.cubic_splines_eq = []
        self.cubic_splines = []

    def _kernel_weights(self, dataset, x):
        dists = np.linalg.norm(dataset - x, axis=1)
        weights = np.exp(- (dists**2) / (2 * self.bandwidth**2))
        w = weights / np.sum(weights)
        return w

    def _local_center_of_mass(self, dataset, x):
        w = self._kernel_weights(dataset, x)
        mu = np.sum(dataset.T * w, axis=1)
        return mu

    def _local_covariance(self, dataset, x, mu):
        w = self._kernel_weights(dataset, x)
        centered = dataset - mu
        # cov = np.zeros((dataset.shape[1], dataset.shape[1]))
        weighted_centered = centered * w[:, np.newaxis]  # shape: (n, d)
        cov = np.dot(weighted_centered.T, centered)  # shape: (d, d)
        # for i in range(len(dataset)):
        #     cov += w[i] * np.outer(centered[i], centered[i])
        return cov

    def _principal_component(self, cov, prev_vec=None):
        vals, vecs = np.linalg.eig(cov)
        idx = np.argsort(vals)[::-1]
        # vals = vals[idx]
        vecs = vecs[:, idx]

        gamma = vecs[:, 0]  # first principal component

        # Sign/direction handling
        if prev_vec is not None and np.linalg.norm(prev_vec) != 0:
            cos_alpha = np.dot(gamma, prev_vec) / (np.linalg.norm(gamma)*np.linalg.norm(prev_vec))
            if cos_alpha < 0:
                gamma = -gamma

            # Angle penalization
            cos_alpha = np.dot(gamma, prev_vec) / (np.linalg.norm(gamma)*np.linalg.norm(prev_vec))
            a_x = (abs(cos_alpha))**self.angle_penalty_exp
            gamma = a_x * gamma + (1 - a_x) * prev_vec
            gamma /= np.linalg.norm(gamma)

        return gamma

    def _forward_run(self, dataset, x_start):
        x = x_start
        path_x = [x]
        prev_gamma = None

        for _ in range(self.max_iter):
            mu = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            x_new = mu + self.h * gamma

            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new)
                break

            path_x.append(x_new)
            x = x_new
            prev_gamma = gamma

        return np.array(path_x)

    def _backward_run(self, dataset, x0, gamma0):
        x = x0
        path_x = [x]
        prev_gamma = -gamma0

        for _ in range(self.max_iter):
            mu = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            x_new = mu + self.h * gamma
            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new)
                break

            path_x.append(x_new)
            x = x_new
            prev_gamma = gamma

        return np.array(path_x)

    def _find_starting_point(self, dataset, start_point):
        if start_point is None:
            idx = np.random.choice(len(dataset))
            return dataset[idx], idx
        else:
            diffs = dataset - start_point
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            closest_pt = dataset[min_idx]
            # if not np.allclose(closest_pt, start_point, rtol=1e-01):
            #     print(f"Starting point not in dataset. Using closest point: {closest_pt}")
            return closest_pt, min_idx

    def fit(self, dataset, start_points=None, end_point=None, num_points=500):
        """
        Fit LPC on the dataset. Optionally provide:
         - start_points: array of shape (d,) or a single point of shape (d,)
         - end_point: single point of shape (d,), only allowed if a start_point is provided.
        """
        dataset = np.array(dataset)
        self.paths = []
        self.initializations = []

        if end_point is not None and start_points is None:
            raise ValueError("end_point provided but no start_points given. end_point only allowed if start_point is provided.")

        # Ensure start_points is a list
        if start_points is not None and not isinstance(start_points, (list, tuple)):
            start_points = [start_points]

        if end_point is not None and (start_points is None or len(start_points) != 1):
            raise ValueError("If end_point is provided, exactly one start_point must be provided.")

        for sp in (start_points if start_points is not None else [None]):
            x0, _ = self._find_starting_point(dataset, sp)

            forward_path = self._forward_run(dataset, x0)
            if len(forward_path) > 1:
                initial_gamma_direction = (forward_path[1] - forward_path[0]) / self.h
            else:
                initial_gamma_direction = np.zeros(dataset.shape[1])
            # Debugging
            # import pdb
            # pdb.set_trace()
            # Debugging
            if np.linalg.norm(initial_gamma_direction) > 0:
                backward_path = self._backward_run(dataset, x0, initial_gamma_direction)
                full_path = np.vstack([backward_path[::-1], forward_path[1:]])
            else:
                full_path = forward_path

            # Check orientation
            dist_start_to_first = np.linalg.norm(x0 - full_path[0])
            dist_start_to_last = np.linalg.norm(x0 - full_path[-1])
            if dist_start_to_last < dist_start_to_first:
                full_path = full_path[::-1]

            self.paths.append(full_path)
            self.initializations.append(x0)

        # Fit splines and compute equal arc-length
        self._fit_cubic_splines_eq()
        self._compute_equal_arc_length_spline_points(num_points=num_points)

        # If end_point provided, correct for the looping back issue
        # if end_point is not None:
        #     try:
        #         # Assuming a single path scenario
        #         spline_points = self.cubic_splines[0]
        #
        #         # 1) Find closest point on cubic_spline to end_point
        #         dists = np.linalg.norm(spline_points - end_point, axis=1)
        #         closest_idx = np.argmin(dists)
        #
        #         # 2) Determine end_direction_vector using points around closest_idx
        #         # We'll take up to 3 points: [closest_idx-1, closest_idx, closest_idx+1]
        #         # If closest_idx is at the boundary, adjust accordingly
        #         if closest_idx == 0:
        #             # At start, use next two points if available
        #             if len(spline_points) > 2:
        #                 p0 = spline_points[closest_idx]
        #                 p1 = spline_points[closest_idx + 1]
        #                 p2 = spline_points[closest_idx + 2]
        #                 end_direction_vector = ((p1 - p0) + (p2 - p1)) / 2.0
        #             else:
        #                 # If very short, just fallback
        #                 end_direction_vector = np.array([1, 0, 0])
        #         elif closest_idx == len(spline_points) - 1:
        #             # At the end, we might not have a point after it
        #             # use the two points before it if possible
        #             if len(spline_points) > 2:
        #                 p_end = spline_points[closest_idx]
        #                 p_endm1 = spline_points[closest_idx - 1]
        #                 p_endm2 = spline_points[closest_idx - 2]
        #                 end_direction_vector = ((p_end - p_endm1) + (p_endm1 - p_endm2)) / 2.0
        #             else:
        #                 end_direction_vector = np.array([1, 0, 0])
        #         else:
        #             # Middle somewhere, use prev and next
        #             p_before = spline_points[closest_idx - 1]
        #             p_mid = spline_points[closest_idx]
        #             p_after = spline_points[closest_idx + 1]
        #             end_direction_vector = ((p_mid - p_before) + (p_after - p_mid)) / 2.0
        #
        #         # Normalize end_direction_vector
        #         norm_edv = np.linalg.norm(end_direction_vector)
        #         if norm_edv > 0:
        #             end_direction_vector = end_direction_vector / norm_edv
        #         else:
        #             warnings.warn("end_direction_vector has zero magnitude. Using default direction.")
        #             end_direction_vector = np.array([1, 0, 0])
        #
        #         # 3) Check directionality after closest_idx
        #         # We'll look at pairs of points (p_j, p_{j+1}) for j > closest_idx
        #         cutoff_index = None
        #         for j in range(closest_idx + 1, len(spline_points) - 1):
        #             seg_vec = spline_points[j + 1] - spline_points[j]
        #             csim = cosine_similarity(seg_vec.reshape(1, -1), end_direction_vector.reshape(1, -1))
        #             if csim < 0.5:
        #                 cutoff_index = j + 1
        #                 break
        #
        #         # If we found a cutoff_index, truncate the spline
        #         if cutoff_index is not None:
        #             spline_points = spline_points[:cutoff_index]
        #
        #             # Refit with truncated spline_points
        #             self.paths = [spline_points]
        #             self._fit_cubic_splines_eq()
        #             self._compute_equal_arc_length_spline_points()
        #
        #     except (ValueError, IndexError, TypeError) as e:
        #         # Log a warning and exit the if block gracefully
        #         warnings.warn(
        #             f"Error processing spline with end_point: {e}. Skipping spline adjustment."
        #         )
        #         # Optionally, you can log more details for debugging
        #         # For example:
        #         # warnings.warn(f"Error processing spline: {e}. spline_points shape: {spline_points.shape}, end_point shape: {np.shape(end_point)}")
        #         return  # Exit the if block

        return self.paths

    def _fit_cubic_splines_eq(self):
        self.cubic_splines_eq = []
        for path in self.paths:
            if len(path) < 4:
                self.cubic_splines_eq.append(None)
                continue
            t = np.arange(len(path))
            splines_dict = {}
            for dim in range(path.shape[1]):
                splines_dict[dim] = CubicSpline(t, path[:, dim])
            self.cubic_splines_eq.append(splines_dict)

    def _compute_cubic_spline_points(self, num_points=500):
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            path = self.paths[i]
            t_values = np.linspace(0, len(path) - 1, num_points)
            spline_points = self.evaluate_cubic_spline(i, t_values)
            self.cubic_splines.append(spline_points)

    def evaluate_cubic_spline(self, path_idx, t_values):
        if path_idx >= len(self.cubic_splines_eq) or self.cubic_splines_eq[path_idx] is None:
            raise ValueError(f"No cubic spline found for path index {path_idx}.")
        spline = self.cubic_splines_eq[path_idx]
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T  # Fixed line
        return points

    def compute_arc_length(self, spline, t_min, t_max, num_samples=10000):
        t_values = np.linspace(t_min, t_max, num_samples)
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T  # Fixed line

        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_length = np.insert(np.cumsum(distances), 0, 0.0)
        return t_values, cumulative_length

    def get_uniformly_spaced_points(self, spline, num_points):
        path_length = len(spline[0].x)
        t_min = 0
        t_max = path_length - 1

        t_vals_dense, cum_length = self.compute_arc_length(spline, t_min, t_max, num_samples=5000)
        total_length = cum_length[-1]
        desired_distances = np.linspace(0, total_length, num_points)
        t_for_dist = interp1d(cum_length, t_vals_dense, kind='linear')(desired_distances)

        uniform_points = np.array([spline[dim](t_for_dist) for dim in sorted(spline.keys())]).T  # Fixed line
        return uniform_points

    def _compute_equal_arc_length_spline_points(self, num_points=500):
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            spline_points = self.get_uniformly_spaced_points(eq, num_points)
            self.cubic_splines.append(spline_points)

    def plot_path_3d(self, path_idx=0, dataset=None):
        dataset = np.array(dataset)
        path = self.paths[path_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if dataset is not None:
            ax.scatter(dataset[:,0], dataset[:,1], dataset[:,2], alpha=0.5, label='Data')
        ax.plot(path[:,0], path[:,1], path[:,2], 'r-', label='Local Principal Curve')
        ax.legend()
        plt.show()

    def plot_cubic_spline_3d(self, path_idx, show_path=True):
        if path_idx >= len(self.paths):
            raise IndexError(f"Path index {path_idx} is out of range. Total paths: {len(self.paths)}.")
        path = self.paths[path_idx]
        spline_points = self.cubic_splines[path_idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if show_path:
            ax.scatter(path[:, 0], path[:, 1], path[:, 2], label="LPC Path", alpha=0.5)
        ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], color="red", label="Cubic Spline")
        ax.legend()
        plt.show()

def assign_points_to_segments(pert_df, segment_list):
    points = pert_df[["PCA_1", "PCA_2", "PCA_3"]].values
    segment_midpoints = np.array([(s + e) / 2 for s, e in segment_list])
    
    # Compute all distances at once
    distances = cdist(points, segment_midpoints)
    segment_ids = distances.argmin(axis=1)
    
    pert_df["segment_id"] = segment_ids
    return pert_df