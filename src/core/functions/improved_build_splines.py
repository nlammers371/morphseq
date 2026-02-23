import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import os

def build_splines_and_segments_with_bootstrap(
    df,
    model_index,
    spline_fit_wrapper,  # Function from spline_fitting_v2.py
    save_dir=None,
    comparisons=None,
    group_by_col="genotype",
    stage_col="predicted_stage_hpf",
    z_mu_biological_columns=None,
    n_components=3,
    # Spline fitting parameters
    bandwidth=0.5,
    h=None,  # Step size parameter for LocalPrincipalCurve
    max_iter=2500,
    tol=1e-5,
    angle_penalty_exp=1,
    n_boots=10,
    boot_size=2500,
    n_spline_points=500,
    time_window=2,
    # Segmentation parameters
    k=50
):
    """
    Enhanced version of build_splines_and_segments that uses spline_fit_wrapper
    for bootstrap uncertainty estimation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least [group_by_col, stage_col] and either
        ["PCA_00_bio", "PCA_01_bio", "PCA_02_bio"] OR z_mu_biological_columns for PCA computation.
    model_index : int
        Model index used in naming output files.
    spline_fit_wrapper : function
        The bootstrap spline fitting function from spline_fitting_v2.py
    save_dir : str, optional
        Directory to save outputs
    comparisons : list, optional
        List of group values to process. If None, uses all unique values.
    group_by_col : str
        Column name to group by (default: "genotype")
    stage_col : str
        Column name for developmental stage (default: "predicted_stage_hpf")
    z_mu_biological_columns : list, optional
        Columns for PCA if PCA columns don't exist
    n_components : int
        Number of PCA components (default: 3)
    bandwidth : float
        Bandwidth for LocalPrincipalCurve (default: 0.5)
    h : float, optional
        Step size parameter for LocalPrincipalCurve (default: None, uses automatic selection)
    max_iter : int
        Max iterations for curve fitting (default: 2500)
    tol : float
        Convergence tolerance (default: 1e-5)
    angle_penalty_exp : int
        Angle penalty exponent (default: 1)
    n_boots : int
        Number of bootstrap iterations (default: 10)
    boot_size : int
        Bootstrap sample size (default: 2500)
    n_spline_points : int
        Number of points in final spline (default: 500)
    time_window : float
        Time window for anchor point selection (default: 2)
    k : int
        Number of segments to create (default: 50)

    Returns
    -------
    pert_splines : pd.DataFrame
        Bootstrap-averaged spline points with uncertainty estimates
    df_augmented : pd.DataFrame
        Original DataFrame with segment assignments
    segment_info_df : pd.DataFrame
        Per-segment analysis results
    """
    
    # ----------------------------
    # 0. Check for PCA columns and apply PCA if needed
    # ----------------------------
    # Look for PCA columns with the pattern used in spline_fit_wrapper
    pca_pattern_cols = [col for col in df.columns if col.startswith("PCA_") and col.endswith("_bio")]
    
    if len(pca_pattern_cols) >= n_components:
        print(f"Found PCA columns: {pca_pattern_cols[:n_components]}")
        fit_cols = pca_pattern_cols[:n_components]
    else:
        print("PCA columns not found. Need to apply PCA first...")
        
        # Auto-detect z_mu_biological_columns if not provided
        if z_mu_biological_columns is None:
            z_mu_b_cols = [col for col in df.columns if col.startswith('z_mu_b_')]
            if z_mu_b_cols:
                z_mu_biological_columns = z_mu_b_cols
                print(f"Auto-detected z_mu_b columns for PCA: {len(z_mu_b_cols)} columns")
            else:
                potential_cols = [col for col in df.columns if any(pattern in col.lower()
                                for pattern in ['z_mu', 'embedding', 'feature', 'latent', 'biological'])]
                if potential_cols:
                    z_mu_biological_columns = potential_cols
                    print(f"Using potential latent columns: {len(potential_cols)} columns")
                else:
                    raise ValueError("No suitable columns found for PCA. Expected z_mu_b_* columns or PCA_*_bio columns.")
        
        print(f"Using {len(z_mu_biological_columns)} columns for PCA")
        
        # Apply PCA using sklearn
        from sklearn.decomposition import PCA
        
        # Clean data first
        df_clean = df.dropna(subset=z_mu_biological_columns)
        print(f"After removing NaN rows: {len(df_clean)} rows remaining")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_values = pca.fit_transform(df_clean[z_mu_biological_columns])
        
        # Add PCA columns to dataframe
        fit_cols = [f"PCA_{i:02d}_bio" for i in range(n_components)]
        for i, col in enumerate(fit_cols):
            df_clean[col] = pca_values[:, i]
        
        # Update df to be the clean version with PCA
        df = df_clean
        
        print(f"Applied PCA. Explained variance ratios: {pca.explained_variance_ratio_}")
        print(f"Created PCA columns: {fit_cols}")

    # ----------------------------
    # 1. Handle comparisons parameter
    # ----------------------------
    if comparisons is None:
        comparisons = list(df[group_by_col].unique())
        print(f"No comparisons specified. Using all available {group_by_col} values: {comparisons}")
    else:
        print(f"Using specified comparisons for spline building: {comparisons}")

    # Filter data to only include specified comparisons
    df = df[df[group_by_col].isin(comparisons)]

    # ----------------------------
    # 2. Build Bootstrap Splines for Each Group
    # ----------------------------
    print(f"Building bootstrap splines for each {group_by_col}...")
    splines_records = []
    spline_uncertainty_records = []

    for idx, pert in enumerate(tqdm(comparisons, desc=f"Creating bootstrap splines for each {group_by_col}")):
        # Filter the DataFrame for the given group
        pert_df = df[df[group_by_col] == pert].copy()
        if pert_df.empty:
            print(f"No data points for {group_by_col}={pert}, skipping...")
            continue

        print(f"Fitting spline for {group_by_col}={pert} with {len(pert_df)} points...")
        
        # Use spline_fit_wrapper for bootstrap fitting
        try:
            spline_result = spline_fit_wrapper(
                df=pert_df,
                fit_cols=fit_cols,
                stage_col=stage_col,
                bandwidth=bandwidth,
                h=h,  # Pass the step size parameter
                max_iter=max_iter,
                tol=tol,
                angle_penalty_exp=angle_penalty_exp,
                n_boots=n_boots,
                boot_size=min(len(pert_df), boot_size),  # Don't exceed available data
                n_spline_points=n_spline_points,
                time_window=time_window
            )
        except Exception as e:
            print(f"Failed to fit spline for {group_by_col}={pert}: {e}")
            continue

        if spline_result is None or spline_result.empty:
            print(f"No spline result for {group_by_col}={pert}, skipping...")
            continue

        # Create spline DataFrame with mean values
        spline_df = spline_result[fit_cols].copy()
        spline_df[group_by_col] = pert
        spline_df['spline_point_index'] = range(len(spline_df))
        
        # Store uncertainty information separately
        se_cols = [col + "_se" for col in fit_cols]
        uncertainty_df = spline_result[se_cols].copy()
        uncertainty_df[group_by_col] = pert
        uncertainty_df['spline_point_index'] = range(len(uncertainty_df))

        splines_records.append(spline_df)
        spline_uncertainty_records.append(uncertainty_df)

    # Concatenate all spline DataFrames
    if splines_records:
        pert_splines = pd.concat(splines_records, ignore_index=True)
        spline_uncertainties = pd.concat(spline_uncertainty_records, ignore_index=True)
        
        # Merge uncertainty back into main splines DataFrame
        pert_splines = pd.merge(
            pert_splines, 
            spline_uncertainties, 
            on=[group_by_col, 'spline_point_index'], 
            how='left'
        )
    else:
        # Fallback to empty DataFrame
        cols = fit_cols + [group_by_col, 'spline_point_index'] + [col + "_se" for col in fit_cols]
        pert_splines = pd.DataFrame(columns=cols)
        print("No splines were successfully fitted!")

    # Optionally save the spline data
    if save_dir and not pert_splines.empty:
        os.makedirs(save_dir, exist_ok=True)
        spline_csv_path = os.path.join(save_dir, f"bootstrap_pert_splines_{model_index}.csv")
        pert_splines.to_csv(spline_csv_path, index=False)
        print(f"Bootstrap spline DataFrame saved to: {spline_csv_path}")

    # ----------------------------
    # 3. Create segments for each group
    # ----------------------------
    print("Assigning segments and building segment_info_df...")
    
    # Rename columns to match expected format for create_spline_segments_for_df
    # Map from PCA_XX_bio to PCA_X format
    splines_for_segmentation = pert_splines.copy()
    
    # Create mapping from fit_cols to standard PCA column names
    pca_mapping = {}
    for i, col in enumerate(fit_cols):
        pca_mapping[col] = f"PCA_{i+1}"
    
    # Rename columns in splines and df
    splines_for_segmentation = splines_for_segmentation.rename(columns=pca_mapping)
    df_for_segmentation = df.rename(columns=pca_mapping)
    
    # Call the existing segmentation function
    try:
        # You'll need to import or copy the create_spline_segments_for_df function
        from spline_morph_spline_metrics import create_spline_segments_for_df
        
        df_augmented, segment_info_df, pert_splines_out = create_spline_segments_for_df(
            df=df_for_segmentation,
            pert_splines=splines_for_segmentation,
            k=k,
            group_by_col=group_by_col
        )
        
        # Rename columns back to original format
        reverse_mapping = {v: k for k, v in pca_mapping.items()}
        df_augmented = df_augmented.rename(columns=reverse_mapping)
        
    except ImportError:
        print("Warning: create_spline_segments_for_df not available. Returning splines only.")
        df_augmented = df
        segment_info_df = pd.DataFrame()
        pert_splines_out = pert_splines

    return pert_splines_out, df_augmented, segment_info_df


def plot_bootstrap_splines_with_uncertainty(
    pert_splines,
    fit_cols=None,
    group_by_col="genotype",
    save_dir=None,
    filename="bootstrap_splines_3d.html",
    confidence_bands=True,
    alpha=0.3
):
    """
    Plot 3D splines with bootstrap uncertainty bands.
    
    Parameters
    ----------
    pert_splines : pd.DataFrame
        DataFrame with spline points and uncertainty estimates
    fit_cols : list
        PCA column names to plot (default: first 3 found)
    group_by_col : str
        Column to group/color by
    save_dir : str, optional
        Directory to save plot
    filename : str
        Filename for saved plot
    confidence_bands : bool
        Whether to show uncertainty bands
    alpha : float
        Transparency for uncertainty bands
    """
    import plotly.graph_objects as go
    
    if fit_cols is None:
        fit_cols = [col for col in pert_splines.columns if col.startswith("PCA_") and col.endswith("_bio")][:3]
    
    if len(fit_cols) < 3:
        raise ValueError("Need at least 3 PCA columns for 3D plotting")
    
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Plotly
    
    for i, group in enumerate(pert_splines[group_by_col].unique()):
        group_data = pert_splines[pert_splines[group_by_col] == group]
        color = colors[i % len(colors)]
        
        # Main spline line
        fig.add_trace(go.Scatter3d(
            x=group_data[fit_cols[0]],
            y=group_data[fit_cols[1]],
            z=group_data[fit_cols[2]],
            mode='lines+markers',
            name=f'{group}',
            line=dict(color=color, width=4),
            marker=dict(size=3, color=color)
        ))
        
        # Add uncertainty bands if available and requested
        if confidence_bands:
            se_cols = [col + "_se" for col in fit_cols]
            if all(col in group_data.columns for col in se_cols):
                # Upper bound
                upper_x = group_data[fit_cols[0]] + group_data[se_cols[0]]
                upper_y = group_data[fit_cols[1]] + group_data[se_cols[1]]
                upper_z = group_data[fit_cols[2]] + group_data[se_cols[2]]
                
                # Lower bound
                lower_x = group_data[fit_cols[0]] - group_data[se_cols[0]]
                lower_y = group_data[fit_cols[1]] - group_data[se_cols[1]]
                lower_z = group_data[fit_cols[2]] - group_data[se_cols[2]]
                
                # Add uncertainty cloud (simplified as points)
                fig.add_trace(go.Scatter3d(
                    x=upper_x,
                    y=upper_y,
                    z=upper_z,
                    mode='markers',
                    name=f'{group} +SE',
                    marker=dict(size=1, color=color, opacity=alpha),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=lower_x,
                    y=lower_y,
                    z=lower_z,
                    mode='markers',
                    name=f'{group} -SE',
                    marker=dict(size=1, color=color, opacity=alpha),
                    showlegend=False
                ))

    fig.update_layout(
        scene=dict(
            xaxis_title=fit_cols[0],
            yaxis_title=fit_cols[1],
            zaxis_title=fit_cols[2],
            aspectmode='data'
        ),
        width=1200,
        height=800,
        title="Bootstrap Splines with Uncertainty Estimates"
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.write_html(os.path.join(save_dir, filename))
        print(f"Bootstrap spline plot saved to: {os.path.join(save_dir, filename)}")

    return fig


# Example usage function
def example_usage():
    """
    Example of how to use the improved function
    """
    # Import the spline_fit_wrapper
    from spline_fitting_v2 import spline_fit_wrapper
    
    # Example call
    pert_splines, df_augmented, segment_info_df = build_splines_and_segments_with_bootstrap(
        df=your_dataframe,
        model_index=1,
        spline_fit_wrapper=spline_fit_wrapper,
        save_dir="./results",
        comparisons=["wt", "mut1", "mut2"],
        group_by_col="genotype",
        stage_col="predicted_stage_hpf",
        # Spline parameters
        bandwidth=0.5,
        n_boots=20,  # More bootstrap iterations for better uncertainty estimates
        boot_size=2000,
        n_spline_points=500,
        # Segmentation parameters
        k=50
    )
    
    # Plot results with uncertainty
    fig = plot_bootstrap_splines_with_uncertainty(
        pert_splines=pert_splines,
        group_by_col="genotype",
        save_dir="./results"
    )
    
    return pert_splines, df_augmented, segment_info_df