"""
PCA transformation utilities for VAE embeddings.

Provides functions to:
1. Fit PCA on VAE embeddings (z_mu_b columns)
2. Transform embeddings to PCA space
3. Compute wildtype reference (time-binned average)
4. Subtract wildtype reference to get deviation trajectories

These utilities enable visualization and clustering of embryos in a reduced
dimensional space that captures the main axes of morphological variation.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple


# ==============================================================================
# PCA Fitting and Transformation
# ==============================================================================

def fit_pca_on_embeddings(
    df: pd.DataFrame,
    z_mu_cols: Optional[List[str]] = None,
    n_components: int = 3,
    scale: bool = True,
) -> Tuple[PCA, Optional[StandardScaler], List[str]]:
    """
    Learn a PCA transformation from VAE embedding data.

    Fits a StandardScaler (computes mean/std from the data) and PCA
    (computes principal components). Returns the fitted objects so
    they can be reused consistently.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing VAE embedding columns
    z_mu_cols : list of str, optional
        Columns to use for PCA. If None, auto-detects z_mu_b* columns.
    n_components : int, default=3
        Number of PCA components to compute
    scale : bool, default=True
        Whether to standardize features before PCA (recommended)

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA object
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler (None if scale=False)
    z_mu_cols : list of str
        Column names used for PCA (useful if auto-detected)

    Examples
    --------
    >>> pca, scaler, cols = fit_pca_on_embeddings(df, n_components=3)
    >>> print(f"Variance explained: {pca.explained_variance_ratio_}")
    """
    # Auto-detect z_mu_b columns if not provided
    if z_mu_cols is None:
        z_mu_cols = [c for c in df.columns if 'z_mu_b' in c]
        if len(z_mu_cols) == 0:
            raise ValueError("No z_mu_b* columns found in DataFrame")
        print(f"Auto-detected {len(z_mu_cols)} z_mu_b columns")

    # Extract embedding matrix
    X = df[z_mu_cols].values

    # Remove rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid = X[valid_mask]

    if len(X_valid) == 0:
        raise ValueError("No valid (non-NaN) rows in embedding data")

    print(f"Fitting PCA on {len(X_valid)} samples with {len(z_mu_cols)} features")

    # Scale if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
    else:
        X_scaled = X_valid

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Report variance explained
    var_explained = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(var_explained)
    print(f"Variance explained by {n_components} components:")
    for i, (v, c) in enumerate(zip(var_explained, cumulative_var)):
        print(f"  PC{i+1}: {v*100:.1f}% (cumulative: {c*100:.1f}%)")

    return pca, scaler, z_mu_cols


def transform_embeddings_to_pca(
    df: pd.DataFrame,
    pca: PCA,
    scaler: Optional[StandardScaler] = None,
    z_mu_cols: Optional[List[str]] = None,
    prefix: str = 'PCA',
) -> pd.DataFrame:
    """
    Apply a fitted PCA transformation to embedding data.

    Transforms embeddings using a pre-fitted PCA (and scaler) and adds
    PCA columns to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing VAE embedding columns
    pca : sklearn.decomposition.PCA
        Fitted PCA object (from fit_pca_on_embeddings)
    scaler : sklearn.preprocessing.StandardScaler, optional
        Fitted scaler (from fit_pca_on_embeddings)
    z_mu_cols : list of str, optional
        Columns to transform. If None, auto-detects z_mu_b* columns.
    prefix : str, default='PCA'
        Prefix for PCA column names (e.g., 'PCA' -> 'PCA_1', 'PCA_2', ...)

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added PCA columns

    Examples
    --------
    >>> df_with_pca = transform_embeddings_to_pca(df, pca, scaler)
    >>> # Now df_with_pca has columns: PCA_1, PCA_2, PCA_3, ...
    """
    # Auto-detect z_mu_b columns if not provided
    if z_mu_cols is None:
        z_mu_cols = [c for c in df.columns if 'z_mu_b' in c]
        if len(z_mu_cols) == 0:
            raise ValueError("No z_mu_b* columns found in DataFrame")

    # Make a copy
    df_result = df.copy()

    # Extract embedding matrix
    X = df[z_mu_cols].values

    # Handle NaN: create output array with NaN for invalid rows
    valid_mask = ~np.isnan(X).any(axis=1)
    X_transformed = np.full((len(X), pca.n_components_), np.nan)

    if valid_mask.sum() > 0:
        X_valid = X[valid_mask]

        # Scale if scaler provided
        if scaler is not None:
            X_scaled = scaler.transform(X_valid)
        else:
            X_scaled = X_valid

        # Transform
        X_transformed[valid_mask] = pca.transform(X_scaled)

    # Add PCA columns
    for i in range(pca.n_components_):
        col_name = f"{prefix}_{i+1}"
        df_result[col_name] = X_transformed[:, i]

    return df_result


# ==============================================================================
# Wildtype Reference Computation
# ==============================================================================

def compute_wt_reference_by_time(
    df: pd.DataFrame,
    pca_cols: List[str],
    time_col: str = 'predicted_stage_hpf',
    wt_embryo_ids: Optional[List[str]] = None,
    embryo_id_col: str = 'embryo_id',
    genotype_col: str = 'genotype',
    wt_genotype_pattern: str = 'wildtype',
    bin_width: float = 2.0,
) -> pd.DataFrame:
    """
    Compute mean wildtype PCA values per time bin.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing PCA columns and time column
    pca_cols : list of str
        PCA column names (e.g., ['PCA_1', 'PCA_2', 'PCA_3'])
    time_col : str, default='predicted_stage_hpf'
        Time column for binning
    wt_embryo_ids : list of str, optional
        Explicit list of wildtype embryo IDs. If None, uses genotype filtering.
    embryo_id_col : str, default='embryo_id'
        Column containing embryo IDs
    genotype_col : str, default='genotype'
        Column containing genotype information
    wt_genotype_pattern : str, default='wildtype'
        Pattern to match in genotype column (case-insensitive contains)
    bin_width : float, default=2.0
        Width of time bins in hours

    Returns
    -------
    pd.DataFrame
        Reference table with columns:
        - time_bin: bin center
        - {pca_col}_wt_mean: mean value per PCA column
        - n_embryos: number of embryos in bin

    Examples
    --------
    >>> wt_ref = compute_wt_reference_by_time(
    ...     df, ['PCA_1', 'PCA_2', 'PCA_3'],
    ...     wt_embryo_ids=wildtype_ids
    ... )
    """
    # Filter to wildtype embryos
    if wt_embryo_ids is not None:
        df_wt = df[df[embryo_id_col].isin(wt_embryo_ids)].copy()
    elif genotype_col in df.columns:
        # Filter by genotype pattern
        mask = df[genotype_col].astype(str).str.lower().str.contains(
            wt_genotype_pattern.lower(), na=False
        )
        df_wt = df[mask].copy()
    else:
        raise ValueError(
            "Must provide either wt_embryo_ids or genotype_col with wt_genotype_pattern"
        )

    n_wt = df_wt[embryo_id_col].nunique()
    print(f"Computing WT reference from {n_wt} wildtype embryos")

    if len(df_wt) == 0:
        raise ValueError("No wildtype data found")

    # Create time bins
    df_wt['_time_bin'] = (
        np.floor(df_wt[time_col] / bin_width) * bin_width + bin_width / 2
    )

    # Compute mean per time bin
    agg_dict = {col: 'mean' for col in pca_cols}
    agg_dict[embryo_id_col] = 'nunique'

    wt_ref = df_wt.groupby('_time_bin').agg(agg_dict).reset_index()
    wt_ref = wt_ref.rename(columns={
        '_time_bin': 'time_bin',
        embryo_id_col: 'n_embryos',
    })

    # Rename PCA columns to indicate they are WT means
    for col in pca_cols:
        wt_ref = wt_ref.rename(columns={col: f'{col}_wt_mean'})

    print(f"Created reference with {len(wt_ref)} time bins")
    return wt_ref


def subtract_wt_reference(
    df: pd.DataFrame,
    wt_reference: pd.DataFrame,
    pca_cols: List[str],
    time_col: str = 'predicted_stage_hpf',
    bin_width: float = 2.0,
    suffix: str = '_delta',
) -> pd.DataFrame:
    """
    Subtract wildtype reference from PCA values to get deviation.

    For each embryo at each timepoint, finds the corresponding time bin
    in the WT reference and subtracts the WT mean PCA values.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing PCA columns and time column
    wt_reference : pd.DataFrame
        Reference table from compute_wt_reference_by_time
    pca_cols : list of str
        PCA column names (e.g., ['PCA_1', 'PCA_2', 'PCA_3'])
    time_col : str, default='predicted_stage_hpf'
        Time column for bin matching
    bin_width : float, default=2.0
        Width of time bins (must match wt_reference)
    suffix : str, default='_delta'
        Suffix for delta column names (e.g., 'PCA_1' -> 'PCA_1_delta')

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added delta columns

    Examples
    --------
    >>> df_with_delta = subtract_wt_reference(
    ...     df, wt_ref, ['PCA_1', 'PCA_2', 'PCA_3']
    ... )
    >>> # Now df_with_delta has columns: PCA_1_delta, PCA_2_delta, PCA_3_delta
    """
    df_result = df.copy()

    # Compute time bin for each row
    df_result['_time_bin'] = (
        np.floor(df_result[time_col] / bin_width) * bin_width + bin_width / 2
    )

    # Merge with WT reference
    wt_cols = ['time_bin'] + [f'{col}_wt_mean' for col in pca_cols]
    df_merged = df_result.merge(
        wt_reference[wt_cols],
        left_on='_time_bin',
        right_on='time_bin',
        how='left'
    )

    # Compute deltas
    for col in pca_cols:
        wt_mean_col = f'{col}_wt_mean'
        delta_col = f'{col}{suffix}'

        if wt_mean_col in df_merged.columns:
            df_merged[delta_col] = df_merged[col] - df_merged[wt_mean_col]
        else:
            print(f"Warning: {wt_mean_col} not found in reference")
            df_merged[delta_col] = np.nan

    # Clean up temporary columns
    cols_to_drop = ['_time_bin', 'time_bin'] + [f'{col}_wt_mean' for col in pca_cols]
    cols_to_drop = [c for c in cols_to_drop if c in df_merged.columns]
    df_result = df_merged.drop(columns=cols_to_drop)

    # Report how many values got WT reference
    n_with_ref = df_result[f'{pca_cols[0]}{suffix}'].notna().sum()
    n_total = len(df_result)
    print(f"WT reference subtracted for {n_with_ref}/{n_total} rows ({n_with_ref/n_total*100:.1f}%)")

    return df_result


# ==============================================================================
# Convenience Functions
# ==============================================================================

def fit_transform_pca(
    df: pd.DataFrame,
    z_mu_cols: Optional[List[str]] = None,
    n_components: int = 3,
    scale: bool = True,
    prefix: str = 'PCA',
) -> Tuple[pd.DataFrame, PCA, Optional[StandardScaler], List[str]]:
    """
    Convenience function: fit PCA and transform in one step.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing VAE embedding columns
    z_mu_cols : list of str, optional
        Columns to use for PCA. If None, auto-detects z_mu_b* columns.
    n_components : int, default=3
        Number of PCA components
    scale : bool, default=True
        Whether to standardize features before PCA
    prefix : str, default='PCA'
        Prefix for PCA column names

    Returns
    -------
    df_with_pca : pd.DataFrame
        DataFrame with added PCA columns
    pca : PCA
        Fitted PCA object
    scaler : StandardScaler or None
        Fitted scaler
    z_mu_cols : list of str
        Column names used

    Examples
    --------
    >>> df_pca, pca, scaler, cols = fit_transform_pca(df, n_components=3)
    """
    pca, scaler, z_mu_cols = fit_pca_on_embeddings(
        df, z_mu_cols, n_components, scale
    )
    df_with_pca = transform_embeddings_to_pca(
        df, pca, scaler, z_mu_cols, prefix
    )
    return df_with_pca, pca, scaler, z_mu_cols
