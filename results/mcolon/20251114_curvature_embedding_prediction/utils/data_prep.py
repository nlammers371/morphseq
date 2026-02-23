"""
Data preparation utilities for regression analysis.

Handles feature/target extraction, validation, and preprocessing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


def prepare_features_and_target(
    df: pd.DataFrame,
    feature_pattern: str = 'embedding_dim_',
    target_col: str = None,
    remove_nan: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Prepare features and target columns for regression.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_pattern : str
        Prefix or pattern to identify feature columns.
        If contains wildcards, uses glob pattern matching.
    target_col : str
        Target column name. If None, uses any remaining numeric column.
    remove_nan : bool
        Drop rows with NaN in features or target
    verbose : bool
        Print preparation details

    Returns
    -------
    tuple
        (df_clean, feature_cols, target_col)
    """
    df_prep = df.copy()

    # Identify feature columns
    if feature_pattern.endswith('_'):
        # Simple prefix match
        feature_cols = sorted([
            col for col in df_prep.columns
            if col.startswith(feature_pattern)
        ])
    else:
        # Assume exact column names
        feature_cols = [col for col in [feature_pattern] if col in df_prep.columns]

    if len(feature_cols) == 0:
        raise ValueError(f"No columns found matching pattern '{feature_pattern}'")

    if verbose:
        print(f"\n  Features identified: {len(feature_cols)}")
        print(f"    First: {feature_cols[0]}")
        print(f"    Last:  {feature_cols[-1]}")

    # Identify target column
    if target_col is None:
        raise ValueError("target_col must be specified")

    if target_col not in df_prep.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    if verbose:
        print(f"\n  Target column: {target_col}")

    # Check data completeness
    n_before = len(df_prep)
    n_feat_nan = df_prep[feature_cols].isna().sum().sum()
    n_target_nan = df_prep[target_col].isna().sum()

    if verbose:
        print(f"\n  Data completeness:")
        print(f"    NaN in features: {n_feat_nan}")
        print(f"    NaN in target: {n_target_nan}")

    # Remove rows with NaN
    if remove_nan:
        cols_to_check = feature_cols + [target_col]
        df_prep = df_prep.dropna(subset=cols_to_check).copy()

        n_after = len(df_prep)
        n_dropped = n_before - n_after

        if verbose and n_dropped > 0:
            print(f"    Dropped {n_dropped} rows with missing values")
            print(f"    Remaining: {n_after} samples")

    # Validate data types
    for col in feature_cols + [target_col]:
        if not np.issubdtype(df_prep[col].dtype, np.number):
            raise ValueError(f"Column '{col}' is not numeric (dtype={df_prep[col].dtype})")

    return df_prep, feature_cols, target_col


def validate_data_completeness(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    embryo_id_col: str = 'embryo_id',
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate data structure and completeness for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate
    feature_cols : list of str
        Feature column names
    target_col : str
        Target column name
    embryo_id_col : str
        Embryo identifier column
    verbose : bool
        Print validation report

    Returns
    -------
    dict
        Validation results with keys:
        - 'is_valid': bool
        - 'n_samples': int
        - 'n_features': int
        - 'n_embryos': int
        - 'messages': list of validation messages
    """
    messages = []
    is_valid = True

    # Check required columns exist
    required_cols = feature_cols + [target_col]
    if embryo_id_col:
        required_cols.append(embryo_id_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        messages.append(f"ERROR: Missing columns: {missing_cols}")
        is_valid = False

    # Check for NaN
    n_feat_nan = df[feature_cols].isna().sum().sum()
    n_target_nan = df[target_col].isna().sum()

    if n_feat_nan > 0:
        messages.append(f"WARNING: {n_feat_nan} NaN values in features")

    if n_target_nan > 0:
        messages.append(f"WARNING: {n_target_nan} NaN values in target")

    # Check data types
    for col in feature_cols + [target_col]:
        if col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                messages.append(f"ERROR: Column '{col}' is not numeric")
                is_valid = False

    # Check min samples per embryo
    if embryo_id_col and embryo_id_col in df.columns:
        samples_per_embryo = df.groupby(embryo_id_col).size()
        min_samples = samples_per_embryo.min()

        if min_samples < 1:
            messages.append(f"ERROR: Some embryos have < 1 sample")
            is_valid = False

        if min_samples < 2:
            messages.append(f"WARNING: Some embryos have only 1 sample (LOEO will be problematic)")

    # Print report
    if verbose:
        print(f"\n  Data Validation Report:")
        print(f"    Samples: {len(df)}")
        print(f"    Features: {len(feature_cols)}")
        if embryo_id_col and embryo_id_col in df.columns:
            print(f"    Embryos: {df[embryo_id_col].nunique()}")
        print(f"    Valid: {is_valid}")

        if messages:
            print(f"\n  Messages:")
            for msg in messages:
                print(f"    {msg}")

    return {
        'is_valid': is_valid,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'n_embryos': df[embryo_id_col].nunique() if embryo_id_col and embryo_id_col in df.columns else None,
        'messages': messages
    }


def get_embedding_columns(
    df: pd.DataFrame,
    pattern: str = 'z_mu_b_'
) -> List[str]:
    """
    Identify embedding columns in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Data potentially containing embedding columns
    pattern : str
        Column name pattern for embeddings

    Returns
    -------
    list of str
        Matching column names, sorted
    """
    cols = sorted([col for col in df.columns if col.startswith(pattern)])
    return cols


def rename_embeddings_to_standard(
    df: pd.DataFrame,
    old_pattern: str = 'z_mu_b_',
    new_prefix: str = 'embedding_dim_'
) -> pd.DataFrame:
    """
    Rename embedding columns to standard naming scheme.

    Parameters
    ----------
    df : pd.DataFrame
        Data with embedding columns
    old_pattern : str
        Current prefix of embedding columns
    new_prefix : str
        New prefix for renamed columns

    Returns
    -------
    pd.DataFrame
        Data with renamed columns
    """
    df_renamed = df.copy()

    embedding_cols = get_embedding_columns(df_renamed, pattern=old_pattern)

    rename_dict = {}
    for i, old_col in enumerate(sorted(embedding_cols)):
        new_col = f'{new_prefix}{i}'
        rename_dict[old_col] = new_col

    df_renamed = df_renamed.rename(columns=rename_dict)

    return df_renamed


def filter_by_genotype(
    df: pd.DataFrame,
    genotypes: List[str],
    genotype_col: str = 'genotype',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter data to specific genotypes.

    Parameters
    ----------
    df : pd.DataFrame
    genotypes : list of str
        Genotype values to keep
    genotype_col : str
        Column name for genotype
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    if genotype_col not in df.columns:
        warnings.warn(f"Column '{genotype_col}' not found in data")
        return df

    n_before = len(df)
    df_filt = df[df[genotype_col].isin(genotypes)].copy()
    n_after = len(df_filt)

    if verbose:
        print(f"\n  Filtered to genotypes: {genotypes}")
        print(f"    Before: {n_before}")
        print(f"    After:  {n_after}")

    return df_filt


def stratify_by_genotype(
    df: pd.DataFrame,
    genotype_col: str = 'genotype'
) -> Dict[str, pd.DataFrame]:
    """
    Split data into separate dataframes per genotype.

    Parameters
    ----------
    df : pd.DataFrame
    genotype_col : str
        Column name for genotype

    Returns
    -------
    dict
        {genotype_value: dataframe, ...}
    """
    dfs_by_genotype = {}

    for genotype in df[genotype_col].unique():
        mask = df[genotype_col] == genotype
        dfs_by_genotype[genotype] = df[mask].copy()

    return dfs_by_genotype
