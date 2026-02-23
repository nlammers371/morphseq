"""
Group-aware data splitting utilities for train/test splits and cross-validation.

These functions ensure no data leakage by keeping all samples from a group
(e.g., embryo) together in either train or test set, never split across both.

Usage:
    from analyze.utils.splitting import (
        train_test_split_by_group,
        leave_one_out_by_group,
        get_group_split_masks,
    )
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple


def train_test_split_by_group(
    df: pd.DataFrame,
    group_col: str = 'embryo_id',
    test_fraction: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/test sets by group (e.g., embryo).

    Ensures all samples from a given group are in either train OR test,
    never split across both. This prevents data leakage in time-series
    or repeated-measures data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column name identifying groups (default: 'embryo_id')
    test_fraction : float
        Fraction of groups to reserve for testing (default: 0.20)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data

    Example
    -------
    >>> train_df, test_df = train_test_split_by_group(df, 'embryo_id', 0.2)
    >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    """
    train_mask, test_mask = get_group_split_masks(
        df, group_col, test_fraction, random_state
    )

    return df[train_mask].copy(), df[test_mask].copy()


def get_group_split_masks(
    df: pd.DataFrame,
    group_col: str = 'embryo_id',
    test_fraction: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get boolean masks for train/test split by group.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column name identifying groups
    test_fraction : float
        Fraction of groups to reserve for testing
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    train_mask : np.ndarray
        Boolean mask for training samples
    test_mask : np.ndarray
        Boolean mask for test samples
    """
    np.random.seed(random_state)

    groups = df[group_col].values
    unique_groups = np.unique(groups)

    n_test_groups = max(1, int(len(unique_groups) * test_fraction))
    test_groups = np.random.choice(
        unique_groups, size=n_test_groups, replace=False
    )
    train_groups = np.setdiff1d(unique_groups, test_groups)

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    return train_mask, test_mask


def leave_one_out_by_group(
    df: pd.DataFrame,
    group_col: str = 'embryo_id'
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, str], None, None]:
    """
    Generator for leave-one-group-out cross-validation.

    Yields train/test splits where each unique group is held out once
    as the test set. All other groups form the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column name identifying groups (default: 'embryo_id')

    Yields
    ------
    train_df : pd.DataFrame
        Training data (all groups except test group)
    test_df : pd.DataFrame
        Test data (single held-out group)
    group_id : str
        Identifier of the held-out group

    Example
    -------
    >>> for train_df, test_df, embryo_id in leave_one_out_by_group(df):
    ...     model.fit(train_df[features], train_df[target])
    ...     preds = model.predict(test_df[features])
    """
    groups = df[group_col].values
    unique_groups = np.unique(groups)

    for test_group in unique_groups:
        train_mask = groups != test_group
        test_mask = groups == test_group

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        yield train_df, test_df, test_group


def get_split_info(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    group_col: str = 'embryo_id'
) -> dict:
    """
    Get summary information about a train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    train_mask : np.ndarray
        Boolean mask for training samples
    test_mask : np.ndarray
        Boolean mask for test samples
    group_col : str
        Column name identifying groups

    Returns
    -------
    dict
        {
            'n_train_samples': int,
            'n_test_samples': int,
            'n_train_groups': int,
            'n_test_groups': int,
            'train_groups': np.ndarray,
            'test_groups': np.ndarray
        }
    """
    train_groups = df.loc[train_mask, group_col].unique()
    test_groups = df.loc[test_mask, group_col].unique()

    return {
        'n_train_samples': train_mask.sum(),
        'n_test_samples': test_mask.sum(),
        'n_train_groups': len(train_groups),
        'n_test_groups': len(test_groups),
        'train_groups': train_groups,
        'test_groups': test_groups,
    }
