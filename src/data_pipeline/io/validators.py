"""Shared validation utilities for schema enforcement."""

import pandas as pd
from typing import Iterable, List, Optional, Set


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    stage_name: str,
    *,
    nullable_columns: Optional[Iterable[str]] = None,
) -> bool:
    """
    Validate DataFrame against schema requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        stage_name: Name of processing stage (for error messages)
        nullable_columns: Columns that are required to exist but may contain null/NaN values.

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with clear message

    Checks:
    1. All required columns exist in DataFrame
    2. No required columns contain null/NaN values (except nullable_columns)
    """
    nullable: Set[str] = set(nullable_columns or [])

    # Check column existence
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {stage_name}: {sorted(missing)}"
        )

    # Check for null values in required columns
    for col in required_columns:
        if col in nullable:
            continue
        if df[col].isna().any():
            null_count = df[col].isna().sum()
            raise ValueError(
                f"Column '{col}' contains {null_count} null/empty values in {stage_name}"
            )

    return True
