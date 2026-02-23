"""
Data preparation utilities - intentionally thin wrappers.

CAUTION
-------
Avoid "wrappers on wrappers": if a given analysis only needs to add the group
column once, call `add_group_column()` directly in that analysis script. Keep
this module for shared, repeated prep logic (e.g. consistent filtering rules).
"""
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add src to path for analyze.* imports
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from analyze.difference_detection.comparison import add_group_column


def load_ids_from_file(filepath: Path) -> List[str]:
    """
    Load embryo IDs from a text file (one ID per line).

    Parameters
    ----------
    filepath : Path
        Path to text file containing embryo IDs

    Returns
    -------
    List[str]
        List of embryo IDs with whitespace stripped
    """
    filepath = Path(filepath)
    return [line.strip() for line in filepath.read_text().strip().split('\n') if line.strip()]


def prepare_comparison_data(
    df: pd.DataFrame,
    group1_ids: List[str],
    group2_ids: List[str],
    group1_label: str,
    group2_label: str,
    subset_by: Optional[str] = None,
    subset_values: Optional[List] = None,
    embryo_id_col: str = 'embryo_id',
) -> pd.DataFrame:
    """
    Prepare data for group comparison.

    Thin wrapper around add_group_column() with optional filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trajectory data with embryo_id column
    group1_ids : List[str]
        Embryo IDs for group1 (positive/phenotype class)
    group2_ids : List[str]
        Embryo IDs for group2 (negative/reference class)
    group1_label : str
        Label for group1 (e.g., 'CE', 'Penetrant')
    group2_label : str
        Label for group2 (e.g., 'WT', 'Control')
    subset_by : Optional[str]
        Column to filter by before grouping (e.g., 'pair')
    subset_values : Optional[List]
        Values to keep in subset_by column
    embryo_id_col : str
        Column containing embryo IDs (default: 'embryo_id')

    Returns
    -------
    pd.DataFrame
        DataFrame with 'group' column added, filtered to specified embryos only

    Example
    -------
    >>> df_prep = prepare_comparison_data(
    ...     df,
    ...     group1_ids=ce_embryo_ids,
    ...     group2_ids=wt_embryo_ids,
    ...     group1_label='CE',
    ...     group2_label='WT'
    ... )
    """
    # Optional filtering first
    if subset_by and subset_values:
        df = df[df[subset_by].isin(subset_values)].copy()

    # Use existing API to add group column
    df_with_groups = add_group_column(
        df,
        groups={group1_label: group1_ids, group2_label: group2_ids},
        column_name='group',
        embryo_id_col=embryo_id_col,
    )

    # Filter to only embryos in our groups (remove NaN groups)
    df_filtered = df_with_groups[df_with_groups['group'].notna()].copy()

    return df_filtered
