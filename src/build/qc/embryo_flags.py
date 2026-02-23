"""
Central QC logic - THE ONLY place that determines use_embryo_flag.

This module contains the authoritative logic for determining which embryos
should be used for analysis. All other code in the pipeline should call
determine_use_embryo_flag() rather than implementing filtering logic directly.
"""

import pandas as pd


def determine_use_embryo_flag(df: pd.DataFrame) -> pd.Series:
    """
    Determine which embryos to use based on QC flags.

    This is THE ONLY function that computes use_embryo_flag.
    All other code should call this function rather than implementing
    filtering logic directly.

    Args:
        df: DataFrame with QC flag columns. Expected columns:
            - dead_flag: Manually marked dead embryos
            - dead_flag2: Death within lead time (computed in Build04)
            - sa_outlier_flag: Surface area outlier (computed in Build04)
            - sam2_qc_flag: SAM2 segmentation issues
            - frame_flag: Out of frame issues
            - no_yolk_flag: No yolk detected
            - focus_flag: Focus issues (informational only)
            - bubble_flag: Bubble presence (informational only)

    Returns:
        Boolean Series: True = use embryo, False = exclude

    Exclusion criteria:
        An embryo is excluded if ANY of these flags are True:
        - dead_flag: Manually marked dead
        - dead_flag2: Death within lead time (Build04)
        - sa_outlier_flag: Surface area outlier (Build04)
        - sam2_qc_flag: SAM2 segmentation issues
        - frame_flag: Out of frame issues
        - no_yolk_flag: No yolk detected

    NOT used for exclusion (informational only):
        - focus_flag: Focus issues (too many false positives)
        - bubble_flag: Bubble presence (too many false positives)

    Examples:
        >>> # In Build04 after computing QC flags
        >>> df["use_embryo_flag"] = determine_use_embryo_flag(df)
    """
    # Ensure all flag columns exist with False as default
    flag_defaults = {
        "dead_flag": False,
        "dead_flag2": False,
        "sa_outlier_flag": False,
        "sam2_qc_flag": False,
        "frame_flag": False,
        "no_yolk_flag": False,
    }

    for col, default in flag_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Exclusion logic: if ANY flag is True, exclude the embryo
    exclude = (
        df["dead_flag"].fillna(False).astype(bool) |
        df["dead_flag2"].fillna(False).astype(bool) |
        df["sa_outlier_flag"].fillna(False).astype(bool) |
        df["sam2_qc_flag"].fillna(False).astype(bool) |
        df["frame_flag"].fillna(False).astype(bool) |
        df["no_yolk_flag"].fillna(False).astype(bool)
    )

    # focus_flag and bubble_flag are NOT used for exclusion
    # They remain in the DataFrame for informational purposes only

    return ~exclude
