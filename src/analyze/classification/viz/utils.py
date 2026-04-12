from __future__ import annotations

import pandas as pd


def _pretty_axis_label(col: str) -> str:
    """Return a human-readable axis label for a known column name, or the column name itself."""
    _KNOWN = {
        "time_bin_center": "Time (hpf)",
        "time_bin": "Time bin",
        "hpf": "Time (hpf)",
    }
    return _KNOWN.get(col, col.replace("_", " "))


def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
    *,
    context: str = "",
) -> None:
    """Raise ValueError if any required columns are missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}missing required columns {missing!r}. "
            f"Available: {df.columns.tolist()}"
        )


def validate_unique_embryo_x(
    df: pd.DataFrame,
    *,
    embryo_col: str = "embryo_id",
    x_col: str = "time_bin_center",
) -> None:
    """Raise ValueError if any (embryo, x) pair appears more than once.

    This is the core invariant for trajectory plots: each embryo should have
    at most one margin value per x position. Duplicates cause silent overplotting.
    """
    validate_required_columns(df, [embryo_col, x_col], context="validate_unique_embryo_x")

    dup = (
        df.groupby([embryo_col, x_col], dropna=False)
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    if not dup.empty:
        preview = dup.head(10).to_dict("records")
        raise ValueError(
            f"Expected one row per ({embryo_col!r}, {x_col!r}) pair, "
            f"but found duplicates. Examples: {preview}. "
            "Filter to a single comparison and/or feature before plotting."
        )


def validate_margin_range(
    df: pd.DataFrame,
    margin_col: str,
    *,
    tol: float = 1e-6,
) -> None:
    """Raise ValueError if margin_col values fall outside [-1, 1] (up to tol)."""
    validate_required_columns(df, [margin_col], context="validate_margin_range")
    vals = df[margin_col].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return
    lo, hi = float(vals.min()), float(vals.max())
    if lo < -1.0 - tol or hi > 1.0 + tol:
        raise ValueError(
            f"margin_col={margin_col!r} has values outside [-1, 1]: "
            f"min={lo:.4f}, max={hi:.4f}. "
            "Use coerce_margin_range() to normalize first, or choose the correct column."
        )
