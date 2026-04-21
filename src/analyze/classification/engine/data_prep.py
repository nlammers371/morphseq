"""Classification-scoped data preparation helpers.

These functions are shared between the main classification loop (engine/loop.py)
and the lightweight direction extractor (directions/extract.py). They live here
so that neither file needs to import from the other.

All three functions are classification-specific: they reference class_col semantics,
ResolvedComparison, or the binary _y label convention. They do not belong in
the global src/analyze/utils/ because they assume classification data contracts.

Public API (used via relative import within the classification package)
-----------------------------------------------------------------------
_resolve_feature_columns  -- feature spec -> concrete column lists
_build_binary_labels      -- pooling-aware binary labeling (_y column)
_bin_and_aggregate        -- floor-bin + per-(id, bin, label) mean aggregation
"""

from __future__ import annotations

import pandas as pd

from ...utils.binning import add_time_bins

from .comparison_resolution import ResolvedComparison


def _resolve_feature_columns(
    df: pd.DataFrame,
    features: dict[str, str | list[str]],
) -> dict[str, list[str]]:
    """Resolve user-facing feature spec into concrete column lists.

    Parameters
    ----------
    df : source DataFrame (columns inspected for prefix matching)
    features : dict mapping feature set name to either:
        - str : prefix-match against df.columns (sorted)
        - list[str] : passed through; missing columns raise ValueError

    Returns
    -------
    dict mapping feature set name -> sorted list of column names
    """
    resolved: dict[str, list[str]] = {}
    for name, spec in features.items():
        if isinstance(spec, str):
            cols = [c for c in df.columns if c.startswith(spec)]
            if not cols:
                raise ValueError(
                    f"Feature set {name!r}: no columns match prefix {spec!r}"
                )
            resolved[name] = sorted(cols)
        elif isinstance(spec, list):
            missing = [c for c in spec if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Feature set {name!r}: missing columns {missing}"
                )
            resolved[name] = list(spec)
        else:
            raise TypeError(
                f"Feature set {name!r}: expected str or list[str], "
                f"got {type(spec).__name__}"
            )
    return resolved


def _build_binary_labels(
    df: pd.DataFrame,
    class_col: str,
    comparison: ResolvedComparison,
) -> pd.DataFrame:
    """Filter *df* to comparison members and assign a binary ``_y`` column.

    ``_y = 1`` for positive members, ``_y = 0`` for negative members.
    Rows from unrelated classes are dropped.

    Parameters
    ----------
    df : source DataFrame; must contain class_col
    class_col : column with group/genotype labels
    comparison : ResolvedComparison defining positive_members and negative_members
    """
    pos_set = set(comparison.positive_members)
    neg_set = set(comparison.negative_members)
    all_members = pos_set | neg_set

    mask = df[class_col].isin(all_members)
    out = df.loc[mask].copy()

    out["_y"] = out[class_col].map(lambda x: 1 if x in pos_set else 0)
    return out


def _bin_and_aggregate(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    bin_width: float,
) -> pd.DataFrame:
    """Bin by *time_col*, then mean-aggregate per (id, bin, label).

    Adds two columns to the result:
      _time_bin      : int floor of the time bin (e.g. 22 for [22, 24) hpf)
      time_bin_center: float center of the bin (e.g. 23.0 for bin_width=2)

    Aggregation: mean of feature_cols per (id_col, _time_bin, time_bin_center, _y).
    The ``_y`` column must be present (added by _build_binary_labels).

    Parameters
    ----------
    df : labeled DataFrame with a ``_y`` column
    id_col : per-embryo unique identifier column
    time_col : continuous time column (hpf)
    feature_cols : feature columns to aggregate
    bin_width : time bin width in hpf
    """
    binned = add_time_bins(df, time_col=time_col, bin_width=bin_width, bin_col="_time_bin")
    return _aggregate_binned(
        binned,
        id_col=id_col,
        feature_cols=feature_cols,
        label_col="_y",
        bin_col="_time_bin",
        bin_width=bin_width,
    )


def _aggregate_binned(
    df_binned: pd.DataFrame,
    id_col: str,
    feature_cols: list[str],
    *,
    label_col: str | None = None,
    bin_col: str = "time_bin",
    bin_width: float,
) -> pd.DataFrame:
    """Mean-aggregate features from a pre-binned DataFrame.

    ``df_binned`` must already contain ``bin_col``. This helper only groups and
    averages; it does not perform flooring/bin assignment.
    """
    out = df_binned.copy()
    out["time_bin_center"] = out[bin_col].astype(float) + bin_width / 2.0

    groupby_cols = [id_col, bin_col, "time_bin_center"]
    if label_col:
        groupby_cols.append(label_col)
    agg_cols = [c for c in feature_cols if c in out.columns]
    return out.groupby(groupby_cols, as_index=False)[agg_cols].mean()
