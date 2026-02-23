"""
Utilities for working with model comparison time matrices.

The functions below are adapted from the exploratory script
``results/mcolon/20251020/compare_3models_full_time_matrix.py`` and made
reusable so that horizon plots and follow-up analyses can share a common data
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ConditionSpec = Union[str, Path]
ConditionMap = Mapping[str, ConditionSpec]
MatrixDict = Dict[str, pd.DataFrame]


@dataclass
class TimeMatrixBundle:
    """
    Convenience container returned by :func:`load_time_matrix_results`.

    Attributes
    ----------
    condition
        Condition label (e.g. model name).
    data
        Either a dataframe of all records or a mapping of group -> dataframe
        when grouping is requested.
    path
        Path to the source CSV file (useful for logging/debugging).
    """

    condition: str
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    path: Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_time_matrix_results(
    root: Union[str, Path],
    conditions: Union[Sequence[str], ConditionMap],
    sub_path: Union[str, Sequence[str]] = (),
    filename: str = "full_time_matrix_metrics.csv",
    group_col: Optional[str] = "genotype",
    require_columns: Optional[Sequence[str]] = None,
) -> Dict[str, TimeMatrixBundle]:
    """
    Load time-matrix CSVs for multiple conditions.

    Parameters
    ----------
    root
        Directory containing the per-condition folders.
    conditions
        Either an iterable of folder names or a mapping of ``label -> path``.
        Relative paths are resolved against ``root``.
    sub_path
        Optional subdirectory inside each condition folder – accepts a string
        or a sequence of path components.
    filename
        CSV file to read within each directory.
    group_col
        Columns such as ``genotype`` used to split the dataframe.  When
        provided, results are returned as ``{group: dataframe}``.
    require_columns
        Optional set of columns which must be present (raises ``KeyError`` if
        any are missing).

    Returns
    -------
    dict
        Mapping of condition label to :class:`TimeMatrixBundle`.
    """

    root = Path(root)

    if isinstance(conditions, Mapping):
        condition_iter = conditions.items()
    else:
        condition_iter = ((name, name) for name in conditions)

    if isinstance(sub_path, (str, Path)):
        sub_components: Iterable[Union[str, Path]] = [sub_path] if sub_path else []
    else:
        sub_components = sub_path

    bundles: Dict[str, TimeMatrixBundle] = {}

    for label, folder in condition_iter:
        cond_path = Path(folder)
        if not cond_path.is_absolute():
            cond_path = root / cond_path

        csv_path = cond_path.joinpath(*sub_components, filename)
        if not csv_path.exists():
            raise FileNotFoundError(f"Time-matrix CSV not found for '{label}': {csv_path}")

        df = pd.read_csv(csv_path)

        if require_columns:
            missing = [col for col in require_columns if col not in df.columns]
            if missing:
                raise KeyError(
                    f"Columns {missing} not found in {csv_path}. "
                    f"Available: {df.columns.tolist()}"
                )

        if group_col and group_col in df.columns:
            grouped = {grp: grp_df.reset_index(drop=True) for grp, grp_df in df.groupby(group_col)}
            data: Union[pd.DataFrame, Dict[str, pd.DataFrame]] = grouped
        else:
            data = df

        bundles[label] = TimeMatrixBundle(condition=label, data=data, path=csv_path)

    return bundles


# ---------------------------------------------------------------------------
# Matrix shaping
# ---------------------------------------------------------------------------


def build_metric_matrices(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    metric: str,
    start_col: str = "start_time",
    target_col: str = "target_time",
    index_col: Optional[str] = None,
    columns_col: Optional[str] = None,
    values_col: Optional[str] = None,
    aggfunc: str = "mean",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Reshape long-form time matrix data into 2-D arrays.

    Parameters mirror :meth:`pandas.DataFrame.pivot_table`.  The resulting
    matrices are sorted along both axes to provide deterministic ordering.
    """

    def _build_single(df: pd.DataFrame) -> pd.DataFrame:
        idx = index_col or start_col
        cols = columns_col or target_col
        vals = values_col or metric
        _validate_time_matrix_columns(df, idx, cols, vals)

        matrix = (
            df.pivot_table(index=idx, columns=cols, values=vals, aggfunc=aggfunc)
            .sort_index()
            .sort_index(axis=1)
        )
        return matrix

    if isinstance(data, Mapping):
        return {label: _build_single(df) for label, df in data.items()}

    return _build_single(data)


def align_matrix_times(
    matrices: MatrixDict,
    time_axis: str = "both",
) -> MatrixDict:
    """
    Reindex matrices so they share the same row/column times.
    """

    if time_axis not in {"rows", "cols", "both"}:
        raise ValueError("time_axis must be one of {'rows', 'cols', 'both'}")

    rows, cols = _get_aligned_time_indices(matrices, axis="both")

    aligned: MatrixDict = {}
    for label, matrix in matrices.items():
        current = matrix
        if time_axis in {"rows", "both"}:
            current = current.reindex(rows)
        if time_axis in {"cols", "both"}:
            current = current.reindex(columns=cols)
        aligned[label] = current

    return aligned


def filter_matrices_by_time_range(
    matrices: MatrixDict,
    start_min: Optional[float] = None,
    start_max: Optional[float] = None,
    target_min: Optional[float] = None,
    target_max: Optional[float] = None,
) -> MatrixDict:
    """
    Subset matrices to a given start/target window.
    """

    filtered: MatrixDict = {}
    for label, matrix in matrices.items():
        current = matrix

        if start_min is not None or start_max is not None:
            row_mask = pd.Series(True, index=current.index)
            if start_min is not None:
                row_mask &= current.index >= start_min
            if start_max is not None:
                row_mask &= current.index <= start_max
            current = current.loc[row_mask[row_mask].index]

        if target_min is not None or target_max is not None:
            col_mask = pd.Series(True, index=current.columns)
            if target_min is not None:
                col_mask &= current.columns >= target_min
            if target_max is not None:
                col_mask &= current.columns <= target_max
            current = current.loc[:, col_mask[col_mask].index]

        filtered[label] = current

    return filtered


def interpolate_missing_times(
    matrices: MatrixDict,
    method: str = "linear",
    axis: str = "both",
) -> MatrixDict:
    """
    Fill gaps in matrices using pandas' interpolation.
    """

    if axis not in {"index", "columns", "both"}:
        raise ValueError("axis must be one of {'index', 'columns', 'both'}")

    interpolated: MatrixDict = {}
    for label, matrix in matrices.items():
        current = matrix
        if axis in {"index", "both"}:
            current = current.interpolate(method=method, axis=0, limit_direction="both")
        if axis in {"columns", "both"}:
            current = current.interpolate(method=method, axis=1, limit_direction="both")
        interpolated[label] = current
    return interpolated


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def compute_matrix_statistics(
    matrices: MatrixDict,
    statistics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, std, etc.) for each matrix.
    """

    if statistics is None:
        statistics = ["mean", "std", "median", "min", "max"]

    rows = []
    for label, matrix in matrices.items():
        flat = matrix.to_numpy().ravel()
        flat = flat[~np.isnan(flat)]
        if flat.size == 0:
            stats = {stat: np.nan for stat in statistics}
            stats["n_observations"] = 0
        else:
            series = pd.Series(flat)
            stats = series.agg(statistics).to_dict()
            stats["n_observations"] = flat.size
        stats["label"] = label
        rows.append(stats)

    return pd.DataFrame(rows).set_index("label")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_time_matrix_columns(
    df: pd.DataFrame,
    start_col: str,
    target_col: str,
    values_col: str,
) -> None:
    missing = [col for col in (start_col, target_col, values_col) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}; available columns: {df.columns.tolist()}"
        )


def _get_aligned_time_indices(
    matrices: MatrixDict,
    axis: str = "rows",
) -> Tuple[np.ndarray, np.ndarray]:
    if axis not in {"rows", "cols", "both"}:
        raise ValueError("axis must be one of {'rows', 'cols', 'both'}")

    row_vals: set = set()
    col_vals: set = set()
    for matrix in matrices.values():
        row_vals.update(matrix.index.tolist())
        col_vals.update(matrix.columns.tolist())

    rows = np.array(sorted(row_vals))
    cols = np.array(sorted(col_vals))

    return rows, cols


__all__ = [
    "TimeMatrixBundle",
    "load_time_matrix_results",
    "build_metric_matrices",
    "align_matrix_times",
    "filter_matrices_by_time_range",
    "interpolate_missing_times",
    "compute_matrix_statistics",
]
