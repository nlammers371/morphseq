"""
High-level orchestration helpers for horizon plot analyses.

These functions glue together the data-loading, metric reshaping, and plotting
utilities exposed in :mod:`analyze.difference_detection.time_matrix` and
:mod:`analyze.difference_detection.horizon_plots`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from . import horizon_plots
from .time_matrix import (
    MatrixDict,
    TimeMatrixBundle,
    align_matrix_times,
    build_metric_matrices,
    load_time_matrix_results,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HorizonPlotContext:
    """
    Bundle holding everything required to render a horizon grid.
    """

    bundles: Dict[str, TimeMatrixBundle]
    matrices: Dict[str, Dict[str, pd.DataFrame]]
    metric: str
    statistics: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def load_and_prepare_time_matrices(
    root: Union[str, Path],
    conditions: Union[Sequence[str], Mapping[str, Union[str, Path]]],
    *,
    metric: str = "mae",
    sub_path: Union[str, Sequence[str]] = (),
    filename: str = "full_time_matrix_metrics.csv",
    group_col: str = "genotype",
    metrics_for_summary: Sequence[str] = ("mae", "r2", "error_std"),
    training_lookup: Optional[Mapping[str, str]] = None,
) -> HorizonPlotContext:
    """
    Load CSV results and reshape them into matrices ready for plotting.

    Parameters
    ----------
    root / conditions / sub_path / filename
        Forwarded to :func:`load_time_matrix_results`.
    metric
        Primary metric to visualise in the horizon plot grid.
    metrics_for_summary
        Metrics to include in the computed summary table.
    training_lookup
        Optional mapping of ``condition -> training_group`` used to tag rows as
        "leave-one-embryo-out" in the summary.
    """

    bundles = load_time_matrix_results(
        root=root,
        conditions=conditions,
        sub_path=sub_path,
        filename=filename,
        group_col=group_col,
    )

    matrices: Dict[str, Dict[str, pd.DataFrame]] = {}
    for label, bundle in bundles.items():
        if isinstance(bundle.data, dict):
            matrices[label] = build_metric_matrices(bundle.data, metric=metric)  # type: ignore[arg-type]
        else:
            matrices[label] = {"all": build_metric_matrices(bundle.data, metric=metric)}

    matrices = _align_nested_matrices(matrices)
    summary = summarise_bundles(bundles, metrics=metrics_for_summary, group_col=group_col, training_lookup=training_lookup)

    return HorizonPlotContext(
        bundles=bundles,
        matrices=matrices,
        metric=metric,
        statistics=summary,
    )


def render_horizon_grid(
    context: HorizonPlotContext,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    **plot_kwargs,
):
    """
    Wrapper around :func:`horizon_plots.plot_horizon_grid` using a prepared context.
    """

    return horizon_plots.plot_horizon_grid(
        context.matrices,
        row_labels=row_labels,
        col_labels=col_labels,
        metric=context.metric,
        **plot_kwargs,
    )


# ---------------------------------------------------------------------------
# Summary logic
# ---------------------------------------------------------------------------


def summarise_bundles(
    bundles: Mapping[str, TimeMatrixBundle],
    *,
    metrics: Iterable[str],
    group_col: str,
    training_lookup: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """
    Compute summary statistics for each condition / group / metric combination.
    """

    rows = []
    for condition, bundle in bundles.items():
        if isinstance(bundle.data, dict):
            groups = bundle.data.items()
        else:
            groups = [("all", bundle.data)]

        for group, df in groups:
            for metric in metrics:
                if metric not in df.columns:
                    continue
                stats = df[metric].agg(["mean", "std", "median", "min", "max"])
                rows.append(
                    {
                        "condition": condition,
                        "group": group,
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "median": stats["median"],
                        "min": stats["min"],
                        "max": stats["max"],
                        "n_timepoints": len(df),
                        "uses_loeo": training_lookup and training_lookup.get(condition) == group,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "condition",
                "group",
                "metric",
                "mean",
                "std",
                "median",
                "min",
                "max",
                "n_timepoints",
                "uses_loeo",
            ]
        )

    summary = pd.DataFrame(rows)
    summary["uses_loeo"] = summary["uses_loeo"].fillna(False)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_nested_matrices(matrices: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Align matrices across conditions for each group key.
    """

    group_keys = set()
    for per_condition in matrices.values():
        group_keys.update(per_condition.keys())

    for group in group_keys:
        to_align: MatrixDict = {
            condition: per_condition[group]
            for condition, per_condition in matrices.items()
            if group in per_condition
        }
        if not to_align:
            continue
        aligned = align_matrix_times(to_align, time_axis="both")
        for condition, matrix in aligned.items():
            matrices[condition][group] = matrix

    return matrices


__all__ = [
    "HorizonPlotContext",
    "load_and_prepare_time_matrices",
    "render_horizon_grid",
    "summarise_bundles",
]

