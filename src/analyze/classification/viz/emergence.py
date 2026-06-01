"""Emergence HTML renderer for interactive phenotype emergence exploration.

This module provides tools for computing and rendering emergence timelines —
summaries of when phenotypic classes durably separate from one another.

Scope
-----
This module is for:
- Computing emergence onset data from tidy pairwise scores DataFrames
- Rendering pre-computed emergence data to standalone browser HTML
- Plotting static emergence heatmaps (matplotlib)

Not for:
- Experiment-specific data loading or path resolution
- Project-specific color conventions
- General emergence analysis orchestration (see analyze.classification.emergence)

Data flow
---------
1. Call ``compute_emergence_data(scores_df, class_order, ...)`` to produce an
   ``EmergenceData`` from a tidy scores table.
2a. Call ``plot_emergence_heatmap(data, level=...)`` for a static matplotlib figure.
2b. Call ``render_emergence_html(data, ...)`` to write a self-contained HTML file.

Convenience: ``render_emergence_html_from_scores(scores_df, class_order, ...)``
calls both steps in one go.

The rendered HTML embeds D3.js and all data inline — fully portable, no server
needed.

Public API
----------
``EmergenceData``                  — science result: onset matrices + axis order
``compute_emergence_data``         — scores DataFrame → EmergenceData
``plot_emergence_heatmap``         — EmergenceData → matplotlib Figure
``render_emergence_html``          — EmergenceData + render kwargs → HTML string
``render_emergence_html_from_scores`` — convenience: scores → HTML in one call
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.classification.emergence.onset import (
    build_onset_matrix,
    classify_pair_state_over_time,
    compute_pair_onsets,
)
from analyze.classification.emergence.onset import OnsetParams as _OnsetParams

__all__ = [
    "EmergenceData",
    "compute_emergence_data",
    "plot_emergence_heatmap",
    "render_emergence_html",
    "render_emergence_html_from_scores",
]

# ---------------------------------------------------------------------------
# Package asset paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_TEMPLATE_PATH = _HERE / "_emergence_template.html"
_D3_BUNDLE_PATH = _HERE / "_d3_bundle.js"

# ---------------------------------------------------------------------------
# Default AUROC levels
# ---------------------------------------------------------------------------

_DEFAULT_AUROC_LEVELS: dict[str, float] = {
    "none": 0.0,
    "0.60": 0.60,
    "0.65": 0.65,
    "0.70": 0.70,
}

# ---------------------------------------------------------------------------
# Public science result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmergenceData:
    """Science result: onset matrices and axis metadata.

    Produced by ``compute_emergence_data``. Sufficient for static plotting
    (``plot_emergence_heatmap``) and interactive HTML rendering
    (``render_emergence_html``).

    Fields
    ------
    onset_matrices_by_level:
        ``{level_name: {class_a: {class_b: onset_hpf | None}}}``.
        Keys of the outer dict match ``auroc_levels`` exactly.
        Inner dicts are square over ``class_order``.
        ``None`` means the pair never durably separated.
        No NaN values — NaN is always converted to None at build time.
    class_order:
        Canonical ordered list of all class names. Defines row/column order
        in every onset matrix.
    auroc_levels:
        Ordered sequence of AUROC level names (e.g. ``["none", "0.60", ...]``).
        Exactly matches the keys of ``onset_matrices_by_level``.
    color_scale_min:
        Global minimum finite onset value across all levels. Useful for
        consistent color scale across plots.
    color_scale_max:
        Global maximum finite onset value across all levels.

    Invariants
    ----------
    - ``set(auroc_levels) == set(onset_matrices_by_level.keys())``
    - Every matrix is square over ``class_order`` (all pairs present)
    - All values are ``float | None`` — no NaN, no infinity
    - ``color_scale_min <= color_scale_max``
    """

    onset_matrices_by_level: dict[str, dict[str, dict[str, float | None]]]
    class_order: list[str]
    auroc_levels: list[str]
    color_scale_min: float
    color_scale_max: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_emergence_data(data: EmergenceData) -> None:
    """Raise ValueError if EmergenceData violates any invariant."""
    cls_set = set(data.class_order)
    level_set = set(data.auroc_levels)
    matrix_keys = set(data.onset_matrices_by_level.keys())

    missing_in_matrices = level_set - matrix_keys
    extra_in_matrices = matrix_keys - level_set
    if missing_in_matrices:
        raise ValueError(
            f"auroc_levels has entries not in onset_matrices_by_level: {sorted(missing_in_matrices)}"
        )
    if extra_in_matrices:
        raise ValueError(
            f"onset_matrices_by_level has keys not in auroc_levels: {sorted(extra_in_matrices)}"
        )

    for level, mat in data.onset_matrices_by_level.items():
        missing_rows = cls_set - set(mat.keys())
        if missing_rows:
            raise ValueError(
                f"Level {level!r}: onset matrix missing rows: {sorted(missing_rows)}"
            )
        for row_cls, row in mat.items():
            if row_cls not in cls_set:
                continue
            missing_cols = cls_set - set(row.keys())
            if missing_cols:
                raise ValueError(
                    f"Level {level!r}, row {row_cls!r}: missing columns: {sorted(missing_cols)}"
                )
            for col_cls, val in row.items():
                if col_cls not in cls_set:
                    continue
                if val is not None and (math.isnan(val) or math.isinf(val)):
                    raise ValueError(
                        f"Level {level!r}, cell ({row_cls!r}, {col_cls!r}): "
                        f"value {val!r} is not finite. Use None for missing onsets."
                    )

    if data.color_scale_min > data.color_scale_max:
        raise ValueError(
            f"color_scale_min ({data.color_scale_min}) > color_scale_max ({data.color_scale_max})"
        )


# ---------------------------------------------------------------------------
# Builder — scores DataFrame → EmergenceData
# ---------------------------------------------------------------------------


def _nan_to_none(val: object) -> float | None:
    """Convert NaN / non-finite to None; pass through finite floats."""
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(f) or not math.isfinite(f):
        return None
    return f


def _mat_df_to_nested_dict(
    mat: pd.DataFrame,
    class_order: Sequence[str],
) -> dict[str, dict[str, float | None]]:
    """Convert a square onset DataFrame to the nested dict used by EmergenceData."""
    result: dict[str, dict[str, float | None]] = {}
    for a in class_order:
        result[a] = {}
        for b in class_order:
            if a in mat.index and b in mat.columns:
                result[a][b] = _nan_to_none(mat.loc[a, b])
            else:
                result[a][b] = None
    return result


def compute_emergence_data(
    scores_df: pd.DataFrame,
    class_order: Sequence[str],
    *,
    auroc_levels: Optional[Mapping[str, float]] = None,
    p_sep: float = 0.05,
    p_ns: float = 0.10,
    subsequent_frac: float = 0.40,
    # Column name overrides — defaults match run_classification output
    time_col: str = "time_bin_center",
    positive_class_col: str = "positive_label",
    negative_class_col: str = "negative_label",
    auroc_col: str = "auroc_obs",
    pvalue_col: str = "pval",
) -> EmergenceData:
    """Compute pairwise emergence onset data from a tidy scores DataFrame.

    Parameters
    ----------
    scores_df:
        Tidy pairwise classification scores. Required columns (by default):
        ``time_bin_center``, ``positive_label``, ``negative_label``,
        ``auroc_obs``, ``pval``. Column names are overridable via the
        ``*_col`` parameters.
    class_order:
        **Required.** Canonical ordered list of all class names. Defines matrix
        row/column order across all AUROC levels. Never inferred from data.
    auroc_levels:
        Mapping of display name → minimum AUROC threshold for the "separated"
        state. ``None`` uses the default:
        ``{"none": 0.0, "0.60": 0.60, "0.65": 0.65, "0.70": 0.70}``.
    p_sep:
        p-value threshold below which a pair is "separated".
    p_ns:
        p-value threshold above which a pair is "not separated".
    subsequent_frac:
        Fraction of time bins from the onset onward that must remain
        "separated" for the onset to be considered durable.
    time_col:
        Column for developmental time. Default ``"time_bin_center"``.
    positive_class_col:
        Column for the positive class. Default ``"positive_label"``.
    negative_class_col:
        Column for the negative class. Default ``"negative_label"``.
    auroc_col:
        Column for the observed AUROC. Default ``"auroc_obs"``.
    pvalue_col:
        Column for the permutation p-value. Default ``"pval"``.

    Returns
    -------
    EmergenceData
        Validated onset matrices indexed by AUROC level, aligned to
        ``class_order``.
    """
    # --- Input validation ---
    required_cols = {time_col, positive_class_col, negative_class_col, auroc_col, pvalue_col}
    missing = required_cols - set(scores_df.columns)
    if missing:
        raise ValueError(
            f"scores_df is missing required columns: {sorted(missing)}. "
            f"Available: {scores_df.columns.tolist()}"
        )
    if not pd.api.types.is_numeric_dtype(scores_df[time_col]):
        raise ValueError(f"Column {time_col!r} must be numeric.")
    if not scores_df[auroc_col].apply(lambda x: pd.notna(x) and math.isfinite(float(x))).all():
        raise ValueError(f"Column {auroc_col!r} must contain only finite numeric values.")
    pvals = scores_df[pvalue_col].astype(float)
    if (pvals < 0).any() or (pvals > 1).any():
        raise ValueError(f"Column {pvalue_col!r} values must be in [0, 1].")
    dup_mask = scores_df.duplicated(
        subset=[time_col, positive_class_col, negative_class_col]
    )
    if dup_mask.any():
        dupes = scores_df[dup_mask][
            [time_col, positive_class_col, negative_class_col]
        ].head(3)
        raise ValueError(
            f"scores_df has duplicate (time, positive_class, negative_class) triplets:\n{dupes}"
        )

    # --- Resolve defaults ---
    levels: dict[str, float] = (
        dict(auroc_levels) if auroc_levels is not None else dict(_DEFAULT_AUROC_LEVELS)
    )
    all_classes = list(class_order)

    # --- Compute onset matrices for each AUROC level ---
    onset_matrices_by_level: dict[str, dict[str, dict[str, float | None]]] = {}

    for level_name, auroc_sep in levels.items():
        params = _OnsetParams(
            p_sep=p_sep,
            p_ns=p_ns,
            auroc_sep=auroc_sep,
            subsequent_frac=subsequent_frac,
        )
        classified = classify_pair_state_over_time(
            scores_df,
            params,
            time_col=time_col,
            class_i_col=positive_class_col,
            class_j_col=negative_class_col,
            pval_col=pvalue_col,
            auroc_col=auroc_col,
        )
        onset_df = compute_pair_onsets(
            classified,
            params,
            time_col=time_col,
            class_i_col=positive_class_col,
            class_j_col=negative_class_col,
        )
        mat = build_onset_matrix(onset_df, all_classes)
        onset_matrices_by_level[level_name] = _mat_df_to_nested_dict(mat, all_classes)

    # --- Compute global color scale bounds ---
    all_finite: list[float] = [
        v
        for mat in onset_matrices_by_level.values()
        for row in mat.values()
        for v in row.values()
        if v is not None
    ]
    color_scale_min = float(min(all_finite)) if all_finite else 0.0
    color_scale_max = float(max(all_finite)) if all_finite else 130.0

    data = EmergenceData(
        onset_matrices_by_level=onset_matrices_by_level,
        class_order=all_classes,
        auroc_levels=list(levels.keys()),
        color_scale_min=color_scale_min,
        color_scale_max=color_scale_max,
    )
    _validate_emergence_data(data)
    return data


# ---------------------------------------------------------------------------
# Static matplotlib heatmap
# ---------------------------------------------------------------------------


def plot_emergence_heatmap(
    data: EmergenceData,
    level: str,
    *,
    class_labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    cmap: str = "YlOrRd",
    output_path: Optional[Union[str, Path]] = None,
) -> "matplotlib.figure.Figure":
    """Plot a static emergence onset heatmap for a single AUROC level.

    Parameters
    ----------
    data:
        An ``EmergenceData`` from ``compute_emergence_data``.
    level:
        One of ``data.auroc_levels`` — selects the onset matrix to display.
    class_labels:
        Display names for axis tick labels. ``None`` → class name as label.
    title:
        Figure title. ``None`` → ``"Emergence onset heatmap (level={level})"``.
    cmap:
        Matplotlib colormap. A sequential map (e.g. ``"YlOrRd"``) is recommended.
    output_path:
        If given, saves PNG at 150 dpi and closes the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if level not in data.auroc_levels:
        raise ValueError(
            f"level={level!r} not in data.auroc_levels: {data.auroc_levels}"
        )

    mat = data.onset_matrices_by_level[level]
    order = data.class_order
    labels = class_labels or {}

    n = len(order)
    matrix = np.full((n, n), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            v = mat[a][b]
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4, n * 0.9)))
    im = ax.imshow(
        matrix,
        aspect="equal",
        vmin=data.color_scale_min,
        vmax=data.color_scale_max,
        cmap=cmap,
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    tick_labels = [labels.get(c, c) for c in order]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Class B")
    ax.set_ylabel("Class A")
    ax.set_title(title or f"Emergence onset heatmap (level={level})")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.0f}",
                    ha="center", va="center", fontsize=7, color="#222",
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Onset (hpf)")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------


def _auto_colors(class_order: Sequence[str]) -> dict[str, str]:
    """Assign tab10 hex colors cycling over class_order."""
    cmap = plt.cm.tab10
    result = {}
    for i, cls in enumerate(class_order):
        rgba = cmap(i % 10)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        result[cls] = f"#{r:02x}{g:02x}{b:02x}"
    return result


def _build_html_spec(
    data: EmergenceData,
    class_labels: dict[str, str],
    class_colors: dict[str, str],
    bin_width: float,
    min_cross_support: float,
    heatmap_font_scale: float,
) -> str:
    """Serialize EmergenceData + render kwargs to the JSON expected by the JS.

    Ordering contract (deterministic):
    - ``all_classes`` / matrix rows / matrix cols: in ``class_order``
    - ``auroc_levels``: in ``data.auroc_levels`` sequence order
    - ``class_labels`` / ``class_colors``: in ``class_order``
    - ``onset_matrices`` outer keys: in ``auroc_levels`` order

    JSON keys use legacy names (``vmin``/``vmax``, ``all_classes``,
    ``onset_matrices``) for backward compatibility with the D3 JavaScript.
    """
    cls_order = list(data.class_order)
    level_order = list(data.auroc_levels)

    onset_matrices: dict[str, dict[str, dict[str, float | None]]] = {}
    for level in level_order:
        mat = data.onset_matrices_by_level[level]
        onset_matrices[level] = {
            a: {b: mat[a][b] for b in cls_order}
            for a in cls_order
        }

    spec = {
        "onset_matrices":    onset_matrices,
        "auroc_levels":      level_order,
        "all_classes":       cls_order,
        "class_labels":      {c: class_labels[c] for c in cls_order},
        "class_colors":      {c: class_colors[c] for c in cls_order},
        # Heatmap color scale — legacy JS keys
        "vmin":              data.color_scale_min,
        "vmax":              data.color_scale_max,
        # Tree Y-axis range — same values, separate keys for semantic clarity
        "tree_tmin":         data.color_scale_min,
        "tree_tmax":         data.color_scale_max,
        # JS client-side algorithm parameters
        "bin_width":          bin_width,
        "min_cross_support":  min_cross_support,
        "heatmap_font_scale": heatmap_font_scale,
    }

    return json.dumps(spec, allow_nan=False, separators=(",", ":"))


def _load_asset(filename: str) -> str:
    """Load a package asset by filename, with editable-install fallback."""
    try:
        from importlib.resources import files as _res_files
        return (
            _res_files("analyze.classification.viz")
            .joinpath(filename)
            .read_text(encoding="utf-8")
        )
    except Exception:
        return (_HERE / filename).read_text(encoding="utf-8")


def render_emergence_html(
    data: EmergenceData,
    *,
    class_labels: Optional[Mapping[str, str]] = None,
    class_colors: Optional[Mapping[str, str]] = None,
    bin_width: float = 4.0,
    min_cross_support: float = 0.5,
    heatmap_font_scale: float = 1.0,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Render an ``EmergenceData`` to a self-contained interactive HTML file.

    The output embeds D3.js and all data inline. No network calls are made;
    the file can be opened in any modern browser without a server.

    Parameters
    ----------
    data:
        An ``EmergenceData`` from ``compute_emergence_data``.
    class_labels:
        Display name per class (e.g. ``{"inj_ctrl": "Inj. Ctrl"}``).
        ``None`` → class name used as its own label.
    class_colors:
        Hex color per class (e.g. ``{"inj_ctrl": "#2166AC"}``).
        ``None`` → auto-assigned from matplotlib tab10 palette.
    bin_width:
        Time bin width in hpf. Controls the JS block-grouping step that
        groups classes with similar onset times before tree construction.
        Default 4.0.
    min_cross_support:
        Minimum fraction of finite cross-partition onset pairs required for a
        bipartition to be accepted during within-block resolution. Default 0.5.
    heatmap_font_scale:
        Multiplier for the interactive heatmap's row/column/cell text sizing.
        Default 1.0 preserves the existing layout.
    output_path:
        If provided, the HTML is written here (UTF-8, overwriting). The HTML
        string is always returned regardless.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    _validate_emergence_data(data)

    # Resolve defaults for render-only kwargs
    resolved_labels = dict(class_labels) if class_labels is not None else {}
    resolved_colors = dict(class_colors) if class_colors is not None else {}

    auto_colors = _auto_colors(data.class_order)
    for cls in data.class_order:
        resolved_labels.setdefault(cls, cls)
        resolved_colors.setdefault(cls, auto_colors[cls])

    template = _load_asset("_emergence_template.html")
    d3_text = _load_asset("_d3_bundle.js")

    data_json = _build_html_spec(
        data,
        class_labels=resolved_labels,
        class_colors=resolved_colors,
        bin_width=bin_width,
        min_cross_support=min_cross_support,
        heatmap_font_scale=heatmap_font_scale,
    )

    html = template.replace("__D3_PLACEHOLDER__", d3_text)
    html = html.replace("__DATA_PLACEHOLDER__", data_json)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")

    return html


def render_emergence_html_from_scores(
    scores_df: pd.DataFrame,
    class_order: Sequence[str],
    *,
    auroc_levels: Optional[Mapping[str, float]] = None,
    p_sep: float = 0.05,
    p_ns: float = 0.10,
    subsequent_frac: float = 0.40,
    class_labels: Optional[Mapping[str, str]] = None,
    class_colors: Optional[Mapping[str, str]] = None,
    bin_width: float = 4.0,
    min_cross_support: float = 0.5,
    heatmap_font_scale: float = 1.0,
    # Column name overrides
    time_col: str = "time_bin_center",
    positive_class_col: str = "positive_label",
    negative_class_col: str = "negative_label",
    auroc_col: str = "auroc_obs",
    pvalue_col: str = "pval",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Compute emergence data and render to HTML in one step.

    Convenience wrapper around ``compute_emergence_data`` +
    ``render_emergence_html``. Accepts all parameters of both functions.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    data = compute_emergence_data(
        scores_df,
        class_order,
        auroc_levels=auroc_levels,
        p_sep=p_sep,
        p_ns=p_ns,
        subsequent_frac=subsequent_frac,
        time_col=time_col,
        positive_class_col=positive_class_col,
        negative_class_col=negative_class_col,
        auroc_col=auroc_col,
        pvalue_col=pvalue_col,
    )
    return render_emergence_html(
        data,
        class_labels=class_labels,
        class_colors=class_colors,
        bin_width=bin_width,
        min_cross_support=min_cross_support,
        heatmap_font_scale=heatmap_font_scale,
        output_path=output_path,
    )
