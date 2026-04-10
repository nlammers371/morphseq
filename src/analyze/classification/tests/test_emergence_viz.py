"""
Tests for analyze.classification.viz.emergence

Tests are grouped into three classes:
- TestEmergenceData: dataclass invariants and validation
- TestComputeEmergenceData: builder input validation and computation
- TestRenderEmergenceHtml: HTML output contract

Synthetic 3-class fixture: A, B, C — simple enough to reason about manually.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze.classification.viz.emergence import (
    EmergenceData,
    _validate_emergence_data,
    compute_emergence_data,
    render_emergence_html,
    render_emergence_html_from_scores,
    plot_emergence_heatmap,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CLASSES = ["A", "B", "C"]


def _minimal_onset_matrices() -> dict[str, dict[str, dict[str, float | None]]]:
    """Return a valid 2-level onset matrix for classes A, B, C."""
    def _mat(vals):
        # vals: upper-triangle floats, diagonal/lower → None
        # order: AA, AB, AC, BA, BB, BC, CA, CB, CC
        a, b, c = CLASSES
        return {
            a: {a: None, b: vals[0], c: vals[1]},
            b: {a: vals[0], b: None, c: vals[2]},
            c: {a: vals[1], b: vals[2], c: None},
        }
    return {
        "none": _mat([22.0, 30.0, 40.0]),
        "0.70": _mat([22.0, None, 40.0]),
    }


def _minimal_data(**kwargs) -> EmergenceData:
    defaults = dict(
        onset_matrices_by_level=_minimal_onset_matrices(),
        class_order=list(CLASSES),
        auroc_levels=["none", "0.70"],
        color_scale_min=22.0,
        color_scale_max=40.0,
    )
    defaults.update(kwargs)
    return EmergenceData(**defaults)


def _make_scores_df() -> pd.DataFrame:
    """Synthetic 3-class scores table with known, simple structure.

    A vs B: separated from t=22 onward
    A vs C: separated from t=30 onward
    B vs C: separated from t=40 onward
    All pairs have p=0.01, auroc=0.80 at all times where separated.
    Non-separated bins have p=0.8, auroc=0.55.
    """
    rows = []
    # 6 time bins, 3 pairs
    times = [18.0, 22.0, 26.0, 30.0, 34.0, 38.0]
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    onset_at = {"A__B": 22.0, "A__C": 30.0, "B__C": 38.0}
    for pos, neg in pairs:
        key = f"{pos}__{neg}"
        for t in times:
            sep = t >= onset_at[key]
            rows.append({
                "time_bin_center": t,
                "positive_label": pos,
                "negative_label": neg,
                "auroc_obs": 0.80 if sep else 0.55,
                "pval": 0.01 if sep else 0.80,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestEmergenceData — dataclass invariants
# ---------------------------------------------------------------------------


class TestEmergenceData:
    def test_valid_data_passes_validation(self):
        data = _minimal_data()
        _validate_emergence_data(data)  # no exception

    def test_rejects_nan_in_matrix(self):
        mats = _minimal_onset_matrices()
        mats["none"]["A"]["B"] = float("nan")
        with pytest.raises(ValueError, match="not finite"):
            _validate_emergence_data(_minimal_data(onset_matrices_by_level=mats))

    def test_rejects_inf_in_matrix(self):
        mats = _minimal_onset_matrices()
        mats["none"]["A"]["B"] = float("inf")
        with pytest.raises(ValueError, match="not finite"):
            _validate_emergence_data(_minimal_data(onset_matrices_by_level=mats))

    def test_rejects_class_missing_from_matrix_row(self):
        mats = _minimal_onset_matrices()
        del mats["none"]["C"]  # remove entire row
        with pytest.raises(ValueError, match="missing rows"):
            _validate_emergence_data(_minimal_data(onset_matrices_by_level=mats))

    def test_rejects_class_missing_from_matrix_col(self):
        mats = _minimal_onset_matrices()
        del mats["none"]["A"]["C"]  # remove one column
        with pytest.raises(ValueError, match="missing columns"):
            _validate_emergence_data(_minimal_data(onset_matrices_by_level=mats))

    def test_rejects_auroc_level_not_in_matrices(self):
        mats = _minimal_onset_matrices()
        del mats["0.70"]  # remove a level from matrices
        with pytest.raises(ValueError, match="not in onset_matrices_by_level"):
            _validate_emergence_data(_minimal_data(onset_matrices_by_level=mats))

    def test_rejects_extra_matrix_key_not_in_auroc_levels(self):
        mats = _minimal_onset_matrices()
        mats["0.99"] = mats["none"]  # extra level not in auroc_levels
        with pytest.raises(ValueError, match="not in auroc_levels"):
            _validate_emergence_data(_minimal_data(onset_matrices_by_level=mats))

    def test_rejects_vmin_gt_vmax(self):
        with pytest.raises(ValueError, match="color_scale_min.*>.*color_scale_max"):
            _validate_emergence_data(_minimal_data(color_scale_min=100.0, color_scale_max=10.0))

    def test_none_is_valid_for_missing_onset(self):
        # None (not NaN) should pass validation
        mats = _minimal_onset_matrices()
        mats["none"]["A"]["B"] = None
        _validate_emergence_data(_minimal_data(onset_matrices_by_level=mats))  # no exception


# ---------------------------------------------------------------------------
# TestComputeEmergenceData — builder
# ---------------------------------------------------------------------------


class TestComputeEmergenceData:
    @pytest.fixture
    def scores(self) -> pd.DataFrame:
        return _make_scores_df()

    def test_produces_valid_data(self, scores):
        data = compute_emergence_data(scores, CLASSES)
        _validate_emergence_data(data)

    def test_class_order_preserved(self, scores):
        order = ["C", "A", "B"]
        data = compute_emergence_data(scores, order)
        assert data.class_order == order

    def test_default_auroc_levels_present(self, scores):
        data = compute_emergence_data(scores, CLASSES)
        assert set(data.auroc_levels) == {"none", "0.60", "0.65", "0.70"}

    def test_custom_auroc_levels(self, scores):
        data = compute_emergence_data(
            scores, CLASSES, auroc_levels={"none": 0.0, "0.75": 0.75}
        )
        assert set(data.auroc_levels) == {"none", "0.75"}
        assert set(data.onset_matrices_by_level.keys()) == {"none", "0.75"}

    def test_matrices_square_over_class_order(self, scores):
        data = compute_emergence_data(scores, CLASSES)
        for level, mat in data.onset_matrices_by_level.items():
            assert set(mat.keys()) == set(CLASSES), f"Level {level}: wrong row keys"
            for row in mat.values():
                assert set(row.keys()) == set(CLASSES)

    def test_no_nan_in_matrices(self, scores):
        data = compute_emergence_data(scores, CLASSES)
        for level, mat in data.onset_matrices_by_level.items():
            for a, row in mat.items():
                for b, val in row.items():
                    assert val is None or (isinstance(val, float) and math.isfinite(val)), \
                        f"Level {level}, ({a},{b}): got {val!r}"

    def test_diagonal_is_none(self, scores):
        data = compute_emergence_data(scores, CLASSES)
        for level, mat in data.onset_matrices_by_level.items():
            for c in CLASSES:
                assert mat[c][c] is None, f"Level {level}: diagonal ({c},{c}) should be None"

    def test_color_scale_min_le_max(self, scores):
        data = compute_emergence_data(scores, CLASSES)
        assert data.color_scale_min <= data.color_scale_max

    def test_column_name_overrides(self, scores):
        renamed = scores.rename(columns={
            "time_bin_center": "t",
            "positive_label": "cls_pos",
            "negative_label": "cls_neg",
            "auroc_obs": "auc",
            "pval": "p",
        })
        data = compute_emergence_data(
            renamed, CLASSES,
            time_col="t",
            positive_class_col="cls_pos",
            negative_class_col="cls_neg",
            auroc_col="auc",
            pvalue_col="p",
        )
        _validate_emergence_data(data)

    def test_rejects_missing_required_column(self, scores):
        bad = scores.drop(columns=["auroc_obs"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_emergence_data(bad, CLASSES)

    def test_rejects_non_numeric_time(self, scores):
        bad = scores.copy()
        bad["time_bin_center"] = bad["time_bin_center"].astype(str)
        with pytest.raises(ValueError, match="must be numeric"):
            compute_emergence_data(bad, CLASSES)

    def test_rejects_invalid_pvalue(self, scores):
        bad = scores.copy()
        bad.loc[0, "pval"] = 1.5
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            compute_emergence_data(bad, CLASSES)

    def test_rejects_nan_auroc(self, scores):
        bad = scores.copy()
        bad.loc[0, "auroc_obs"] = float("nan")
        with pytest.raises(ValueError, match="finite numeric"):
            compute_emergence_data(bad, CLASSES)

    def test_rejects_duplicate_triplets(self, scores):
        bad = pd.concat([scores, scores.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            compute_emergence_data(bad, CLASSES)


# ---------------------------------------------------------------------------
# TestRenderEmergenceHtml — HTML output contract
# ---------------------------------------------------------------------------


_REQUIRED_JSON_KEYS = {
    "onset_matrices", "auroc_levels", "all_classes",
    "class_labels", "class_colors",
    "vmin", "vmax", "tree_tmin", "tree_tmax",
    "bin_width", "min_cross_support",
}


def _extract_data_json(html: str) -> dict:
    """Extract the DATA JSON object from rendered HTML."""
    marker = "const DATA      = "
    start = html.index(marker) + len(marker)
    # Find the matching closing brace
    depth = 0
    i = start
    while i < len(html):
        if html[i] == "{":
            depth += 1
        elif html[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return json.loads(html[start:i + 1])


class TestRenderEmergenceHtml:
    @pytest.fixture
    def data(self) -> EmergenceData:
        return _minimal_data()

    def test_returns_html_string(self, data):
        html = render_emergence_html(data)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_all_required_json_keys_present(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        missing = _REQUIRED_JSON_KEYS - set(parsed.keys())
        assert not missing, f"Missing JSON keys: {missing}"

    def test_class_order_preserved_in_all_classes(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        assert parsed["all_classes"] == data.class_order

    def test_auroc_levels_order_preserved(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        assert parsed["auroc_levels"] == list(data.auroc_levels)

    def test_onset_matrices_keys_match_auroc_levels(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        assert set(parsed["onset_matrices"].keys()) == set(data.auroc_levels)

    def test_onset_matrix_rows_in_class_order(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        for level, mat in parsed["onset_matrices"].items():
            assert list(mat.keys()) == data.class_order, \
                f"Level {level}: row order mismatch"
            for row in mat.values():
                assert list(row.keys()) == data.class_order

    def test_default_labels_are_identity(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        for c in data.class_order:
            assert parsed["class_labels"][c] == c

    def test_custom_labels_applied(self, data):
        html = render_emergence_html(data, class_labels={"A": "Alpha", "B": "Beta", "C": "Gamma"})
        parsed = _extract_data_json(html)
        assert parsed["class_labels"]["A"] == "Alpha"
        assert parsed["class_labels"]["B"] == "Beta"

    def test_custom_colors_applied(self, data):
        colors = {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}
        html = render_emergence_html(data, class_colors=colors)
        parsed = _extract_data_json(html)
        assert parsed["class_colors"]["A"] == "#ff0000"

    def test_vmin_vmax_match_color_scale(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        assert parsed["vmin"] == data.color_scale_min
        assert parsed["vmax"] == data.color_scale_max

    def test_tree_tmin_tmax_match_color_scale(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        assert parsed["tree_tmin"] == data.color_scale_min
        assert parsed["tree_tmax"] == data.color_scale_max

    def test_bin_width_and_min_cross_support_passed_through(self, data):
        html = render_emergence_html(data, bin_width=2.0, min_cross_support=0.3)
        parsed = _extract_data_json(html)
        assert parsed["bin_width"] == 2.0
        assert parsed["min_cross_support"] == 0.3

    def test_deterministic_output(self, data):
        html1 = render_emergence_html(data)
        html2 = render_emergence_html(data)
        assert html1 == html2

    def test_writes_file(self, data, tmp_path):
        out = tmp_path / "test_emergence.html"
        html = render_emergence_html(data, output_path=out)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == html

    def test_creates_parent_dirs(self, data, tmp_path):
        out = tmp_path / "nested" / "dir" / "out.html"
        render_emergence_html(data, output_path=out)
        assert out.exists()

    def test_no_nan_in_data_json(self, data):
        # The injected DATA JSON must not contain NaN/Infinity literals.
        # (The D3 bundle and template JS may legitimately contain the word "NaN"
        # in source code; we only check the serialized data payload.)
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        # Round-trip through json to verify no NaN escaped as a literal
        round_tripped = json.dumps(parsed, allow_nan=False)  # would raise if NaN present
        assert round_tripped  # non-empty

    def test_class_names_embedded_in_json(self, data):
        html = render_emergence_html(data)
        parsed = _extract_data_json(html)
        for c in data.class_order:
            assert c in parsed["all_classes"]

    def test_from_scores_convenience(self):
        scores = _make_scores_df()
        html = render_emergence_html_from_scores(scores, CLASSES)
        assert html.strip().startswith("<!DOCTYPE html>")
        parsed = _extract_data_json(html)
        assert set(parsed.keys()) >= _REQUIRED_JSON_KEYS


# ---------------------------------------------------------------------------
# TestPlotEmergenceHeatmap — static matplotlib figure
# ---------------------------------------------------------------------------


class TestPlotEmergenceHeatmap:
    @pytest.fixture
    def data(self) -> EmergenceData:
        return _minimal_data()

    def test_returns_figure(self, data):
        import matplotlib.figure
        fig = plot_emergence_heatmap(data, level="none")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_rejects_invalid_level(self, data):
        with pytest.raises(ValueError, match="not in data.auroc_levels"):
            plot_emergence_heatmap(data, level="0.99")

    def test_writes_file(self, data, tmp_path):
        import matplotlib.pyplot as plt
        out = tmp_path / "heatmap.png"
        fig = plot_emergence_heatmap(data, level="none", output_path=out)
        assert out.exists()
