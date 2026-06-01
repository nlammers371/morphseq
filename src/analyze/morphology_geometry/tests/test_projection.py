"""Tests for morphology_geometry/projection.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyze.morphology_geometry.projection import project_binned_features
from .conftest import FEATURE_NAMES, N_FEATURES, BIN_WIDTH


def _make_df(
    *,
    n_embryos: int = 6,
    n_frames: int = 12,
    n_bins: int = 3,
    feature_names: list[str] | None = None,
    rng: np.random.Generator | None = None,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Build a synthetic per-frame embryo DataFrame."""
    if rng is None:
        rng = np.random.default_rng(0)
    if feature_names is None:
        feature_names = FEATURE_NAMES

    embryo_ids = [f"e{i:02d}" for i in range(n_embryos)]
    classes = ["wt"] * (n_embryos // 2) + ["het"] * (n_embryos - n_embryos // 2)
    rows = []
    for eid, cls in zip(embryo_ids, classes):
        for frame in range(n_frames):
            t = float(frame) * (BIN_WIDTH * n_bins / n_frames) + 22.0  # starts at 22 hpf
            row = {"embryo_id": eid, "hpf": t, "genotype": cls}
            for fname in feature_names:
                row[fname] = rng.standard_normal()
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)
    return pd.DataFrame(rows)


class TestProjectBinnedFeatures:
    def test_output_shape(self, validated):
        df = _make_df()
        axis = validated.vectors[0]
        out = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH,
        )
        assert "phenotype_direction_score" in out.columns
        assert "time_bin_center" in out.columns
        assert "embryo_id" in out.columns
        assert len(out) > 0

    def test_column_order_invariance(self, validated):
        """Shuffling df column order must not change projection scores."""
        rng = np.random.default_rng(5)
        df = _make_df(rng=rng)
        axis = validated.vectors[0]

        out_orig = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH,
        )

        # Shuffle feature columns (keep non-feature cols at the front)
        non_feat = [c for c in df.columns if c not in FEATURE_NAMES]
        feat_shuffled = list(rng.permutation(FEATURE_NAMES))
        df_shuffled = df[non_feat + feat_shuffled]

        out_shuffled = project_binned_features(
            df_shuffled, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH,
        )

        merged = out_orig.merge(
            out_shuffled,
            on=["embryo_id", "time_bin_center"],
            suffixes=("_orig", "_shuf"),
        )
        np.testing.assert_allclose(
            merged["phenotype_direction_score_orig"].to_numpy(),
            merged["phenotype_direction_score_shuf"].to_numpy(),
            atol=1e-10,
        )

    def test_missing_feature_column_raises(self, validated):
        df = _make_df()
        df = df.drop(columns=[FEATURE_NAMES[0]])
        axis = validated.vectors[0]
        with pytest.raises(ValueError, match="missing feature columns"):
            project_binned_features(
                df, vd=validated, axis=axis, id_col="embryo_id",
                time_col="hpf", bin_width=BIN_WIDTH,
            )

    def test_axis_length_mismatch_raises(self, validated):
        df = _make_df()
        bad_axis = np.ones(N_FEATURES + 1)
        with pytest.raises(ValueError, match="Axis length"):
            project_binned_features(
                df, vd=validated, axis=bad_axis, id_col="embryo_id",
                time_col="hpf", bin_width=BIN_WIDTH,
            )

    def test_class_col_preserved(self, validated):
        df = _make_df()
        axis = validated.vectors[0]
        out = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH, class_col="genotype",
        )
        assert "genotype" in out.columns

    def test_extra_group_cols_preserved(self, validated):
        df = _make_df(extra_cols={"batch": "A"})
        axis = validated.vectors[0]
        out = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH, extra_group_cols=["batch"],
        )
        assert "batch" in out.columns

    def test_bin_center_arithmetic(self, validated):
        """time_bin_center == floor(t / bin_width) * bin_width + bin_width / 2."""
        df = _make_df()
        axis = validated.vectors[0]
        out = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH,
        )
        # All centers should be of the form N + bin_width/2 for integer N*(bin_width)
        half = BIN_WIDTH / 2.0
        for center in out["time_bin_center"].unique():
            remainder = (center - half) % BIN_WIDTH
            assert remainder == pytest.approx(0.0, abs=1e-9), (
                f"Unexpected bin center {center}"
            )

    def test_custom_output_col(self, validated):
        df = _make_df()
        axis = validated.vectors[0]
        out = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH, output_col="my_score",
        )
        assert "my_score" in out.columns
        assert "phenotype_direction_score" not in out.columns

    def test_no_internal_time_bin_col(self, validated):
        """_time_bin must be dropped from the output."""
        df = _make_df()
        axis = validated.vectors[0]
        out = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH,
        )
        assert "_time_bin" not in out.columns

    def test_projection_math(self, validated):
        """Manual projection matches output."""
        rng = np.random.default_rng(99)
        n_embryos = 2
        embryo_ids = ["e0", "e1"]
        # One frame per embryo per bin so mean == raw value
        t_centers = [24.0 + BIN_WIDTH / 2, 28.0 + BIN_WIDTH / 2]
        rows = []
        for eid in embryo_ids:
            for tc in t_centers:
                row = {"embryo_id": eid, "hpf": tc}
                for fname in FEATURE_NAMES:
                    row[fname] = rng.standard_normal()
                rows.append(row)
        df = pd.DataFrame(rows)
        axis = validated.vectors[0]

        out = project_binned_features(
            df, vd=validated, axis=axis, id_col="embryo_id",
            time_col="hpf", bin_width=BIN_WIDTH,
        )

        for _, row in df.iterrows():
            eid = row["embryo_id"]
            t = row["hpf"]
            expected_center = float(int(t // BIN_WIDTH) * BIN_WIDTH) + BIN_WIDTH / 2.0
            x = np.array([row[f] for f in FEATURE_NAMES], dtype=float)
            expected_score = float(x @ axis)
            match = out[
                (out["embryo_id"] == eid) &
                (np.abs(out["time_bin_center"] - expected_center) < 1e-9)
            ]
            assert len(match) == 1
            assert match["phenotype_direction_score"].iloc[0] == pytest.approx(
                expected_score, abs=1e-10
            )
