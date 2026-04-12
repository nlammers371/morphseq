import math

import pandas as pd

from analyze.classification.emergence.onset import (
    AMBIGUOUS,
    NOT_SEPARATED,
    SEPARATED,
    OnsetParams,
    build_onset_matrix,
    classify_pair_state,
    compute_pair_onsets,
)
from analyze.classification.emergence.transitivity import build_transitivity_report


class TestOnsetHelpers:
    def test_classify_pair_state(self):
        params = OnsetParams(p_sep=0.05, p_ns=0.10, auroc_sep=0.6)

        assert classify_pair_state(0.01, 0.8, params) == SEPARATED
        assert classify_pair_state(0.20, 0.8, params) == NOT_SEPARATED
        assert classify_pair_state(0.07, 0.8, params) == AMBIGUOUS
        assert classify_pair_state(0.01, 0.4, params) == AMBIGUOUS

    def test_compute_pair_onsets(self):
        params = OnsetParams(subsequent_frac=0.75)
        classified_df = pd.DataFrame(
            {
                "pair_key": ["a__b", "a__b", "a__b", "a__b"],
                "time_bin_center": [10.0, 12.0, 14.0, 16.0],
                "positive_label": ["b", "b", "b", "b"],
                "negative_label": ["a", "a", "a", "a"],
                "edge_state": [NOT_SEPARATED, SEPARATED, SEPARATED, SEPARATED],
            }
        )

        onset_df = compute_pair_onsets(classified_df, params)

        assert len(onset_df) == 1
        row = onset_df.iloc[0]
        assert row["class_i"] == "a"
        assert row["class_j"] == "b"
        assert row["onset_hpf"] == 12.0
        assert row["n_separated_bins"] == 3
        assert row["first_separated_bin"] == 12.0

    def test_build_onset_matrix(self):
        onset_df = pd.DataFrame(
            {
                "class_i": ["a", "a", "b"],
                "class_j": ["b", "c", "c"],
                "onset_hpf": [12.0, 16.0, 20.0],
            }
        )

        mat = build_onset_matrix(onset_df, ["a", "b", "c"])

        assert mat.loc["a", "b"] == 12.0
        assert mat.loc["b", "a"] == 12.0
        assert mat.loc["a", "c"] == 16.0
        assert mat.loc["c", "a"] == 16.0
        assert mat.loc["b", "c"] == 20.0
        assert mat.loc["c", "b"] == 20.0
        assert math.isnan(mat.loc["a", "a"])


class TestTransitivityReport:
    def test_build_transitivity_report_smoke(self):
        scores_df = pd.DataFrame(
            {
                "positive_label": [
                    "b", "b", "c",
                    "b", "b", "c",
                    "b", "b", "c",
                ],
                "negative_label": [
                    "a", "a", "a",
                    "a", "a", "a",
                    "c", "c", "b",
                ],
                "time_bin_center": [10.0, 14.0, 10.0, 14.0, 10.0, 14.0, 10.0, 14.0, 10.0],
                "pval": [0.4, 0.001, 0.5, 0.002, 0.45, 0.003, 0.35, 0.001, 0.3],
                "auroc_obs": [0.55, 0.8, 0.52, 0.81, 0.51, 0.82, 0.54, 0.79, 0.53],
            }
        )

        report = build_transitivity_report(scores_df)

        assert list(report.onset_matrix.index) == ["a", "b", "c"]
        assert report.onset_matrix.loc["a", "b"] == 14.0
        assert report.onset_matrix.loc["a", "c"] == 14.0
        assert report.onset_df.shape[0] == 3
        assert report.classified_df.shape[0] == scores_df.shape[0]
        assert report.timebin_summary.shape[0] == 2
        assert "edge_state" in report.classified_df.columns

    def test_build_transitivity_report_nondefault_score_columns(self):
        scores_df = pd.DataFrame(
            {
                "positive_label": ["b", "b", "c", "c"],
                "negative_label": ["a", "a", "a", "a"],
                "time_bin_center": [10.0, 14.0, 10.0, 14.0],
                "pval": [0.99, 0.99, 0.99, 0.99],
                "auroc_obs": [0.0, 0.0, 0.0, 0.0],
                "p_custom": [0.4, 0.001, 0.4, 0.001],
                "auroc_custom": [0.55, 0.82, 0.56, 0.81],
            }
        )

        report = build_transitivity_report(
            scores_df,
            pval_col="p_custom",
            auroc_col="auroc_custom",
        )

        assert set(report.classified_df["edge_state"]) == {NOT_SEPARATED, SEPARATED}
        assert report.onset_matrix.loc["a", "b"] == 14.0
        assert report.onset_matrix.loc["a", "c"] == 14.0
