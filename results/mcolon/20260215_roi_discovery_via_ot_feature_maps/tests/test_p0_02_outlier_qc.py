"""Phase 0 outlier detection tests."""

import numpy as np

from outlier_detection import (
    OutlierDetectionConfig,
    detect_outliers,
    detect_outliers_iqr,
    detect_outliers_mad,
)


def test_iqr_detector_flags_high_cost_tail_outlier():
    values = np.array([1.0, 1.1, 0.9, 1.05, 10.0], dtype=np.float32)
    result = detect_outliers_iqr(values, multiplier=1.5)
    assert result.n_total == 5
    assert result.n_outliers == 1
    assert result.outlier_flag[-1]


def test_dispatch_supports_multiple_methods():
    values = np.array([0.0, 0.1, -0.1, 0.05, 4.0], dtype=np.float32)

    iqr = detect_outliers(values, OutlierDetectionConfig(method="iqr", iqr_multiplier=1.5))
    mad = detect_outliers(values, OutlierDetectionConfig(method="mad", mad_threshold=3.5))

    assert iqr.outlier_flag.shape == values.shape
    assert mad.outlier_flag.shape == values.shape
    assert iqr.n_outliers >= 0
    assert mad.n_outliers >= 0


def test_mad_detector_returns_no_outliers_for_constant_signal():
    values = np.ones(8, dtype=np.float32)
    result = detect_outliers_mad(values, threshold=3.5)
    assert result.n_outliers == 0
    assert not np.any(result.outlier_flag)
