"""Compatibility wrapper for moved classification APIs.

The canonical implementation now lives in ``analyze.classification.classification_test``.
"""

from __future__ import annotations

import warnings

from analyze.classification.classification_test import (
    _make_logistic_classifier,
    extract_temporal_confusion_profile,
    run_classification_test,
    run_multiclass_classification_test,
)

warnings.warn(
    "analyze.difference_detection.classification_test_multiclass is deprecated; "
    "use analyze.classification.classification_test instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "_make_logistic_classifier",
    "run_multiclass_classification_test",
    "run_classification_test",
    "extract_temporal_confusion_profile",
]
