"""Compatibility wrapper for moved classification visualization APIs."""

from __future__ import annotations

import warnings

from analyze.classification.viz.classification import *  # noqa: F401,F403

warnings.warn(
    "analyze.difference_detection.classification_test_viz is deprecated; "
    "use analyze.classification.viz.classification instead.",
    DeprecationWarning,
    stacklevel=2,
)
