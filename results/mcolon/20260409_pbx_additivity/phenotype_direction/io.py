"""I/O helpers for classifier direction artifacts."""

from __future__ import annotations

from pathlib import Path

from analyze.classification.engine.analysis import (
    ClassificationAnalysis,
    ClassifierDirections,
)


def load_classifier_directions(path: str | Path) -> ClassifierDirections:
    """Load the classifier_directions layer from a saved classification run."""
    analysis = ClassificationAnalysis.load(path)
    return analysis.layers["classifier_directions"]
