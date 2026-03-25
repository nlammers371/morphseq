# Primary API
from .run_classification import run_classification
from .engine.analysis import ClassificationAnalysis

# Submodules
from . import viz
from . import misclassification

# Misclassification pipeline (unchanged)
from .misclassification import run_misclassification_pipeline, run_stage_geometry

# Legacy (FutureWarning on call, not on import)
from .classification_test import (
    extract_temporal_confusion_profile,
    run_classification_test,
    run_multiclass_classification_test,
)
from .results import ComparisonSpec, MulticlassOVRResults
from .classification_results import ClassificationResults

__all__ = [
    # Primary
    "run_classification",
    "ClassificationAnalysis",
    # Submodules
    "viz",
    "misclassification",
    # Misclassification pipeline
    "run_misclassification_pipeline",
    "run_stage_geometry",
    # Legacy
    "run_classification_test",
    "run_multiclass_classification_test",
    "extract_temporal_confusion_profile",
    "MulticlassOVRResults",
    "ComparisonSpec",
    "ClassificationResults",
]
