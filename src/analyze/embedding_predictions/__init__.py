"""
Embedding-based prediction utilities.

This package hosts proof-of-concept regression models that predict continuous
metrics from embedding spaces (e.g., curvature from VAE latents).
"""

from .curvature_regression import (
    CurvatureRegressionOutputs,
    prepare_curvature_dataframe,
    predict_curvature_from_embeddings,
    save_regression_outputs,
)

__all__ = [
    'CurvatureRegressionOutputs',
    'prepare_curvature_dataframe',
    'predict_curvature_from_embeddings',
    'save_regression_outputs',
]
