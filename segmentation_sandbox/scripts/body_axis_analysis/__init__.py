"""
Body Axis Analysis Package

Consolidated suite for zebrafish embryo centerline extraction and curvature analysis.

Quick Start:
    >>> from body_axis_analysis import extract_centerline
    >>> spline_x, spline_y, curvature, arc_length = extract_centerline(mask)

Methods Available:
    - Geodesic (primary): Robust for curved embryos
    - PCA (fallback): Fast for normal shapes
    - Auto: Automatic selection based on morphology

Documentation:
    See METHODS_DECISIONS.md for detailed information about method choices,
    parameters, and reasoning behind decisions made during optimization.
"""

from .centerline_extraction import (
    extract_centerline,
    compare_methods,
)
from .geodesic_method import GeodesicCenterlineAnalyzer
from .pca_method import PCACenterlineAnalyzer
from .mask_preprocessing import apply_preprocessing, apply_gaussian_preprocessing
from .spline_utils import (
    identify_head_by_taper,
    orient_spline_head_to_tail,
    align_spline_orientation,
)

__all__ = [
    # Main API
    'extract_centerline',
    'compare_methods',

    # Individual analyzers (for advanced use)
    'GeodesicCenterlineAnalyzer',
    'PCACenterlineAnalyzer',

    # Preprocessing and utilities
    'apply_preprocessing',
    'apply_gaussian_preprocessing',
    'identify_head_by_taper',
    'orient_spline_head_to_tail',
    'align_spline_orientation',
]

__version__ = '1.0.0'
