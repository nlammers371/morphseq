"""
Schema definition for analysis-ready table.

This module defines required columns for the final analysis-ready table,
which combines features, QC flags, plate/scope metadata, and embeddings.
"""

from .features import REQUIRED_COLUMNS_FEATURES
from .quality_control import REQUIRED_COLUMNS_QC
from .plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
from .scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA

REQUIRED_COLUMNS_ANALYSIS_READY = [
    # Core IDs (must be preserved from features)
    'snip_id',
    'embryo_id',
    'experiment_id',
    'time_int',
    'well_id',
    'well_index',

    # Embedding status
    'embedding_calculated',  # Boolean: True if embeddings present

    # Note: Embedding columns (z0...z{dim-1}) are optional and checked separately
] + REQUIRED_COLUMNS_FEATURES + REQUIRED_COLUMNS_QC + REQUIRED_COLUMNS_PLATE_METADATA + REQUIRED_COLUMNS_SCOPE_METADATA
