"""
Embryo Metadata Management Submodule

This submodule handles all embryo-specific metadata operations including:
- EmbryoMetadata class for managing biological annotations
- UnifiedManagers for business logic
- AnnotationBatch for batch operations
- Tutorial notebooks and utilities

Usage:
    from metadata.embryo_metadata import EmbryoMetadata
    from metadata.embryo_metadata.unified_managers import UnifiedManagers
    from metadata.embryo_metadata.annotation_batch import AnnotationBatch
"""

# Main classes
from .embryo_metadata import EmbryoMetadata

# Utilities (when implemented)
# from .unified_managers import UnifiedManagers
# from .annotation_batch import AnnotationBatch

__all__ = [
    'EmbryoMetadata',
    # 'UnifiedManagers',
    # 'AnnotationBatch'
]
