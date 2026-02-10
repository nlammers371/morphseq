"""
Metadata Management Module - Module 1

Handles all metadata management including:
- Experiment metadata with validation and schema support  
- Embryo metadata submodule for biological annotations
"""

from .experiment_metadata import ExperimentMetadata
from .schema_manager import SchemaManager
# Embryo metadata available as submodule: from metadata.embryo_metadata import EmbryoMetadata

__all__ = ['ExperimentMetadata', 'SchemaManager']
