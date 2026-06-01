"""Analysis-ready table assembly module."""

from .core.assemble import assemble_analysis_ready

# Backwards-compatible aliases for older call sites.
assemble_features_qc_embeddings = assemble_analysis_ready

__all__ = [
    "assemble_analysis_ready",
    "assemble_features_qc_embeddings",
]
