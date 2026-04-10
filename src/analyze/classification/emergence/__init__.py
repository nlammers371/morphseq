"""Reference-rooted phenotype emergence timeline."""

from .algorithm import build_emergence_timeline
from .emergence import compute_emergence_scores, form_emergence_blocks
from .partition import resolve_block
from .reference import validate_reference
from .types import (
    EmergenceBlock,
    EmergenceScore,
    EmergenceTimeline,
    ReferenceValidation,
    ResolutionNode,
)

__all__ = [
    "ReferenceValidation",
    "EmergenceScore",
    "EmergenceBlock",
    "ResolutionNode",
    "EmergenceTimeline",
    "validate_reference",
    "compute_emergence_scores",
    "form_emergence_blocks",
    "resolve_block",
    "build_emergence_timeline",
]
