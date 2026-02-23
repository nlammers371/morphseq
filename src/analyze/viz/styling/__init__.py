"""
Visualization styling utilities and shared color palettes.
"""

from .lookup import ColorLookup, build_suffix_color_lookup
from .color_utils import (
    STANDARD_PALETTE,
    normalize_color,
    to_rgba_string,
    create_color_lookup,
    resolve_color_lookup,
)
from .color_mapping_config import (
    GENOTYPE_SUFFIX_COLORS,
    GENOTYPE_SUFFIX_ORDER,
    GENOTYPE_COLORS,
)

__all__ = [
    'ColorLookup',
    'build_suffix_color_lookup',
    'STANDARD_PALETTE',
    'normalize_color',
    'to_rgba_string',
    'create_color_lookup',
    'resolve_color_lookup',
    'GENOTYPE_SUFFIX_COLORS',
    'GENOTYPE_SUFFIX_ORDER',
    'GENOTYPE_COLORS',
]
