"""
Visualization styling utilities and shared color palettes.
"""

from .lookup import ColorLookup, build_suffix_color_lookup
from .color_utils import (
    STANDARD_PALETTE,
    normalize_color,
    to_rgba_string,
    create_color_lookup,
    apply_label_map,
    ordered_present_values,
    build_ordered_color_lookup,
    resolve_color_lookup,
    build_genotype_color_lookup,
)
from .genotype_colors import (
    SPECIAL_GENOTYPE_COLORS,
    extract_genotype_suffix,
    get_known_genotype_color,
    get_color_for_genotype,
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
    'apply_label_map',
    'ordered_present_values',
    'build_ordered_color_lookup',
    'resolve_color_lookup',
    'build_genotype_color_lookup',
    'SPECIAL_GENOTYPE_COLORS',
    'extract_genotype_suffix',
    'get_known_genotype_color',
    'get_color_for_genotype',
    'GENOTYPE_SUFFIX_COLORS',
    'GENOTYPE_SUFFIX_ORDER',
    'GENOTYPE_COLORS',
]
