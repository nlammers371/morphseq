"""
Genotype Styling Utilities

Provides suffix-based color and style mapping for genotype visualization.
Colors are determined by genotype SUFFIX (wildtype, heterozygous, homozygous)
independent of gene PREFIX (cep290, b9d2, tmem67, etc.).

This enables consistent styling across all genes without hardcoding.
"""

from typing import Dict, List, Optional

from analyze.trajectory_analysis.config import MEMBERSHIP_COLORS
from analyze.viz.styling import GENOTYPE_SUFFIX_COLORS, GENOTYPE_SUFFIX_ORDER


def extract_genotype_suffix(genotype: str) -> str:
    """Extract suffix category from full genotype name.

    Looks for known suffixes at the end of the genotype string (case-insensitive).
    Returns the canonical suffix name.

    Args:
        genotype: Full genotype name (e.g., 'cep290_homozygous', 'b9d2_het')

    Returns:
        Canonical suffix ('wildtype', 'heterozygous', 'homozygous', or 'unknown')

    Examples:
        >>> extract_genotype_suffix('cep290_homozygous')
        'homozygous'
        >>> extract_genotype_suffix('b9d2_het')
        'heterozygous'
        >>> extract_genotype_suffix('tmem67_wildtype')
        'wildtype'
        >>> extract_genotype_suffix('unknown_gene')
        'unknown'
    """
    # Known suffixes to match (order matters for longest match)
    suffix_patterns = [
        ('crispant', ['crispant', 'crisp']),
        ('homozygous', ['homozygous', 'homo']),
        ('heterozygous', ['heterozygous', 'het']),
        ('wildtype', ['wildtype', 'wt', 'wild_type', 'ab', 'wik', 'wik-ab', 'ab-wik']),
    ]

    genotype_lower = genotype.lower()

    for canonical, patterns in suffix_patterns:
        for pattern in patterns:
            if genotype_lower.endswith(pattern):
                return canonical

    return 'unknown'


def extract_genotype_prefix(genotype: str) -> str:
    """Extract gene prefix from full genotype name.

    Uses internal list of valid suffixes to robustly extract gene name.
    Format expected: {gene}_{suffix} where suffix is one of the known types.

    Args:
        genotype: Full genotype name (e.g., 'cep290_homozygous', 'tmem67_unknown')

    Returns:
        Gene prefix (e.g., 'cep290', 'tmem67')

    Examples:
        >>> extract_genotype_prefix('cep290_homozygous')
        'cep290'
        >>> extract_genotype_prefix('b9d2_het')
        'b9d2'
        >>> extract_genotype_prefix('tmem67_unknown')
        'tmem67'
    """
    # Internal list of all valid suffix patterns (exhaustive)
    ALL_SUFFIX_PATTERNS = [
        'homozygous', 'homo',
        'heterozygous', 'het',
        'wildtype', 'wt', 'wild_type',
        'unknown',  # Include unknown as a valid suffix
    ]

    genotype_lower = genotype.lower()

    # Try to match each suffix pattern from longest to shortest
    # (longest first to handle 'wild_type' before 'wt')
    sorted_patterns = sorted(ALL_SUFFIX_PATTERNS, key=len, reverse=True)

    for pattern in sorted_patterns:
        if genotype_lower.endswith(f'_{pattern}'):
            # Remove _{suffix} from end
            prefix = genotype[:-(len(pattern) + 1)]
            return prefix
        elif genotype_lower.endswith(pattern) and len(genotype) > len(pattern):
            # Handle case without separator (edge case)
            prefix = genotype[:-(len(pattern))]
            return prefix.rstrip('_- ')

    # No known suffix found - return whole string as prefix
    return genotype


def get_color_for_genotype(
    genotype: str,
    suffix_colors: Optional[Dict[str, str]] = None
) -> str:
    """Get color for a genotype based on its suffix.

    Args:
        genotype: Full genotype name
        suffix_colors: Optional custom color mapping (uses defaults if None)

    Returns:
        Hex color string

    Examples:
        >>> get_color_for_genotype('cep290_homozygous')
        '#D32F2F'
        >>> get_color_for_genotype('b9d2_wildtype')
        '#2E7D32'
    """
    if suffix_colors is None:
        suffix_colors = GENOTYPE_SUFFIX_COLORS

    suffix = extract_genotype_suffix(genotype)
    return suffix_colors.get(suffix, '#808080')  # Gray fallback for unknown


def get_membership_category_colors(categories: List[str]) -> Dict[str, str]:
    """Get standardized colors for membership categories.

    Args:
        categories: List of membership category names.

    Returns:
        Mapping of category to hex color.
    """
    return {cat: MEMBERSHIP_COLORS.get(cat, '#95a5a6') for cat in categories}


def sort_genotypes_by_suffix(
    genotypes: List[str],
    suffix_order: Optional[List[str]] = None
) -> List[str]:
    """Sort genotypes by their suffix order.

    Args:
        genotypes: List of full genotype names
        suffix_order: Optional custom order (uses defaults if None)

    Returns:
        Sorted list of genotypes

    Example:
        >>> sort_genotypes_by_suffix(['b9d2_homo', 'b9d2_wt', 'b9d2_het'])
        ['b9d2_wt', 'b9d2_het', 'b9d2_homo']
    """
    if suffix_order is None:
        suffix_order = GENOTYPE_SUFFIX_ORDER

    def sort_key(g):
        suffix = extract_genotype_suffix(g)
        try:
            return suffix_order.index(suffix)
        except ValueError:
            return 999  # Unknown suffixes go last

    return sorted(genotypes, key=sort_key)


def build_genotype_style_config(
    genotypes: List[str],
    suffix_colors: Optional[Dict[str, str]] = None,
    suffix_order: Optional[List[str]] = None
) -> Dict:
    """Build complete style configuration for a set of genotypes.

    Args:
        genotypes: List of full genotype names to style
        suffix_colors: Optional custom colors
        suffix_order: Optional custom order

    Returns:
        Dict with 'order' (sorted genotypes), 'colors' (genotype->color mapping),
        and other config details

    Example:
        >>> config = build_genotype_style_config(
        ...     ['b9d2_wildtype', 'b9d2_homozygous', 'b9d2_heterozygous']
        ... )
        >>> config['order']
        ['b9d2_wildtype', 'b9d2_heterozygous', 'b9d2_homozygous']
        >>> config['colors']['b9d2_homozygous']
        '#D32F2F'
    """
    ordered = sort_genotypes_by_suffix(genotypes, suffix_order)
    colors = {g: get_color_for_genotype(g, suffix_colors) for g in genotypes}

    return {
        'order': ordered,
        'colors': colors,
        'suffix_colors': suffix_colors or GENOTYPE_SUFFIX_COLORS,
        'suffix_order': suffix_order or GENOTYPE_SUFFIX_ORDER,
    }


def format_genotype_label(genotype: str, include_prefix: bool = False) -> str:
    """Format genotype for display labels.

    Args:
        genotype: Full genotype name
        include_prefix: If True, include gene prefix in label

    Returns:
        Formatted label string

    Examples:
        >>> format_genotype_label('cep290_homozygous')
        'Homozygous'
        >>> format_genotype_label('cep290_homozygous', include_prefix=True)
        'Cep290 Homozygous'
    """
    suffix = extract_genotype_suffix(genotype)

    if include_prefix:
        prefix = extract_genotype_prefix(genotype)
        return f"{prefix.title()} {suffix.title()}"

    return suffix.title()
