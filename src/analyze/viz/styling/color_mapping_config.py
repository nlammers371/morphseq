"""
Shared color mappings and ordering for visualization.
"""

from .lookup import ColorLookup


GENOTYPE_SUFFIX_COLORS = {
    'wildtype': '#2E7D32',      # Green
    'heterozygous': '#FFA500',  # Orange
    'homozygous': '#D32F2F',    # Red
    'crispant': '#9467bd',      # Purple
    'unknown': '#808080',       # Gray
}

GENOTYPE_SUFFIX_ORDER = ['crispant', 'homozygous', 'heterozygous', 'wildtype', 'unknown']

GENOTYPE_COLORS = ColorLookup(
    suffix_colors=GENOTYPE_SUFFIX_COLORS,
    suffix_order=GENOTYPE_SUFFIX_ORDER,
)

B9D2_PHENOTYPE_COLORS = {
    'CE': '#5B7C99',            # Slate blue
    'HTA': '#7FA87F',           # Sage green
    'BA_rescue': '#C4956A',     # Terracotta/tan
    'non_penetrant': '#9E9E9E', # Warm gray
}

B9D2_PHENOTYPE_ORDER = ['CE', 'HTA', 'BA_rescue', 'non_penetrant']


__all__ = [
    'GENOTYPE_SUFFIX_COLORS',
    'GENOTYPE_SUFFIX_ORDER',
    'GENOTYPE_COLORS',
    'B9D2_PHENOTYPE_COLORS',
    'B9D2_PHENOTYPE_ORDER',
]
