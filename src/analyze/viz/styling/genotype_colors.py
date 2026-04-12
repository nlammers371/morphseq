"""
Shared genotype-aware color helpers.

These helpers operate on already-canonical labels. They do not rename or
normalize semantic labels beyond light key normalization for exact-match lookup.
"""

from __future__ import annotations

from typing import Dict, Optional

from .color_mapping_config import GENOTYPE_SUFFIX_COLORS

SPECIAL_GENOTYPE_COLORS = {
    "pbx1b_pbx4": "#d62728",
    "pbx4": "#d62728",
    "pbx1b": "#7b3294",
    "inj_ctrl": "#7f7f7f",
    "inj-ctrl": "#7f7f7f",
    "non_inj_ctrl": "#1f77b4",
    "wik_ab": "#1f77b4",
    "wik-ab": "#1f77b4",
}

_SUFFIX_PATTERNS = [
    ("crispant", ["crispant", "crisp"]),
    ("homozygous", ["homozygous", "homo"]),
    ("heterozygous", ["heterozygous", "het"]),
    ("wildtype", ["wildtype", "wt", "wild_type", "ab", "wik", "wik-ab", "ab-wik"]),
]


def _normalize_genotype_key(genotype: str) -> str:
    key = str(genotype).strip().lower().replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def extract_genotype_suffix(genotype: str) -> str:
    """Return the canonical genotype suffix for a label, or ``'unknown'``."""
    genotype_lower = str(genotype).lower()
    for canonical, patterns in _SUFFIX_PATTERNS:
        for pattern in patterns:
            if genotype_lower.endswith(pattern):
                return canonical
    return "unknown"


def get_known_genotype_color(
    genotype: str,
    suffix_colors: Optional[Dict[str, str]] = None,
    special_colors: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Return a known genotype color or ``None`` when no genotype rule matches.

    This is intended for default resolver use in generic plotting code. Unknown
    labels return ``None`` so the caller can continue to a palette fallback.
    """
    if suffix_colors is None:
        suffix_colors = GENOTYPE_SUFFIX_COLORS
    if special_colors is None:
        special_colors = SPECIAL_GENOTYPE_COLORS

    genotype_key = _normalize_genotype_key(genotype)
    if genotype_key in special_colors:
        return special_colors[genotype_key]

    suffix = extract_genotype_suffix(genotype)
    if suffix == "unknown":
        return None
    return suffix_colors.get(suffix)


def get_color_for_genotype(
    genotype: str,
    suffix_colors: Optional[Dict[str, str]] = None,
    special_colors: Optional[Dict[str, str]] = None,
    unknown_color: str = "#808080",
) -> str:
    """Return the canonical genotype color for a label."""
    color = get_known_genotype_color(
        genotype,
        suffix_colors=suffix_colors,
        special_colors=special_colors,
    )
    if color is not None:
        return color
    return unknown_color


__all__ = [
    "SPECIAL_GENOTYPE_COLORS",
    "extract_genotype_suffix",
    "get_known_genotype_color",
    "get_color_for_genotype",
]
