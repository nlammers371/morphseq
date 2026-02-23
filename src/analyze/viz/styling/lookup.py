"""
Generic styling helpers for visualization.
"""

from typing import Dict, List, Any, Optional


class ColorLookup:
    """
    A dict-like object that performs suffix-based color matching.
    """

    def __init__(
        self,
        suffix_colors: Dict[str, str],
        suffix_order: List[str],
        fallback_palette: Optional[List[str]] = None,
    ):
        self.suffix_colors = suffix_colors
        self.suffix_order = suffix_order
        if fallback_palette is None:
            fallback_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]
        self.fallback_palette = fallback_palette
        self._cache: Dict[Any, str] = {}
        self._fallback_index = 0

    def __getitem__(self, key: Any) -> str:
        if key in self._cache:
            return self._cache[key]

        key_str = str(key)
        for suffix in self.suffix_order:
            if key_str.endswith('_' + suffix) or key_str == suffix:
                color = self.suffix_colors[suffix]
                self._cache[key] = color
                return color

        color = self.fallback_palette[self._fallback_index % len(self.fallback_palette)]
        self._fallback_index += 1
        self._cache[key] = color
        return color

    def get(self, key: Any, default: Optional[str] = None) -> Optional[str]:
        try:
            return self[key]
        except Exception:
            return default

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
            return True
        except Exception:
            return False

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()


def build_suffix_color_lookup(
    values: List[Any],
    suffix_colors: Dict[str, str],
    suffix_order: List[str],
    fallback_palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Build color lookup by matching value suffixes."""
    if fallback_palette is None:
        fallback_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

    lookup = {}
    fallback_idx = 0

    for val in values:
        val_str = str(val)
        matched = False

        for suffix in suffix_order:
            if val_str.endswith('_' + suffix) or val_str == suffix:
                lookup[val] = suffix_colors[suffix]
                matched = True
                break

        if not matched:
            lookup[val] = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1

    return lookup


__all__ = [
    'ColorLookup',
    'build_suffix_color_lookup',
]
