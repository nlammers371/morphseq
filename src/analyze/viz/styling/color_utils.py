"""
Generic color utilities for visualization.

These helpers are intentionally plotting-backend agnostic and can be shared
across faceted, time-series, 3D, or other plot implementations.
"""

import colorsys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import matplotlib.colors as mcolors

from .genotype_colors import get_known_genotype_color


STANDARD_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def normalize_color(color: Any) -> str:
    """Convert any supported color representation to hex string."""
    try:
        return mcolors.to_hex(color)
    except (ValueError, TypeError):
        return str(color)


def to_rgba_string(color: Any, alpha: float = 1.0) -> str:
    """Convert any supported color to `rgba(r,g,b,a)` string."""
    try:
        r, g, b, _ = mcolors.to_rgba(color)
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"
    except (ValueError, TypeError):
        return f"rgba(128,128,128,{alpha})"


def create_color_lookup(
    unique_values: Sequence[Any],
    palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Create a value->color mapping from a list of category values."""
    palette = [normalize_color(c) for c in (palette or STANDARD_PALETTE)]
    return {v: palette[i % len(palette)] for i, v in enumerate(unique_values)}


def _generate_distinct_color(index: int) -> str:
    """Generate additional distinct colors when the palette is exhausted."""
    hue = (index * 0.618033988749895) % 1.0  # Golden-ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return normalize_color((r, g, b))


def _next_unused_color(
    used_colors: set,
    palette: List[str],
    palette_idx: List[int],
    generated_idx: List[int],
) -> str:
    """Return the next color not already used in this mapping."""
    n = len(palette)
    attempts = 0
    while attempts < n:
        color = palette[palette_idx[0] % n]
        palette_idx[0] += 1
        attempts += 1
        if color not in used_colors:
            return color

    while True:
        color = _generate_distinct_color(generated_idx[0])
        generated_idx[0] += 1
        if color not in used_colors:
            return color


def resolve_color_lookup(
    unique_values: Sequence[Any],
    color_lookup: Optional[Mapping[Any, Any]] = None,
    palette: Optional[List[str]] = None,
    default_resolver: Optional[Callable[[Any], Optional[Any]]] = get_known_genotype_color,
    enforce_distinct: bool = True,
    warn_on_collision: bool = True,
) -> Dict[Any, str]:
    """
    Resolve per-value colors with optional collision handling.

    Rules:
    - Use explicit `color_lookup` entries first.
    - Then try `default_resolver` if provided.
    - Fill remaining values from `palette`.
    - If `enforce_distinct=True`, ensure every final value gets a distinct color.
      Explicit and default-resolved colors are treated as preferred starting
      assignments, but later collisions are reassigned to unused colors.
    """
    values = list(unique_values)
    if not values:
        return {}

    palette_norm = [normalize_color(c) for c in (palette or STANDARD_PALETTE)]
    assigned: Dict[Any, str] = {}
    used_colors = set()
    palette_idx = [0]
    generated_idx = [0]

    get_color = (
        color_lookup.get
        if color_lookup is not None and hasattr(color_lookup, "get")
        else None
    )

    for val in values:
        provided = get_color(val) if get_color else None
        if provided not in (None, ""):
            color = normalize_color(provided)
        else:
            default_color = default_resolver(val) if default_resolver else None
            if default_color not in (None, ""):
                color = normalize_color(default_color)
            else:
                color = _next_unused_color(used_colors, palette_norm, palette_idx, generated_idx)
        assigned[val] = color
        used_colors.add(color)

    if not enforce_distinct:
        return assigned

    seen: Dict[str, Any] = {}
    reassign_count = 0
    colliders: list = []
    for val in values:
        color = assigned[val]
        if color not in seen:
            seen[color] = val
            continue
        # Color already taken by another value — pick a new distinct color.
        # Do NOT discard the original color; it is still held by the first owner.
        new_color = _next_unused_color(used_colors, palette_norm, palette_idx, generated_idx)
        assigned[val] = new_color
        used_colors.add(new_color)
        seen[new_color] = val
        reassign_count += 1
        colliders.append(val)

    if reassign_count > 0 and warn_on_collision:
        print(
            f"Warning: {reassign_count} group(s) shared a default color and were "
            f"reassigned to keep colors distinct: {colliders}"
        )

    return assigned


def build_genotype_color_lookup(
    genotypes: Sequence[Any],
    color_lookup: Optional[Mapping[Any, Any]] = None,
    palette: Optional[List[str]] = None,
    enforce_distinct: bool = True,
    warn_on_collision: bool = True,
) -> Dict[Any, str]:
    """
    Build a genotype-aware color lookup for a set of labels.

    This applies exact-match genotype defaults first, then breaks any color
    collisions across the final mapping when ``enforce_distinct=True``.
    """
    return resolve_color_lookup(
        genotypes,
        color_lookup=color_lookup,
        palette=palette,
        default_resolver=get_known_genotype_color,
        enforce_distinct=enforce_distinct,
        warn_on_collision=warn_on_collision,
    )


__all__ = [
    "STANDARD_PALETTE",
    "normalize_color",
    "to_rgba_string",
    "create_color_lookup",
    "resolve_color_lookup",
    "build_genotype_color_lookup",
]
