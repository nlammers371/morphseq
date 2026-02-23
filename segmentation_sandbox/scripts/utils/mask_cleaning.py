"""
Mask Cleaning Utilities for Embryo Segmentation

Provides functions to clean embryo masks before downstream analysis:
- Remove small debris
- Connect disconnected components
- Fill holes
- Remove spindly protrusions
- Ensure single connected component

Usage:
    from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask

    # Clean mask before analysis
    mask_cleaned, cleaning_stats = clean_embryo_mask(mask_original, verbose=True)

    # Use cleaned mask
    analyzer = GeodesicBSplineAnalyzer(mask_cleaned)
    results = analyzer.analyze()
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure


def clean_embryo_mask(mask: np.ndarray, verbose: bool = False):
    """
    Clean embryo mask to remove artifacts before centerline extraction.

    Cleaning pipeline:
    1. Remove small debris (<10% of total area)
    2. Iterative adaptive closing to connect components
    3. Fill holes
    4. Conditional adaptive morphological opening (only if solidity < 0.6)
       - Skipped for already-solid masks (solidity >= 0.6)
       - Removes spindly protrusions while preserving thin tails
    5. Keep largest component (safety check)

    Args:
        mask: Binary mask (H, W) as numpy array
        verbose: If True, print cleaning statistics

    Returns:
        cleaned_mask: Cleaned binary mask
        cleaning_stats: Dictionary with cleaning metrics:
            - original_area: Area before cleaning
            - n_components_initial: Initial number of components
            - debris_removed: Number of small components removed
            - n_components_after_debris: Components after debris removal
            - closing_iterations: Number of closing iterations performed
            - closing_radius: Final closing radius used
            - n_components_after_closing: Components after closing
            - holes_filled: Number of pixels filled in holes
            - adaptive_radius: Opening radius used (0 if skipped)
            - opening_skipped: True if opening was skipped due to high solidity
            - solidity_before: Solidity before opening
            - solidity_after: Solidity after opening
            - area_after: Final area after cleaning
            - area_removed: Total pixels removed
            - area_removed_pct: Percentage of area removed

    Example:
        >>> mask = decode_mask_rle(rle_data)
        >>> cleaned, stats = clean_embryo_mask(mask, verbose=True)
        >>> print(f"Removed {stats['debris_removed']} debris components")
        >>> print(f"Opening skipped: {stats['opening_skipped']}")
    """
    original_area = mask.sum()
    total_area = original_area

    # Count initial components
    initial_labeled = measure.label(mask)
    n_components_initial = initial_labeled.max()

    # Step 1: Remove small debris (<10% of total area)
    labeled = measure.label(mask)
    debris_removed = 0
    if labeled.max() > 1:
        component_areas = [(i, np.sum(labeled == i)) for i in range(1, labeled.max() + 1)]
        threshold = 0.10 * total_area

        # Keep components >= 10% of total area
        keep_labels = [label for label, area in component_areas if area >= threshold]

        if len(keep_labels) == 0:
            # Safety: if all components are small, keep largest
            keep_labels = [max(component_areas, key=lambda x: x[1])[0]]

        debris_removed = labeled.max() - len(keep_labels)
        mask_filtered = np.zeros_like(mask, dtype=bool)
        for label in keep_labels:
            mask_filtered |= (labeled == label)
        mask = mask_filtered

    n_components_after_debris = measure.label(mask).max()

    # Step 2: Iterative adaptive CLOSING to connect components
    closing_iterations = 0
    closing_radius = 0
    n_components_after_closing = n_components_after_debris

    if n_components_after_debris > 1:
        # Calculate initial adaptive radius
        props_temp = measure.regionprops(measure.label(mask))[0]
        perimeter_temp = props_temp.perimeter
        closing_radius = max(5, int(perimeter_temp / 100))

        closed = mask.copy()
        max_iterations = 5
        max_radius = 50

        for iteration in range(max_iterations):
            selem_close = morphology.disk(closing_radius)
            closed = morphology.binary_closing(closed, selem_close)

            # Check if now 1 component
            closed_labeled = measure.label(closed)
            n_components_after_closing = closed_labeled.max()
            closing_iterations = iteration + 1

            if n_components_after_closing == 1:
                break

            # Increase radius for next iteration
            closing_radius = min(closing_radius + 5, max_radius)

            if closing_radius >= max_radius:
                break

        mask = closed

    # Step 3: Fill holes
    filled = ndimage.binary_fill_holes(mask)
    holes_filled = filled.sum() - mask.sum()

    # Step 4: Conditional adaptive morphological opening
    # Only apply opening if solidity < 0.6 (based on morphology analysis)
    # Masks with solidity >= 0.6 are already solid and don't need smoothing
    props = measure.regionprops(measure.label(filled))[0]
    perimeter = props.perimeter
    solidity_before = props.solidity

    opening_skipped = False
    if solidity_before < 0.6:
        # Use perimeter/150 for gentler smoothing
        # This preserves thin tails while still removing spindly protrusions
        adaptive_radius = max(3, int(perimeter / 150))
        selem_open = morphology.disk(adaptive_radius)
        cleaned = morphology.binary_opening(filled, selem_open)
    else:
        # Skip opening for solid masks (solidity >= 0.6)
        adaptive_radius = 0
        cleaned = filled
        opening_skipped = True

    # Step 5: Final safety check - keep largest component
    final_labeled = measure.label(cleaned)
    if final_labeled.max() > 1:
        component_sizes = [(i, np.sum(final_labeled == i)) for i in range(1, final_labeled.max() + 1)]
        largest_label = max(component_sizes, key=lambda x: x[1])[0]
        cleaned = (final_labeled == largest_label)

    # Get final stats
    final_props = measure.regionprops(measure.label(cleaned))[0]
    solidity_after = final_props.solidity
    area_after = cleaned.sum()

    cleaning_stats = {
        'original_area': original_area,
        'n_components_initial': n_components_initial,
        'debris_removed': debris_removed,
        'n_components_after_debris': n_components_after_debris,
        'closing_iterations': closing_iterations,
        'closing_radius': closing_radius,
        'n_components_after_closing': n_components_after_closing,
        'holes_filled': holes_filled,
        'adaptive_radius': adaptive_radius,
        'opening_skipped': opening_skipped,
        'solidity_before': solidity_before,
        'solidity_after': solidity_after,
        'area_after': area_after,
        'area_removed': original_area - area_after,
        'area_removed_pct': (original_area - area_after) / original_area * 100
    }

    if verbose:
        print(f"  Mask Cleaning:")
        print(f"    Initial components: {n_components_initial}")
        print(f"    Debris removed (<10%): {debris_removed} → {n_components_after_debris} components")
        if closing_iterations > 0:
            print(f"    Closing: {closing_iterations} iterations, radius: {closing_radius} px → {n_components_after_closing} component(s)")
        print(f"    Holes filled: {holes_filled} px")
        if opening_skipped:
            print(f"    Opening: SKIPPED (solidity {solidity_before:.3f} >= 0.6 threshold)")
        else:
            print(f"    Opening radius: {adaptive_radius} px")
        print(f"    Solidity: {solidity_before:.3f} → {solidity_after:.3f}")
        print(f"    Area: {original_area:,} → {area_after:,} px ({cleaning_stats['area_removed_pct']:.1f}% removed)")

    return cleaned, cleaning_stats
