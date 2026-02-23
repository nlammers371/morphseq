#!/usr/bin/env python3
"""
Extract embryos using the full rotation/cropping pipeline.

This script takes the SAM2 metadata and processes all embryos through:
1. Rotation correction (align to principal axis)
2. Cropping to standard size
3. Both old (6.5 μm/px) and new (7.8 μm/px) resolutions

Outputs comparison figures showing the extraction quality.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.transform import rescale, resize
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage.filters
import skimage.exposure
from scipy.stats import truncnorm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.functions.image_utils import get_embryo_angle, crop_embryo_image
from src.build.build03A_process_images import resolve_sandbox_embryo_mask_from_csv, rotate_image
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from skimage.measure import regionprops


def crop_embryo_image_with_metrics(im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape):
    """Wrapper around build03's crop_embryo_image that returns metrics."""
    im_cropped, emb_mask_cropped, yolk_mask_cropped, out_of_frame = crop_embryo_image(
        im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape=outshape, return_metrics=True
    )

    # Calculate area retained for reference
    mask_area_before = np.sum(emb_mask_rotated > 0.5)
    mask_area_after = np.sum(emb_mask_cropped > 0.5)
    area_retained = mask_area_after / mask_area_before if mask_area_before > 0 else 0.0

    metrics = {
        "out_of_frame": out_of_frame,
        "area_retained": area_retained,
    }

    return im_cropped, emb_mask_cropped, yolk_mask_cropped, metrics


def process_embryo(im_ff, im_mask, im_yolk, outscale, outshape=[576, 256], dl_rad_um=50, px_dim_raw=None):
    """Process embryo through full pipeline at given resolution (matching build03).

    Args:
        px_dim_raw: Micrometers per pixel in original image. If None, defaults to 0.65.
                   Should be calculated from metadata: row["Height (um)"] / row["Height (px)"]
    """
    if px_dim_raw is None:
        px_dim_raw = 0.65  # Fallback, but should use metadata value
    scale_factor = px_dim_raw / outscale

    im_ff_rs = rescale(im_ff, (scale_factor, scale_factor), order=1, preserve_range=True)
    mask_emb_rs = resize(im_mask.astype(float), im_ff_rs.shape, order=1)
    mask_yolk_rs = resize(im_yolk.astype(float), im_ff_rs.shape, order=1)

    angle_to_use = get_embryo_angle(
        (mask_emb_rs > 0.5).astype(np.uint8),
        (mask_yolk_rs > 0.5).astype(np.uint8)
    )

    im_ff_rotated = rotate_image(im_ff_rs, np.rad2deg(angle_to_use))
    emb_mask_rotated = rotate_image(mask_emb_rs, np.rad2deg(angle_to_use))
    im_yolk_rotated = rotate_image(mask_yolk_rs, np.rad2deg(angle_to_use))

    im_cropped, emb_mask_cropped, yolk_mask_cropped, metrics = crop_embryo_image_with_metrics(
        im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape=outshape
    )

    # Calculate fill fraction
    emb_mask_cropped2 = scipy.ndimage.binary_fill_holes(emb_mask_cropped > 0.5).astype(np.uint8)
    embryo_props = regionprops((emb_mask_cropped2 > 0).astype(int))
    if embryo_props:
        bbox = embryo_props[0].bbox
        embryo_length_px = bbox[2] - bbox[0]
        embryo_width_px = bbox[3] - bbox[1]
        embryo_length_mm = (embryo_length_px * outscale) / 1000
        embryo_width_mm = (embryo_width_px * outscale) / 1000
        fill_fraction_h = embryo_length_px / outshape[0]
        fill_fraction_w = embryo_width_px / outshape[1]
    else:
        embryo_length_mm = 0
        embryo_width_mm = 0
        fill_fraction_h = 0
        fill_fraction_w = 0

    metrics.update({
        "outscale_um_per_px": outscale,
        "rotation_angle_deg": np.rad2deg(angle_to_use),
        "embryo_length_mm": embryo_length_mm,
        "embryo_width_mm": embryo_width_mm,
        "fill_fraction_height": fill_fraction_h,
        "fill_fraction_width": fill_fraction_w,
    })

    # --- POST-PROCESSING: Match build03 exactly (lines 407-409) ---
    # 1. Adaptive histogram equalization (CLAHE)
    # equalize_adapthist expects float in [0, 1] range, returns [0, 1]
    im_cropped_eq = skimage.exposure.equalize_adapthist(im_cropped.astype(float) / 255.0) * 255

    # 2. Gaussian smooth the mask: sigma = dl_rad_um / outscale
    sigma_px = dl_rad_um / outscale
    mask_cropped_gauss = skimage.filters.gaussian(emb_mask_cropped2.astype(float), sigma=sigma_px)

    # 3. Generate synthetic noise to fill background
    px_mean = np.mean(im_cropped[im_cropped > 0]) if np.sum(im_cropped > 0) > 0 else 100
    px_std = np.std(im_cropped[im_cropped > 0]) if np.sum(im_cropped > 0) > 0 else 50
    noise_array_raw = np.reshape(truncnorm.rvs(-px_mean/px_std, 4, size=outshape[0]*outshape[1]), outshape)
    noise_array = noise_array_raw * px_std + px_mean
    noise_array[noise_array < 0] = 0

    # 4. Blend: embryo region from equalized image + background from noise
    im_cropped_gauss = np.multiply(im_cropped_eq.astype(float), mask_cropped_gauss) + \
                       np.multiply(noise_array, 1 - mask_cropped_gauss)

    return {
        "cropped": im_cropped.astype(np.uint8),           # Raw extraction
        "cropped_eq": im_cropped_eq.astype(np.uint8),     # With histogram equalization
        "cropped_gauss": im_cropped_gauss.astype(np.uint8),  # Final post-processed (matches bf_embryo_snips)
        "cropped_mask": emb_mask_cropped2 * 255,
        "metrics": metrics
    }


def save_comparison_figure(snip_id, im_ff, im_mask, result_old, result_new, output_dir):
    """Save comparison figure with original + both extractions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original FF with mask overlay - use alpha blending to show mask clearly
    axes[0, 0].imshow(im_ff, cmap='gray')
    # Create colored mask overlay
    mask_rgb = np.zeros((*im_mask.shape, 3))
    mask_rgb[im_mask > 127, 0] = 1  # Red channel for mask
    axes[0, 0].imshow(mask_rgb, alpha=0.4)
    # Also add contour for clarity
    axes[0, 0].contour(im_mask > 127, levels=[0.5], colors='yellow', linewidths=2)
    axes[0, 0].set_title("Original Image + Mask (red overlay + yellow contour)")
    axes[0, 0].axis('off')

    # Old extraction (6.5 μm/px) - POST-PROCESSED
    axes[0, 1].imshow(result_old["cropped_gauss"], cmap='gray')
    axes[0, 1].set_title(
        f"OLD (6.5 μm/px) - POST-PROCESSED\n"
        f"Fill: {result_old['metrics']['fill_fraction_height']:.1%}, "
        f"OOF: {result_old['metrics']['out_of_frame']}"
    )
    axes[0, 1].axis('off')

    # New extraction (7.8 μm/px) - POST-PROCESSED
    axes[1, 0].imshow(result_new["cropped_gauss"], cmap='gray')
    axes[1, 0].set_title(
        f"NEW (7.8 μm/px) - POST-PROCESSED\n"
        f"Fill: {result_new['metrics']['fill_fraction_height']:.1%}, "
        f"OOF: {result_new['metrics']['out_of_frame']}"
    )
    axes[1, 0].axis('off')

    # Metrics text
    ax_text = axes[1, 1]
    ax_text.axis('off')
    metrics_text = (
        f"Embryo: {snip_id}\n\n"
        f"OLD (6.5 μm/px):\n"
        f"  • Size: {result_old['metrics']['embryo_length_mm']:.2f} × "
        f"{result_old['metrics']['embryo_width_mm']:.2f} mm\n"
        f"  • Fill: {result_old['metrics']['fill_fraction_height']:.1%}\n"
        f"  • Retained: {result_old['metrics']['area_retained']:.1%}\n"
        f"  • Out of frame: {result_old['metrics']['out_of_frame']}\n\n"
        f"NEW (7.8 μm/px):\n"
        f"  • Size: {result_new['metrics']['embryo_length_mm']:.2f} × "
        f"{result_new['metrics']['embryo_width_mm']:.2f} mm\n"
        f"  • Fill: {result_new['metrics']['fill_fraction_height']:.1%}\n"
        f"  • Retained: {result_new['metrics']['area_retained']:.1%}\n"
        f"  • Out of frame: {result_new['metrics']['out_of_frame']}"
    )
    ax_text.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f"{snip_id}_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    output_dir = Path(__file__).parent / "extraction_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM2 metadata
    metadata_files = [
        root / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20251106.csv",
        root / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20250512.csv",
    ]

    # Verify metadata files exist
    available_files = [f for f in metadata_files if f.exists()]
    if not available_files:
        print(f"❌ ERROR: No metadata files found!")
        print(f"Looking in: {root / 'sam2_pipeline_files' / 'sam2_expr_files'}")
        exit(1)

    df_meta = pd.concat([pd.read_csv(f) for f in available_files], ignore_index=True)

    # Test snip IDs (problematic cases)
    test_snip_ids = [
        "20251106_A02_e01_t0049",
        "20251106_A03_e01_t0085",
        "20250512_A03_e01_t0112",
        "20251106_H12_e01_t0093",
    ]

    print("="*80)
    print("EMBRYO EXTRACTION - FULL PIPELINE TEST")
    print("="*80)

    results = []

    for snip_id in test_snip_ids:
        print(f"\n{snip_id}...")
        matching = df_meta[df_meta["snip_id"] == snip_id]
        if len(matching) == 0:
            print(f"  ❌ Not found")
            continue

        row = matching.iloc[0].to_dict()

        # Add missing columns that build03 expects
        if "experiment_date" not in row:
            row["experiment_date"] = row["experiment_id"]
        if "region_label" not in row:
            embryo_id = row["embryo_id"]
            region_label = int(embryo_id.split("_e")[-1])
            row["region_label"] = region_label

        # Load mask
        try:
            mask_path = resolve_sandbox_embryo_mask_from_csv(root, row)
            im_mask_int = io.imread(mask_path)
            lbi = int(row["region_label"])
            im_mask = ((im_mask_int == lbi) * 255).astype(np.uint8)
            im_mask, _ = clean_embryo_mask(im_mask, verbose=False)
            print(f"  ✓ Mask loaded")
        except Exception as e:
            print(f"  ❌ Mask load failed: {e}")
            continue

        # Load FF
        try:
            date = str(row["experiment_date"])
            ff_dir = root / "built_image_data" / "stitched_FF_images" / date
            ff_paths = sorted(ff_dir.glob(f"{row['image_id']}*"))
            if not ff_paths:
                import re
                m = re.search(r'_([A-H]\d{2})_.*_(t\d{4})$', row['image_id'])
                if m:
                    well, time = m.groups()
                    ff_paths = sorted(ff_dir.glob(f"{well}_{time}*"))

            if not ff_paths:
                print(f"  ❌ FF not found")
                continue

            im_ff = io.imread(ff_paths[0])
            print(f"  ✓ FF loaded")
        except Exception as e:
            print(f"  ❌ FF load failed: {e}")
            continue

        im_yolk = np.zeros_like(im_mask)

        # Get px_dim_raw from metadata (critical - not hardcoded!)
        if 'Height (um)' in row and 'Height (px)' in row and row['Height (px)'] > 0:
            px_dim_raw = row["Height (um)"] / row["Height (px)"]
        else:
            px_dim_raw = 0.65  # Fallback
        print(f"  Using px_dim_raw = {px_dim_raw:.4f} μm/px from metadata")

        # Process both scales
        print(f"  Processing 6.5 μm/px...")
        result_old = process_embryo(im_ff, im_mask, im_yolk, outscale=6.5, px_dim_raw=px_dim_raw)

        print(f"  Processing 7.8 μm/px...")
        result_new = process_embryo(im_ff, im_mask, im_yolk, outscale=7.8, px_dim_raw=px_dim_raw)

        # Save extracted images (POST-PROCESSED versions that match bf_embryo_snips)
        old_dir = output_dir / "old_6.5um"
        new_dir = output_dir / "new_7.8um"
        old_dir.mkdir(parents=True, exist_ok=True)
        new_dir.mkdir(parents=True, exist_ok=True)

        # Save post-processed versions (these match bf_embryo_snips format)
        io.imsave(old_dir / f"{snip_id}.jpg", result_old["cropped_gauss"], check_contrast=False)
        io.imsave(new_dir / f"{snip_id}.jpg", result_new["cropped_gauss"], check_contrast=False)

        # Save comparison figure
        figures_dir = output_dir / "comparisons"
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_comparison_figure(snip_id, im_ff, im_mask, result_old, result_new, figures_dir)

        print(f"  ✓ Extraction complete")
        print(f"    OLD: Fill={result_old['metrics']['fill_fraction_height']:.1%}, OOF={result_old['metrics']['out_of_frame']}")
        print(f"    NEW: Fill={result_new['metrics']['fill_fraction_height']:.1%}, OOF={result_new['metrics']['out_of_frame']}")

        results.append({
            "snip_id": snip_id,
            "old_fill": result_old["metrics"]["fill_fraction_height"],
            "old_retained": result_old["metrics"]["area_retained"],
            "old_oof": result_old["metrics"]["out_of_frame"],
            "new_fill": result_new["metrics"]["fill_fraction_height"],
            "new_retained": result_new["metrics"]["area_retained"],
            "new_oof": result_new["metrics"]["out_of_frame"],
        })

    # Save metrics
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_dir / "extraction_metrics.csv", index=False)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to {output_dir}")
    print(f"  • Extracted images: {output_dir}/old_6.5um/ and {output_dir}/new_7.8um/")
    print(f"  • Comparison figures: {output_dir}/comparisons/")
    print(f"  • Metrics: {output_dir}/extraction_metrics.csv")
    print(f"{'='*80}\n")
    print(df_res.to_string())
