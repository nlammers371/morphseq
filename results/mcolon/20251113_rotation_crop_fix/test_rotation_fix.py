#!/usr/bin/env python3
"""
Test script for embryo rotation/cropping fix.

Tests 4 problematic images with both old (6.5 μm/px) and new (7.8 μm/px) parameters
to verify that the updated resolution prevents embryo clipping.

Problem: Large embryos (~3.5mm) were filling 100% of the 3.74mm capture window,
causing clipping when rotated. New resolution (7.8 μm/px) gives 3.0mm window,
ensuring embryos fill max 80%.

Test images:
1. 20251106_A02_e01_t0049 - embryo cut at top
2. 20251106_A03_e01_t0085 - embryo cut at bottom
3. 20250512_A03_e01_t0112 - embryo cut at side
4. 20251106_H12_e01_t0093 - large embryo (reference)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io as io
from skimage.transform import rescale, resize
import scipy.ndimage
import cv2

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.functions.image_utils import get_embryo_angle
from skimage.measure import regionprops


def rotate_image(mat, angle):
    """Rotate image with canvas expansion to avoid cropping."""
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def crop_embryo_image_with_metrics(im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape):
    """
    Crop embryo image and return metrics about the crop quality.

    Returns:
        im_cropped, emb_mask_cropped, yolk_mask_cropped, metrics_dict
    """
    if np.sum(emb_mask_rotated) == 0:
        return (np.zeros(outshape), np.zeros(outshape), np.zeros(outshape),
                {"out_of_frame": True, "area_retained": 0.0, "crop_incomplete": True})

    y_indices = np.where(np.max(emb_mask_rotated, axis=1) > 0.5)[0]
    x_indices = np.where(np.max(emb_mask_rotated, axis=0) > 0.5)[0]

    if y_indices.size == 0 or x_indices.size == 0:
        return (np.zeros(outshape), np.zeros(outshape), np.zeros(outshape),
                {"out_of_frame": True, "area_retained": 0.0, "crop_incomplete": True})

    y_mean = int(np.mean(y_indices))
    x_mean = int(np.mean(x_indices))

    fromshape = emb_mask_rotated.shape
    raw_range_y = [y_mean - int(outshape[0] / 2), y_mean + int(outshape[0] / 2)]
    from_range_y = np.asarray([np.max([raw_range_y[0], 0]), np.min([raw_range_y[1], fromshape[0]])])
    to_range_y = [0 + (from_range_y[0] - raw_range_y[0]), outshape[0] + (from_range_y[1] - raw_range_y[1])]

    raw_range_x = [x_mean - int(outshape[1] / 2), x_mean + int(outshape[1] / 2)]
    from_range_x = np.asarray([np.max([raw_range_x[0], 0]), np.min([raw_range_x[1], fromshape[1]])])
    to_range_x = [0 + (from_range_x[0] - raw_range_x[0]), outshape[1] + (from_range_x[1] - raw_range_x[1])]

    # Calculate crop metrics
    requested_area = outshape[0] * outshape[1]
    actual_h = to_range_y[1] - to_range_y[0]
    actual_w = to_range_x[1] - to_range_x[0]
    actual_area = actual_h * actual_w
    crop_incomplete = actual_area < 0.95 * requested_area

    # Calculate mask area retention
    mask_area_before = np.sum(emb_mask_rotated > 0.5)

    if len(im_ff_rotated.shape) == 2:
        im_cropped = np.zeros(outshape).astype(np.uint8)
        im_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_ff_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]
    else:
        im_cropped = np.zeros((im_ff_rotated.shape[0], outshape[0], outshape[1]), dtype=im_ff_rotated.dtype)
        im_cropped[:, to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
            im_ff_rotated[:, from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    emb_mask_cropped = np.zeros(outshape)
    emb_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        emb_mask_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    yolk_mask_cropped = np.zeros(outshape)
    yolk_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        im_yolk_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    mask_area_after = np.sum(emb_mask_cropped > 0.5)
    area_retained = mask_area_after / mask_area_before if mask_area_before > 0 else 0.0

    # Flag if <98% of mask area is retained
    out_of_frame_flag = area_retained < 0.98

    metrics = {
        "out_of_frame": out_of_frame_flag,
        "area_retained": area_retained,
        "crop_incomplete": crop_incomplete,
        "requested_area": requested_area,
        "actual_area": actual_area,
        "crop_center_y": y_mean,
        "crop_center_x": x_mean,
        "from_bounds": f"y[{from_range_y[0]}:{from_range_y[1]}] x[{from_range_x[0]}:{from_range_x[1]}]",
        "to_bounds": f"y[{to_range_y[0]}:{to_range_y[1]}] x[{to_range_x[0]}:{to_range_x[1]}]"
    }

    return im_cropped, emb_mask_cropped, yolk_mask_cropped, metrics


def process_single_embryo(im_ff, im_mask, im_yolk, outscale, outshape=[576, 256]):
    """
    Process a single embryo through the full pipeline.

    Returns:
        Dictionary with all intermediate images and metrics
    """
    # Rescale to target resolution
    px_dim_raw = 0.65  # Assume original is 0.65 μm/px for Keyence
    scale_factor = px_dim_raw / outscale

    im_ff_rs = rescale(im_ff, (scale_factor, scale_factor), order=1, preserve_range=True)
    mask_emb_rs = resize(im_mask.astype(float), im_ff_rs.shape, order=1)
    mask_yolk_rs = resize(im_yolk.astype(float), im_ff_rs.shape, order=1)

    # Get rotation angle
    angle_to_use = get_embryo_angle(
        (mask_emb_rs > 0.5).astype(np.uint8),
        (mask_yolk_rs > 0.5).astype(np.uint8)
    )

    # Rotate with expansion
    im_ff_rotated = rotate_image(im_ff_rs, np.rad2deg(angle_to_use))
    emb_mask_rotated = rotate_image(mask_emb_rs, np.rad2deg(angle_to_use))
    im_yolk_rotated = rotate_image(mask_yolk_rs, np.rad2deg(angle_to_use))

    # Crop to final size
    im_cropped, emb_mask_cropped, yolk_mask_cropped, metrics = crop_embryo_image_with_metrics(
        im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape=outshape
    )

    # Fill holes in masks
    emb_mask_cropped2 = scipy.ndimage.binary_fill_holes(emb_mask_cropped > 0.5).astype(np.uint8)
    yolk_mask_cropped = scipy.ndimage.binary_fill_holes(yolk_mask_cropped > 0.5).astype(np.uint8)

    # Calculate physical dimensions
    embryo_props = regionprops((emb_mask_cropped2 > 0).astype(int))
    if embryo_props:
        bbox = embryo_props[0].bbox  # (min_row, min_col, max_row, max_col)
        embryo_length_px = bbox[2] - bbox[0]  # Height in pixels
        embryo_width_px = bbox[3] - bbox[1]   # Width in pixels
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
        "capture_window_h_mm": (outshape[0] * outscale) / 1000,
        "capture_window_w_mm": (outshape[1] * outscale) / 1000,
    })

    return {
        "rescaled": im_ff_rs.astype(np.uint8),
        "rescaled_mask": (mask_emb_rs * 255).astype(np.uint8),
        "rotated": im_ff_rotated.astype(np.uint8),
        "rotated_mask": (emb_mask_rotated * 255).astype(np.uint8),
        "cropped": im_cropped.astype(np.uint8),
        "cropped_mask": emb_mask_cropped2 * 255,
        "metrics": metrics
    }


def create_overlay_image(image, mask, alpha=0.3):
    """Create RGB overlay of mask on grayscale image."""
    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    else:
        rgb_image = image.copy()

    # Make mask red overlay
    mask_rgb = np.zeros_like(rgb_image)
    mask_rgb[:, :, 0] = (mask > 127).astype(np.uint8) * 255  # Red channel

    # Blend
    overlay = (rgb_image * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return overlay


def run_test(root_path, metadata_csv, output_dir):
    """
    Run full test suite on problematic images loaded from raw FF + metadata.

    Parameters
    ----------
    root_path : Path
        Project root directory
    metadata_csv : Path
        Path to embryo_metadata_df01.csv (from Build03 output)
    output_dir : Path
        Output directory for results
    """
    root = Path(root_path)
    output = Path(output_dir)

    # Load metadata
    try:
        df_meta = pd.read_csv(metadata_csv)
        print(f"Loaded metadata: {len(df_meta)} rows")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return None

    # Define test cases by snip_id
    test_snip_ids = [
        "20251106_A02_e01_t0049",
        "20251106_A03_e01_t0085",
        "20250512_A03_e01_t0112",
        "20251106_H12_e01_t0093"
    ]

    results = []

    print("="*80)
    print("EMBRYO ROTATION/CROPPING FIX TEST")
    print("="*80)
    print(f"Output directory: {output}")
    print(f"Metadata file: {metadata_csv}")
    print(f"Test cases: {len(test_snip_ids)}")
    print()

    for snip_id in test_snip_ids:
        print(f"\n{'─'*80}")
        print(f"Processing: {snip_id}")
        print(f"{'─'*80}")

        # Find row in metadata
        matching_rows = df_meta[df_meta["snip_id"] == snip_id]
        if len(matching_rows) == 0:
            print(f"   ❌ snip_id not found in metadata: {snip_id}")
            continue

        row = matching_rows.iloc[0]
        date = str(row["experiment_date"])

        # Load raw FF image
        try:
            from src.build.build03A_process_images import resolve_sandbox_embryo_mask_from_csv
            mask_path = resolve_sandbox_embryo_mask_from_csv(root, row)
            im_mask_int = io.imread(mask_path)
            lbi = int(row["region_label"])
            im_mask = ((im_mask_int == lbi) * 255).astype(np.uint8)
            print(f"   ✓ Loaded mask from: {mask_path}")
        except Exception as e:
            print(f"   ❌ Failed to load mask: {e}")
            continue

        # Load FF image
        try:
            ff_dir = root / 'built_image_data' / 'stitched_FF_images' / date
            full_stub = f"{row['image_id']}*"
            ff_image_paths = sorted(list(ff_dir.glob(full_stub)))

            if not ff_image_paths:
                # Try legacy format
                import re
                image_id = row['image_id']
                match = re.search(r'_([A-H]\d{2})_.*_(t\d{4})$', image_id)
                if match:
                    well_part, time_part = match.groups()
                    legacy_stub = f"{well_part}_{time_part}*"
                    ff_image_paths = sorted(list(ff_dir.glob(legacy_stub)))

            if ff_image_paths:
                im_ff = io.imread(ff_image_paths[0])
                print(f"   ✓ Loaded FF image: {ff_image_paths[0].name} ({im_ff.shape})")
            else:
                print(f"   ❌ FF image not found for stub: {full_stub}")
                continue
        except Exception as e:
            print(f"   ❌ Failed to load FF image: {e}")
            continue

        # Create dummy yolk mask
        im_yolk = np.zeros_like(im_mask)

        print(f"   ✓ Loaded images: {im_ff.shape}")

        # Process with OLD parameters (6.5 μm/px)
        print(f"   Processing OLD (6.5 μm/px)...")
        result_old = process_single_embryo(im_ff, im_mask, im_yolk, outscale=6.5)

        # Process with NEW parameters (7.8 μm/px)
        print(f"   Processing NEW (7.8 μm/px)...")
        result_new = process_single_embryo(im_ff, im_mask, im_yolk, outscale=7.8)

        # Save outputs
        io.imsave(output / "old_6.5um" / f"{snip_id}.jpg", result_old["cropped"], check_contrast=False)
        io.imsave(output / "new_7.8um" / f"{snip_id}.jpg", result_new["cropped"], check_contrast=False)

        # Save debug images
        io.imsave(output / "debug" / f"{snip_id}_old_rescaled.jpg", result_old["rescaled"], check_contrast=False)
        io.imsave(output / "debug" / f"{snip_id}_old_rotated.jpg", result_old["rotated"], check_contrast=False)
        io.imsave(output / "debug" / f"{snip_id}_old_cropped.jpg", result_old["cropped"], check_contrast=False)

        io.imsave(output / "debug" / f"{snip_id}_new_rescaled.jpg", result_new["rescaled"], check_contrast=False)
        io.imsave(output / "debug" / f"{snip_id}_new_rotated.jpg", result_new["rotated"], check_contrast=False)
        io.imsave(output / "debug" / f"{snip_id}_new_cropped.jpg", result_new["cropped"], check_contrast=False)

        # Save mask overlays
        overlay_old_rotated = create_overlay_image(result_old["rotated"], result_old["rotated_mask"])
        overlay_old_cropped = create_overlay_image(result_old["cropped"], result_old["cropped_mask"])
        overlay_new_rotated = create_overlay_image(result_new["rotated"], result_new["rotated_mask"])
        overlay_new_cropped = create_overlay_image(result_new["cropped"], result_new["cropped_mask"])

        io.imsave(output / "debug" / f"{snip_id}_old_rotated_overlay.jpg", overlay_old_rotated, check_contrast=False)
        io.imsave(output / "debug" / f"{snip_id}_old_cropped_overlay.jpg", overlay_old_cropped, check_contrast=False)
        io.imsave(output / "debug" / f"{snip_id}_new_rotated_overlay.jpg", overlay_new_rotated, check_contrast=False)
        io.imsave(output / "debug" / f"{snip_id}_new_cropped_overlay.jpg", overlay_new_cropped, check_contrast=False)

        # Print metrics
        print(f"\n   OLD (6.5 μm/px) Metrics:")
        print(f"      • Embryo size: {result_old['metrics']['embryo_length_mm']:.2f} × {result_old['metrics']['embryo_width_mm']:.2f} mm")
        print(f"      • Fill fraction: {result_old['metrics']['fill_fraction_height']:.1%} (H) × {result_old['metrics']['fill_fraction_width']:.1%} (W)")
        print(f"      • Area retained: {result_old['metrics']['area_retained']:.1%}")
        print(f"      • Out of frame: {result_old['metrics']['out_of_frame']}")
        print(f"      • Crop incomplete: {result_old['metrics']['crop_incomplete']}")

        print(f"\n   NEW (7.8 μm/px) Metrics:")
        print(f"      • Embryo size: {result_new['metrics']['embryo_length_mm']:.2f} × {result_new['metrics']['embryo_width_mm']:.2f} mm")
        print(f"      • Fill fraction: {result_new['metrics']['fill_fraction_height']:.1%} (H) × {result_new['metrics']['fill_fraction_width']:.1%} (W)")
        print(f"      • Area retained: {result_new['metrics']['area_retained']:.1%}")
        print(f"      • Out of frame: {result_new['metrics']['out_of_frame']}")
        print(f"      • Crop incomplete: {result_new['metrics']['crop_incomplete']}")

        # Store results
        results.append({
            "snip_id": snip_id,
            "issue": test_case["issue"],
            **{f"old_{k}": v for k, v in result_old["metrics"].items()},
            **{f"new_{k}": v for k, v in result_new["metrics"].items()}
        })

    # Save metrics to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output / "metrics.csv", index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output}")
    print(f"Metrics CSV: {output / 'metrics.csv'}")
    print(f"\nTest complete!")

    return df_results


if __name__ == "__main__":
    root_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")

    # Try multiple possible metadata locations for 20251106 and 20250512 experiments
    possible_metadata = [
        root_path / "metadata" / "embryo_metadata_files" / "20251106_embryo_metadata.csv",
        root_path / "metadata" / "embryo_metadata_files" / "20250512_embryo_metadata.csv",
        root_path / "metadata" / "combined_metadata_files" / "embryo_metadata_df01.csv",
        root_path / "metadata" / "embryo_metadata_files" / "20250612_30hpf_ctrl_atf6_embryo_metadata.csv",
    ]

    # Also check for glob patterns
    import glob as glob_module
    embryo_meta_dir = root_path / "metadata" / "embryo_metadata_files"
    if embryo_meta_dir.exists():
        all_meta_files = sorted(glob_module.glob(str(embryo_meta_dir / "*embryo_metadata.csv")))
        possible_metadata.extend([Path(f) for f in all_meta_files])

    metadata_csv = None
    for possible_path in possible_metadata:
        if possible_path.exists():
            metadata_csv = possible_path
            print(f"✓ Found metadata: {metadata_csv}")
            break

    if metadata_csv is None:
        print(f"❌ Could not find metadata in any of:")
        for p in possible_metadata:
            print(f"   - {p}")
        print(f"\n📁 Available files in {embryo_meta_dir}:")
        if embryo_meta_dir.exists():
            for f in sorted(embryo_meta_dir.glob("*.csv")):
                print(f"   - {f.name}")
        exit(1)

    output_dir = Path(__file__).parent

    results = run_test(root_path, metadata_csv, output_dir)

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    for _, row in results.iterrows():
        print(f"\n{row['snip_id']} ({row['issue']}):")
        print(f"   OLD: Fill={row['old_fill_fraction_height']:.1%}, Retained={row['old_area_retained']:.1%}, Flagged={row['old_out_of_frame']}")
        print(f"   NEW: Fill={row['new_fill_fraction_height']:.1%}, Retained={row['new_area_retained']:.1%}, Flagged={row['new_out_of_frame']}")

        if row['old_out_of_frame'] and not row['new_out_of_frame']:
            print(f"   ✅ FIXED: No longer out of frame!")
        elif row['old_fill_fraction_height'] > 0.95 and row['new_fill_fraction_height'] < 0.85:
            print(f"   ✅ IMPROVED: Embryo now fits with margin")
