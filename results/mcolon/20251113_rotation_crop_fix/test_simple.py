#!/usr/bin/env python3
"""
Simplified test for embryo rotation/cropping fix.

This test loads already-extracted snips and their masks, then re-processes them
through the extraction pipeline with both old (6.5) and new (7.8) μm/px parameters.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
    """Crop and return metrics."""
    if np.sum(emb_mask_rotated) == 0:
        return (np.zeros(outshape), np.zeros(outshape), np.zeros(outshape),
                {"out_of_frame": True, "area_retained": 0.0})

    y_indices = np.where(np.max(emb_mask_rotated, axis=1) > 0.5)[0]
    x_indices = np.where(np.max(emb_mask_rotated, axis=0) > 0.5)[0]

    if y_indices.size == 0 or x_indices.size == 0:
        return (np.zeros(outshape), np.zeros(outshape), np.zeros(outshape),
                {"out_of_frame": True, "area_retained": 0.0})

    y_mean = int(np.mean(y_indices))
    x_mean = int(np.mean(x_indices))

    fromshape = emb_mask_rotated.shape
    raw_range_y = [y_mean - int(outshape[0] / 2), y_mean + int(outshape[0] / 2)]
    from_range_y = np.asarray([np.max([raw_range_y[0], 0]), np.min([raw_range_y[1], fromshape[0]])])
    to_range_y = [0 + (from_range_y[0] - raw_range_y[0]), outshape[0] + (from_range_y[1] - raw_range_y[1])]

    raw_range_x = [x_mean - int(outshape[1] / 2), x_mean + int(outshape[1] / 2)]
    from_range_x = np.asarray([np.max([raw_range_x[0], 0]), np.min([raw_range_x[1], fromshape[1]])])
    to_range_x = [0 + (from_range_x[0] - raw_range_x[0]), outshape[1] + (from_range_x[1] - raw_range_x[1])]

    mask_area_before = np.sum(emb_mask_rotated > 0.5)

    im_cropped = np.zeros(outshape).astype(np.uint8)
    im_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        im_ff_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    emb_mask_cropped = np.zeros(outshape)
    emb_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        emb_mask_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    yolk_mask_cropped = np.zeros(outshape)
    yolk_mask_cropped[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        im_yolk_rotated[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    mask_area_after = np.sum(emb_mask_cropped > 0.5)
    area_retained = mask_area_after / mask_area_before if mask_area_before > 0 else 0.0
    out_of_frame_flag = area_retained < 0.98

    metrics = {
        "out_of_frame": out_of_frame_flag,
        "area_retained": area_retained,
    }

    return im_cropped, emb_mask_cropped, yolk_mask_cropped, metrics


def process_embryo(im_ff, im_mask, im_yolk, outscale, outshape=[576, 256]):
    """Process embryo through pipeline at given resolution."""
    # Assume input masks are at same resolution as im_ff
    # Rescale to target μm/px (assuming 0.65 μm/px original)
    px_dim_raw = 0.65
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

    return {
        "cropped": im_cropped.astype(np.uint8),
        "cropped_mask": emb_mask_cropped2 * 255,
        "metrics": metrics
    }


def run_test(root_path, output_dir):
    """Test using actual extract_embryo_snips workflow."""
    from src.build.build03A_process_images import resolve_sandbox_embryo_mask_from_csv
    from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask

    root = Path(root_path)
    output = Path(output_dir)

    # Test snip IDs from the problematic images
    # For this simple test, we manually define minimal metadata rows
    test_cases = [
        {
            "snip_id": "20251106_A02_e01_t0049",
            "experiment_date": "20251106",
            "image_id": "20251106_A02_ch00_t0049",
            "region_label": 1,
            "issue": "cut_at_top"
        },
        {
            "snip_id": "20251106_A03_e01_t0085",
            "experiment_date": "20251106",
            "image_id": "20251106_A03_ch00_t0085",
            "region_label": 1,
            "issue": "cut_at_bottom"
        },
        {
            "snip_id": "20250512_A03_e01_t0112",
            "experiment_date": "20250512",
            "image_id": "20250512_A03_ch00_t0112",
            "region_label": 1,
            "issue": "cut_at_side"
        },
        {
            "snip_id": "20251106_H12_e01_t0093",
            "experiment_date": "20251106",
            "image_id": "20251106_H12_ch00_t0093",
            "region_label": 1,
            "issue": "large_reference"
        }
    ]

    results = []

    print("="*80)
    print("EMBRYO ROTATION/CROPPING FIX TEST")
    print("="*80)
    print(f"Output directory: {output}")
    print(f"Test cases: {len(test_cases)}")
    print()

    for test_row in test_cases:
        snip_id = test_row["snip_id"]
        issue = test_row["issue"]

        print(f"\n{'─'*80}")
        print(f"Processing: {snip_id} ({issue})")
        print(f"{'─'*80}")

        # Try to load mask using actual Build03 function
        try:
            mask_path = resolve_sandbox_embryo_mask_from_csv(root, test_row)
            im_mask_int = io.imread(mask_path)
            lbi = int(test_row["region_label"])
            im_mask = ((im_mask_int == lbi) * 255).astype(np.uint8)
            # Clean mask
            im_mask, _ = clean_embryo_mask(im_mask, verbose=False)
            print(f"   ✓ Loaded and cleaned mask")
        except Exception as e:
            print(f"   ❌ Failed to load mask: {e}")
            continue

        # Load FF image
        try:
            date = str(test_row["experiment_date"])
            ff_image_path = root / 'built_image_data' / 'stitched_FF_images'
            full_stub = f"{test_row['image_id']}*"
            ff_image_paths = sorted((ff_image_path / date).glob(full_stub))

            if ff_image_paths:
                im_ff = io.imread(ff_image_paths[0])
                print(f"   ✓ Loaded FF image: {ff_image_paths[0].name}")
            else:
                # Try legacy format
                import re
                image_id = test_row['image_id']
                match = re.search(r'_([A-H]\d{2})_.*_(t\d{4})$', image_id)
                if match:
                    well_part, time_part = match.groups()
                    legacy_stub = f"{well_part}_{time_part}*"
                    ff_image_paths = sorted((ff_image_path / date).glob(legacy_stub))

                if ff_image_paths:
                    im_ff = io.imread(ff_image_paths[0])
                    print(f"   ✓ Loaded FF image (legacy): {ff_image_paths[0].name}")
                else:
                    print(f"   ❌ FF image not found")
                    continue
        except Exception as e:
            print(f"   ❌ Failed to load FF image: {e}")
            continue

        im_yolk = np.zeros_like(im_mask)

        print(f"   ✓ Loaded images: {im_ff.shape}")

        # Process with OLD parameters
        print(f"   Processing OLD (6.5 μm/px)...")
        result_old = process_embryo(im_ff, im_mask, im_yolk, outscale=6.5)

        # Process with NEW parameters
        print(f"   Processing NEW (7.8 μm/px)...")
        result_new = process_embryo(im_ff, im_mask, im_yolk, outscale=7.8)

        # Save outputs
        io.imsave(output / "old_6.5um" / f"{snip_id}.jpg", result_old["cropped"], check_contrast=False)
        io.imsave(output / "new_7.8um" / f"{snip_id}.jpg", result_new["cropped"], check_contrast=False)

        # Print metrics
        print(f"\n   OLD (6.5 μm/px) Metrics:")
        print(f"      • Embryo size: {result_old['metrics']['embryo_length_mm']:.2f} × {result_old['metrics']['embryo_width_mm']:.2f} mm")
        print(f"      • Fill fraction: {result_old['metrics']['fill_fraction_height']:.1%} (H)")
        print(f"      • Area retained: {result_old['metrics']['area_retained']:.1%}")
        print(f"      • Out of frame: {result_old['metrics']['out_of_frame']}")

        print(f"\n   NEW (7.8 μm/px) Metrics:")
        print(f"      • Embryo size: {result_new['metrics']['embryo_length_mm']:.2f} × {result_new['metrics']['embryo_width_mm']:.2f} mm")
        print(f"      • Fill fraction: {result_new['metrics']['fill_fraction_height']:.1%} (H)")
        print(f"      • Area retained: {result_new['metrics']['area_retained']:.1%}")
        print(f"      • Out of frame: {result_new['metrics']['out_of_frame']}")

        # Store results
        results.append({
            "snip_id": snip_id,
            "issue": issue,
            **{f"old_{k}": v for k, v in result_old["metrics"].items()},
            **{f"new_{k}": v for k, v in result_new["metrics"].items()}
        })

    # Save metrics
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
    root_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    output_dir = Path(__file__).parent

    # Load metadata for the test dates
    metadata_files = [
        root_path / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20251106.csv",
        root_path / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20250512.csv",
    ]

    # Verify metadata files exist
    missing_files = [f for f in metadata_files if not f.exists()]
    if missing_files:
        print(f"❌ ERROR: Missing metadata files:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"\nAvailable metadata files in {root_path / 'sam2_pipeline_files' / 'sam2_expr_files'}:")
        expr_dir = root_path / 'sam2_pipeline_files' / 'sam2_expr_files'
        if expr_dir.exists():
            for f in sorted(expr_dir.glob('sam2_metadata_*.csv'))[:5]:
                print(f"   - {f.name}")
        exit(1)

    # Combine metadata from both dates
    df_all = pd.concat([pd.read_csv(f) for f in metadata_files if f.exists()], ignore_index=True)

    # Filter to test snip IDs
    test_snip_ids = [
        "20251106_A02_e01_t0049",
        "20251106_A03_e01_t0085",
        "20250512_A03_e01_t0112",
        "20251106_H12_e01_t0093",
    ]

    # Extract rows for test snips - add missing columns that build03 expects
    test_rows = []
    for snip_id in test_snip_ids:
        matching = df_all[df_all["snip_id"] == snip_id]
        if len(matching) > 0:
            row = matching.iloc[0].to_dict()
            # Add experiment_date = experiment_id (they're the same - the date)
            if "experiment_date" not in row:
                row["experiment_date"] = row["experiment_id"]
            # Add region_label extracted from embryo_id (e.g., "20251106_A02_e01" -> 1)
            if "region_label" not in row:
                embryo_id = row["embryo_id"]
                region_label = int(embryo_id.split("_e")[-1])  # Extract eNN -> NN
                row["region_label"] = region_label
            test_rows.append((row, snip_id))

    if not test_rows:
        print(f"❌ Could not find test snip_ids in metadata")
        print(f"Available snip_ids (sample):")
        print(df_all["snip_id"].head(10).tolist())
        exit(1)

    print(f"✓ Found {len(test_rows)} test snips in metadata")
    results = run_test(root_path, output_dir)

    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)

    for _, row in results.iterrows():
        print(f"\n{row['snip_id']} ({row['issue']}):")
        print(f"   OLD: Fill={row['old_fill_fraction_height']:.1%}, Retained={row['old_area_retained']:.1%}, OOF={row['old_out_of_frame']}")
        print(f"   NEW: Fill={row['new_fill_fraction_height']:.1%}, Retained={row['new_area_retained']:.1%}, OOF={row['new_out_of_frame']}")

        if row['old_out_of_frame'] and not row['new_out_of_frame']:
            print(f"   ✅ FIXED: No longer out of frame!")
        elif row['old_fill_fraction_height'] > 0.90 and row['new_fill_fraction_height'] < 0.85:
            print(f"   ✅ IMPROVED: Embryo now fits with margin")
