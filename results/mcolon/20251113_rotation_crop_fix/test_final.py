#!/usr/bin/env python3
"""
Final test for embryo rotation/cropping fix using SAM2 metadata.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.transform import rescale, resize
import scipy.ndimage
import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.functions.image_utils import get_embryo_angle
from src.build.build03A_process_images import resolve_sandbox_embryo_mask_from_csv
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from skimage.measure import regionprops


def rotate_image(mat, angle):
    """Rotate with canvas expansion."""
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))


def crop_embryo_with_metrics(im_ff_rotated, emb_mask_rotated, im_yolk_rotated, outshape):
    """Crop and compute metrics."""
    if np.sum(emb_mask_rotated) == 0:
        return np.zeros(outshape), np.zeros(outshape), np.zeros(outshape), True

    y_indices = np.where(np.max(emb_mask_rotated, axis=1) > 0.5)[0]
    x_indices = np.where(np.max(emb_mask_rotated, axis=0) > 0.5)[0]

    if y_indices.size == 0 or x_indices.size == 0:
        return np.zeros(outshape), np.zeros(outshape), np.zeros(outshape), True

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

    mask_area_after = np.sum(emb_mask_cropped > 0.5)
    area_retained = mask_area_after / mask_area_before if mask_area_before > 0 else 0.0
    out_of_frame = area_retained < 0.98

    return im_cropped, emb_mask_cropped, np.zeros(outshape), out_of_frame


def process_at_scale(im_ff, im_mask, outscale, outshape=[576, 256]):
    """Process embryo at given scale."""
    px_dim_raw = 0.65
    scale_factor = px_dim_raw / outscale

    im_ff_rs = rescale(im_ff, (scale_factor, scale_factor), order=1, preserve_range=True)
    mask_emb_rs = resize(im_mask.astype(float), im_ff_rs.shape, order=1)

    angle = get_embryo_angle((mask_emb_rs > 0.5).astype(np.uint8), np.zeros_like(mask_emb_rs))

    im_ff_rot = rotate_image(im_ff_rs, np.rad2deg(angle))
    mask_emb_rot = rotate_image(mask_emb_rs, np.rad2deg(angle))

    im_crop, mask_crop, _, oof = crop_embryo_with_metrics(im_ff_rot, mask_emb_rot, np.zeros_like(mask_emb_rot), outshape)

    mask_crop_filled = scipy.ndimage.binary_fill_holes(mask_crop > 0.5).astype(np.uint8)
    props = regionprops((mask_crop_filled > 0).astype(int))
    if props:
        bbox = props[0].bbox
        embryo_len_px = bbox[2] - bbox[0]
        embryo_width_px = bbox[3] - bbox[1]
        fill_h = embryo_len_px / outshape[0]
    else:
        embryo_len_px = 0
        fill_h = 0

    return {
        "cropped": im_crop,
        "fill_fraction": fill_h,
        "area_retained": np.sum(mask_crop > 0.5) / np.sum(mask_emb_rot > 0.5) if np.sum(mask_emb_rot > 0.5) > 0 else 0,
        "out_of_frame": oof
    }


if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    output_dir = Path(__file__).parent

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
        expr_dir = root / 'sam2_pipeline_files' / 'sam2_expr_files'
        if expr_dir.exists():
            print(f"\nAvailable metadata files:")
            for f in sorted(expr_dir.glob('sam2_metadata_*.csv'))[:5]:
                print(f"   - {f.name}")
        exit(1)

    df_meta = pd.concat([pd.read_csv(f) for f in available_files], ignore_index=True)

    test_snip_ids = [
        "20251106_A02_e01_t0049",
        "20251106_A03_e01_t0085",
        "20250512_A03_e01_t0112",
        "20251106_H12_e01_t0093",
    ]

    print("="*80)
    print("EMBRYO ROTATION/CROPPING FIX - FINAL TEST")
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
        # experiment_date = experiment_id (they're the same - the date)
        if "experiment_date" not in row:
            row["experiment_date"] = row["experiment_id"]

        # region_label = extract from embryo_id (e.g., "20251106_A02_e01" -> 1)
        if "region_label" not in row:
            embryo_id = row["embryo_id"]
            region_label = int(embryo_id.split("_e")[-1])  # Extract eNN -> NN
            row["region_label"] = region_label

        # Load mask (exactly like build03 does at line 287-298)
        try:
            mask_path = resolve_sandbox_embryo_mask_from_csv(root, row)
            im_mask_int = io.imread(mask_path)
            lbi = int(row["region_label"])
            im_mask = ((im_mask_int == lbi) * 255).astype(np.uint8)
            im_mask, _ = clean_embryo_mask(im_mask, verbose=False)
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
        except Exception as e:
            print(f"  ❌ FF load failed: {e}")
            continue

        # Save mask overlay
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            overlay_dir = output_dir / "mask_overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 1, figsize=(8, 6))

            # Display FF image
            axes.imshow(im_ff, cmap='gray')
            # Overlay embryo mask in blue
            axes.contour(im_mask > 127, levels=[0.5], colors='blue', linewidths=2)
            axes.set_title(f"Embryo: {snip_id}\nMask Overlay")
            axes.axis('off')

            plt.tight_layout()
            plt.savefig(overlay_dir / f"{snip_id}_mask_overlay.png", dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved mask overlay")
        except Exception as e:
            print(f"  ⚠️  Failed to save mask overlay: {e}")

        # Process both scales
        print(f"  Processing 6.5 μm/px...")
        res_old = process_at_scale(im_ff, im_mask, outscale=6.5)
        io.imsave(output_dir / "old_6.5um" / f"{snip_id}.jpg", res_old["cropped"], check_contrast=False)

        print(f"  Processing 7.8 μm/px...")
        res_new = process_at_scale(im_ff, im_mask, outscale=7.8)
        io.imsave(output_dir / "new_7.8um" / f"{snip_id}.jpg", res_new["cropped"], check_contrast=False)

        print(f"  OLD: fill={res_old['fill_fraction']:.1%}, retained={res_old['area_retained']:.1%}, oof={res_old['out_of_frame']}")
        print(f"  NEW: fill={res_new['fill_fraction']:.1%}, retained={res_new['area_retained']:.1%}, oof={res_new['out_of_frame']}")

        results.append({
            "snip_id": snip_id,
            "old_fill": res_old["fill_fraction"],
            "old_retained": res_old["area_retained"],
            "old_oof": res_old["out_of_frame"],
            "new_fill": res_new["fill_fraction"],
            "new_retained": res_new["area_retained"],
            "new_oof": res_new["out_of_frame"],
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\n✓ Results saved to {output_dir / 'metrics.csv'}")
    print(df_res.to_string())
