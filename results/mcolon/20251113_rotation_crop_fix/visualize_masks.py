#!/usr/bin/env python3
"""
Visualize masks from SAM2 metadata - show embryo masks clearly.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.build.build03A_process_images import resolve_sandbox_embryo_mask_from_csv
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask


if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    output_dir = Path(__file__).parent / "mask_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM2 metadata
    metadata_files = [
        root / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20251106.csv",
        root / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20250512.csv",
    ]

    available_files = [f for f in metadata_files if f.exists()]
    if not available_files:
        print(f"❌ ERROR: No metadata files found!")
        exit(1)

    df_meta = pd.concat([pd.read_csv(f) for f in available_files], ignore_index=True)

    # Test snip IDs
    test_snip_ids = [
        "20251106_A02_e01_t0049",
        "20251106_A03_e01_t0085",
        "20250512_A03_e01_t0112",
        "20251106_H12_e01_t0093",
    ]

    print("="*80)
    print("MASK DIAGNOSTICS - VISUAL INSPECTION")
    print("="*80)

    for snip_id in test_snip_ids:
        print(f"\n{snip_id}...")
        matching = df_meta[df_meta["snip_id"] == snip_id]
        if len(matching) == 0:
            print(f"  ❌ Not found")
            continue

        row = matching.iloc[0].to_dict()

        # Add missing columns
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
            im_mask_clean, _ = clean_embryo_mask(im_mask, verbose=False)
            print(f"  ✓ Mask loaded from {mask_path.name}")
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

        # Create detailed mask visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # Row 1: Raw mask images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(im_mask_int, cmap='viridis')
        ax1.set_title(f"Raw Mask Integer Labels\nUnique values: {np.unique(im_mask_int)}")
        plt.colorbar(ax1.images[0], ax=ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(im_mask, cmap='gray')
        ax2.set_title(f"Extracted Region (label={lbi})\nPixels: {np.sum(im_mask > 0)}")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(im_mask_clean, cmap='gray')
        ax3.set_title("After Cleaning")

        # Row 2: FF image with mask overlays
        ax4 = fig.add_subplot(gs[1, :])
        ax4.imshow(im_ff, cmap='gray')
        # Red overlay for raw mask
        mask_rgb = np.zeros((*im_mask.shape, 3))
        mask_rgb[im_mask > 127, 0] = 1
        ax4.imshow(mask_rgb, alpha=0.3)
        # Yellow contour
        ax4.contour(im_mask > 127, levels=[0.5], colors='yellow', linewidths=2)
        ax4.set_title(f"FF Image with Mask Overlay (red) and Contour (yellow)")
        ax4.axis('off')

        # Row 3: Mask statistics
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        stats_text = f"""
MASK DIAGNOSTICS FOR {snip_id}

Image ID: {row['image_id']}
Embryo ID: {row['embryo_id']}
Region Label: {lbi}

Raw Mask:
  • Shape: {im_mask_int.shape}
  • Unique values: {np.unique(im_mask_int)}
  • Min/Max: {im_mask_int.min()}/{im_mask_int.max()}

Extracted Mask (label={lbi}):
  • Pixels with label: {np.sum(im_mask > 0):,}
  • Percentage of image: {100 * np.sum(im_mask > 0) / im_mask.size:.2f}%

Cleaned Mask:
  • Pixels after cleaning: {np.sum(im_mask_clean > 0):,}
  • Pixels removed: {np.sum(im_mask > 0) - np.sum(im_mask_clean > 0):,}

FF Image:
  • Shape: {im_ff.shape}
  • Dtype: {im_ff.dtype}
  • Min/Max: {im_ff.min()}/{im_ff.max()}
        """

        ax5.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.savefig(output_dir / f"{snip_id}_mask_diagnostic.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved diagnostic figure")

    print(f"\n{'='*80}")
    print(f"✓ Diagnostic figures saved to {output_dir}")
    print(f"{'='*80}")
