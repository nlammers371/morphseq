"""
Simple test to decode and visualize RLE masks using existing mask_utils

Required CSV columns:
- mask_rle: Base64-encoded RLE compressed mask data
- mask_height_px, mask_width_px: Dimensions needed for decoding RLE
- height_um, width_um: For pixel-to-micron conversion
  - um_per_pixel_y = height_um / height_px
  - um_per_pixel_x = width_um / width_px
  - Useful for reporting curvature in physical units (1/μm) instead of (1/pixel)
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


def test_mask_loading():
    """Load and visualize a few masks from the CSV."""

    # Load the CSV
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251020.csv")

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows")

    # Test with first 5 embryos
    n_test = 5

    fig, axes = plt.subplots(1, n_test, figsize=(20, 4))

    for i in range(n_test):
        row = df.iloc[i]

        print(f"\n--- Test {i+1} ---")
        print(f"Embryo ID: {row['embryo_id']}")
        print(f"Snip ID: {row['snip_id']}")

        # Decode the mask
        try:
            # Create RLE dict from CSV columns
            rle_data = {
                'counts': row['mask_rle'],
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])]
            }

            # Decode using existing utility
            mask = decode_mask_rle(rle_data)

            print(f"Mask shape: {mask.shape}")
            print(f"Mask dtype: {mask.dtype}")
            print(f"Mask values: min={mask.min()}, max={mask.max()}")
            print(f"Non-zero pixels: {np.sum(mask > 0)}")

            # Visualize
            axes[i].imshow(mask, cmap='gray')
            axes[i].set_title(f"{row['embryo_id']}\nFrame {row['frame_index']}")
            axes[i].axis('off')

            print("✓ Successfully decoded and visualized")

        except Exception as e:
            print(f"✗ Error decoding mask: {e}")
            axes[i].text(0.5, 0.5, f"Error:\n{str(e)}",
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')

    plt.tight_layout()

    # Save the figure
    output_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/test_masks.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.close()

    return df


if __name__ == "__main__":
    print("="*60)
    print("Testing RLE Mask Decoding")
    print("="*60)

    df = test_mask_loading()

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
