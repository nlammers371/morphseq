#!/usr/bin/env python3
"""
Simple test of mask encoding/decoding utilities.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.mask_utils import encode_mask_rle, decode_mask_rle

def test_mask_utils():
    """Test basic mask encoding and decoding."""
    
    # Create a simple square mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1  # 50x50 square in center
    
    print(f"Original mask: shape={mask.shape}, sum={mask.sum()}")
    
    # Save original
    cv2.imwrite("temp/original_mask.png", mask * 255)
    print("Saved original_mask.png")
    
    # Encode
    try:
        encoded = encode_mask_rle(mask)
        print(f"Encoded: {encoded}")
    except Exception as e:
        print(f"Encoding failed: {e}")
        return 1
    
    # Decode
    try:
        decoded = decode_mask_rle(encoded)
        print(f"Decoded mask: shape={decoded.shape}, sum={decoded.sum()}")
    except Exception as e:
        print(f"Decoding failed: {e}")
        return 1
    
    # Save decoded
    cv2.imwrite("temp/decoded_mask.png", decoded * 255)
    print("Saved decoded_mask.png")
    
    # Check if they match
    if np.array_equal(mask, decoded):
        print("✅ Masks match perfectly")
    else:
        print("❌ Masks don't match")
        print(f"Difference: {np.sum(mask != decoded)} pixels")
    
    return 0

if __name__ == "__main__":
    Path("temp").mkdir(exist_ok=True)
    sys.exit(test_mask_utils())
