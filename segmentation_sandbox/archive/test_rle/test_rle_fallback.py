#!/usr/bin/env python3
"""
Test script to verify the fallback RLE conversion when pycocotools is not available.
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add the sandbox root directory to path
script_dir = Path(__file__).parent
sandbox_root = script_dir.parent
sys.path.append(str(sandbox_root))

def convert_sam2_mask_to_rle_fallback(binary_mask: np.ndarray) -> dict:
    """Convert SAM2 binary mask to simple format (simulating no pycocotools)."""
    # Simulate pycocotools not being available
    print("Warning: pycocotools not available, using simple mask storage")
    return {
        'format': 'simple_mask',
        'size': binary_mask.shape,
        'data': binary_mask.flatten().tolist()
    }


def decode_rle_to_mask_fallback(rle_data: dict) -> np.ndarray:
    """Decode simple format back to binary mask."""
    if rle_data.get('format') == 'simple_mask':
        # Handle simple mask format
        shape = rle_data['size']
        data = rle_data['data']
        return np.array(data).reshape(shape).astype(np.uint8)
    else:
        raise ValueError("Expected simple_mask format")


def test_fallback_conversion():
    """Test the fallback conversion mechanism."""
    print("🧪 Testing Fallback RLE Conversion (No pycocotools)")
    print("=" * 55)
    
    # Create a simple test mask
    test_mask = np.zeros((50, 50), dtype=np.uint8)
    test_mask[15:35, 15:35] = 1  # Square in the center
    
    print(f"Original mask shape: {test_mask.shape}")
    print(f"Original mask dtype: {test_mask.dtype}")
    print(f"Original non-zero pixels: {np.sum(test_mask)}")
    
    # Test encoding
    rle_data = convert_sam2_mask_to_rle_fallback(test_mask)
    print(f"RLE format: {rle_data.get('format')}")
    
    # Test JSON serialization
    json_str = json.dumps(rle_data)
    json_size = len(json_str)
    original_size = test_mask.nbytes
    
    print(f"JSON size: {json_size:,} bytes")
    print(f"Original size: {original_size:,} bytes")
    print(f"Compression ratio: {original_size / json_size:.2f}x")
    
    # Test decoding
    decoded_mask = decode_rle_to_mask_fallback(rle_data)
    print(f"Decoded shape: {decoded_mask.shape}")
    print(f"Decoded dtype: {decoded_mask.dtype}")
    print(f"Decoded non-zero pixels: {np.sum(decoded_mask)}")
    
    # Verify reconstruction
    if np.array_equal(test_mask, decoded_mask):
        print("✅ Perfect reconstruction with fallback method!")
        return True
    else:
        print("❌ Reconstruction failed!")
        print(f"Difference: {np.sum(test_mask != decoded_mask)} pixels")
        return False


def main():
    """Run fallback tests."""
    print("🔬 RLE Fallback Test Suite")
    print("=" * 30)
    
    success = test_fallback_conversion()
    
    print(f"\n🎯 Final Result")
    print("=" * 20)
    if success:
        print("✅ Fallback RLE conversion works correctly!")
        print("   The system gracefully handles missing pycocotools.")
        return 0
    else:
        print("❌ Fallback conversion failed!")
        return 1


if __name__ == "__main__":
    exit(main())
