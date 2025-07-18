#!/usr/bin/env python3
"""
Test script to verify RLE mask conversion functionality.

This script creates dummy binary masks and tests the RLE encoding/decoding process
to ensure it works correctly for the SAM2 video processing pipeline.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add the sandbox root directory to path
script_dir = Path(__file__).parent
sandbox_root = script_dir.parent
sys.path.append(str(sandbox_root))

def convert_sam2_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert SAM2 binary mask to RLE format for compact storage."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        # Fallback to simple dict if pycocotools not available
        print("Warning: pycocotools not available, using simple mask storage")
        return {
            'format': 'simple_mask',
            'size': binary_mask.shape,
            'data': binary_mask.flatten().tolist()
        }
    
    # Convert to uint8 if needed
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Convert to Fortran order for COCO tools
    binary_mask_fortran = np.asfortranarray(binary_mask)
    
    # Encode to RLE
    rle = mask_utils.encode(binary_mask_fortran)
    
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


def decode_rle_to_mask(rle_data: dict) -> np.ndarray:
    """Decode RLE format back to binary mask."""
    if rle_data.get('format') == 'simple_mask':
        # Handle simple mask format
        shape = rle_data['size']
        data = rle_data['data']
        return np.array(data).reshape(shape).astype(np.uint8)
    
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise ImportError("pycocotools required for RLE decoding")
    
    # Convert string back to bytes for decoding
    rle_data_copy = rle_data.copy()
    rle_data_copy['counts'] = rle_data_copy['counts'].encode('utf-8')
    
    # Decode RLE back to binary mask
    binary_mask = mask_utils.decode(rle_data_copy)
    
    return binary_mask


def create_test_masks():
    """Create various test masks for validation."""
    test_masks = {}
    
    # Test 1: Simple rectangular mask
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:80, 30:70] = 1
    test_masks['rectangle'] = mask1
    
    # Test 2: Circular mask
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    y, x = np.ogrid[:100, :100]
    center_y, center_x = 50, 50
    mask2[(x - center_x)**2 + (y - center_y)**2 <= 25**2] = 1
    test_masks['circle'] = mask2
    
    # Test 3: Multiple disconnected regions
    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[10:30, 10:30] = 1  # Top-left square
    mask3[70:90, 70:90] = 1  # Bottom-right square
    mask3[40:60, 20:40] = 1  # Middle-left rectangle
    test_masks['multiple_regions'] = mask3
    
    # Test 4: Complex shape with holes
    mask4 = np.zeros((100, 100), dtype=np.uint8)
    mask4[20:80, 20:80] = 1  # Large square
    mask4[30:70, 30:70] = 0  # Hole in the middle
    mask4[40:60, 40:60] = 1  # Small square in the hole
    test_masks['complex_with_holes'] = mask4
    
    # Test 5: Edge case - empty mask
    mask5 = np.zeros((100, 100), dtype=np.uint8)
    test_masks['empty'] = mask5
    
    # Test 6: Edge case - full mask
    mask6 = np.ones((100, 100), dtype=np.uint8)
    test_masks['full'] = mask6
    
    # Test 7: Different sizes
    mask7 = np.zeros((50, 200), dtype=np.uint8)
    mask7[10:40, 50:150] = 1
    test_masks['different_size'] = mask7
    
    return test_masks


def test_rle_conversion():
    """Test RLE conversion with various mask types."""
    print("🧪 Testing RLE Mask Conversion")
    print("=" * 40)
    
    # Create test masks
    test_masks = create_test_masks()
    
    results = {}
    all_passed = True
    
    for mask_name, original_mask in test_masks.items():
        print(f"\n📋 Testing {mask_name} mask...")
        print(f"   Original shape: {original_mask.shape}")
        print(f"   Original dtype: {original_mask.dtype}")
        print(f"   Non-zero pixels: {np.sum(original_mask)}")
        
        try:
            # Test encoding
            rle_data = convert_sam2_mask_to_rle(original_mask)
            print(f"   RLE format: {rle_data.get('format', 'coco_rle')}")
            
            if rle_data.get('format') == 'simple_mask':
                # Simple format
                size_bytes = len(str(rle_data['data']))
                print(f"   Simple format size: {size_bytes} bytes")
            else:
                # COCO RLE format
                counts_length = len(rle_data['counts'])
                print(f"   RLE counts length: {counts_length} characters")
            
            # Test JSON serialization
            json_str = json.dumps(rle_data)
            json_size = len(json_str)
            original_size = original_mask.nbytes
            compression_ratio = original_size / json_size
            print(f"   JSON size: {json_size} bytes")
            print(f"   Original size: {original_size} bytes")
            print(f"   Compression ratio: {compression_ratio:.2f}x")
            
            # Test decoding
            decoded_mask = decode_rle_to_mask(rle_data)
            print(f"   Decoded shape: {decoded_mask.shape}")
            print(f"   Decoded dtype: {decoded_mask.dtype}")
            print(f"   Decoded non-zero pixels: {np.sum(decoded_mask)}")
            
            # Verify reconstruction
            if np.array_equal(original_mask, decoded_mask):
                print("   ✅ Perfect reconstruction!")
                test_passed = True
            else:
                print("   ❌ Reconstruction failed!")
                print(f"   Difference: {np.sum(original_mask != decoded_mask)} pixels")
                test_passed = False
                all_passed = False
            
            results[mask_name] = {
                'passed': test_passed,
                'original_size': original_size,
                'compressed_size': json_size,
                'compression_ratio': compression_ratio,
                'rle_format': rle_data.get('format', 'coco_rle')
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[mask_name] = {
                'passed': False,
                'error': str(e)
            }
            all_passed = False
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 40)
    passed_tests = sum(1 for r in results.values() if r.get('passed', False))
    total_tests = len(results)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed!")
    
    # Compression analysis
    print(f"\n📦 Compression Analysis")
    print("=" * 40)
    successful_results = [r for r in results.values() if r.get('passed', False)]
    if successful_results:
        avg_compression = np.mean([r['compression_ratio'] for r in successful_results])
        total_original = sum(r['original_size'] for r in successful_results)
        total_compressed = sum(r['compressed_size'] for r in successful_results)
        overall_compression = total_original / total_compressed
        
        print(f"Average compression ratio: {avg_compression:.2f}x")
        print(f"Overall compression ratio: {overall_compression:.2f}x")
        print(f"Total original size: {total_original:,} bytes")
        print(f"Total compressed size: {total_compressed:,} bytes")
        print(f"Space saved: {total_original - total_compressed:,} bytes ({(1 - total_compressed/total_original)*100:.1f}%)")
    
    return all_passed, results


def create_visual_test():
    """Create a visual test to verify masks look correct."""
    print(f"\n🎨 Creating Visual Test")
    print("=" * 40)
    
    test_masks = create_test_masks()
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Test a few masks visually
    test_cases = ['rectangle', 'circle', 'multiple_regions', 'complex_with_holes']
    
    for mask_name in test_cases:
        if mask_name not in test_masks:
            continue
            
        original_mask = test_masks[mask_name]
        
        # Convert to RLE and back
        rle_data = convert_sam2_mask_to_rle(original_mask)
        reconstructed_mask = decode_rle_to_mask(rle_data)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original mask
        axes[0].imshow(original_mask, cmap='gray')
        axes[0].set_title(f'Original {mask_name}')
        axes[0].axis('off')
        
        # Reconstructed mask
        axes[1].imshow(reconstructed_mask, cmap='gray')
        axes[1].set_title(f'Reconstructed {mask_name}')
        axes[1].axis('off')
        
        # Difference
        diff = np.abs(original_mask.astype(float) - reconstructed_mask.astype(float))
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f'Difference (max: {np.max(diff)})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / f"rle_test_{mask_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved visual test: {output_path}")
    
    print(f"   📁 Visual tests saved to: {output_dir}")


def main():
    """Run all tests."""
    print("🔬 RLE Mask Conversion Test Suite")
    print("=" * 50)
    
    # Check if pycocotools is available
    try:
        from pycocotools import mask as mask_utils
        print("✅ pycocotools available - testing COCO RLE format")
    except ImportError:
        print("⚠️  pycocotools not available - testing simple format fallback")
    
    # Run conversion tests
    all_passed, results = test_rle_conversion()
    
    # Create visual tests
    try:
        create_visual_test()
    except Exception as e:
        print(f"⚠️  Visual test failed: {e}")
    
    # Final result
    print(f"\n🎯 Final Result")
    print("=" * 20)
    if all_passed:
        print("✅ All RLE conversion tests passed!")
        print("   The RLE mask conversion is working correctly.")
        return 0
    else:
        print("❌ Some tests failed!")
        print("   Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
