"""
Create a dummy GSAM JSON file that deliberately violates each QC check
for testing purposes.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from pycocotools import mask as mask_utils

def create_dummy_gsam_with_violations(output_path: str = "test_gsam_violations.json"):
    """
    Create a GSAM JSON file with deliberate QC violations for testing.
    
    Violations included:
    1. HIGH_SEGMENTATION_VAR - Embryo with wildly varying areas across frames
    2. MASK_ON_EDGE - Mask touching image boundaries
    3. OVERLAPPING_MASKS - Two masks with significant overlap
    4. LARGE_MASK - Mask covering >15% of image
    5. SMALL_MASK - Mask covering <0.1% of image
    6. DETECTION_FAILURE - Missing embryos in some frames
    7. DISCONTINUOUS_MASK - Mask with multiple disconnected components
    """
    
    # Image dimensions for our test
    height, width = 512, 512
    
    # Helper function to create RLE mask from binary array
    def create_rle(binary_mask):
        import base64
        fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
        rle = mask_utils.encode(fortran_mask)
        return {
            'counts': base64.b64encode(rle['counts']).decode('utf-8'),  # Base64 encode bytes to string
            'size': rle['size'],
            'format': 'rle'
        }
    
    # Create test masks with violations
    
    # 1. Normal mask (no violations) - for embryo e01 in most frames
    normal_mask = np.zeros((height, width), dtype=np.uint8)
    normal_mask[200:300, 200:300] = 1  # 100x100 = 10,000 pixels (~3.8% of image)
    normal_rle = create_rle(normal_mask)
    normal_area = float(np.sum(normal_mask))
    
    # 2. Edge-touching mask - for embryo e02
    edge_mask = np.zeros((height, width), dtype=np.uint8)
    edge_mask[0:100, 0:100] = 1  # Top-left corner (touches edges)
    edge_rle = create_rle(edge_mask)
    edge_area = float(np.sum(edge_mask))
    
    # 3. Large mask - for embryo e03  
    large_mask = np.zeros((height, width), dtype=np.uint8)
    large_mask[100:400, 100:400] = 1  # 300x300 = 90,000 pixels (~34% of image)
    large_rle = create_rle(large_mask)
    large_area = float(np.sum(large_mask))
    
    # 4. Small mask - for embryo e04
    small_mask = np.zeros((height, width), dtype=np.uint8)
    small_mask[250:252, 250:252] = 1  # 2x2 = 4 pixels (~0.0015% of image)
    small_rle = create_rle(small_mask)
    small_area = float(np.sum(small_mask))
    
    # 5. Discontinuous mask - for embryo e05
    discontinuous_mask = np.zeros((height, width), dtype=np.uint8)
    discontinuous_mask[100:150, 100:150] = 1  # First component
    discontinuous_mask[300:350, 300:350] = 1  # Second component (disconnected)
    discontinuous_mask[400:420, 100:120] = 1  # Third component
    discontinuous_rle = create_rle(discontinuous_mask)
    discontinuous_area = float(np.sum(discontinuous_mask))
    
    # 6. Overlapping masks - for embryos e06 and e07
    overlap_mask1 = np.zeros((height, width), dtype=np.uint8)
    overlap_mask1[200:280, 200:280] = 1
    overlap_rle1 = create_rle(overlap_mask1)
    overlap_area1 = float(np.sum(overlap_mask1))
    
    overlap_mask2 = np.zeros((height, width), dtype=np.uint8)
    overlap_mask2[240:320, 240:320] = 1  # Overlaps with mask1
    overlap_rle2 = create_rle(overlap_mask2)
    overlap_area2 = float(np.sum(overlap_mask2))
    
    # 7. Variable size masks for segmentation variability - embryo e08
    var_small = np.zeros((height, width), dtype=np.uint8)
    var_small[200:250, 200:250] = 1  # 50x50 = 2,500 pixels
    var_small_rle = create_rle(var_small)
    var_small_area = float(np.sum(var_small))
    
    var_large = np.zeros((height, width), dtype=np.uint8)
    var_large[180:320, 180:320] = 1  # 140x140 = 19,600 pixels (>7x larger!)
    var_large_rle = create_rle(var_large)
    var_large_area = float(np.sum(var_large))
    
    # Build the GSAM structure
    gsam_data = {
        "file_info": {
            "created": datetime.now().isoformat(),
            "purpose": "QC testing with deliberate violations",
            "last_updated": datetime.now().isoformat()
        },
        "experiments": {
            "20240411_test": {
                "videos": {
                    "20240411_test_H01": {
                        "image_ids": {}
                    }
                }
            }
        }
    }
    
    # Add 10 frames to the video
    video_data = gsam_data["experiments"]["20240411_test"]["videos"]["20240411_test_H01"]["image_ids"]
    
    for frame_num in range(10):
        image_id = f"20240411_test_H01_t{frame_num:04d}"
        video_data[image_id] = {
            "embryos": {}
        }
        
        # Frame-specific violations
        if frame_num == 0:
            # Frame 0: All embryos present with various violations
            
            # Normal embryo e01
            video_data[image_id]["embryos"]["e01"] = {
                "snip_id": f"20240411_test_H01_e01_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 300/width, 300/height],
                "segmentation": normal_rle,
                "segmentation_format": "rle",
                "area": normal_area
            }
            
            # Edge-touching embryo e02
            video_data[image_id]["embryos"]["e02"] = {
                "snip_id": f"20240411_test_H01_e02_s{frame_num:04d}",
                "bbox": [0.0, 0.0, 100/width, 100/height],  # Touches top and left edges
                "segmentation": edge_rle,
                "segmentation_format": "rle",
                "area": edge_area
            }
            
            # Large embryo e03
            video_data[image_id]["embryos"]["e03"] = {
                "snip_id": f"20240411_test_H01_e03_s{frame_num:04d}",
                "bbox": [100/width, 100/height, 400/width, 400/height],
                "segmentation": large_rle,
                "segmentation_format": "rle",
                "area": large_area
            }
            
            # Small embryo e04
            video_data[image_id]["embryos"]["e04"] = {
                "snip_id": f"20240411_test_H01_e04_s{frame_num:04d}",
                "bbox": [250/width, 250/height, 252/width, 252/height],
                "segmentation": small_rle,
                "segmentation_format": "rle",
                "area": small_area
            }
            
            # Discontinuous embryo e05
            video_data[image_id]["embryos"]["e05"] = {
                "snip_id": f"20240411_test_H01_e05_s{frame_num:04d}",
                "bbox": [100/width, 100/height, 420/width, 350/height],
                "segmentation": discontinuous_rle,
                "segmentation_format": "rle",
                "area": discontinuous_area
            }
            
            # Overlapping embryos e06 and e07
            video_data[image_id]["embryos"]["e06"] = {
                "snip_id": f"20240411_test_H01_e06_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 280/width, 280/height],
                "segmentation": overlap_rle1,
                "segmentation_format": "rle",
                "area": overlap_area1
            }
            
            video_data[image_id]["embryos"]["e07"] = {
                "snip_id": f"20240411_test_H01_e07_s{frame_num:04d}",
                "bbox": [240/width, 240/height, 320/width, 320/height],
                "segmentation": overlap_rle2,
                "segmentation_format": "rle",
                "area": overlap_area2
            }
            
            # High variability embryo e08 (starts small)
            video_data[image_id]["embryos"]["e08"] = {
                "snip_id": f"20240411_test_H01_e08_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 250/width, 250/height],
                "segmentation": var_small_rle,
                "segmentation_format": "rle",
                "area": var_small_area
            }
            
        elif frame_num == 1:
            # Frame 1: e08 suddenly gets HUGE (segmentation variability)
            video_data[image_id]["embryos"]["e08"] = {
                "snip_id": f"20240411_test_H01_e08_s{frame_num:04d}",
                "bbox": [180/width, 180/height, 320/width, 320/height],
                "segmentation": var_large_rle,
                "segmentation_format": "rle",
                "area": var_large_area  # 7x larger than frame 0!
            }
            
            # Keep e01 consistent
            video_data[image_id]["embryos"]["e01"] = {
                "snip_id": f"20240411_test_H01_e01_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 300/width, 300/height],
                "segmentation": normal_rle,
                "segmentation_format": "rle",
                "area": normal_area
            }
            
            # Add e02 again to ensure it gets processed
            video_data[image_id]["embryos"]["e02"] = {
                "snip_id": f"20240411_test_H01_e02_s{frame_num:04d}",
                "bbox": [0.0, 0.0, 100/width, 100/height],
                "segmentation": edge_rle,
                "segmentation_format": "rle",
                "area": edge_area
            }
            
        elif frame_num == 2:
            # Frame 2: e08 back to small (more variability)
            video_data[image_id]["embryos"]["e08"] = {
                "snip_id": f"20240411_test_H01_e08_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 250/width, 250/height],
                "segmentation": var_small_rle,
                "segmentation_format": "rle",
                "area": var_small_area  # Back to small
            }
            
            # e01 stays consistent
            video_data[image_id]["embryos"]["e01"] = {
                "snip_id": f"20240411_test_H01_e01_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 300/width, 300/height],
                "segmentation": normal_rle,
                "segmentation_format": "rle",
                "area": normal_area
            }
            
            # Add key violation embryos to ensure processing
            video_data[image_id]["embryos"]["e03"] = {
                "snip_id": f"20240411_test_H01_e03_s{frame_num:04d}",
                "bbox": [100/width, 100/height, 400/width, 400/height],
                "segmentation": large_rle,
                "segmentation_format": "rle",
                "area": large_area
            }
            
            video_data[image_id]["embryos"]["e04"] = {
                "snip_id": f"20240411_test_H01_e04_s{frame_num:04d}",
                "bbox": [250/width, 250/height, 252/width, 252/height],
                "segmentation": small_rle,
                "segmentation_format": "rle",
                "area": small_area
            }
            
            video_data[image_id]["embryos"]["e05"] = {
                "snip_id": f"20240411_test_H01_e05_s{frame_num:04d}",
                "bbox": [100/width, 100/height, 420/width, 350/height],
                "segmentation": discontinuous_rle,
                "segmentation_format": "rle",
                "area": discontinuous_area
            }
            
            video_data[image_id]["embryos"]["e06"] = {
                "snip_id": f"20240411_test_H01_e06_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 280/width, 280/height],
                "segmentation": overlap_rle1,
                "segmentation_format": "rle",
                "area": overlap_area1
            }
            
            video_data[image_id]["embryos"]["e07"] = {
                "snip_id": f"20240411_test_H01_e07_s{frame_num:04d}",
                "bbox": [240/width, 240/height, 320/width, 320/height],
                "segmentation": overlap_rle2,
                "segmentation_format": "rle",
                "area": overlap_area2
            }
            
        elif frame_num == 3:
            # Frame 3: e08 huge again
            video_data[image_id]["embryos"]["e08"] = {
                "snip_id": f"20240411_test_H01_e08_s{frame_num:04d}",
                "bbox": [180/width, 180/height, 320/width, 320/height],
                "segmentation": var_large_rle,
                "segmentation_format": "rle",
                "area": var_large_area
            }
            
            # e01 consistent
            video_data[image_id]["embryos"]["e01"] = {
                "snip_id": f"20240411_test_H01_e01_s{frame_num:04d}",
                "bbox": [200/width, 200/height, 300/width, 300/height],
                "segmentation": normal_rle,
                "segmentation_format": "rle",
                "area": normal_area
            }
            
        else:
            # Frames 4-9: Create some empty frames for detection failure  
            if frame_num >= 7:
                # Frames 7-9: Empty frames (detection failure)
                pass  # No embryos added - this will trigger DETECTION_FAILURE
            else:
                # Frames 4-6: Just keep e01 and e08 with e08 alternating sizes
                video_data[image_id]["embryos"]["e01"] = {
                    "snip_id": f"20240411_test_H01_e01_s{frame_num:04d}",
                    "bbox": [200/width, 200/height, 300/width, 300/height],
                    "segmentation": normal_rle,
                    "segmentation_format": "rle",
                    "area": normal_area
                }
                
                # e08 alternates between small and large
                if frame_num % 2 == 0:
                    video_data[image_id]["embryos"]["e08"] = {
                        "snip_id": f"20240411_test_H01_e08_s{frame_num:04d}",
                        "bbox": [200/width, 200/height, 250/width, 250/height],
                        "segmentation": var_small_rle,
                        "segmentation_format": "rle",
                        "area": var_small_area
                    }
                else:
                    video_data[image_id]["embryos"]["e08"] = {
                        "snip_id": f"20240411_test_H01_e08_s{frame_num:04d}",
                        "bbox": [180/width, 180/height, 320/width, 320/height],
                        "segmentation": var_large_rle,
                        "segmentation_format": "rle",
                        "area": var_large_area
                    }
    
    # Save the file
    output = Path(output_path)
    with open(output, 'w') as f:
        json.dump(gsam_data, f, indent=2)
    
    print(f"✅ Created test GSAM file: {output}")
    print("\nExpected violations:")
    print("1. HIGH_SEGMENTATION_VAR: embryo e08 (varies 7x in size)")
    print("2. MASK_ON_EDGE: embryo e02 (touches top-left corner)")
    print("3. OVERLAPPING_MASKS: embryos e06 & e07 (40x40 pixel overlap)")
    print("4. LARGE_MASK: embryo e03 (~34% of image)")
    print("5. SMALL_MASK: embryo e04 (~0.0015% of image)")
    print("6. DETECTION_FAILURE: frames 1-9 missing embryos e02-e07")
    print("7. DISCONTINUOUS_MASK: embryo e05 (3 separate components)")
    
    return str(output)


def test_qc_with_violations():
    """Test the QC system with the violation file."""
    import sys
    import importlib.util
    
    # Load the QC module directly
    qc_path = Path(__file__).parent.parent / "pipelines" / "05_sam2_qc_analysis.py"
    spec = importlib.util.spec_from_file_location("qc_analysis", qc_path)
    qc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qc_module)
    GSAMQualityControl = qc_module.GSAMQualityControl
    
    # Create test file
    test_file = create_dummy_gsam_with_violations("test_gsam_violations.json")
    
    print("\n" + "="*60)
    print("RUNNING QC CHECKS ON TEST FILE")
    print("="*60)
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=True)
    
    # Run all checks (process_all=True to ensure we check everything)
    qc.run_all_checks(author="test_violations", process_all=True, save_in_place=False)
    
    # Print detailed results
    print("\n" + "="*60)
    print("QC RESULTS")
    print("="*60)
    
    qc.print_summary()
    
    # Check each flag type
    print("\n" + "="*60)
    print("DETAILED FLAG RESULTS")
    print("="*60)
    
    flag_types = [
        "HIGH_SEGMENTATION_VAR_EMBRYO",
        "HIGH_SEGMENTATION_VAR_SNIP", 
        "MASK_ON_EDGE",
        "OVERLAPPING_MASKS",
        "LARGE_MASK",
        "SMALL_MASK",
        "DETECTION_FAILURE",
        "DISCONTINUOUS_MASK"
    ]
    
    for flag_type in flag_types:
        flags = qc.get_flags_by_type(flag_type)
        if flags:
            print(f"\n{flag_type}: {len(flags)} violations found")
            # Show first violation details
            if flags:
                print(f"  Example: {json.dumps(flags[0], indent=4)[:200]}...")
        else:
            print(f"\n{flag_type}: ❌ NO VIOLATIONS FOUND (should have found some!)")
    
    return qc


def validate_qc_results(qc):
    """Simple validation that expected violations were found."""
    expected = {"MASK_ON_EDGE": 1, "LARGE_MASK": 1, "SMALL_MASK": 1, "OVERLAPPING_MASKS": 1, "DISCONTINUOUS_MASK": 1}
    summary = qc.get_flags_summary()
    flags = summary.get('flag_categories', {})
    
    print(f"\n{'='*40}\nVALIDATION RESULTS\n{'='*40}")
    all_passed = True
    for flag_type, expected_count in expected.items():
        actual = flags.get(flag_type, 0)
        status = "PASS" if actual >= expected_count else "FAIL"
        print(f"{flag_type}: {actual} found (expected {expected_count}) - {status}")
        if status == "FAIL":
            all_passed = False
    
    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

if __name__ == "__main__":
    qc = test_qc_with_violations()
    success = validate_qc_results(qc)
    exit(0 if success else 1)