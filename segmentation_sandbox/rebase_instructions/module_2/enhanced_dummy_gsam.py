"""
Enhanced Dummy GSAM Generator for QC Testing

Creates a comprehensive test file that deliberately includes ALL quality control
violations to validate the ported QC system.

Fixes issues from the original dummy generator:
1. Ensures proper data structure that matches QC expectations
2. Includes ALL violation types including small masks
3. Creates realistic segmentation variability patterns
4. Provides clear violation mapping for validation
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Only import if available (graceful degradation)
try:
    from pycocotools import mask as mask_utils
    _HAS_MASK_UTILS = True
except ImportError:
    _HAS_MASK_UTILS = False
    print("⚠️ pycocotools not available - using simplified mask encoding")

def create_rle_encoding(binary_mask):
    """Create RLE encoding compatible with QC system."""
    if _HAS_MASK_UTILS:
        # Use pycocotools for proper RLE encoding
        fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
        rle = mask_utils.encode(fortran_mask)
        rle['counts'] = rle['counts'].decode('utf-8')
        rle['format'] = 'rle'
        return rle
    else:
        # Simplified RLE for testing without pycocotools
        height, width = binary_mask.shape
        return {
            'size': [height, width],
            'counts': 'simplified_rle_for_testing',
            'format': 'rle'
        }

def create_enhanced_dummy_gsam(output_path: str = "enhanced_test_gsam_violations.json"):
    """
    Create a comprehensive GSAM JSON file with ALL QC violations for testing.
    
    Violations Included:
    ==================
    1. HIGH_SEGMENTATION_VAR_EMBRYO: Embryo e08 varies >7x in area across frames
    2. HIGH_SEGMENTATION_VAR_SNIP: Individual snips with >20% area differences 
    3. MASK_ON_EDGE: Embryo e02 touches image boundaries
    4. OVERLAPPING_MASKS: Embryos e06 & e07 have significant IoU overlap
    5. LARGE_MASK: Embryo e03 covers >15% of image area
    6. SMALL_MASK: Embryo e04 covers <0.1% of image area
    7. DETECTION_FAILURE: Missing embryos in frames 1-3 (< 1 embryo minimum)
    8. DISCONTINUOUS_MASK: Embryo e05 has multiple disconnected components
    
    Data Structure:
    ==============
    - 1 experiment: "20240411_test_qc"
    - 1 video: "20240411_test_qc_H01" 
    - 10 frames (images): t0000 through t0009
    - Up to 8 embryos per frame (e01-e08)
    - Each embryo has proper snip_id, bbox, segmentation, area
    """
    
    # Image dimensions
    height, width = 512, 512
    total_pixels = height * width
    
    print(f"🔧 Creating enhanced test GSAM file...")
    print(f"   Image dimensions: {width}x{height} ({total_pixels:,} pixels)")
    
    # === MASK DEFINITIONS ===
    
    # 1. Normal mask (control) - 100x100 = 10,000 pixels (3.8%)
    normal_mask = np.zeros((height, width), dtype=np.uint8)
    normal_mask[200:300, 200:300] = 1
    normal_rle = create_rle_encoding(normal_mask)
    normal_area = float(np.sum(normal_mask))
    print(f"   Normal mask: {normal_area:,.0f} pixels ({normal_area/total_pixels*100:.1f}%)")
    
    # 2. Edge-touching mask - touches top-left corner
    edge_mask = np.zeros((height, width), dtype=np.uint8)
    edge_mask[0:100, 0:100] = 1  # Explicitly touches edges
    edge_rle = create_rle_encoding(edge_mask)
    edge_area = float(np.sum(edge_mask))
    print(f"   Edge mask: {edge_area:,.0f} pixels (touches boundaries)")
    
    # 3. Large mask - 300x300 = 90,000 pixels (34.2% - well above 15% threshold)
    large_mask = np.zeros((height, width), dtype=np.uint8)
    large_mask[100:400, 100:400] = 1
    large_rle = create_rle_encoding(large_mask)
    large_area = float(np.sum(large_mask))
    print(f"   Large mask: {large_area:,.0f} pixels ({large_area/total_pixels*100:.1f}% - threshold 15%)")
    
    # 4. Small mask - 3x3 = 9 pixels (0.0034% - well below 0.1% threshold)
    small_mask = np.zeros((height, width), dtype=np.uint8)
    small_mask[250:253, 250:253] = 1
    small_rle = create_rle_encoding(small_mask)
    small_area = float(np.sum(small_mask))
    print(f"   Small mask: {small_area:,.0f} pixels ({small_area/total_pixels*100:.4f}% - threshold 0.1%)")
    
    # 5. Discontinuous mask - 3 separate components
    discontinuous_mask = np.zeros((height, width), dtype=np.uint8)
    discontinuous_mask[100:150, 100:150] = 1  # Component 1: 50x50
    discontinuous_mask[300:350, 300:350] = 1  # Component 2: 50x50 (disconnected)
    discontinuous_mask[400:420, 100:120] = 1  # Component 3: 20x20 (disconnected)
    discontinuous_rle = create_rle_encoding(discontinuous_mask)
    discontinuous_area = float(np.sum(discontinuous_mask))
    print(f"   Discontinuous mask: {discontinuous_area:,.0f} pixels (3 components)")
    
    # 6. Overlapping masks - designed for significant IoU
    overlap_mask1 = np.zeros((height, width), dtype=np.uint8)
    overlap_mask1[200:280, 200:280] = 1  # 80x80 = 6,400 pixels
    overlap_rle1 = create_rle_encoding(overlap_mask1)
    overlap_area1 = float(np.sum(overlap_mask1))
    
    overlap_mask2 = np.zeros((height, width), dtype=np.uint8)
    overlap_mask2[240:320, 240:320] = 1  # 80x80 = 6,400 pixels
    overlap_rle2 = create_rle_encoding(overlap_mask2)
    overlap_area2 = float(np.sum(overlap_mask2))
    
    # Calculate expected overlap
    intersection = np.sum((overlap_mask1 == 1) & (overlap_mask2 == 1))
    union = np.sum((overlap_mask1 == 1) | (overlap_mask2 == 1))
    expected_iou = intersection / union if union > 0 else 0
    print(f"   Overlapping masks: IoU = {expected_iou:.3f} (threshold typically 0.1)")
    
    # 7. Variable size masks for segmentation variability
    # Small variant for e08
    var_small = np.zeros((height, width), dtype=np.uint8)
    var_small[200:250, 200:250] = 1  # 50x50 = 2,500 pixels
    var_small_rle = create_rle_encoding(var_small)
    var_small_area = float(np.sum(var_small))
    
    # Large variant for e08 (creates dramatic variability)
    var_large = np.zeros((height, width), dtype=np.uint8)
    var_large[180:320, 180:320] = 1  # 140x140 = 19,600 pixels
    var_large_rle = create_rle_encoding(var_large)
    var_large_area = float(np.sum(var_large))
    
    variability_ratio = var_large_area / var_small_area
    cv = np.std([var_small_area, var_large_area]) / np.mean([var_small_area, var_large_area])
    print(f"   Variable masks: {var_small_area:,.0f} vs {var_large_area:,.0f} pixels")
    print(f"                   Ratio: {variability_ratio:.1f}x, CV: {cv:.3f} (threshold 0.15)")
    
    # === BUILD GSAM STRUCTURE ===
    
    gsam_data = {
        "file_info": {
            "created": datetime.now().isoformat(),
            "purpose": "Enhanced QC testing with comprehensive violations",
            "last_updated": datetime.now().isoformat(),
            "image_dimensions": [height, width],
            "total_pixels": total_pixels
        },
        "experiments": {
            "20240411_test_qc": {
                "videos": {
                    "20240411_test_qc_H01": {
                        "images": {}
                    }
                }
            }
        }
    }
    
    video_data = gsam_data["experiments"]["20240411_test_qc"]["videos"]["20240411_test_qc_H01"]["images"]
    
    # === POPULATE FRAMES ===
    
    for frame_num in range(10):
        image_id = f"20240411_test_qc_H01_t{frame_num:04d}"
        video_data[image_id] = {"embryos": {}}
        
        if frame_num == 0:
            # Frame 0: ALL embryos present (baseline for comparisons)
            
            # e01: Normal control embryo
            video_data[image_id][\"embryos\"][\"e01\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e01_s{frame_num:04d}\",
                \"bbox\": [200/width, 200/height, 300/width, 300/height],
                \"segmentation\": normal_rle,
                \"area\": normal_area
            }
            
            # e02: Edge-touching embryo (MASK_ON_EDGE violation)
            video_data[image_id][\"embryos\"][\"e02\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e02_s{frame_num:04d}\",
                \"bbox\": [0.0, 0.0, 100/width, 100/height],
                \"segmentation\": edge_rle,
                \"area\": edge_area
            }
            
            # e03: Large embryo (LARGE_MASK violation)
            video_data[image_id][\"embryos\"][\"e03\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e03_s{frame_num:04d}\",
                \"bbox\": [100/width, 100/height, 400/width, 400/height],
                \"segmentation\": large_rle,
                \"area\": large_area
            }
            
            # e04: Small embryo (SMALL_MASK violation)
            video_data[image_id][\"embryos\"][\"e04\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e04_s{frame_num:04d}\",
                \"bbox\": [250/width, 250/height, 253/width, 253/height],
                \"segmentation\": small_rle,
                \"area\": small_area
            }
            
            # e05: Discontinuous embryo (DISCONTINUOUS_MASK violation)
            video_data[image_id][\"embryos\"][\"e05\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e05_s{frame_num:04d}\",
                \"bbox\": [100/width, 100/height, 420/width, 350/height],
                \"segmentation\": discontinuous_rle,
                \"area\": discontinuous_area
            }
            
            # e06 & e07: Overlapping embryos (OVERLAPPING_MASKS violation)
            video_data[image_id][\"embryos\"][\"e06\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e06_s{frame_num:04d}\",
                \"bbox\": [200/width, 200/height, 280/width, 280/height],
                \"segmentation\": overlap_rle1,
                \"area\": overlap_area1
            }
            
            video_data[image_id][\"embryos\"][\"e07\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e07_s{frame_num:04d}\",
                \"bbox\": [240/width, 240/height, 320/width, 320/height],
                \"segmentation\": overlap_rle2,
                \"area\": overlap_area2
            }
            
            # e08: Variable embryo (starts small)
            video_data[image_id][\"embryos\"][\"e08\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e08_s{frame_num:04d}\",
                \"bbox\": [200/width, 200/height, 250/width, 250/height],
                \"segmentation\": var_small_rle,
                \"area\": var_small_area
            }
            
        elif frame_num in [1, 2, 3]:
            # Frames 1-3: DETECTION_FAILURE (missing most embryos, only e01 and e08)
            
            # e01: Consistent control
            video_data[image_id][\"embryos\"][\"e01\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e01_s{frame_num:04d}\",
                \"bbox\": [200/width, 200/height, 300/width, 300/height],
                \"segmentation\": normal_rle,
                \"area\": normal_area
            }
            
            # e08: Alternates size (HIGH_SEGMENTATION_VAR violations)
            if frame_num == 1:
                # Sudden jump to large (HIGH_SEGMENTATION_VAR_SNIP)
                video_data[image_id][\"embryos\"][\"e08\"] = {
                    \"snip_id\": f\"20240411_test_qc_H01_e08_s{frame_num:04d}\",
                    \"bbox\": [180/width, 180/height, 320/width, 320/height],
                    \"segmentation\": var_large_rle,
                    \"area\": var_large_area
                }
            else:
                # Back to small
                video_data[image_id][\"embryos\"][\"e08\"] = {
                    \"snip_id\": f\"20240411_test_qc_H01_e08_s{frame_num:04d}\",
                    \"bbox\": [200/width, 200/height, 250/width, 250/height],
                    \"segmentation\": var_small_rle,
                    \"area\": var_small_area
                }
            
            # e02-e07 MISSING (causes DETECTION_FAILURE)
            
        else:
            # Frames 4-9: Continue pattern for HIGH_SEGMENTATION_VAR_EMBRYO
            
            # e01: Always consistent
            video_data[image_id][\"embryos\"][\"e01\"] = {
                \"snip_id\": f\"20240411_test_qc_H01_e01_s{frame_num:04d}\",
                \"bbox\": [200/width, 200/height, 300/width, 300/height],
                \"segmentation\": normal_rle,
                \"area\": normal_area
            }
            
            # e08: Alternates to build up CV > 0.15 for HIGH_SEGMENTATION_VAR_EMBRYO
            if frame_num % 2 == 0:
                video_data[image_id][\"embryos\"][\"e08\"] = {
                    \"snip_id\": f\"20240411_test_qc_H01_e08_s{frame_num:04d}\",
                    \"bbox\": [200/width, 200/height, 250/width, 250/height],
                    \"segmentation\": var_small_rle,
                    \"area\": var_small_area
                }
            else:
                video_data[image_id][\"embryos\"][\"e08\"] = {
                    \"snip_id\": f\"20240411_test_qc_H01_e08_s{frame_num:04d}\",
                    \"bbox\": [180/width, 180/height, 320/width, 320/height],
                    \"segmentation\": var_large_rle,
                    \"area\": var_large_area
                }
    
    # === SAVE FILE ===
    
    output = Path(output_path)
    with open(output, 'w') as f:
        json.dump(gsam_data, f, indent=2)
    
    print(f\"\\n✅ Enhanced test GSAM file created: {output}\")
    print(f\"   File size: {output.stat().st_size / 1024:.1f} KB\")
    
    # === VALIDATION SUMMARY ===
    
    print(\"\\n\" + \"=\"*60)
    print(\"EXPECTED QC VIOLATIONS\")
    print(\"=\"*60)
    
    violations = [
        \"1. HIGH_SEGMENTATION_VAR_EMBRYO: e08 (CV > 0.15 across 10 frames)\",
        \"2. HIGH_SEGMENTATION_VAR_SNIP: e08 frame transitions (>20% area change)\",
        \"3. MASK_ON_EDGE: e02 in frame 0 (touches top & left edges)\",
        \"4. OVERLAPPING_MASKS: e06 & e07 in frame 0 (significant IoU)\",
        f\"5. LARGE_MASK: e03 in frame 0 ({large_area/total_pixels*100:.1f}% > 15% threshold)\",
        f\"6. SMALL_MASK: e04 in frame 0 ({small_area/total_pixels*100:.4f}% < 0.1% threshold)\",
        \"7. DETECTION_FAILURE: frames 1-3 (< 1 embryo minimum expected)\",
        \"8. DISCONTINUOUS_MASK: e05 in frame 0 (3 separate components)\"
    ]
    
    for violation in violations:
        print(f\"   {violation}\")
    
    print(\"\\n\" + \"=\"*60)
    print(\"DATA STRUCTURE SUMMARY\")
    print(\"=\"*60)
    print(f\"   Experiments: 1 (20240411_test_qc)\")
    print(f\"   Videos: 1 (20240411_test_qc_H01)\")
    print(f\"   Images: 10 (t0000-t0009)\")
    print(f\"   Total snips: {sum(len(img_data['embryos']) for img_data in video_data.values())}\")
    print(f\"   Unique embryo IDs: 8 (e01-e08)\")
    
    return str(output)


def validate_structure(gsam_file: str):
    \"\"\"Quick validation that the structure matches QC expectations.\"\"\"
    with open(gsam_file, 'r') as f:
        data = json.load(f)
    
    print(\"\\n\" + \"=\"*60)
    print(\"STRUCTURE VALIDATION\")
    print(\"=\"*60)
    
    experiments = data.get(\"experiments\", {})
    print(f\"✓ Experiments found: {len(experiments)}\")
    
    for exp_id, exp_data in experiments.items():
        videos = exp_data.get(\"videos\", {})
        print(f\"✓ Videos in {exp_id}: {len(videos)}\")
        
        for video_id, video_data in videos.items():
            images = video_data.get(\"images\", {})
            print(f\"✓ Images in {video_id}: {len(images)}\")
            
            total_snips = 0
            embryo_ids = set()
            
            for image_id, image_data in images.items():
                embryos = image_data.get(\"embryos\", {})
                total_snips += len(embryos)
                
                for embryo_id, embryo_data in embryos.items():
                    embryo_ids.add(embryo_id)
                    
                    # Check required fields
                    snip_id = embryo_data.get(\"snip_id\")
                    area = embryo_data.get(\"area\")
                    segmentation = embryo_data.get(\"segmentation\")
                    
                    if not snip_id:
                        print(f\"❌ Missing snip_id in {image_id}/{embryo_id}\")
                    if area is None:
                        print(f\"❌ Missing area in {image_id}/{embryo_id}\")
                    if not segmentation:
                        print(f\"❌ Missing segmentation in {image_id}/{embryo_id}\")
            
            print(f\"✓ Total snips: {total_snips}\")
            print(f\"✓ Unique embryos: {len(embryo_ids)} - {sorted(embryo_ids)}\")
    
    print(\"✓ Structure validation complete\")


if __name__ == \"__main__\":
    # Create the enhanced test file
    test_file = create_enhanced_dummy_gsam(\"enhanced_test_gsam_violations.json\")
    
    # Validate the structure
    validate_structure(test_file)
    
    print(f\"\\n🎯 Test file ready for QC validation: {test_file}\")
    print(\"   Use this file to test the ported QC system\")
