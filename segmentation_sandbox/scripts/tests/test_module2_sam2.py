#!/usr/bin/env python3
"""
Test SAM2 Utils Refactoring (Module 2)
======================================

Test the refactored SAM2 utilities to ensure:
1. GroundedSamAnnotations class works with modular utilities
2. Entity tracking is properly integrated
3. Snip IDs use '_s' prefix format
4. ExperimentMetadata integration works
5. Parsing utils are used consistently

Run with: python test_module2_sam2.py
"""

import sys
import json
from pathlib import Path

# Add the project root to the path
SANDBOX_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(SANDBOX_ROOT))

from scripts.detection_segmentation import GroundedSamAnnotations, create_snip_id
from scripts.utils.entity_id_tracker import EntityIDTracker
from scripts.utils.parsing_utils import extract_frame_number


def test_sam2_initialization():
    """Test SAM2 class initialization with validation."""
    print("🧪 Testing SAM2 initialization...")
    
    # Test paths (using correct metadata location)
    test_detections_path = SANDBOX_ROOT / "data" / "annotation_and_masks" / "test_detections.json"
    test_metadata_path = SANDBOX_ROOT / "scripts" / "pipelines" / "data" / "raw_data_organized" / "experiment_metadata.json"
    test_sam2_output = SANDBOX_ROOT / "temp" / "test_sam2_output.json"
    
    # Create temp directory
    test_sam2_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Create minimal test detection file if it doesn't exist
    if not test_detections_path.exists():
        test_detections_path.parent.mkdir(parents=True, exist_ok=True)
        minimal_detections = {
            "script_version": "test",
            "creation_time": "2025-08-04T00:00:00",
            "high_quality_annotations": {
                "individual embryo": {
                    "20250703_chem3_28C_T00_1325_A01_t0000": [
                        {
                            "bbox": [0.1, 0.1, 0.3, 0.3],
                            "confidence": 0.8,
                            "class_name": "individual embryo"
                        }
                    ]
                }
            }
        }
        with open(test_detections_path, 'w') as f:
            json.dump(minimal_detections, f, indent=2)
        print(f"   📝 Created minimal test detections: {test_detections_path}")
    
    try:
        # Test initialization
        gsam = GroundedSamAnnotations(
            filepath=test_sam2_output,
            seed_annotations_path=test_detections_path,
            experiment_metadata_path=test_metadata_path,
            verbose=True
        )
        
        print(f"   ✅ GroundedSamAnnotations initialized successfully")
        print(f"   📋 GSAM ID: {gsam.get_gsam_id()}")
        print(f"   🎯 Target prompt: {gsam.target_prompt}")
        print(f"   📊 Format: {gsam.segmentation_format}")
        
        # Test that it uses ExperimentMetadata
        if hasattr(gsam, 'exp_metadata') and gsam.exp_metadata is not None:
            print(f"   ✅ ExperimentMetadata integration working")
        else:
            print(f"   ❌ ExperimentMetadata not loaded")
        
        return gsam
        
    except Exception as e:
        print(f"   ❌ SAM2 initialization failed: {e}")
        return None


def test_snip_id_format():
    """Test that snip_id creation uses '_s' prefix format."""
    print("\n🧪 Testing snip_id format...")
    
    test_cases = [
        ("20240411_A01_e01", "20240411_A01_t0000", "20240411_A01_e01_s0000"),
        ("20240411_A01_e02", "20240411_A01_t0123", "20240411_A01_e02_s0123"),
        ("20250703_chem3_28C_T00_1325_A01_e01", "20250703_chem3_28C_T00_1325_A01_t0045", 
         "20250703_chem3_28C_T00_1325_A01_e01_s0045")
    ]
    
    for embryo_id, image_id, expected_snip_id in test_cases:
        actual_snip_id = create_snip_id(embryo_id, image_id)
        
        if actual_snip_id == expected_snip_id:
            print(f"   ✅ {image_id} → {actual_snip_id}")
        else:
            print(f"   ❌ {image_id} → {actual_snip_id} (expected: {expected_snip_id})")
        
        # Verify '_s' format
        if "_s" in actual_snip_id and actual_snip_id.count("_s") == 1:
            print(f"      ✅ Correct '_s' format")
        else:
            print(f"      ❌ Incorrect format - should contain exactly one '_s'")


def test_entity_tracking():
    """Test entity tracking integration."""
    print("\n🧪 Testing entity tracking integration...")
    
    # Create a minimal SAM2 result structure
    test_results = {
        "script_version": "sam2_utils.py (refactored)",
        "creation_time": "2025-08-04T00:00:00",
        "snip_ids": [
            "20240411_A01_e01_s0000",
            "20240411_A01_e01_s0001", 
            "20240411_A01_e02_s0000"
        ],
        "experiments": {
            "20240411": {
                "images": {
                    "20240411_A01_t0000": {
                        "image_id": "20240411_A01_t0000",
                        "embryos": {
                            "20240411_A01_e01": {
                                "snip_id": "20240411_A01_e01_s0000"
                            }
                        }
                    }
                }
            }
        }
    }
    
    try:
        # Test entity tracker update (FIXED: assign return value)
        test_results = EntityIDTracker.update_entity_tracker(
            test_results,
            pipeline_step="module_2_segmentation"
        )
        
        # Check for entity_tracker section (not entity_tracking)
        if "entity_tracker" in test_results:
            print(f"   ✅ Entity tracking added to results")
            
            # Extract and validate entities
            entities = EntityIDTracker.extract_entities(test_results)
            entity_counts = EntityIDTracker.get_counts(entities)
            print(f"   📊 Entity counts: {entity_counts}")
            
            # Validate hierarchy (FIXED: removed raise_on_violations parameter)
            validation_result = EntityIDTracker.validate_hierarchy(entities)
            if validation_result.get('valid', True):
                print(f"   ✅ Entity hierarchy validation passed")
            else:
                print(f"   ⚠️ Entity hierarchy warnings: {validation_result.get('violations', [])}")
            
            # Check snip_id formats
            snip_ids = entities.get("snips", [])
            valid_formats = all("_s" in sid and sid.count("_s") == 1 for sid in snip_ids)
            if valid_formats:
                print(f"   ✅ All snip_ids use correct '_s' format")
            else:
                invalid_snips = [sid for sid in snip_ids if not ("_s" in sid and sid.count("_s") == 1)]
                print(f"   ❌ Invalid snip_id formats: {invalid_snips}")
                
        else:
            print(f"   ❌ Entity tracking not added to results")
            
    except Exception as e:
        print(f"   ❌ Entity tracking test failed: {e}")


def test_video_grouping(gsam):
    """Test video grouping functionality."""
    if not gsam:
        print("\n⏭️ Skipping video grouping test (no GSAM instance)")
        return
        
    print("\n🧪 Testing video grouping...")
    
    try:
        video_groups = gsam.group_annotations_by_video()
        
        if video_groups:
            print(f"   ✅ Video grouping successful: {len(video_groups)} videos found")
            for video_id, images in video_groups.items():
                print(f"      📹 {video_id}: {len(images)} images")
        else:
            print(f"   ⚠️ No video groups found (may be expected with minimal test data)")
            
    except Exception as e:
        print(f"   ❌ Video grouping failed: {e}")


def test_save_functionality(gsam):
    """Test save functionality with entity validation."""
    if not gsam:
        print("\n⏭️ Skipping save test (no GSAM instance)")
        return
        
    print("\n🧪 Testing save functionality...")
    
    try:
        # Add some test data to results
        gsam.results["snip_ids"] = ["test_embryo_s0000"]
        
        # Test save (should include entity validation)
        gsam.save()
        
        # Verify file was created and contains entity_tracking
        if gsam.filepath.exists():
            with open(gsam.filepath, 'r') as f:
                saved_data = json.load(f)
            
            if "entity_tracking" in saved_data:
                print(f"   ✅ Save successful with entity tracking")
                print(f"   📊 Pipeline step: {saved_data['entity_tracking'].get('pipeline_step', 'unknown')}")
            else:
                print(f"   ⚠️ Save successful but no entity tracking found")
        else:
            print(f"   ❌ Save failed - file not created")
            
    except Exception as e:
        print(f"   ❌ Save test failed: {e}")


def main():
    """Run all SAM2 refactoring tests."""
    print("🚀 Starting SAM2 Utils Refactoring Tests")
    print("=" * 50)
    
    # Test 1: Initialization
    gsam = test_sam2_initialization()
    
    # Test 2: Snip ID format
    test_snip_id_format() 
    
    # Test 3: Entity tracking
    test_entity_tracking()
    
    # Test 4: Video grouping
    test_video_grouping(gsam)
    
    # Test 5: Save functionality
    test_save_functionality(gsam)
    
    print("\n" + "=" * 50)
    print("🏁 SAM2 Utils Refactoring Tests Complete")
    
    if gsam:
        print(f"\n📊 Summary:")
        summary = gsam.get_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
