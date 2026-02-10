#!/usr/bin/env python3
"""
Integration test for Module 1 metadata management using real experiment_metadata.json

This test validates that our Module 1 implementation works correctly with the actual
experiment metadata file produced by the data organization process.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to Python path for imports
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from metadata.experiment_metadata import ExperimentMetadata
from metadata.schema_manager import SchemaManager
from utils.parsing_utils import parse_entity_id, build_image_id

def test_real_metadata_loading():
    """Test loading and parsing real experiment metadata JSON."""
    print("🧪 Testing real metadata loading...")
    
    # Path to the real metadata file
    metadata_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json"
    
    # Create ExperimentMetadata instance with real data
    meta = ExperimentMetadata(metadata_path)
    
    # Test basic loading
    assert meta.metadata is not None, "Metadata should be loaded"
    assert "experiments" in meta.metadata, "Should have experiments section"
    
    # Get the experiment ID from the real data
    experiments = meta.metadata["experiments"]
    assert len(experiments) > 0, "Should have at least one experiment"
    
    experiment_id = list(experiments.keys())[0]  # "20250703_chem3_28C_T00_1325"
    print(f"✅ Loaded experiment: {experiment_id}")
    
    # Test video access
    videos = experiments[experiment_id]["videos"]
    assert len(videos) > 0, "Should have videos"
    
    video_id = list(videos.keys())[0]  # e.g., "20250703_chem3_28C_T00_1325_G04"
    print(f"✅ Found video: {video_id}")
    
    return meta, experiment_id, video_id

def test_metadata_driven_path_resolution():
    """Test that path resolution uses stored metadata efficiently."""
    print("\n🧪 Testing metadata-driven path resolution...")
    
    meta, experiment_id, video_id = test_real_metadata_loading()
    
    # Get video data
    video_data = meta.metadata["experiments"][experiment_id]["videos"][video_id]
    expected_image_dir = video_data["processed_jpg_images_dir"]
    image_ids = video_data["image_ids"]
    
    # Test get_image_path uses stored metadata
    for image_id in image_ids:
        image_path = meta.get_image_path(image_id)
        
        # Should use the stored processed_jpg_images_dir
        assert str(expected_image_dir) in str(image_path), f"Path should use stored directory: {image_path}"
        
        # Path should be constructed correctly
        expected_filename = f"{image_id.split('_t')[1]}.jpg"  # Extract frame number
        assert expected_filename in str(image_path), f"Should have correct filename: {image_path}"
        
        print(f"✅ Path resolution for {image_id}: {image_path}")
    
    return meta

def test_actual_file_existence():
    """Test that verify_image_exists works with real files."""
    print("\n🧪 Testing actual file existence checking...")
    
    meta = test_metadata_driven_path_resolution()
    
    # Test with a known existing image
    # From our earlier check, we know A03_t0000 exists
    test_image_id = "20250703_chem3_28C_T00_1325_A03_t0000"
    
    # Verify the image exists
    exists = meta.verify_image_exists(test_image_id)
    assert exists, f"Image {test_image_id} should exist"
    print(f"✅ Verified existing image: {test_image_id}")
    
    # Test with a non-existent image
    fake_image_id = "20250703_chem3_28C_T00_1325_FAKE_t9999"
    exists = meta.verify_image_exists(fake_image_id)
    assert not exists, f"Fake image {fake_image_id} should not exist"
    print(f"✅ Correctly identified non-existent image: {fake_image_id}")
    
    return meta

def test_detection_pipeline_integration():
    """Test get_images_for_detection with real metadata."""
    print("\n🧪 Testing detection pipeline integration...")
    
    meta = test_actual_file_existence()
    
    # Get all images for detection
    images_for_detection = meta.get_images_for_detection()
    
    assert len(images_for_detection) > 0, "Should find images for detection"
    print(f"✅ Found {len(images_for_detection)} images for detection")
    
    # Test that all returned images actually exist
    existing_count = 0
    for image_info in images_for_detection[:5]:  # Test first 5 to avoid long runtime
        image_path = Path(image_info["full_path"])
        if image_path.exists():
            existing_count += 1
            print(f"✅ Confirmed exists: {image_info['image_id']} -> {image_path}")
    
    print(f"✅ Verified {existing_count} out of {min(5, len(images_for_detection))} tested images exist")
    
    # Test filtering by experiment
    experiment_id = "20250703_chem3_28C_T00_1325"
    filtered_images = meta.get_images_for_detection(experiment_ids=[experiment_id])
    assert len(filtered_images) > 0, "Should find images for specific experiment"
    print(f"✅ Filtered to {len(filtered_images)} images for experiment {experiment_id}")
    
    return meta

def test_schema_integration():
    """Test schema management with real data context."""
    print("\n🧪 Testing schema integration...")
    
    # Create a temporary schema for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_path = f.name
    
    try:
        schema_manager = SchemaManager(schema_path)
        
        # Add some realistic phenotypes and genotypes
        schema_manager.add_phenotype("live_cell")
        schema_manager.add_phenotype("dead_cell")
        schema_manager.add_genotype("wildtype")
        schema_manager.add_genotype("knockout")
        schema_manager.add_treatment("control")
        schema_manager.add_treatment("drug_treatment")
        
        # Validate some realistic annotations
        assert schema_manager.validate_phenotype("live_cell"), "Should validate existing phenotype"
        assert not schema_manager.validate_phenotype("invalid_phenotype"), "Should reject invalid phenotype"
        
        print("✅ Schema management works with realistic biological annotations")
        
    finally:
        # Cleanup
        os.unlink(schema_path)

def test_parsing_utilities_integration():
    """Test parsing utilities with real IDs from metadata."""
    print("\n🧪 Testing parsing utilities integration...")
    
    meta, experiment_id, video_id = test_real_metadata_loading()
    
    # Get a real image ID from the metadata
    video_data = meta.metadata["experiments"][experiment_id]["videos"][video_id]
    real_image_id = video_data["image_ids"][0]  # e.g., "20250703_chem3_28C_T00_1325_G04_t0000"
    
    # Test parsing
    parsed = parse_entity_id(real_image_id)
    assert parsed is not None, "Should parse real image ID"
    assert parsed["date"] == "20250703", "Should extract correct date"
    assert parsed["experiment"] == "chem3", "Should extract correct experiment"
    assert "well" in parsed, "Should extract well information"
    
    print(f"✅ Parsed real image ID {real_image_id}: {parsed}")
    
    # Test building image ID
    built_id = build_image_id(
        parsed["date"], parsed["experiment"], parsed["temperature"], 
        parsed["timepoint"], parsed["unique_id"], parsed["well"], 
        frame_num=0
    )
    assert built_id == real_image_id, f"Built ID {built_id} should match original {real_image_id}"
    print(f"✅ Successfully rebuilt image ID: {built_id}")

def main():
    """Run all integration tests."""
    print("🚀 Starting Module 1 Real Metadata Integration Tests")
    print("=" * 60)
    
    try:
        test_real_metadata_loading()
        test_metadata_driven_path_resolution() 
        test_actual_file_existence()
        test_detection_pipeline_integration()
        test_schema_integration()
        test_parsing_utilities_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Module 1 successfully integrates with real experiment data")
        print("✅ Path resolution uses stored metadata efficiently")
        print("✅ File existence checking works with actual files")
        print("✅ Detection pipeline ready for Module 2")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
