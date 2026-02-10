#!/usr/bin/env python3
"""
Test GroundingDINO Integration with Module 0/1 Utilities
======================================================

This test verifies that the refactored GroundedDinoAnnotations class works correctly
with the entity tracking and metadata systems.
"""

import sys
from pathlib import Path

# Add project root to path
SANDBOX_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(SANDBOX_ROOT))

from scripts.detection_segmentation import GroundedDinoAnnotations
from scripts.metadata import ExperimentMetadata


def test_gdino_annotations_basic():
    """Test basic GroundedDinoAnnotations functionality."""
    print("🧪 Testing GroundedDinoAnnotations basic functionality...")
    
    # Test initialization
    test_path = SANDBOX_ROOT / "temp" / "test_detections.json"
    gdino = GroundedDinoAnnotations(test_path, verbose=True)
    
    print(f"✅ Initialized GroundedDinoAnnotations at: {test_path}")
    print(f"📊 Initial summary:")
    gdino.print_summary()
    
    return gdino


def test_metadata_integration():
    """Test integration with experiment metadata."""
    print("\n🧪 Testing metadata integration...")
    
    # Use the existing experiment metadata
    metadata_path = SANDBOX_ROOT / "data" / "raw_data_organized" / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"❌ Experiment metadata not found at: {metadata_path}")
        return None
    
    # Test initialization with metadata
    test_path = SANDBOX_ROOT / "temp" / "test_detections_with_metadata.json"
    gdino = GroundedDinoAnnotations(test_path, verbose=True, metadata_path=metadata_path)
    
    print(f"✅ Initialized with metadata integration")
    
    # Test metadata functionality
    all_images = gdino.get_all_metadata_image_ids()
    print(f"📊 Total images in metadata: {len(all_images)}")
    
    if len(all_images) > 0:
        print(f"📋 First few image IDs: {all_images[:5]}")
        
        # Test get_images_for_detection
        detection_images = gdino.get_images_for_detection()
        print(f"📁 Images available for detection: {len(detection_images)}")
        
        if len(detection_images) > 0:
            print(f"📋 First detection image: {detection_images[0]}")
    
    return gdino


def test_missing_annotations():
    """Test missing annotation detection."""
    print("\n🧪 Testing missing annotation detection...")
    
    metadata_path = SANDBOX_ROOT / "data" / "raw_data_organized" / "experiment_metadata.json"
    if not metadata_path.exists():
        print(f"❌ Experiment metadata not found, skipping test")
        return
    
    test_path = SANDBOX_ROOT / "temp" / "test_missing_annotations.json"
    gdino = GroundedDinoAnnotations(test_path, verbose=True, metadata_path=metadata_path)
    
    # Test missing annotations detection
    missing = gdino.get_missing_annotations(
        prompts=["individual embryo"],
        experiment_ids=["20250612_30hpf_ctrl_atf6"]  # Use experiment from metadata
    )
    
    print(f"📊 Missing annotations: {len(missing.get('individual embryo', []))}")
    
    return gdino


def test_entity_validation():
    """Test entity ID validation integration."""
    print("\n🧪 Testing entity validation integration...")
    
    test_path = SANDBOX_ROOT / "temp" / "test_entity_validation.json"
    gdino = GroundedDinoAnnotations(test_path, verbose=True)
    
    # Test valid image ID format
    valid_image_id = "20250612_30hpf_ctrl_atf6_A01_t0000"
    print(f"📝 Testing with valid image ID: {valid_image_id}")
    
    # Create mock detection data
    import numpy as np
    mock_boxes = np.array([[0.1, 0.1, 0.9, 0.9]])
    mock_logits = np.array([0.85])
    mock_phrases = ["individual embryo"]
    
    # Create mock model object
    class MockModel:
        def __init__(self):
            self._annotation_metadata = {
                "model_config_path": "GroundingDINO_SwinT_OGC.py",
                "model_weights_path": "groundingdino_swint_ogc.pth",
                "loading_timestamp": "2025-08-04T12:00:00.000000",
                "model_architecture": "GroundedDINO"
            }
    
    mock_model = MockModel()
    
    # Test adding annotation
    gdino.add_annotation(
        image_id=valid_image_id,
        prompt="individual embryo",
        model=mock_model,
        boxes=mock_boxes,
        logits=mock_logits,
        phrases=mock_phrases
    )
    
    print(f"✅ Successfully added annotation for {valid_image_id}")
    
    # Test save with entity validation
    gdino.save()
    print(f"✅ Successfully saved with entity validation")
    
    gdino.print_summary()
    
    return gdino


def main():
    """Run all tests."""
    print("🚀 Testing Module 2 GroundingDINO Integration")
    print("=" * 50)
    
    # Create temp directory
    temp_dir = SANDBOX_ROOT / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Run tests
        test_gdino_annotations_basic()
        test_metadata_integration()
        test_missing_annotations()
        test_entity_validation()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
