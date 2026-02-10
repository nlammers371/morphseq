#!/usr/bin/env python3
"""
Test Module 2 GroundingDINO with Real Data
==========================================

Quick test script to verify our Module 2 implementation works with real experiment data.
"""

import sys
from pathlib import Path

# Add project root to path
SANDBOX_ROOT = Path(__file__).parent.parent
sys.path.append(str(SANDBOX_ROOT))

from scripts.detection_segmentation.grounded_dino_utils import GroundedDinoAnnotations
from scripts.metadata.experiment_metadata import ExperimentMetadata

def main():
    print("🧪 Testing Module 2 GroundingDINO with Real Data")
    print("=" * 50)
    
    # Paths
    metadata_path = "data/raw_data_organized/experiment_metadata.json"
    annotations_path = "temp/test_real_gdino_annotations.json"
    
    # Load metadata
    print("📂 Loading experiment metadata...")
    metadata = ExperimentMetadata(metadata_path, verbose=True)
    
    # Get a small subset of images for testing
    all_images = metadata.list_images()
    test_images = all_images[:5]  # Just test 5 images
    
    print(f"📊 Testing with {len(test_images)} images:")
    for img_id in test_images:
        print(f"   - {img_id}")
    
    # Initialize annotations manager
    print("\n🔧 Initializing annotations manager...")
    annotations = GroundedDinoAnnotations(
        annotations_path, 
        verbose=True, 
        metadata_path=metadata_path
    )
    
    # Test path resolution
    print("\n🔍 Testing image path resolution...")
    for img_id in test_images[:2]:
        try:
            # This should work by getting the experiment and using the image path
            # We'll need to use a different approach since get_images_for_detection
            # doesn't support filtering by specific image IDs
            from scripts.utils.parsing_utils import extract_experiment_id
            exp_id = extract_experiment_id(img_id)
            print(f"   ✅ {img_id} belongs to experiment: {exp_id}")
        except Exception as e:
            print(f"   ❌ Error resolving {img_id}: {e}")
    
    # Test missing annotations detection
    print("\n🔍 Testing missing annotations detection...")
    missing = annotations.get_missing_annotations(
        prompts=["individual embryo"],
        image_ids=test_images
    )
    
    for prompt, missing_ids in missing.items():
        print(f"   '{prompt}': {len(missing_ids)} missing annotations")
    
    # Test summary
    print("\n📊 Testing summary...")
    annotations.print_summary()
    
    print("\n✅ Module 2 real data test completed!")
    print(f"💾 Test annotations saved to: {annotations_path}")

if __name__ == "__main__":
    main()
