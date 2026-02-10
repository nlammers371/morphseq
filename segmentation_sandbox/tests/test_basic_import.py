#!/usr/bin/env python3
"""
Basic test to verify SAM2 import functionality and Python imports
"""
import sys
import os
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_sam2_data_structure():
    """Test that we can read and understand SAM2 data structure"""
    print("🔍 Testing SAM2 data structure...")
    
    sam2_path = project_root / "data" / "segmentation" / "grounded_sam_segmentations.json"
    
    if not sam2_path.exists():
        print(f"❌ SAM2 file not found: {sam2_path}")
        return False
    
    try:
        with open(sam2_path, 'r') as f:
            sam2_data = json.load(f)
        
        print(f"✅ SAM2 file loaded successfully")
        print(f"   File size: {sam2_path.stat().st_size / 1024:.1f} KB")
        
        # Analyze structure
        if "experiments" not in sam2_data:
            print("❌ Missing 'experiments' key in SAM2 data")
            return False
            
        experiments = sam2_data["experiments"]
        print(f"✅ Found {len(experiments)} experiments")
        
        # Look at first experiment
        first_exp_id = next(iter(experiments.keys()))
        first_exp = experiments[first_exp_id]
        print(f"   First experiment: {first_exp_id}")
        
        if "videos" not in first_exp:
            print("❌ Missing 'videos' key in experiment")
            return False
            
        videos = first_exp["videos"]
        print(f"✅ Found {len(videos)} videos in first experiment")
        
        # Look for embryo data
        first_video_id = next(iter(videos.keys()))
        first_video = videos[first_video_id]
        print(f"   First video: {first_video_id}")
        
        # Check for embryo_ids or image data
        embryo_count = 0
        image_count = 0
        
        if "embryo_ids" in first_video:
            embryo_count = len(first_video["embryo_ids"])
            print(f"✅ Found embryo_ids: {embryo_count}")
            
        if "images" in first_video:
            images = first_video["images"]
            image_count = len(images)
            print(f"✅ Found images: {image_count}")
            
            # Look at first image for embryo data
            if images:
                first_image_id = next(iter(images.keys()))
                first_image = images[first_image_id]
                print(f"   First image: {first_image_id}")
                
                if "embryos" in first_image:
                    embryos_in_image = len(first_image["embryos"])
                    print(f"✅ Found {embryos_in_image} embryos in first image")
                    
                    # Show embryo IDs
                    embryo_ids = list(first_image["embryos"].keys())
                    print(f"   Embryo IDs: {embryo_ids[:3]}{'...' if len(embryo_ids) > 3 else ''}")
        
        print("✅ SAM2 data structure validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error reading SAM2 file: {e}")
        return False

def test_import_paths():
    """Test that we can set up import paths correctly"""
    print("🔍 Testing import paths...")
    
    try:
        # Test if we can import our modules
        from scripts.utils.parsing_utils import parse_entity_id, get_entity_type
        print("✅ parsing_utils import successful")
        
        # Test parsing with a real ID format
        test_snip_id = "20250612_30hpf_ctrl_atf6_C11_e01_s0000"
        parsed = parse_entity_id(test_snip_id)
        entity_type = get_entity_type(test_snip_id)
        
        print(f"✅ ID parsing test successful:")
        print(f"   Input: {test_snip_id}")
        print(f"   Type: {entity_type}")
        print(f"   Parsed: {parsed}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("   This is expected - imports will be fixed when implementing modules")
        return False
    except Exception as e:
        print(f"❌ Parsing error: {e}")
        return False

def main():
    """Run basic tests"""
    print("=" * 60)
    print("Module 3 Basic Test Suite")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print("=" * 60)
    
    # Run tests
    sam2_test = test_sam2_data_structure()
    import_test = test_import_paths()
    
    print("=" * 60)
    print("Test Results:")
    print(f"{'✅' if sam2_test else '❌'} SAM2 data structure test")
    print(f"{'✅' if import_test else '❌'} Import paths test")
    
    if sam2_test:
        print("✅ Ready to proceed with SAM2 integration")
    else:
        print("❌ SAM2 data issues need resolution")
        
    if import_test:
        print("✅ Import system working correctly")
    else:
        print("⚠️  Import issues - expected during development")
    
    print("=" * 60)

if __name__ == "__main__":
    main()