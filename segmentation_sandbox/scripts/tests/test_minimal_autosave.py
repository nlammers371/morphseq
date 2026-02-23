#!/usr/bin/env python3
"""
Minimal test with just 1 image to verify autosave works properly
"""

import sys
from pathlib import Path
import json
import tempfile

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def test_single_image_autosave():
    """Test autosave with just one image addition."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metadata_file = temp_path / "minimal_test.json"
        
        print("🧪 Testing autosave with single image...")
        
        try:
            from metadata.experiment_metadata import ExperimentMetadata
            
            # Set autosave to trigger after just 1 operation
            print("1️⃣ Creating ExperimentMetadata with auto_save_interval=1")
            em = ExperimentMetadata(metadata_file, verbose=True, auto_save_interval=1)
            
            print(f"   📁 Initial file exists: {metadata_file.exists()}")
            print(f"   📊 Operations counter: {em._operations_since_save}")
            
            # Add single experiment - this should trigger autosave immediately
            print("\n2️⃣ Adding single experiment (should trigger autosave)...")
            em.add_experiment("exp001", condition="test", researcher="autosave_test")
            
            print(f"   📁 File exists after experiment: {metadata_file.exists()}")
            print(f"   📁 File size: {metadata_file.stat().st_size if metadata_file.exists() else 0} bytes")
            print(f"   📊 Operations counter after save: {em._operations_since_save}")
            
            # Add video - should trigger autosave again
            print("\n3️⃣ Adding single video (should trigger autosave)...")
            em.add_video_to_experiment("exp001", "exp001_A01", well="A01")
            
            print(f"   📁 File size after video: {metadata_file.stat().st_size} bytes")
            print(f"   📊 Operations counter after save: {em._operations_since_save}")
            
            # Add single image - should trigger autosave again
            print("\n4️⃣ Adding single image (should trigger autosave)...")
            em.add_images_to_video("exp001", "exp001_A01", ["exp001_A01_t0001"], format="jpg")
            
            print(f"   📁 File size after image: {metadata_file.stat().st_size} bytes")
            print(f"   📊 Operations counter after save: {em._operations_since_save}")
            
            # Verify content
            print("\n5️⃣ Verifying saved content...")
            with open(metadata_file, 'r') as f:
                saved_data = json.load(f)
            
            experiments = saved_data.get('experiments', {})
            videos = experiments.get('exp001', {}).get('videos', {})
            images = videos.get('exp001_A01', {}).get('images', {})
            
            print(f"   📊 Experiments in file: {len(experiments)}")
            print(f"   📊 Videos in file: {len(videos)}")
            print(f"   📊 Images in file: {len(images)}")
            print(f"   🔍 Image IDs: {list(images.keys())}")
            
            # Check for backup files
            backup_files = list(temp_path.glob("*.backup.*"))
            print(f"   🔐 Backup files created: {len(backup_files)}")
            for backup in backup_files:
                print(f"      - {backup.name} ({backup.stat().st_size} bytes)")
            
            print("\n✅ Single image autosave test passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Single image test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_autosave_disabled():
    """Test behavior when autosave is disabled (auto_save_interval=None)."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metadata_file = temp_path / "no_autosave_test.json"
        
        print("\n🚫 Testing with autosave DISABLED...")
        
        try:
            from metadata.experiment_metadata import ExperimentMetadata
            
            # Create without autosave
            print("1️⃣ Creating ExperimentMetadata with NO autosave")
            em = ExperimentMetadata(metadata_file, verbose=True, auto_save_interval=None)
            
            print(f"   📁 Initial file exists: {metadata_file.exists()}")
            print(f"   ⚙️ Autosave interval: {em.auto_save_interval}")
            
            # Add multiple operations
            print("\n2️⃣ Adding multiple operations (should NOT autosave)...")
            em.add_experiment("exp001", condition="test")
            print(f"   📁 File exists after experiment: {metadata_file.exists()}")
            
            em.add_video_to_experiment("exp001", "exp001_A01")
            print(f"   📁 File exists after video: {metadata_file.exists()}")
            
            em.add_images_to_video("exp001", "exp001_A01", ["exp001_A01_t0001"])
            print(f"   📁 File exists after image: {metadata_file.exists()}")
            print(f"   📊 Operations counter: {em._operations_since_save}")
            
            # Manual save should work
            print("\n3️⃣ Manual save...")
            em.save()
            print(f"   📁 File exists after manual save: {metadata_file.exists()}")
            print(f"   📁 File size: {metadata_file.stat().st_size} bytes")
            print(f"   📊 Operations counter after manual save: {em._operations_since_save}")
            
            print("\n✅ Disabled autosave test passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Disabled autosave test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("🚀 Starting minimal autosave tests...")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Single image with autosave
    if test_single_image_autosave():
        success_count += 1
    
    # Test 2: Autosave disabled
    if test_autosave_disabled():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All minimal tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed!")
        sys.exit(1)
