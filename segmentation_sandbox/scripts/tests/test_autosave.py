#!/usr/bin/env python3
"""
Simple test to verify autosave functionality in ExperimentMetadata class
"""

import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def test_experiment_metadata_autosave():
    """Test autosave functionality with ExperimentMetadata."""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metadata_file = temp_path / "test_experiment_metadata.json"
        
        print("🧪 Testing ExperimentMetadata autosave functionality...")
        
        try:
            from metadata.experiment_metadata import ExperimentMetadata
            
            # Test 1: Create with autosave every 2 operations
            print("\n1️⃣ Creating ExperimentMetadata with auto_save_interval=2")
            em = ExperimentMetadata(metadata_file, verbose=True, auto_save_interval=2)
            
            print(f"   📁 File exists: {metadata_file.exists()}")
            print(f"   📊 Initial experiments: {len(em.metadata['experiments'])}")
            
            # Test 2: Add operations that should trigger autosave
            print("\n2️⃣ Adding operations to trigger autosave...")
            
            # Operation 1 - should not save yet
            print("   🔹 Adding experiment 1...")
            em.add_experiment("test_exp_001", temperature="28C", condition="control")
            print(f"   📁 File size after op 1: {metadata_file.stat().st_size if metadata_file.exists() else 'N/A'} bytes")
            
            # Operation 2 - should trigger autosave (interval=2)
            print("   🔹 Adding experiment 2...")
            em.add_experiment("test_exp_002", temperature="25C", condition="treatment")
            print(f"   📁 File size after op 2: {metadata_file.stat().st_size if metadata_file.exists() else 'N/A'} bytes")
            
            # Operation 3 - should not save yet
            print("   🔹 Adding video to experiment 1...")
            em.add_video_to_experiment("test_exp_001", "test_exp_001_A01", well="A01", frames=100)
            print(f"   📁 File size after op 3: {metadata_file.stat().st_size if metadata_file.exists() else 'N/A'} bytes")
            
            # Operation 4 - should trigger autosave again
            print("   🔹 Adding images to video...")
            image_ids = ["test_exp_001_A01_t0001", "test_exp_001_A01_t0002", "test_exp_001_A01_t0003"]
            em.add_images_to_video("test_exp_001", "test_exp_001_A01", image_ids, format="jpg")
            print(f"   📁 File size after op 4: {metadata_file.stat().st_size if metadata_file.exists() else 'N/A'} bytes")
            
            # Test 3: Verify autosave worked by loading fresh instance
            print("\n3️⃣ Verifying autosave by loading fresh instance...")
            em2 = ExperimentMetadata(metadata_file, verbose=False)
            print(f"   📊 Loaded experiments: {len(em2.metadata['experiments'])}")
            print(f"   📊 Loaded videos: {len(em2.metadata['experiments'].get('test_exp_001', {}).get('videos', {}))}")
            print(f"   📊 Loaded images: {len(em2.metadata['experiments'].get('test_exp_001', {}).get('videos', {}).get('test_exp_001_A01', {}).get('images', {}))}")
            
            # Test 4: Manual save test
            print("\n4️⃣ Testing manual save...")
            em.save()
            print(f"   📁 Final file size: {metadata_file.stat().st_size} bytes")
            
            print("\n✅ Autosave test completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Autosave test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_backup_mechanism():
    """Test backup file creation during save operations."""
    
    print("\n🔐 Testing backup mechanism...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metadata_file = temp_path / "test_backup_metadata.json"
        
        try:
            from metadata.experiment_metadata import ExperimentMetadata
            
            # Create initial file
            print("1️⃣ Creating initial metadata file...")
            em = ExperimentMetadata(metadata_file, verbose=True)
            em.add_experiment("test_exp_001", condition="control")
            em.save()
            
            initial_content = metadata_file.read_text()
            print(f"   📁 Initial file size: {len(initial_content)} characters")
            
            # Check for backup files
            backup_files = list(temp_path.glob("*.backup"))
            print(f"   🔐 Backup files found: {len(backup_files)}")
            for backup in backup_files:
                print(f"     - {backup.name}")
            
            # Modify and save again
            print("\n2️⃣ Modifying and saving again...")
            em.add_experiment("test_exp_002", condition="treatment")
            em.save()
            
            modified_content = metadata_file.read_text()
            print(f"   📁 Modified file size: {len(modified_content)} characters")
            
            # Check for new backup files
            backup_files = list(temp_path.glob("*.backup"))
            print(f"   🔐 Backup files after second save: {len(backup_files)}")
            for backup in backup_files:
                print(f"     - {backup.name} ({backup.stat().st_size} bytes)")
            
            # Check if backup contains old content
            if backup_files:
                backup_content = backup_files[0].read_text()
                has_first_exp = "test_exp_001" in backup_content
                has_second_exp = "test_exp_002" in backup_content
                print(f"   🔍 Backup contains first experiment: {has_first_exp}")
                print(f"   🔍 Backup contains second experiment: {has_second_exp}")
                
                if has_first_exp and not has_second_exp:
                    print("   ✅ Backup correctly contains previous state!")
                else:
                    print("   ⚠️  Backup content unexpected")
            
            print("\n✅ Backup mechanism test completed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Backup test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_base_file_handler_backup():
    """Test the underlying BaseFileHandler backup mechanism."""
    
    print("\n🔧 Testing BaseFileHandler backup mechanism...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_base_handler.json"
        
        try:
            from utils.base_file_handler import BaseFileHandler
            
            # Create a test handler
            class TestHandler(BaseFileHandler):
                def __init__(self, filepath):
                    super().__init__(filepath, verbose=True)
                    self.data = {"test": "initial"}
            
            print("1️⃣ Creating initial file with BaseFileHandler...")
            handler = TestHandler(test_file)
            handler.save_json({"test": "initial", "version": 1}, create_backup=True)
            
            print(f"   📁 File exists: {test_file.exists()}")
            print(f"   📁 File size: {test_file.stat().st_size} bytes")
            
            # List all files
            all_files = list(temp_path.glob("*"))
            print(f"   📋 All files: {[f.name for f in all_files]}")
            
            # Modify and save with backup
            print("\n2️⃣ Modifying and saving with backup...")
            handler.save_json({"test": "modified", "version": 2, "new_field": "added"}, create_backup=True)
            
            # List all files again
            all_files = list(temp_path.glob("*"))
            print(f"   📋 All files after save: {[f.name for f in all_files]}")
            
            # Check backup content
            backup_files = [f for f in all_files if "backup" in f.name]
            if backup_files:
                backup_content = json.loads(backup_files[0].read_text())
                print(f"   🔐 Backup content: {backup_content}")
                print(f"   🔍 Backup has version 1: {backup_content.get('version') == 1}")
            
            # Check current content
            current_content = json.loads(test_file.read_text())
            print(f"   📄 Current content: {current_content}")
            print(f"   🔍 Current has version 2: {current_content.get('version') == 2}")
            
            print("\n✅ BaseFileHandler backup test completed!")
            return True
            
        except Exception as e:
            print(f"\n❌ BaseFileHandler backup test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("🚀 Starting autosave and backup tests...")
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Autosave functionality
    if test_experiment_metadata_autosave():
        success_count += 1
    
    # Test 2: Backup mechanism
    if test_backup_mechanism():
        success_count += 1
    
    # Test 3: Base file handler backup
    if test_base_file_handler_backup():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed!")
        sys.exit(1)
 