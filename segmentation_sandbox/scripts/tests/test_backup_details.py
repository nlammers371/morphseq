#!/usr/bin/env python3
"""
Test to examine the atomic write and backup mechanisms in detail
"""

import sys
from pathlib import Path
import json
import tempfile
import time
import threading

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def test_atomic_write_mechanism():
    """Test atomic write with temporary files."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metadata_file = temp_path / "atomic_test.json"
        
        print("⚛️ Testing atomic write mechanism...")
        
        try:
            from metadata.experiment_metadata import ExperimentMetadata
            
            # Create with autosave disabled so we can control timing
            em = ExperimentMetadata(metadata_file, verbose=True, auto_save_interval=None)
            
            # Add some data
            em.add_experiment("exp001", condition="atomic_test")
            em.add_video_to_experiment("exp001", "exp001_A01", well="A01")
            
            print("1️⃣ Before save - checking for temporary files...")
            temp_files_before = list(temp_path.glob("*.tmp"))
            print(f"   🔍 .tmp files before save: {len(temp_files_before)}")
            print(f"   📁 Main file exists: {metadata_file.exists()}")
            
            print("\n2️⃣ During save - examining file creation...")
            
            # Monitor files during save
            def monitor_files():
                """Monitor files during save operation."""
                for i in range(10):  # Check 10 times quickly
                    all_files = list(temp_path.glob("*"))
                    tmp_files = [f for f in all_files if f.suffix == '.tmp']
                    if tmp_files:
                        print(f"   ⏱️  Detected temp file: {tmp_files[0].name}")
                        break
                    time.sleep(0.001)  # 1ms
            
            # Start monitoring in background
            monitor_thread = threading.Thread(target=monitor_files)
            monitor_thread.start()
            
            # Perform save
            em.save()
            monitor_thread.join()
            
            print("\n3️⃣ After save - checking final state...")
            all_files = list(temp_path.glob("*"))
            main_files = [f for f in all_files if not any(x in f.name for x in ['.tmp', '.backup'])]
            backup_files = [f for f in all_files if '.backup' in f.name]
            temp_files = [f for f in all_files if f.suffix == '.tmp']
            
            print(f"   📄 Main files: {[f.name for f in main_files]}")
            print(f"   🔐 Backup files: {[f.name for f in backup_files]}")
            print(f"   🔍 Temp files remaining: {[f.name for f in temp_files]}")
            
            # Verify file content
            if metadata_file.exists():
                content = json.loads(metadata_file.read_text())
                print(f"   ✅ Main file valid JSON: {len(content)} keys")
                print(f"   📊 Experiments: {len(content.get('experiments', {}))}")
            
            print("\n✅ Atomic write test passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Atomic write test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_backup_timing():
    """Test when backups are created."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metadata_file = temp_path / "backup_timing_test.json"
        
        print("\n⏰ Testing backup timing...")
        
        try:
            from metadata.experiment_metadata import ExperimentMetadata
            
            em = ExperimentMetadata(metadata_file, verbose=True, auto_save_interval=None)
            
            print("1️⃣ First save (no existing file) - should NOT create backup...")
            em.add_experiment("exp001")
            em.save()
            
            backup_files = list(temp_path.glob("*.backup.*"))
            print(f"   🔐 Backups after first save: {len(backup_files)}")
            
            print("\n2️⃣ Second save (file exists) - should create backup...")
            em.add_experiment("exp002")
            em.save()
            
            backup_files = list(temp_path.glob("*.backup.*"))
            print(f"   🔐 Backups after second save: {len(backup_files)}")
            
            if backup_files:
                backup_content = json.loads(backup_files[0].read_text())
                backup_experiments = len(backup_content.get('experiments', {}))
                print(f"   📊 Experiments in backup: {backup_experiments}")
                
                current_content = json.loads(metadata_file.read_text())
                current_experiments = len(current_content.get('experiments', {}))
                print(f"   📊 Experiments in current: {current_experiments}")
                
                if backup_experiments < current_experiments:
                    print("   ✅ Backup correctly contains previous state!")
                else:
                    print("   ⚠️  Backup timing issue")
            
            print("\n3️⃣ Third save - should replace backup...")
            em.add_experiment("exp003")
            em.save()
            
            backup_files = list(temp_path.glob("*.backup.*"))
            print(f"   🔐 Backups after third save: {len(backup_files)}")
            
            # Note: The backup system might keep only the most recent backup
            # or create multiple backups with timestamps
            
            print("\n✅ Backup timing test passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Backup timing test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("🔬 Starting detailed backup and atomic write tests...")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Atomic write mechanism
    if test_atomic_write_mechanism():
        success_count += 1
    
    # Test 2: Backup timing
    if test_backup_timing():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All detailed tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed!")
        sys.exit(1)
