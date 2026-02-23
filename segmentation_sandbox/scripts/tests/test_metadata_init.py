#!/usr/bin/env python3
"""
Test the metadata initialization functionality.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_organization.data_organizer import DataOrganizer

def test_empty_directory_metadata_init():
    """Test that empty metadata file is created when no experiments found."""
    print("🧪 Testing empty directory metadata initialization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create empty source directory (no experiments)
        empty_source = Path(temp_dir) / "empty_source"
        empty_source.mkdir()
        
        # Create output directory
        output_dir = Path(temp_dir) / "output"
        
        # Run DataOrganizer on empty directory
        DataOrganizer.process_experiments(
            source_dir=empty_source,
            output_dir=output_dir,
            verbose=True
        )
        
        # Check if metadata file was created
        metadata_path = output_dir / "raw_data_organized" / "experiment_metadata.json"
        
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            print("✅ Empty metadata file created successfully!")
            print(f"   📋 Experiments count: {len(metadata.get('experiments', {}))}")
            print(f"   📅 Creation time: {metadata.get('file_info', {}).get('creation_time', 'N/A')}")
            return True
        else:
            print("❌ Metadata file was not created!")
            return False

def test_no_stitch_files_metadata_init():
    """Test metadata init when experiment dirs exist but have no stitch files."""
    print("\n🧪 Testing directory with no stitch files...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create source directory with experiment folders but no stitch files
        source_dir = Path(temp_dir) / "source"
        exp_dir = source_dir / "20240101"
        exp_dir.mkdir(parents=True)
        
        # Add some non-stitch files
        (exp_dir / "random_file.txt").write_text("test")
        (exp_dir / "another_file.jpg").write_text("test")  # No "_stitch" in name
        
        output_dir = Path(temp_dir) / "output"
        
        # Run DataOrganizer
        DataOrganizer.process_experiments(
            source_dir=source_dir,
            output_dir=output_dir,
            verbose=True
        )
        
        # Check if metadata file was created
        metadata_path = output_dir / "raw_data_organized" / "experiment_metadata.json"
        
        if metadata_path.exists():
            print("✅ Empty metadata file created for directory with no stitch files!")
            return True
        else:
            print("❌ Metadata file was not created!")
            return False

def main():
    """Run metadata initialization tests."""
    print("🚀 Testing DataOrganizer Metadata Initialization")
    print("=" * 50)
    
    tests = [
        test_empty_directory_metadata_init,
        test_no_stitch_files_metadata_init
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   ✅ Passed: {sum(results)}/{len(results)}")
    print(f"   ❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All metadata initialization tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check implementation.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
