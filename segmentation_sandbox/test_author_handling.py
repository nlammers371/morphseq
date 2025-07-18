#!/usr/bin/env python3
"""
Quick test to verify that author handling is working correctly
"""

import sys
import tempfile
import os
from pathlib import Path

# Add the utils directory to the path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/utils')

from experiment_data_qc_utils import ExperimentDataQC

def test_author_handling():
    """Test that author parameter handling works correctly"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in temporary directory: {temp_dir}")
        
        # Initialize QC system with a default author
        qc = ExperimentDataQC(temp_dir, author_designation="test_author")
        
        # Initialize QC structure (create basic structure for testing)
        print("Setting up test QC structure...")
        qc._qc_data = {
            "valid_qc_flag_categories": {
                "image_level": {
                    "BLUR": "Image is blurry",
                    "DRY_WELL": "Well dried out"
                }
            },
            "experiments": {
                "20241215": {
                    "flags": [], "authors": [], "notes": [],
                    "videos": {
                        "20241215_A01": {
                            "flags": [], "authors": [], "notes": [],
                            "images": {
                                "20241215_A01_t001": {"flags": [], "authors": [], "notes": []},
                                "20241215_A01_t002": {"flags": [], "authors": [], "notes": []},
                                "20241215_A01_t003": {"flags": [], "authors": [], "notes": []}
                            },
                            "embryos": {}
                        }
                    }
                }
            }
        }
        
        # Test 1: Using default author (None passed to gen_flag_batch)
        print("\n1. Testing gen_flag_batch with explicit None author...")
        try:
            batch = qc.gen_flag_batch("image", "20241215_A01_t001", "BLUR", author=None, notes="Test with None author")
            # This should NOT work as expected - author=None should create None author
            print(f"   Generated batch entry: {batch[0] if batch else 'None'}")
            expected_author = batch[0]['author'] if batch else None
            if expected_author is None:
                print("   ✅ gen_flag_batch correctly preserves None when explicitly passed")
            else:
                print(f"   ❌ gen_flag_batch should preserve None but got: {expected_author}")
        except Exception as e:
            print(f"❌ gen_flag_batch with author=None failed: {e}")
            return False
        
        # Test 2: Using class method with default author (no author parameter)
        print("\n2. Testing flag_image with default author (no author parameter)...")
        try:
            # This should use the instance's author_designation
            qc.flag_image("20241215_A01_t002", "DRY_WELL", notes="Test with default author")
            print(f"✅ flag_image without author parameter succeeded")
        except Exception as e:
            print(f"❌ flag_image without author parameter failed: {e}")
            return False
        
        # Test 3: Using explicit author
        print("\n3. Testing flag_image with explicit author...")
        try:
            qc.flag_image("20241215_A01_t003", "BLUR", author="explicit_author", notes="Test with explicit author")
            print(f"✅ flag_image with explicit author succeeded")
        except Exception as e:
            print(f"❌ flag_image with explicit author failed: {e}")
            return False
        
        # Test 4: Using class method with None author (should use default)
        print("\n4. Testing flag_image with None author (should use default)...")
        try:
            qc.flag_image("20241215_A01_t001", "DRY_WELL", author=None, notes="Test with None author should use default")
            print(f"✅ flag_image with author=None succeeded")
        except Exception as e:
            print(f"❌ flag_image with author=None failed: {e}")
            return False
        
        # Test 5: Check the actual data to verify authors are correct
        print("\n5. Checking stored author data...")
        try:
            # Check the actual QC data structure
            exp_data = qc._qc_data["experiments"]["20241215"]["videos"]["20241215_A01"]["images"]
            
            print(f"   Raw t001 data: {exp_data.get('20241215_A01_t001', 'Not found')}")
            print(f"   Raw t002 data: {exp_data.get('20241215_A01_t002', 'Not found')}")
            print(f"   Raw t003 data: {exp_data.get('20241215_A01_t003', 'Not found')}")
            
            # Also check using the get_image_flags method
            image_flags_1 = qc.get_image_flags("20241215_A01_t001")
            image_flags_2 = qc.get_image_flags("20241215_A01_t002")
            image_flags_3 = qc.get_image_flags("20241215_A01_t003")
            
            print(f"   get_image_flags t001: {image_flags_1}")
            print(f"   get_image_flags t002: {image_flags_2}")
            print(f"   get_image_flags t003: {image_flags_3}")
            
            # Check that default author was used for t001 (None -> default)
            t001_authors = exp_data.get('20241215_A01_t001', {}).get('authors', [])
            if 'test_author' in t001_authors:
                print("   ✅ Default author correctly applied to t001 (None -> default)")
            else:
                print(f"   ❌ Default author not correctly applied to t001: {t001_authors}")
                return False
                
            # Check that default author was used for t002 (no author param)
            t002_authors = exp_data.get('20241215_A01_t002', {}).get('authors', [])
            if 'test_author' in t002_authors:
                print("   ✅ Default author correctly applied to t002 (no param)")
            else:
                print(f"   ❌ Default author not correctly applied to t002: {t002_authors}")
                return False
                
            # Check that explicit author was used for t003
            t003_authors = exp_data.get('20241215_A01_t003', {}).get('authors', [])
            if 'explicit_author' in t003_authors:
                print("   ✅ Explicit author correctly applied to t003")
            else:
                print(f"   ❌ Explicit author not correctly applied to t003: {t003_authors}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking stored authors: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 All author handling tests passed!")
        return True

if __name__ == "__main__":
    success = test_author_handling()
    sys.exit(0 if success else 1)
