#!/usr/bin/env python3
"""
Test data access helpers implementation
"""
import sys
import os
import tempfile
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_test_sam2_data():
    """Create minimal SAM2 data for testing"""
    return {
        "experiments": {
            "test_exp": {
                "videos": {
                    "test_exp_A01": {
                        "embryo_ids": ["test_exp_A01_e01", "test_exp_A01_e02"],
                        "images": {
                            "test_exp_A01_ch00_t0100": {
                                "embryos": {
                                    "test_exp_A01_e01": {"snip_id": "test_exp_A01_e01_s0100"},
                                    "test_exp_A01_e02": {"snip_id": "test_exp_A01_e02_s0100"}
                                }
                            },
                            "test_exp_A01_ch00_t0101": {
                                "embryos": {
                                    "test_exp_A01_e01": {"snip_id": "test_exp_A01_e01_s0101"},
                                    "test_exp_A01_e02": {"snip_id": "test_exp_A01_e02_s0101"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

def test_data_helpers():
    """Test the data access helpers"""
    print("🔍 Testing data access helpers...")
    
    try:
        from scripts.annotations.embryo_metadata import EmbryoMetadata
        
        # Create temporary SAM2 file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(create_test_sam2_data(), f)
            sam2_path = f.name
        
        try:
            # Create metadata from SAM2
            metadata = EmbryoMetadata(sam2_path, gen_if_no_file=True, verbose=False)
            
            print("✅ EmbryoMetadata created successfully")
            print(f"   Embryo count: {metadata.embryo_count}")
            print(f"   Snip count: {metadata.snip_count}")
            
            # Test _get_embryo_data helper
            embryo_id = "test_exp_A01_e01"
            embryo_data = metadata._get_embryo_data(embryo_id)
            print(f"✅ _get_embryo_data works for {embryo_id}")
            print(f"   Keys: {list(embryo_data.keys())}")
            
            # Test _get_snip_data helper
            snip_id = "test_exp_A01_e01_s0100"
            snip_data = metadata._get_snip_data(snip_id)
            print(f"✅ _get_snip_data works for {snip_id}")
            print(f"   Keys: {list(snip_data.keys())}")
            
            # Test error handling - non-existent embryo
            try:
                metadata._get_embryo_data("nonexistent_embryo")
                print("❌ Should have raised KeyError for non-existent embryo")
                return False
            except KeyError as e:
                print(f"✅ Proper error handling for non-existent embryo: {e}")
            
            # Test error handling - non-existent snip
            try:
                metadata._get_snip_data("nonexistent_snip_s0000")
                print("❌ Should have raised KeyError for non-existent snip")
                return False
            except KeyError as e:
                print(f"✅ Proper error handling for non-existent snip: {e}")
            
            print("✅ All data helper tests passed!")
            return True
            
        finally:
            # Clean up temporary file
            os.unlink(sam2_path)
            # Clean up any generated metadata file
            metadata_path = Path(sam2_path).with_name(Path(sam2_path).stem + '_embryo_metadata.json')
            if metadata_path.exists():
                metadata_path.unlink()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run data helpers test"""
    print("=" * 60)
    print("Data Access Helpers Test")
    print("=" * 60)
    
    success = test_data_helpers()
    
    print("=" * 60)
    if success:
        print("✅ Data access helpers implementation working correctly!")
    else:
        print("❌ Data access helpers need fixes")
    print("=" * 60)

if __name__ == "__main__":
    main()