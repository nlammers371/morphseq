#!/usr/bin/env python3
"""
Quick validation test for the fixes we made to address API inconsistencies.
"""

import tempfile
import json
from pathlib import Path

# Create test data
def create_test_data():
    sam_data = {
        "experiments": {
            "20240411": {
                "videos": {
                    "20240411_A01": {
                        "embryo_ids": ["20240411_A01_e01"],
                        "images": {
                            "20240411_A01_0000": {
                                "embryos": {
                                    "20240411_A01_e01": {
                                        "snip_id": "20240411_A01_e01_0000"
                                    }
                                }
                            },
                            "20240411_A01_0001": {
                                "embryos": {
                                    "20240411_A01_e01": {
                                        "snip_id": "20240411_A01_e01_0001"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "embryo_ids": ["20240411_A01_e01"],
        "snip_ids": ["20240411_A01_e01_0000", "20240411_A01_e01_0001"]
    }
    return sam_data

def test_api_methods():
    """Test that all expected API methods work correctly."""
    print("🧪 Testing API method fixes...")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sam_path = temp_path / "sam.json"
        
        # Write test data
        with open(sam_path, 'w') as f:
            json.dump(create_test_data(), f)
        
        # Import and test
        from embryo_metadata_refactored import EmbryoMetadata
        
        try:
            # Initialize
            em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
            print("✅ Initialization successful")
            
            # Test phenotype operations
            embryo_id = "20240411_A01_e01"
            snip_id = "20240411_A01_e01_0000"
            
            # Add phenotype
            success = em.add_phenotype(snip_id, "NORMAL", "test_user")
            assert success, "Failed to add phenotype"
            print("✅ add_phenotype() works")
            
            # Get phenotype
            phenotype = em.get_phenotype(snip_id)
            assert phenotype is not None, "Failed to get phenotype"
            print("✅ get_phenotype() works")
            
            # Test genotype operations
            success = em.add_genotype(embryo_id, "lmx1b", "WT", notes="test_user")
            assert success, "Failed to add genotype"
            print("✅ add_genotype() works")
            
            # Get genotype
            genotype = em.get_genotype(embryo_id)
            assert genotype is not None, "Failed to get genotype"
            print("✅ get_genotype() works")
            
            # Test flag operations
            success = em.add_flag(embryo_id, "quality", "Test flag")
            assert success, "Failed to add flag"
            print("✅ add_flag() works")
            
            # Get flag
            flag = em.get_flag(embryo_id, "quality")
            assert flag is not None, "Failed to get flag"
            print("✅ get_flag() works")
            
            print("\n🎉 All API methods working correctly!")
            return True
            
        except Exception as e:
            print(f"❌ API test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    test_api_methods()
