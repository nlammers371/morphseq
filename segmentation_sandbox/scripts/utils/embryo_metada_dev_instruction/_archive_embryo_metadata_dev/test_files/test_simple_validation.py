#!/usr/bin/env python3
"""
Simple, efficient test to validate core refactored system functionality.
Focuses on essential validation without extensive compute.
"""

import tempfile
import json
from pathlib import Path
from embryo_metadata_refactored import EmbryoMetadata

def create_minimal_sam_data():
    """Create minimal SAM annotation data for testing."""
    return {
        "experiments": {
            "20240411": {
                "videos": {
                    "20240411_A01": {
                        "embryo_ids": ["20240411_A01_e01"],
                        "images": {
                            "20240411_A01_0001": {
                                "embryos": {
                                    "20240411_A01_e01": {
                                        "snips": {
                                            "20240411_A01_e01_0001": {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

def test_basic_functionality():
    """Test basic functionality efficiently."""
    print("🚀 Simple validation test for refactored system...")
    
    # Create temp SAM file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(create_minimal_sam_data(), f)
        sam_path = Path(f.name)
    
    try:
        # Test 1: Initialization (disable auto validation for simple test)
        em = EmbryoMetadata(sam_annotation_path=sam_path, gen_if_no_file=True, verbose=False, auto_validate=False)
        print("✅ Initialization successful")
        
        embryo_id = "20240411_A01_e01"
        snip_id = "20240411_A01_e01_0001"
        
        # Test 2: Single genotype enforcement
        result1 = em.add_genotype(embryo_id, "WT", "wild-type", "homozygous")
        assert result1, "Should add first genotype"
        
        try:
            em.add_genotype(embryo_id, "lmx1b", "knockout", "homozygous")
            assert False, "Should not allow second genotype"
        except ValueError as e:
            assert "Cannot add additional gene" in str(e)
        print("✅ Single genotype enforcement working")
        
        # Test 3: Multiple treatments (should work with warnings)
        result2 = em.add_treatment(embryo_id, "shh-i", concentration="5μM")
        result3 = em.add_treatment(embryo_id, "BMP4-i", concentration="10μM")
        assert result2 and result3, "Should allow multiple treatments"
        print("✅ Multiple treatments working")
        
        # Test 4: Basic phenotype
        result4 = em.add_phenotype(snip_id, "NORMAL", "test_author")
        assert result4, "Should add phenotype"
        print("✅ Phenotype operations working")
        
        # Test 5: Basic flag
        result5 = em.add_flag(embryo_id, "quality", "Test flag", priority="medium")
        assert result5, "Should add flag"
        print("✅ Flag operations working")
        
        print("🎉 ALL CORE FUNCTIONALITY VALIDATED!")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False
    finally:
        sam_path.unlink(missing_ok=True)

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
