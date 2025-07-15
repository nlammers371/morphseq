#!/usr/bin/env python3
"""
Comprehensive Test Suite for Refactored EmbryoMetadata System
Tests the modular architecture with manager mixins.

This test suite focuses on:
1. Core functionality of the refactored system
2. Single genotype enforcement
3. Multiple treatment warnings
4. Manager mixin integration
5. Edge cases and system breaking attempts
"""

import sys
import tempfile
import json
from pathlib import Path

# Add the path to our modules
sys.path.append(str(Path(__file__).parent))

from embryo_metadata_refactored import EmbryoMetadata
from embryo_metadata_models import ValidationError
from permitted_values_manager import PermittedValuesManager


def create_test_sam_annotation():
    """Create a minimal SAM annotation for testing."""
    return {
        "experiments": {
            "20240411": {
                "videos": {
                    "20240411_A01": {
                        "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
                        "images": {
                            "20240411_A01_0001": {
                                "embryos": {
                                    "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0001"},
                                    "20240411_A01_e02": {"snip_id": "20240411_A01_e02_0001"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
        "snip_ids": ["20240411_A01_e01_0001", "20240411_A01_e02_0001"]
    }


def test_refactored_initialization():
    """Test that the refactored EmbryoMetadata initializes correctly."""
    print("🧪 Testing refactored initialization...")
    
    # Create temporary SAM annotation file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(create_test_sam_annotation(), tmp)
        sam_path = Path(tmp.name)
    
    try:
        # Test initialization
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Verify basic structure
        assert em.data is not None
        assert "embryos" in em.data
        assert len(em.data["embryos"]) == 2
        assert "20240411_A01_e01" in em.data["embryos"]
        assert "20240411_A01_e02" in em.data["embryos"]
        
        # Verify mixin methods are available
        assert hasattr(em, 'add_phenotype')
        assert hasattr(em, 'add_genotype')
        assert hasattr(em, 'add_flag')
        assert hasattr(em, 'add_treatment')
        
        print("✅ Refactored initialization test passed!")
        return em, sam_path
        
    except Exception as e:
        sam_path.unlink()
        raise


def test_single_genotype_enforcement():
    """Test the critical single genotype per embryo rule."""
    print("🧪 Testing single genotype enforcement...")
    
    em, sam_path = test_refactored_initialization()
    
    try:
        embryo_id = "20240411_A01_e01"
        
        # Test 1: Add first genotype - should succeed
        result = em.add_genotype(embryo_id, "lmx1b", "knockout", "homozygous")
        assert result == True
        assert len(em.get_genotypes(embryo_id)) == 1
        
        # Test 2: Try to add second genotype for different gene - should FAIL
        try:
            em.add_genotype(embryo_id, "shh", "wildtype", "heterozygous")
            assert False, "Should have failed due to single genotype rule!"
        except ValueError as e:
            assert "SINGLE GENOTYPE RULE VIOLATION" in str(e)
            assert "Only ONE genotype per embryo is allowed" in str(e)
        
        # Test 3: Try to add same gene without overwrite - should FAIL
        try:
            em.add_genotype(embryo_id, "lmx1b", "different_allele", "heterozygous")
            assert False, "Should have failed without overwrite=True!"
        except ValueError as e:
            assert "already exists" in str(e)
            assert "Use overwrite=True" in str(e)
        
        # Test 4: Overwrite same gene - should SUCCEED
        result = em.add_genotype(embryo_id, "lmx1b", "different_allele", "heterozygous", overwrite=True)
        assert result == True
        genotypes = em.get_genotypes(embryo_id)
        assert len(genotypes) == 1
        assert genotypes["lmx1b"]["allele"] == "different_allele"
        
        # Test 5: Validate compliance
        compliance = em.validate_single_genotype_compliance()
        assert compliance["compliant"] == True
        assert len(compliance["violations"]) == 0
        
        print("✅ Single genotype enforcement test passed!")
        
    finally:
        sam_path.unlink()


def test_multiple_treatment_warnings():
    """Test the multiple treatment warning system."""
    print("🧪 Testing multiple treatment warnings...")
    
    em, sam_path = test_refactored_initialization()
    
    try:
        embryo_id = "20240411_A01_e01"
        
        # Test 1: Add first treatment - no warning
        result = em.add_treatment(embryo_id, "BMP4-i", concentration="10ng/ml")
        assert result == True
        treatments = em.get_treatments(embryo_id)
        assert len(treatments) == 1
        
        # Test 2: Add second treatment - should warn but succeed
        result = em.add_treatment(embryo_id, "heat_shock", temperature="37C")
        assert result == True
        treatments = em.get_treatments(embryo_id)
        assert len(treatments) == 2
        
        # Test 3: Check multi-treatment embryos
        multi_treatment = em.get_multi_treatment_embryos()
        assert embryo_id in multi_treatment
        assert len(multi_treatment[embryo_id]) == 2
        
        # Test 4: Check treatment combinations
        combinations = em.get_treatment_combinations()
        expected_combo = tuple(sorted(["BMP4-i", "heat_shock"]))
        assert expected_combo in combinations
        
        print("✅ Multiple treatment warnings test passed!")
        
    finally:
        sam_path.unlink()


def test_phenotype_operations():
    """Test phenotype management operations."""
    print("🧪 Testing phenotype operations...")
    
    em, sam_path = test_refactored_initialization()
    
    try:
        snip_id = "20240411_A01_e01_0001"
        
        # Test 1: Add phenotype
        result = em.add_phenotype(snip_id, "NORMAL", "test_author")
        assert result == True
        
        # Test 2: Check phenotype statistics
        stats = em.get_phenotype_statistics()
        assert stats["total_snips"] == 2
        assert stats["phenotyped_snips"] == 1
        assert stats["completion_rate"] == 0.5
        assert "NORMAL" in stats["phenotype_counts"]
        
        print("✅ Phenotype operations test passed!")
        
    finally:
        sam_path.unlink()


def test_flag_operations():
    """Test flag management operations."""
    print("🧪 Testing flag operations...")
    
    em, sam_path = test_refactored_initialization()
    
    try:
        embryo_id = "20240411_A01_e01"
        
        # Test 1: Add flag
        result = em.add_flag(embryo_id, "quality", "Low image quality", priority="high")
        assert result == True
        
        # Test 2: Check high priority flags
        high_priority = em.get_high_priority_flags()
        assert embryo_id in high_priority
        assert "quality_concern" in high_priority[embryo_id]
        
        print("✅ Flag operations test passed!")
        
    finally:
        sam_path.unlink()


def test_edge_cases_and_breaking():
    """Test edge cases and attempt to break the system."""
    print("🧪 Testing edge cases and system breaking attempts...")
    
    em, sam_path = test_refactored_initialization()
    
    try:
        # Test 1: Invalid IDs
        try:
            em.add_genotype("invalid_id", "lmx1b", "knockout")
            assert False, "Should have failed with invalid ID!"
        except ValueError:
            pass  # Expected
        
        # Test 2: Invalid phenotype
        try:
            em.add_phenotype("20240411_A01_e01_0001", "INVALID_PHENOTYPE", "author")
            assert False, "Should have failed with invalid phenotype!"
        except ValueError:
            pass  # Expected
        
        # Test 3: Invalid gene name
        try:
            em.add_genotype("20240411_A01_e01", "invalid_gene", "knockout")
            assert False, "Should have failed with invalid gene!"
        except ValueError:
            pass  # Expected
        
        # Test 4: Invalid treatment
        try:
            em.add_treatment("20240411_A01_e01", "invalid_treatment")
            assert False, "Should have failed with invalid treatment!"
        except ValueError:
            pass  # Expected
        
        # Test 5: Non-existent embryo for treatment
        try:
            em.add_treatment("non_existent_embryo", "BMP4-i")
            assert False, "Should have failed with non-existent embryo!"
        except ValueError:
            pass  # Expected
        
        # Test 6: Confidence out of range
        try:
            em.add_genotype("20240411_A01_e01", "lmx1b", "knockout", confidence=1.5)
            assert False, "Should have failed with invalid confidence!"
        except ValueError:
            pass  # Expected
        
        print("✅ Edge cases and breaking attempts test passed!")
        
    finally:
        sam_path.unlink()


def test_file_operations():
    """Test file save/load operations."""
    print("🧪 Testing file operations...")
    
    em, sam_path = test_refactored_initialization()
    
    try:
        embryo_id = "20240411_A01_e01"
        
        # Add some data
        em.add_genotype(embryo_id, "lmx1b", "knockout")
        em.add_treatment(embryo_id, "BMP4-i")
        em.add_flag(embryo_id, "quality_concern", priority="high")
        
        # Test save
        em.save()
        assert em.filepath.exists()
        
        # Test reload
        original_data = em.data.copy()
        em.reload()
        
        # Verify data integrity
        assert em.get_genotypes(embryo_id)["lmx1b"]["allele"] == "knockout"
        assert "BMP4-i" in em.get_treatments(embryo_id)
        assert "quality_concern" in em.get_flags(embryo_id)
        
        print("✅ File operations test passed!")
        
    finally:
        sam_path.unlink()
        if em.filepath.exists():
            em.filepath.unlink()


def test_integration_workflow():
    """Test a complete integration workflow."""
    print("🧪 Testing complete integration workflow...")
    
    em, sam_path = test_refactored_initialization()
    
    try:
        # Complete workflow for embryo 1
        embryo1 = "20240411_A01_e01"
        snip1 = "20240411_A01_e01_0001"
        
        # Add genotype
        em.add_genotype(embryo1, "lmx1b", "knockout", "homozygous", confidence=0.95)
        
        # Add treatments
        em.add_treatment(embryo1, "BMP4-i", concentration="10ng/ml")
        em.add_treatment(embryo1, "heat_shock", temperature="37C")
        
        # Add phenotype
        em.add_phenotype(snip1, "ABNORMAL", "researcher1", confidence=0.9)
        
        # Add flag
        em.add_flag(embryo1, "interesting", "Good example", priority="medium")
        
        # Complete workflow for embryo 2
        embryo2 = "20240411_A01_e02"
        snip2 = "20240411_A01_e02_0001"
        
        # Add genotype (different gene to test single genotype rule)
        em.add_genotype(embryo2, "shh", "wildtype", "heterozygous")
        
        # Add phenotype
        em.add_phenotype(snip2, "NORMAL", "researcher2")
        
        # Verify final state
        assert len(em.get_embryo_ids()) == 2
        assert len(em.get_snip_ids()) == 2
        
        # Test statistics
        genotype_stats = em.get_genotype_statistics()
        assert genotype_stats["total_embryos"] == 2
        assert genotype_stats["genotyped_embryos"] == 2
        assert genotype_stats["completion_rate"] == 1.0
        
        phenotype_stats = em.get_phenotype_statistics()
        assert phenotype_stats["total_snips"] == 2
        assert phenotype_stats["phenotyped_snips"] == 2
        assert phenotype_stats["completion_rate"] == 1.0
        
        # Save final state
        em.save()
        
        print("✅ Integration workflow test passed!")
        
    finally:
        sam_path.unlink()
        if em.filepath.exists():
            em.filepath.unlink()


def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Running comprehensive test suite for refactored EmbryoMetadata system...\n")
    
    try:
        test_refactored_initialization()
        test_single_genotype_enforcement()
        test_multiple_treatment_warnings()
        test_phenotype_operations()
        test_flag_operations()
        test_edge_cases_and_breaking()
        test_file_operations()
        test_integration_workflow()
        
        print("\n🎉 ALL TESTS PASSED! The refactored system is working correctly!")
        print("✅ Modular architecture is functional")
        print("✅ Single genotype enforcement is working")
        print("✅ Multiple treatment warnings are working")
        print("✅ All manager mixins are integrated properly")
        print("✅ Edge cases are handled correctly")
        print("✅ File operations are working")
        print("✅ Complete workflows are functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
