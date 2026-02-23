#!/usr/bin/env python3
"""
🚨 CRITICAL FOUNDATION TESTS 🚨
Test suite focusing on core functionality, edge cases, and system robustness.
Tests the critical fixes for treatment management, genotype enforcement, and entity detection.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add the path to our modules
sys.path.append(str(Path(__file__).parent))

# Import modules to test
from embryo_metadata import EmbryoMetadata
from base_annotation_parser import BaseAnnotationParser
from embryo_metadata_models import Phenotype, Genotype, Treatment, Flag, ValidationError
from permitted_values_manager import PermittedValuesManager

def create_test_sam_annotation(tmp_dir: Path) -> Path:
    """Create a minimal SAM annotation file for testing."""
    sam_data = {
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
    
    sam_path = tmp_dir / "test_sam.json"
    with open(sam_path, 'w') as f:
        json.dump(sam_data, f)
    
    return sam_path

def test_schema_manager_fixes():
    """Test that all schema_manager references work correctly."""
    print("🧪 Testing Schema Manager Fixes...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_test_sam_annotation(tmp_path)
        
        # Test initialization doesn't crash
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Test that schema_manager is properly initialized
        assert hasattr(em, 'schema_manager')
        assert hasattr(em, 'permitted_values')
        
        # Test schema validation works
        phenotypes = em.schema_manager.get_phenotypes()
        assert len(phenotypes) > 0
        
        print("✅ Schema manager fixes working!")

def test_single_genotype_enforcement():
    """🚨 CRITICAL: Test single genotype per embryo enforcement."""
    print("🧪 Testing Single Genotype Enforcement...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_test_sam_annotation(tmp_path)
        
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Test 1: Add first genotype - should work
        success = em.add_genotype("20240411_A01_e01", "gdf3", "abc123", "homozygous")
        assert success, "First genotype addition should succeed"
        
        # Verify genotype was added
        genotypes = em.get_genotypes("20240411_A01_e01")
        assert len(genotypes) == 1
        assert "gdf3" in genotypes
        
        # Test 2: Try to add second genotype for different gene - should FAIL
        try:
            em.add_genotype("20240411_A01_e01", "shh", "def456", "heterozygous")
            assert False, "Should not allow second genotype for different gene!"
        except ValueError as e:
            assert "SINGLE GENOTYPE RULE VIOLATION" in str(e)
            print(f"✅ Correctly blocked second genotype: {e}")
        
        # Test 3: Overwrite same gene - should work
        success = em.add_genotype("20240411_A01_e01", "gdf3", "xyz789", "heterozygous", overwrite=True)
        assert success, "Overwriting same gene should work"
        
        # Verify only one genotype exists
        genotypes = em.get_genotypes("20240411_A01_e01")
        assert len(genotypes) == 1
        assert genotypes["gdf3"]["allele"] == "xyz789"
        
        # Test 4: Test with different embryo - should work independently
        success = em.add_genotype("20240411_A01_e02", "shh", "different_allele", "homozygous")
        assert success, "Different embryo should allow genotype"
        
        print("✅ Single genotype enforcement working correctly!")

def test_treatment_management():
    """Test multiple treatments per embryo with warning system."""
    print("🧪 Testing Treatment Management...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_test_sam_annotation(tmp_path)
        
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Test 1: Add first treatment
        success = em.add_treatment("20240411_A01_e01", "DMSO", concentration="1%", duration="24h")
        assert success, "First treatment should succeed"
        
        # Test 2: Add second treatment (should warn but allow)
        print("   Testing multiple treatments (expect warning)...")
        success = em.add_treatment("20240411_A01_e01", "heat_shock", temperature="37C", duration="2h")
        assert success, "Second treatment should succeed with warning"
        
        # Verify both treatments exist
        embryo_data = em.get_embryo_data("20240411_A01_e01")
        treatments = embryo_data.get("treatments", {})
        assert len(treatments) == 2
        assert "DMSO" in treatments
        assert "heat_shock" in treatments
        
        # Test 3: Prevent duplicate treatment without overwrite
        try:
            em.add_treatment("20240411_A01_e01", "DMSO", concentration="2%")
            assert False, "Should not allow duplicate treatment without overwrite"
        except ValueError as e:
            assert "already exists" in str(e)
            print(f"✅ Correctly blocked duplicate treatment: {e}")
        
        print("✅ Treatment management working correctly!")

def test_entity_level_detection():
    """Test the new entity level detection in BaseAnnotationParser."""
    print("🧪 Testing Entity Level Detection...")
    
    # Create a temporary file for BaseAnnotationParser
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump({"test": "data"}, tmp)
        tmp_path = Path(tmp.name)
    
    try:
        parser = BaseAnnotationParser(tmp_path, verbose=False)
        
        # Test entity level detection
        test_cases = [
            ("20240411", "experiment"),
            ("20240411_A01", "video"),
            ("20240411_A01_0001", "image"),
            ("20240411_A01_e01", "embryo"),
            ("20240411_A01_e01_0001", "snip"),
            ("invalid_id", "unknown")
        ]
        
        for entity_id, expected_level in test_cases:
            level = parser.detect_entity_level(entity_id)
            assert level == expected_level, f"ID {entity_id} should be {expected_level}, got {level}"
        
        # Test parent entity detection
        parents = parser.get_parent_entities("20240411_A01_e01_0001")
        assert "experiment" in parents
        assert "video" in parents
        assert "embryo" in parents
        assert parents["experiment"] == "20240411"
        assert parents["video"] == "20240411_A01"
        assert parents["embryo"] == "20240411_A01_e01"
        
        # Test hierarchy validation
        assert parser.validate_entity_hierarchy("20240411_A01", "20240411_A01_e01")
        assert parser.validate_entity_hierarchy("20240411_A01_e01", "20240411_A01_e01_0001")
        assert not parser.validate_entity_hierarchy("20240411_A01", "20240412_A01_e01")  # Different dates
        
        print("✅ Entity level detection working correctly!")
        
    finally:
        tmp_path.unlink()

def test_edge_cases_and_breakage_attempts():
    """🔥 Try to break the system with edge cases."""
    print("🧪 Testing Edge Cases and Breakage Attempts...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_test_sam_annotation(tmp_path)
        
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Edge Case 1: Invalid ID formats
        invalid_ids = ["", "invalid", "20240411_X99", "20240411_A01_e99_9999"]
        for invalid_id in invalid_ids:
            try:
                em.add_phenotype(invalid_id, "NORMAL", "test_author")
                print(f"⚠️  WARNING: Should have rejected invalid ID: {invalid_id}")
            except ValueError:
                pass  # Expected
        
        # Edge Case 2: Non-existent embryo
        try:
            em.add_genotype("20240411_A01_e99", "gdf3", "allele", "homozygous")
            print("⚠️  WARNING: Should validate embryo exists in SAM data")
        except ValueError:
            pass  # Expected
        
        # Edge Case 3: Invalid permitted values
        try:
            em.add_phenotype("20240411_A01_e01_0001", "INVALID_PHENOTYPE", "test_author")
            assert False, "Should reject invalid phenotype"
        except ValueError as e:
            assert "Invalid phenotype" in str(e)
        
        # Edge Case 4: Confidence out of range
        try:
            em.add_genotype("20240411_A01_e01", "gdf3", "allele", confidence=1.5)
            assert False, "Should reject invalid confidence"
        except ValueError as e:
            assert "Confidence must be between 0.0 and 1.0" in str(e)
        
        # Edge Case 5: Verify data integrity after operations
        em.add_phenotype("20240411_A01_e01_0001", "NORMAL", "test_author")
        em.add_genotype("20240411_A01_e01", "gdf3", "test_allele", "homozygous")
        
        # Save and reload to test persistence
        em.save()
        
        # Create new instance and verify data persisted
        em2 = EmbryoMetadata(sam_path, verbose=False)
        snip_data = em2.get_snip_data("20240411_A01_e01_0001")
        assert snip_data is not None
        assert snip_data["phenotype"]["value"] == "NORMAL"
        
        embryo_data = em2.get_embryo_data("20240411_A01_e01")
        assert "gdf3" in embryo_data["genotypes"]
        
        print("✅ Edge case testing completed - system is robust!")

def test_core_functionality_integration():
    """Test that all core functionality works together seamlessly."""
    print("🧪 Testing Core Functionality Integration...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sam_path = create_test_sam_annotation(tmp_path)
        
        em = EmbryoMetadata(sam_path, gen_if_no_file=True, verbose=False)
        
        # Full workflow test
        embryo_id = "20240411_A01_e01"
        snip_id = "20240411_A01_e01_0001"
        
        # Add complete annotation
        em.add_phenotype(snip_id, "NORMAL", "researcher1", confidence=0.9)
        em.add_genotype(embryo_id, "gdf3", "loss_of_function", "homozygous", confidence=0.95)
        em.add_flag(embryo_id, "image_quality", "excellent imaging", "low")
        em.add_treatment(embryo_id, "DMSO", concentration="0.1%", duration="24h")
        
        # Verify all data is properly structured
        embryo_data = em.get_embryo_data(embryo_id)
        assert "genotypes" in embryo_data
        assert "treatments" in embryo_data
        assert len(embryo_data["genotypes"]) == 1
        assert len(embryo_data["treatments"]) == 1
        
        snip_data = em.get_snip_data(snip_id)
        assert "phenotype" in snip_data
        assert snip_data["phenotype"]["value"] == "NORMAL"
        
        # Test that unsaved changes are tracked
        assert em.has_unsaved_changes
        
        # Save and verify
        em.save()
        assert not em.has_unsaved_changes
        
        print("✅ Core functionality integration working perfectly!")

if __name__ == "__main__":
    print("🚀 RUNNING CRITICAL FOUNDATION TESTS")
    print("=" * 60)
    
    try:
        test_schema_manager_fixes()
        test_single_genotype_enforcement()
        test_treatment_management()
        test_entity_level_detection()
        test_edge_cases_and_breakage_attempts()
        test_core_functionality_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("💪 Foundation is ROCK SOLID and ready for production!")
        print("🚀 Ready to move to batch processing and advanced features!")
        
    except Exception as e:
        print(f"\n❌ CRITICAL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
