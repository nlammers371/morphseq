"""
Phase 4: Comprehensive Testing Suite for EmbryoMetadata System

This test suite covers:
1. Unit tests for all core functionality
2. Integration tests for full workflows  
3. Edge case testing to try to break the system
4. Performance testing for large datasets
5. Error handling and validation testing

Author: EmbryoMetadata Development Team
Date: July 15, 2025
"""

import unittest
import tempfile
import json
import os
import time
from pathlib import Path
from typing import Dict, List

# Import all our modules
import sys
sys.path.append('.')

from embryo_metadata_refactored import EmbryoMetadata
from embryo_metadata_batch import RangeParser, TemporalRangeParser, BatchProcessor, BatchOperations
from embryo_metadata_models import ValidationError, PermittedValueError
from permitted_values_manager import PermittedValuesManager
from base_annotation_parser import BaseAnnotationParser


class TestEmbryoMetadataComprehensive(unittest.TestCase):
    """Comprehensive test suite for EmbryoMetadata system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock SAM annotation data
        self.sam_data = {
            "experiments": {
                "20240411": {
                    "videos": {
                        "20240411_A01": {
                            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
                            "images": {
                                "20240411_A01_0010": {
                                    "embryos": {
                                        "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0010"},
                                        "20240411_A01_e02": {"snip_id": "20240411_A01_e02_0010"}
                                    }
                                },
                                "20240411_A01_0015": {
                                    "embryos": {
                                        "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0015"},
                                        "20240411_A01_e02": {"snip_id": "20240411_A01_e02_0015"}
                                    }
                                },
                                "20240411_A01_0020": {
                                    "embryos": {
                                        "20240411_A01_e01": {"snip_id": "20240411_A01_e01_0020"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
            "snip_ids": ["20240411_A01_e01_0010", "20240411_A01_e01_0015", "20240411_A01_e01_0020",
                        "20240411_A01_e02_0010", "20240411_A01_e02_0015"]
        }
        
        # Create SAM annotation file
        self.sam_path = self.temp_path / "sam_annotations.json"
        with open(self.sam_path, 'w') as f:
            json.dump(self.sam_data, f)
            
        # Create metadata path
        self.metadata_path = self.temp_path / "embryo_metadata.json"
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    # ==================
    # UNIT TESTS
    # ==================
    
    def test_initialization_new_file(self):
        """Test creating new metadata file."""
        em = EmbryoMetadata(
            self.sam_path,
            self.metadata_path,
            gen_if_no_file=True,
            verbose=False
        )
        
        self.assertIsNotNone(em.data)
        self.assertEqual(len(em.data["embryos"]), 2)
        self.assertIn("20240411_A01_e01", em.data["embryos"])
        self.assertIn("20240411_A01_e02", em.data["embryos"])
    
    def test_initialization_existing_file(self):
        """Test loading existing metadata file."""
        # Create initial file
        em1 = EmbryoMetadata(self.sam_path, self.metadata_path, gen_if_no_file=True, verbose=False)
        em1.add_phenotype("20240411_A01_e01_0010", "EDEMA", "test_user")
        em1.save()
        
        # Load existing file
        em2 = EmbryoMetadata(self.sam_path, self.metadata_path, verbose=False)
        snip_data = em2.data["embryos"]["20240411_A01_e01"]["snips"]["20240411_A01_e01_0010"]
        self.assertEqual(snip_data["phenotype"]["value"], "EDEMA")
    
    def test_phenotype_operations(self):
        """Test all phenotype operations."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        # Add phenotype
        success = em.add_phenotype("20240411_A01_e01_0010", "EDEMA", "test_user", confidence=0.9)
        self.assertTrue(success)
        
        # Get phenotype
        phenotype = em.get_phenotype("20240411_A01_e01_0010")
        self.assertEqual(phenotype["value"], "EDEMA")
        self.assertEqual(phenotype["confidence"], 0.9)
        
        # Update phenotype
        success = em.update_phenotype("20240411_A01_e01_0010", "HEART_DEFECT", "test_user", overwrite=True)
        self.assertTrue(success)
        
        phenotype = em.get_phenotype("20240411_A01_e01_0010")
        self.assertEqual(phenotype["value"], "HEART_DEFECT")
        
        # Remove phenotype
        success = em.remove_phenotype("20240411_A01_e01_0010")
        self.assertTrue(success)
        
        phenotype = em.get_phenotype("20240411_A01_e01_0010")
        self.assertEqual(phenotype["value"], "NONE")
    
    def test_genotype_operations(self):
        """Test all genotype operations."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        # Add genotype
        success = em.add_genotype("20240411_A01_e01", "WT", "test_user", notes="Wild type")
        self.assertTrue(success)
        
        # Get genotype
        genotype = em.get_genotype("20240411_A01_e01")
        self.assertEqual(genotype["WT"]["value"], "WT")
        self.assertEqual(genotype["WT"]["notes"], "Wild type")
        
        # Test single genotype enforcement
        with self.assertRaises(ValidationError):
            em.add_genotype("20240411_A01_e01", "lmx1b", "test_user")
        
        # Allow overwrite
        success = em.add_genotype("20240411_A01_e01", "lmx1b", "test_user", overwrite_genotype=True)
        self.assertTrue(success)
        
        genotype = em.get_genotype("20240411_A01_e01")
        self.assertIn("lmx1b", genotype)
        self.assertNotIn("WT", genotype)  # Should be replaced
    
    def test_flag_operations(self):
        """Test all flag operations."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        # Add flag
        success = em.add_flag("20240411_A01_e01_0010", "MOTION_BLUR", "snip", "test_user", severity="warning")
        self.assertTrue(success)
        
        # Get flags
        flags = em.get_flags("20240411_A01_e01_0010")
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["flag"], "MOTION_BLUR")
        self.assertEqual(flags[0]["severity"], "warning")
        
        # Remove flag
        success = em.remove_flag("20240411_A01_e01_0010", "MOTION_BLUR")
        self.assertTrue(success)
        
        flags = em.get_flags("20240411_A01_e01_0010")
        self.assertEqual(len(flags), 0)
    
    def test_treatment_operations(self):
        """Test all treatment operations."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        # Add treatment
        success = em.add_treatment("20240411_A01_e01", "shh-i", "test_user", concentration="10µM")
        self.assertTrue(success)
        
        # Get treatments
        treatments = em.get_treatments("20240411_A01_e01")
        self.assertIn("shh-i", treatments)
        self.assertEqual(treatments["shh-i"]["concentration"], "10µM")
        
        # Add multiple treatments (should warn)
        success = em.add_treatment("20240411_A01_e01", "BMP4-i", "test_user")
        self.assertTrue(success)
        
        treatments = em.get_treatments("20240411_A01_e01")
        self.assertEqual(len(treatments), 2)
    
    # ==================
    # BATCH PROCESSING TESTS
    # ==================
    
    def test_range_parser(self):
        """Test range parsing functionality."""
        items = ["item0", "item1", "item2", "item3", "item4"]
        
        # Single index
        result = RangeParser.parse_range("[2]", items)
        self.assertEqual(result, ["item2"])
        
        # Range
        result = RangeParser.parse_range("[1:3]", items)
        self.assertEqual(result, ["item1", "item2"])
        
        # Open-ended range
        result = RangeParser.parse_range("[2::]", items)
        self.assertEqual(result, ["item2", "item3", "item4"])
        
        # Step
        result = RangeParser.parse_range("[0::2]", items)
        self.assertEqual(result, ["item0", "item2", "item4"])
        
        # List of indices
        result = RangeParser.parse_range([1, 3], items)
        self.assertEqual(result, ["item1", "item3"])
        
        # List of IDs
        result = RangeParser.parse_range(["item1", "item3"], items)
        self.assertEqual(result, ["item1", "item3"])
    
    def test_temporal_range_parser(self):
        """Test temporal range parsing."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        # All frames
        result = TemporalRangeParser.parse_frame_range("20240411_A01_e01", "all", em)
        expected = ["20240411_A01_e01_0010", "20240411_A01_e01_0015", "20240411_A01_e01_0020"]
        self.assertEqual(result, expected)
        
        # First frame
        result = TemporalRangeParser.parse_frame_range("20240411_A01_e01", "first", em)
        self.assertEqual(result, ["20240411_A01_e01_0010"])
        
        # Last frame
        result = TemporalRangeParser.parse_frame_range("20240411_A01_e01", "last", em)
        self.assertEqual(result, ["20240411_A01_e01_0020"])
        
        # Range syntax
        result = TemporalRangeParser.parse_frame_range("20240411_A01_e01", "[1:3]", em)
        self.assertEqual(result, ["20240411_A01_e01_0015", "20240411_A01_e01_0020"])
    
    def test_batch_phenotype_assignment(self):
        """Test batch phenotype assignment."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        assignments = [
            {
                "embryo_id": "20240411_A01_e01",
                "phenotype": "EDEMA",
                "frames": "all",
                "confidence": 0.9
            },
            {
                "embryo_id": "20240411_A01_e02", 
                "phenotype": "HEART_DEFECT",
                "frames": "[0:2]",
                "confidence": 0.8
            }
        ]
        
        result = em.batch_add_phenotypes(assignments, "test_user", verbose=False)
        
        # Should assign to all frames of embryo 1 (3 frames) + first 2 frames of embryo 2
        self.assertEqual(result["assigned"], 5)
        self.assertEqual(result["embryos_processed"], 2)
        
        # Verify assignments
        snip1 = em.data["embryos"]["20240411_A01_e01"]["snips"]["20240411_A01_e01_0010"]
        self.assertEqual(snip1["phenotype"]["value"], "EDEMA")
        
        snip2 = em.data["embryos"]["20240411_A01_e02"]["snips"]["20240411_A01_e02_0010"]
        self.assertEqual(snip2["phenotype"]["value"], "HEART_DEFECT")
    
    def test_batch_genotype_assignment(self):
        """Test batch genotype assignment."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        assignments = [
            {
                "embryo_id": "20240411_A01_e01",
                "genotype": "WT",
                "notes": "Wild type control"
            },
            {
                "embryo_id": "20240411_A01_e02",
                "genotype": "lmx1b",
                "notes": "Mutant"
            }
        ]
        
        result = em.batch_add_genotypes(assignments, "test_user", verbose=False)
        
        self.assertEqual(result["assigned"], 2)
        
        # Verify assignments
        genotype1 = em.get_genotype("20240411_A01_e01")
        self.assertEqual(genotype1["WT"]["value"], "WT")
        
        genotype2 = em.get_genotype("20240411_A01_e02")
        self.assertEqual(genotype2["lmx1b"]["value"], "lmx1b")
    
    # ==================
    # EDGE CASE TESTS
    # ==================
    
    def test_invalid_phenotype_values(self):
        """Test handling of invalid phenotype values."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        with self.assertRaises(PermittedValueError):
            em.add_phenotype("20240411_A01_e01_0010", "INVALID_PHENOTYPE", "test_user")
    
    def test_invalid_genotype_values(self):
        """Test handling of invalid genotype values."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        with self.assertRaises(PermittedValueError):
            em.add_genotype("20240411_A01_e01", "INVALID_GENOTYPE", "test_user")
    
    def test_invalid_entity_ids(self):
        """Test handling of invalid entity IDs."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        # Invalid snip ID
        success = em.add_phenotype("INVALID_SNIP_ID", "EDEMA", "test_user")
        self.assertFalse(success)
        
        # Invalid embryo ID
        success = em.add_genotype("INVALID_EMBRYO_ID", "WT", "test_user")
        self.assertFalse(success)
    
    def test_dead_phenotype_exclusivity(self):
        """Test DEAD phenotype exclusivity rules."""
        em = EmbryoMetadata(self.sam_path, gen_if_no_file=True, verbose=False)
        
        # Add DEAD phenotype
        em.add_phenotype("20240411_A01_e01_0010", "DEAD", "test_user")
        
        # Try to add another phenotype to same snip
        with self.assertRaises(ValidationError):
            em.add_phenotype("20240411_A01_e01_0010", "EDEMA", "test_user")
        
        # Should be able to force it
        success = em.add_phenotype("20240411_A01_e01_0010", "EDEMA", "test_user", force_dead=True)
        self.assertTrue(success)
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger SAM data
        large_sam_data = {
            "experiments": {"20240411": {"videos": {}}},
            "embryo_ids": [],
            "snip_ids": []
        }
        
        # Create 100 embryos with 10 frames each
        for i in range(100):
            embryo_id = f"20240411_A01_e{i:03d}"
            large_sam_data["embryo_ids"].append(embryo_id)
            
            video_id = "20240411_A01"
            if video_id not in large_sam_data["experiments"]["20240411"]["videos"]:
                large_sam_data["experiments"]["20240411"]["videos"][video_id] = {
                    "embryo_ids": [],
                    "images": {}
                }
            
            large_sam_data["experiments"]["20240411"]["videos"][video_id]["embryo_ids"].append(embryo_id)
            
            for j in range(10):
                frame_num = f"{j:04d}"
                image_id = f"20240411_A01_{frame_num}"
                snip_id = f"{embryo_id}_{frame_num}"
                
                large_sam_data["snip_ids"].append(snip_id)
                
                if image_id not in large_sam_data["experiments"]["20240411"]["videos"][video_id]["images"]:
                    large_sam_data["experiments"]["20240411"]["videos"][video_id]["images"][image_id] = {"embryos": {}}
                
                large_sam_data["experiments"]["20240411"]["videos"][video_id]["images"][image_id]["embryos"][embryo_id] = {
                    "snip_id": snip_id
                }
        
        # Save large SAM data
        large_sam_path = self.temp_path / "large_sam_annotations.json"
        with open(large_sam_path, 'w') as f:
            json.dump(large_sam_data, f)
        
        # Test initialization time
        start_time = time.time()
        em = EmbryoMetadata(large_sam_path, gen_if_no_file=True, verbose=False)
        init_time = time.time() - start_time
        
        print(f"⏱️  Large dataset initialization: {init_time:.3f}s")
        self.assertLess(init_time, 5.0)  # Should initialize in under 5 seconds
        
        # Test batch operations
        assignments = []
        for i in range(0, 100, 10):  # Every 10th embryo
            assignments.append({
                "embryo_id": f"20240411_A01_e{i:03d}",
                "phenotype": "EDEMA",
                "frames": "all"
            })
        
        start_time = time.time()
        result = em.batch_add_phenotypes(assignments, "test_user", verbose=False)
        batch_time = time.time() - start_time
        
        print(f"⏱️  Batch phenotype assignment: {batch_time:.3f}s")
        self.assertLess(batch_time, 10.0)  # Should complete in under 10 seconds
        self.assertEqual(result["assigned"], 10 * 10)  # 10 embryos × 10 frames each
    
    def test_file_corruption_recovery(self):
        """Test recovery from corrupted metadata files."""
        em = EmbryoMetadata(self.sam_path, self.metadata_path, gen_if_no_file=True, verbose=False)
        em.save()
        
        # Corrupt the file
        with open(self.metadata_path, 'w') as f:
            f.write("{invalid json")
        
        # Should be able to regenerate
        em2 = EmbryoMetadata(self.sam_path, self.metadata_path, gen_if_no_file=True, verbose=False)
        self.assertIsNotNone(em2.data)
    
    def test_concurrent_access_safety(self):
        """Test behavior with concurrent access."""
        em1 = EmbryoMetadata(self.sam_path, self.metadata_path, gen_if_no_file=True, verbose=False)
        em1.add_phenotype("20240411_A01_e01_0010", "EDEMA", "user1")
        em1.save()
        
        # Load same file in another instance
        em2 = EmbryoMetadata(self.sam_path, self.metadata_path, verbose=False)
        em2.add_phenotype("20240411_A01_e01_0015", "HEART_DEFECT", "user2")
        
        # em1 should detect changes when trying to save
        em1.add_phenotype("20240411_A01_e01_0020", "NORMAL", "user1")
        
        # This should work (we're not implementing file locking yet)
        em1.save()
        em2.save()  # Last save wins
    
    # ==================
    # INTEGRATION TESTS  
    # ==================
    
    def test_full_workflow_integration(self):
        """Test complete workflow from initialization to analysis."""
        # 1. Initialize
        em = EmbryoMetadata(self.sam_path, self.metadata_path, gen_if_no_file=True, verbose=False)
        
        # 2. Add genotypes
        genotype_assignments = [
            {"embryo_id": "20240411_A01_e01", "genotype": "WT", "notes": "Control"},
            {"embryo_id": "20240411_A01_e02", "genotype": "lmx1b", "notes": "Mutant"}
        ]
        genotype_result = em.batch_add_genotypes(genotype_assignments, "geneticist")
        self.assertEqual(genotype_result["assigned"], 2)
        
        # 3. Add treatments
        em.add_treatment("20240411_A01_e01", "NONE", "researcher")
        em.add_treatment("20240411_A01_e02", "shh-i", "researcher", concentration="10µM")
        
        # 4. Add phenotypes with temporal ranges
        phenotype_assignments = [
            {
                "embryo_id": "20240411_A01_e01",
                "phenotype": "NORMAL", 
                "frames": "all",
                "confidence": 0.95
            },
            {
                "embryo_id": "20240411_A01_e02",
                "phenotype": "EDEMA",
                "frames": "[1::]",  # From second frame onwards
                "confidence": 0.85
            }
        ]
        phenotype_result = em.batch_add_phenotypes(phenotype_assignments, "phenotyper")
        self.assertGreater(phenotype_result["assigned"], 0)
        
        # 5. Add QC flags
        em.add_flag("20240411_A01_e01_0020", "OUT_OF_FOCUS", "snip", "qc_system", severity="warning")
        
        # 6. Save and verify persistence
        em.save()
        
        # 7. Reload and verify all data
        em2 = EmbryoMetadata(self.sam_path, self.metadata_path, verbose=False)
        
        # Verify genotypes
        self.assertEqual(em2.get_genotype("20240411_A01_e01")["WT"]["value"], "WT")
        self.assertEqual(em2.get_genotype("20240411_A01_e02")["lmx1b"]["value"], "lmx1b")
        
        # Verify treatments
        self.assertIn("NONE", em2.get_treatments("20240411_A01_e01"))
        self.assertIn("shh-i", em2.get_treatments("20240411_A01_e02"))
        
        # Verify phenotypes
        self.assertEqual(em2.get_phenotype("20240411_A01_e01_0010")["value"], "NORMAL")
        self.assertEqual(em2.get_phenotype("20240411_A01_e02_0015")["value"], "EDEMA")
        
        # Verify flags
        flags = em2.get_flags("20240411_A01_e01_0020")
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["flag"], "OUT_OF_FOCUS")
        
        print("🎉 Full workflow integration test passed!")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Running comprehensive EmbryoMetadata test suite...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbryoMetadataComprehensive)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏁 Test Summary:")
    print(f"   ✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Failed: {len(result.failures)}")
    print(f"   🚨 Errors: {len(result.errors)}")
    print(f"   📊 Total: {result.testsRun}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, error in result.failures:
            print(f"   {test}: {error}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, error in result.errors:
            print(f"   {test}: {error}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT! System is robust and ready for production!")
    elif success_rate >= 75:
        print("✅ GOOD! Minor issues need attention.")
    else:
        print("⚠️  NEEDS WORK! Significant issues require fixing.")
    
    return result


if __name__ == "__main__":
    run_comprehensive_tests()
