"""
Test Module 6: Batch Processing Engine
Comprehensive testing for batch processing capabilities

Author: EmbryoMetadata Development Team  
Date: July 15, 2025
"""

import unittest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import time

# Test imports
import sys
sys.path.append('/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/utils/embryo_metada_dev_instruction')

from embryo_metadata_batch import (
    RangeParser, TemporalRangeParser, BatchProcessor, BatchOperations,
    create_progress_callback, estimate_batch_time
)


class TestRangeParser(unittest.TestCase):
    """Test the RangeParser functionality."""
    
    def setUp(self):
        self.test_items = [f"item_{i:03d}" for i in range(20)]  # item_000 to item_019
    
    def test_single_index(self):
        """Test single index parsing."""
        result = RangeParser.parse_range("[5]", self.test_items)
        self.assertEqual(result, ["item_005"])
        
        result = RangeParser.parse_range("10", self.test_items)
        self.assertEqual(result, ["item_010"])
    
    def test_range_syntax(self):
        """Test range syntax parsing."""
        # Basic range
        result = RangeParser.parse_range("[5:10]", self.test_items)
        expected = ["item_005", "item_006", "item_007", "item_008", "item_009"]
        self.assertEqual(result, expected)
        
        # Open-ended range
        result = RangeParser.parse_range("[17::]", self.test_items)
        expected = ["item_017", "item_018", "item_019"]
        self.assertEqual(result, expected)
        
        # Range with step
        result = RangeParser.parse_range("[0:10:2]", self.test_items)
        expected = ["item_000", "item_002", "item_004", "item_006", "item_008"]
        self.assertEqual(result, expected)
    
    def test_list_inputs(self):
        """Test list inputs."""
        # List of indices
        result = RangeParser.parse_range([2, 5, 8], self.test_items)
        expected = ["item_002", "item_005", "item_008"]
        self.assertEqual(result, expected)
        
        # List of IDs
        specific_items = ["item_003", "item_007", "item_012"]
        result = RangeParser.parse_range(specific_items, self.test_items)
        self.assertEqual(result, specific_items)
    
    def test_negative_indexing(self):
        """Test negative indexing."""
        result = RangeParser.parse_range("[-3::]", self.test_items)
        expected = ["item_017", "item_018", "item_019"]
        self.assertEqual(result, expected)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty range
        result = RangeParser.parse_range("[]", self.test_items)
        self.assertEqual(result, [])
        
        # Out of bounds
        result = RangeParser.parse_range("[100]", self.test_items)
        self.assertEqual(result, [])
        
        # Invalid ID
        result = RangeParser.parse_range(["nonexistent"], self.test_items)
        self.assertEqual(result, [])


class TestTemporalRangeParser(unittest.TestCase):
    """Test the TemporalRangeParser functionality."""
    
    def setUp(self):
        """Create mock metadata for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create mock SAM annotation
        sam_data = {
            "experiments": {
                "20240411": {
                    "videos": {
                        "20240411_A01": {
                            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
                            "images": {}
                        }
                    }
                }
            },
            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
            "snip_ids": []
        }
        
        # Add multiple frames per embryo
        snip_ids = []
        images = {}
        for embryo_idx in [1, 2]:
            embryo_id = f"20240411_A01_e0{embryo_idx}"
            for frame in range(25):  # 25 frames each
                snip_id = f"{embryo_id}_{frame:04d}"
                snip_ids.append(snip_id)
                
                image_id = f"20240411_A01_{frame:04d}"
                if image_id not in images:
                    images[image_id] = {"embryos": {}}
                
                images[image_id]["embryos"][embryo_id] = {"snip_id": snip_id}
        
        sam_data["snip_ids"] = snip_ids
        sam_data["experiments"]["20240411"]["videos"]["20240411_A01"]["images"] = images
        
        self.sam_path = self.temp_path / "sam_annotations.json"
        with open(self.sam_path, 'w') as f:
            json.dump(sam_data, f)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_special_keywords(self):
        """Test special temporal keywords."""
        # Import here to avoid circular dependency issues
        from embryo_metadata_refactored import EmbryoMetadata
        
        # Create metadata instance
        em = EmbryoMetadata(
            self.sam_path,
            self.temp_path / "embryo_metadata.json",
            gen_if_no_file=True,
            verbose=False
        )
        
        embryo_id = "20240411_A01_e01"
        
        # Test "all"
        result = TemporalRangeParser.parse_frame_range(embryo_id, "all", em)
        self.assertEqual(len(result), 25)
        
        # Test "first"
        result = TemporalRangeParser.parse_frame_range(embryo_id, "first", em)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].endswith("_0000"))
        
        # Test "last" 
        result = TemporalRangeParser.parse_frame_range(embryo_id, "last", em)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].endswith("_0024"))
    
    def test_frame_number_extraction(self):
        """Test frame number extraction."""
        frame_num = TemporalRangeParser._extract_frame_number("20240411_A01_e01_0015")
        self.assertEqual(frame_num, 15)
        
        with self.assertRaises(ValueError):
            TemporalRangeParser._extract_frame_number("invalid_id")


class TestBatchProcessor(unittest.TestCase):
    """Test the BatchProcessor functionality."""
    
    def setUp(self):
        """Create mock metadata for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create minimal SAM annotation
        sam_data = {
            "experiments": {"20240411": {"videos": {"20240411_A01": {"embryo_ids": ["test_e01"]}}}},
            "embryo_ids": ["test_e01"],
            "snip_ids": ["test_snip_001", "test_snip_002", "test_snip_003"]
        }
        
        self.sam_path = self.temp_path / "sam_annotations.json"
        with open(self.sam_path, 'w') as f:
            json.dump(sam_data, f)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_sequential_processing(self):
        """Test sequential batch processing."""
        from embryo_metadata_refactored import EmbryoMetadata
        
        em = EmbryoMetadata(
            self.sam_path,
            self.temp_path / "embryo_metadata.json", 
            gen_if_no_file=True,
            verbose=False
        )
        
        processor = BatchProcessor(em, parallel=False, verbose=False)
        
        # Create test items
        items = [(f"item_{i}", {"value": i}) for i in range(10)]
        
        # Define test operation
        def test_operation(item_id, data):
            return data["value"] * 2
        
        # Process
        results = processor.process_batch(items, test_operation, chunk_size=3)
        
        # Verify results
        self.assertEqual(len(results["successful"]), 10)
        self.assertEqual(len(results["failed"]), 0)
        self.assertEqual(len(results["skipped"]), 0)
        self.assertIn("elapsed_seconds", results)
        self.assertIn("items_per_second", results)
    
    def test_parallel_processing(self):
        """Test parallel batch processing."""
        from embryo_metadata_refactored import EmbryoMetadata
        
        em = EmbryoMetadata(
            self.sam_path,
            self.temp_path / "embryo_metadata.json",
            gen_if_no_file=True,
            verbose=False
        )
        
        processor = BatchProcessor(em, parallel=True, num_workers=2, verbose=False)
        
        # Create test items
        items = [(f"item_{i}", {"value": i}) for i in range(20)]
        
        # Define test operation with small delay
        def test_operation(item_id, data):
            time.sleep(0.01)  # Small delay to test parallelization
            return data["value"] * 2
        
        # Process
        start_time = time.time()
        results = processor.process_batch(items, test_operation, chunk_size=5)
        elapsed = time.time() - start_time
        
        # Verify results
        self.assertEqual(len(results["successful"]), 20)
        self.assertEqual(len(results["failed"]), 0)
        
        # Parallel should be faster than 20 * 0.01 = 0.2 seconds
        self.assertLess(elapsed, 0.15)  # Should be significantly faster
    
    def test_error_handling(self):
        """Test error handling in batch processing."""
        from embryo_metadata_refactored import EmbryoMetadata
        
        em = EmbryoMetadata(
            self.sam_path,
            self.temp_path / "embryo_metadata.json",
            gen_if_no_file=True,
            verbose=False
        )
        
        processor = BatchProcessor(em, parallel=False, verbose=False)
        
        # Create test items
        items = [(f"item_{i}", {"value": i}) for i in range(5)]
        
        # Define operation that fails on item_2
        def test_operation(item_id, data):
            if item_id == "item_2":
                raise ValueError("Test error")
            return data["value"] * 2
        
        # Process
        results = processor.process_batch(items, test_operation)
        
        # Verify results
        self.assertEqual(len(results["successful"]), 4)
        self.assertEqual(len(results["failed"]), 1)
        self.assertEqual(results["failed"][0][0], "item_2")


class TestBatchOperations(unittest.TestCase):
    """Test specialized batch operations."""
    
    def setUp(self):
        """Create comprehensive test setup."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create comprehensive SAM annotation
        sam_data = {
            "experiments": {
                "20240411": {
                    "videos": {
                        "20240411_A01": {
                            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
                            "images": {}
                        }
                    }
                }
            },
            "embryo_ids": ["20240411_A01_e01", "20240411_A01_e02"],
            "snip_ids": []
        }
        
        # Add frames for each embryo
        snip_ids = []
        images = {}
        for embryo_idx in [1, 2]:
            embryo_id = f"20240411_A01_e0{embryo_idx}"
            for frame in range(10):  # 10 frames each
                snip_id = f"{embryo_id}_{frame:04d}"
                snip_ids.append(snip_id)
                
                image_id = f"20240411_A01_{frame:04d}"
                if image_id not in images:
                    images[image_id] = {"embryos": {}}
                
                images[image_id]["embryos"][embryo_id] = {"snip_id": snip_id}
        
        sam_data["snip_ids"] = snip_ids
        sam_data["experiments"]["20240411"]["videos"]["20240411_A01"]["images"] = images
        
        self.sam_path = self.temp_path / "sam_annotations.json"
        with open(self.sam_path, 'w') as f:
            json.dump(sam_data, f)
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_batch_phenotype_assignment(self):
        """Test batch phenotype assignment."""
        from embryo_metadata_refactored import EmbryoMetadata
        
        # Create metadata
        em = EmbryoMetadata(
            self.sam_path,
            self.temp_path / "embryo_metadata.json",
            gen_if_no_file=True,
            verbose=False
        )
        
        # Create assignments
        assignments = [
            {
                "embryo_id": "20240411_A01_e01",
                "phenotype": "EDEMA",
                "frames": "[2:5]",
                "confidence": 0.9
            },
            {
                "embryo_id": "20240411_A01_e02", 
                "phenotype": "CONVERGENCE_EXTENSION",
                "frames": "all",
                "notes": "Strong phenotype"
            }
        ]
        
        # Process assignments
        results = em.batch_add_phenotypes(assignments, "test_author", validate_ranges=True)
        
        # Verify results
        self.assertGreater(results["assigned"], 0)
        self.assertEqual(results["embryos_processed"], 2)
        
        # Check that phenotypes were actually added
        embryo1_data = em.data["embryos"]["20240411_A01_e01"]
        embryo2_data = em.data["embryos"]["20240411_A01_e02"]
        
        # Embryo 1 should have EDEMA in frames 2-4 (3 total)
        edema_count = 0
        for snip_data in embryo1_data["snips"].values():
            if "phenotypes" in snip_data:
                for phenotype in snip_data["phenotypes"].values():
                    if phenotype["value"] == "EDEMA":
                        edema_count += 1
        self.assertEqual(edema_count, 3)  # Frames 2, 3, 4
        
        # Embryo 2 should have CONVERGENCE_EXTENSION in all frames (10 total)
        ce_count = 0
        for snip_data in embryo2_data["snips"].values():
            if "phenotypes" in snip_data:
                for phenotype in snip_data["phenotypes"].values():
                    if phenotype["value"] == "CONVERGENCE_EXTENSION":
                        ce_count += 1
        self.assertEqual(ce_count, 10)  # All 10 frames
    
    def test_batch_genotype_assignment(self):
        """Test batch genotype assignment."""
        from embryo_metadata_refactored import EmbryoMetadata
        
        # Create metadata
        em = EmbryoMetadata(
            self.sam_path,
            self.temp_path / "embryo_metadata.json",
            gen_if_no_file=True,
            verbose=False
        )
        
        # Create assignments
        assignments = [
            {
                "embryo_id": "20240411_A01_e01",
                "genotype": "WT",
                "notes": "PCR confirmed"
            },
            {
                "embryo_id": "20240411_A01_e02",
                "genotype": "lmx1b"
            }
        ]
        
        # Process assignments
        results = em.batch_add_genotypes(assignments, "test_author")
        
        # Verify results
        self.assertEqual(results["assigned"], 2)
        
        # Check genotypes were added
        self.assertEqual(em.data["embryos"]["20240411_A01_e01"]["genotypes"]["value"], "WT")
        self.assertEqual(em.data["embryos"]["20240411_A01_e02"]["genotypes"]["value"], "lmx1b")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_progress_callback_creation(self):
        """Test progress callback creation."""
        callback = create_progress_callback(verbose=False)
        
        # Should not raise an error
        callback(50, 100)
        
        self.assertTrue(callable(callback))
    
    def test_time_estimation(self):
        """Test batch time estimation."""
        # Test different scales
        result = estimate_batch_time(50, 100.0)
        self.assertIn("second", result)
        
        result = estimate_batch_time(5000, 100.0)
        self.assertIn("second", result)
        
        result = estimate_batch_time(500000, 100.0)
        self.assertIn("hour", result)


def run_batch_processing_tests():
    """Run all batch processing tests."""
    print("🧪 Running Module 6: Batch Processing Engine Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestRangeParser,
        TestTemporalRangeParser,
        TestBatchProcessor,
        TestBatchOperations,
        TestUtilityFunctions
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 ALL BATCH PROCESSING TESTS PASSED!")
    else:
        print("\n⚠️  Some tests failed. Please review and fix.")
    
    return success


if __name__ == "__main__":
    run_batch_processing_tests()
