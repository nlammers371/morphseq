#!/usr/bin/env python3
"""
Test Suite for GroundedDINO Utilities
=====================================

This test suite provides comprehensive testing for the grounded_sam_utils module,
covering:
- IoU calculation functions
- Annotation management and persistence
- High-quality annotation filtering
- Metadata integration
- Error handling and edge cases

Usage:
    python test_grounded_sam_utils.py
    python test_grounded_sam_utils.py --subset 5  # Run tests on 5 random samples
    python test_grounded_sam_utils.py --verbose   # Verbose output
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
TEST_DIR = Path(__file__).parent
SCRIPTS_DIR = TEST_DIR.parent
SANDBOX_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Import modules under test
from utils.grounded_sam_utils import (
    calculate_detection_iou,
    get_model_metadata,
    GroundedDinoAnnotations,
    load_config
)

class TestIoUCalculation(unittest.TestCase):
    """Test IoU calculation functions."""
    
    def test_calculate_detection_iou_perfect_overlap(self):
        """Test IoU calculation with perfect overlap."""
        box1 = [0.5, 0.5, 0.2, 0.2]  # center_x, center_y, width, height
        box2 = [0.5, 0.5, 0.2, 0.2]  # identical box
        iou = calculate_detection_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0, places=5)
    
    def test_calculate_detection_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = [0.2, 0.2, 0.2, 0.2]  # left box
        box2 = [0.8, 0.8, 0.2, 0.2]  # right box
        iou = calculate_detection_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.0, places=5)
    
    def test_calculate_detection_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = [0.4, 0.4, 0.4, 0.4]  # box from 0.2 to 0.6
        box2 = [0.6, 0.6, 0.4, 0.4]  # box from 0.4 to 0.8
        iou = calculate_detection_iou(box1, box2)
        
        # Expected calculation:
        # box1: (0.2, 0.2, 0.6, 0.6) - area = 0.16
        # box2: (0.4, 0.4, 0.8, 0.8) - area = 0.16
        # intersection: (0.4, 0.4, 0.6, 0.6) - area = 0.04
        # union: 0.16 + 0.16 - 0.04 = 0.28
        # IoU = 0.04 / 0.28 ≈ 0.1429
        expected_iou = 0.04 / 0.28
        self.assertAlmostEqual(iou, expected_iou, places=4)
    
    def test_calculate_detection_iou_edge_cases(self):
        """Test IoU calculation with edge cases."""
        # Zero width/height boxes
        box1 = [0.5, 0.5, 0.0, 0.2]
        box2 = [0.5, 0.5, 0.2, 0.2]
        iou = calculate_detection_iou(box1, box2)
        self.assertEqual(iou, 0.0)
        
        # Very small boxes
        box1 = [0.5, 0.5, 0.001, 0.001]
        box2 = [0.5, 0.5, 0.001, 0.001]
        iou = calculate_detection_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0, places=3)


class TestModelMetadata(unittest.TestCase):
    """Test model metadata extraction."""
    
    def test_get_model_metadata_with_annotation_metadata(self):
        """Test metadata extraction from model with annotation metadata."""
        mock_model = Mock()
        mock_model._annotation_metadata = {
            "model_config_path": "/path/to/config.py",
            "model_weights_path": "/path/to/weights.pth",
            "loading_timestamp": "2024-01-01T00:00:00",
            "model_architecture": "GroundedDINO"
        }
        
        metadata = get_model_metadata(mock_model)
        
        self.assertEqual(metadata["model_config_path"], "config.py")
        self.assertEqual(metadata["model_weights_path"], "weights.pth")
        self.assertEqual(metadata["loading_timestamp"], "2024-01-01T00:00:00")
        self.assertEqual(metadata["model_architecture"], "GroundedDINO")
    
    def test_get_model_metadata_without_annotation_metadata(self):
        """Test metadata extraction from model without annotation metadata."""
        mock_model = Mock()
        delattr(mock_model, '_annotation_metadata')
        
        metadata = get_model_metadata(mock_model)
        
        self.assertEqual(metadata["model_config_path"], "unknown")
        self.assertEqual(metadata["model_weights_path"], "unknown")
        self.assertEqual(metadata["model_architecture"], "GroundedDINO")
        self.assertIn("T", metadata["loading_timestamp"])  # Check ISO format


class TestGroundedDinoAnnotations(unittest.TestCase):
    """Test GroundedDINO annotations manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.annotations_file = Path(self.test_dir) / "test_annotations.json"
        self.metadata_file = Path(self.test_dir) / "test_metadata.json"
        
        # Create test metadata
        self.test_metadata = {
            "image_ids": ["img_001", "img_002", "img_003"],
            "experiments": {
                "exp_001": {
                    "videos": {
                        "vid_001": {
                            "image_ids": ["img_001", "img_002"]
                        }
                    }
                },
                "exp_002": {
                    "videos": {
                        "vid_002": {
                            "image_ids": ["img_003"]
                        }
                    }
                }
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.test_metadata, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_annotations_initialization_new_file(self):
        """Test initialization with new annotations file."""
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        
        # Save to create the file
        annotations.save()
        
        self.assertTrue(self.annotations_file.exists())
        self.assertIn("file_info", annotations.annotations)
        self.assertIn("images", annotations.annotations)
        self.assertEqual(len(annotations.annotations["images"]), 0)
    
    def test_annotations_initialization_existing_file(self):
        """Test initialization with existing annotations file."""
        # Create initial annotations file
        initial_data = {
            "file_info": {
                "creation_time": "2024-01-01T00:00:00",
                "last_updated": "2024-01-01T00:00:00"
            },
            "images": {
                "img_001": {
                    "annotations": []
                }
            }
        }
        
        with open(self.annotations_file, 'w') as f:
            json.dump(initial_data, f)
        
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        
        self.assertEqual(len(annotations.annotations["images"]), 1)
        self.assertIn("img_001", annotations.annotations["images"])
    
    def test_metadata_integration(self):
        """Test metadata integration."""
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path(self.metadata_file)
        
        metadata_image_ids = annotations.get_all_metadata_image_ids()
        self.assertEqual(len(metadata_image_ids), 3)
        self.assertIn("img_001", metadata_image_ids)
    
    def test_add_annotation(self):
        """Test adding single annotation."""
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        
        # Mock model
        mock_model = Mock()
        mock_model._annotation_metadata = {
            "model_config_path": "config.py",
            "model_weights_path": "weights.pth",
            "loading_timestamp": "2024-01-01T00:00:00",
            "model_architecture": "GroundedDINO"
        }
        
        # Test data
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
        logits = np.array([0.85])
        phrases = ["test object"]
        
        annotations.add_annotation(
            "img_001", "test prompt", mock_model, boxes, logits, phrases
        )
        
        self.assertIn("img_001", annotations.annotations["images"])
        image_annotations = annotations.annotations["images"]["img_001"]["annotations"]
        self.assertEqual(len(image_annotations), 1)
        self.assertEqual(image_annotations[0]["prompt"], "test prompt")
        self.assertEqual(image_annotations[0]["num_detections"], 1)
    
    def test_high_quality_annotation_filtering(self):
        """Test high-quality annotation filtering."""
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path(self.metadata_file)
        
        # Add test annotations with varying confidence
        mock_model = Mock()
        mock_model._annotation_metadata = {
            "model_config_path": "config.py",
            "model_weights_path": "weights.pth",
            "loading_timestamp": "2024-01-01T00:00:00",
            "model_architecture": "GroundedDINO"
        }
        
        # High confidence detection
        boxes_high = np.array([[0.5, 0.5, 0.2, 0.2]])
        logits_high = np.array([0.85])
        phrases_high = ["individual embryo"]
        
        annotations.add_annotation(
            "img_001", "individual embryo", mock_model, boxes_high, logits_high, phrases_high
        )
        
        # Low confidence detection
        boxes_low = np.array([[0.3, 0.3, 0.1, 0.1]])
        logits_low = np.array([0.25])
        phrases_low = ["individual embryo"]
        
        annotations.add_annotation(
            "img_002", "individual embryo", mock_model, boxes_low, logits_low, phrases_low
        )
        
        # Generate high-quality annotations
        result = annotations.generate_high_quality_annotations(
            ["img_001", "img_002"],
            prompt="individual embryo",
            confidence_threshold=0.5,
            iou_threshold=0.5
        )
        
        stats = result["statistics"]
        self.assertEqual(stats["original_detections"], 2)
        self.assertEqual(stats["final_detections"], 1)  # Only high confidence remains
        self.assertEqual(stats["confidence_removed"], 1)
    
    def test_iou_filtering_nms(self):
        """Test IoU-based filtering (Non-Maximum Suppression)."""
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path(self.metadata_file)
        
        mock_model = Mock()
        mock_model._annotation_metadata = {
            "model_config_path": "config.py",
            "model_weights_path": "weights.pth",
            "loading_timestamp": "2024-01-01T00:00:00",
            "model_architecture": "GroundedDINO"
        }
        
        # Add overlapping detections (same image)
        boxes = np.array([
            [0.5, 0.5, 0.2, 0.2],   # Higher confidence
            [0.52, 0.52, 0.2, 0.2], # Lower confidence, high overlap
            [0.8, 0.8, 0.1, 0.1]    # Different location
        ])
        logits = np.array([0.85, 0.75, 0.65])
        phrases = ["individual embryo"] * 3
        
        annotations.add_annotation(
            "img_001", "individual embryo", mock_model, boxes, logits, phrases
        )
        
        # Generate high-quality annotations with IoU filtering
        result = annotations.generate_high_quality_annotations(
            ["img_001"],
            prompt="individual embryo",
            confidence_threshold=0.5,
            iou_threshold=0.5
        )
        
        stats = result["statistics"]
        self.assertEqual(stats["original_detections"], 3)
        self.assertEqual(stats["final_detections"], 2)  # Two non-overlapping detections
        self.assertEqual(stats["iou_removed"], 1)
    
    def test_export_import_high_quality_annotations(self):
        """Test export and import of high-quality annotations."""
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path(self.metadata_file)
        
        # Add test data and generate high-quality annotations
        mock_model = Mock()
        mock_model._annotation_metadata = {
            "model_config_path": "config.py",
            "model_weights_path": "weights.pth",
            "loading_timestamp": "2024-01-01T00:00:00",
            "model_architecture": "GroundedDINO"
        }
        
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
        logits = np.array([0.85])
        phrases = ["individual embryo"]
        
        annotations.add_annotation(
            "img_001", "individual embryo", mock_model, boxes, logits, phrases
        )
        
        annotations.generate_high_quality_annotations(
            ["img_001"],
            prompt="individual embryo",
            confidence_threshold=0.5,
            iou_threshold=0.5
        )
        
        # Export
        export_path = Path(self.test_dir) / "exported_hq.json"
        annotations.export_high_quality_annotations(export_path)
        
        self.assertTrue(export_path.exists())
        
        # Import to new annotations instance
        new_annotations = GroundedDinoAnnotations(
            Path(self.test_dir) / "new_annotations.json", verbose=False
        )
        new_annotations.import_high_quality_annotations(export_path)
        
        hq_annotations = new_annotations.annotations.get("high_quality_annotations", {})
        self.assertGreater(len(hq_annotations), 0)
    
    def test_save_and_load_persistence(self):
        """Test saving and loading annotations persistence."""
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        
        # Add test data
        mock_model = Mock()
        mock_model._annotation_metadata = {
            "model_config_path": "config.py",
            "model_weights_path": "weights.pth",
            "loading_timestamp": "2024-01-01T00:00:00",
            "model_architecture": "GroundedDINO"
        }
        
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
        logits = np.array([0.85])
        phrases = ["test object"]
        
        annotations.add_annotation(
            "img_001", "test prompt", mock_model, boxes, logits, phrases
        )
        
        # Save
        annotations.save()
        
        # Load in new instance
        new_annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        
        self.assertIn("img_001", new_annotations.annotations["images"])
        image_annotations = new_annotations.annotations["images"]["img_001"]["annotations"]
        self.assertEqual(len(image_annotations), 1)
        self.assertEqual(image_annotations[0]["prompt"], "test prompt")


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = Path(self.test_dir) / "test_config.yaml"
        
        # Create test config
        test_config = {
            "models": {
                "groundingdino": {
                    "config": "models/GroundingDINO/config.py",
                    "weights": "models/GroundingDINO/weights.pth"
                }
            },
            "inference": {
                "box_threshold": 0.35,
                "text_threshold": 0.25
            }
        }
        
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_load_config(self):
        """Test loading configuration from YAML file."""
        config = load_config(self.config_file)
        
        self.assertIn("models", config)
        self.assertIn("groundingdino", config["models"])
        self.assertEqual(config["inference"]["box_threshold"], 0.35)


class TestSubsetRunner:
    """Utility class for running tests on random subsets."""
    
    @staticmethod
    def run_subset_tests(test_classes: List, subset_size: int = 5, verbose: bool = False):
        """Run tests on a random subset of test cases."""
        print(f"🔬 Running subset tests ({subset_size} random samples)")
        
        # Collect all test methods
        all_tests = []
        for test_class in test_classes:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            for test in suite:
                all_tests.append(test)
        
        # Select random subset
        if len(all_tests) <= subset_size:
            selected_tests = all_tests
        else:
            selected_tests = random.sample(all_tests, subset_size)
        
        print(f"📋 Selected {len(selected_tests)} tests from {len(all_tests)} total")
        
        # Run selected tests
        suite = unittest.TestSuite(selected_tests)
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(suite)
        
        return result


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GroundedDINO utilities")
    parser.add_argument("--subset", type=int, help="Run tests on random subset of this size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Test classes to run
    test_classes = [
        TestIoUCalculation,
        TestModelMetadata,
        TestGroundedDinoAnnotations,
        TestConfigLoading
    ]
    
    if args.subset:
        # Run subset tests
        result = TestSubsetRunner.run_subset_tests(
            test_classes, 
            subset_size=args.subset, 
            verbose=args.verbose
        )
        success = result.wasSuccessful()
    else:
        # Run all tests
        print("🔬 Running all tests...")
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        success = result.wasSuccessful()
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
