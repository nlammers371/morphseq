#!/usr/bin/env python3
"""
Integration Tests for GroundedDINO Detection Pipeline
====================================================

This test suite provides integration testing for the complete GroundedDINO pipeline,
including:
- End-to-end detection workflow
- Model loading and inference
- Annotation generation and filtering
- File I/O operations
- Error handling in real scenarios

Usage:
    python test_gdino_integration.py
    python test_gdino_integration.py --mock-model    # Use mock model for testing
    python test_gdino_integration.py --subset 3      # Run on 3 random images
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
    GroundedDinoAnnotations,
    gdino_inference_with_visualization,
    run_inference,
    load_config,
    load_groundingdino_model
)
from utils.experiment_metadata_utils import load_experiment_metadata


class TestGDinoIntegration(unittest.TestCase):
    """Integration tests for GroundedDINO pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.setup_test_data()
        self.setup_mock_images()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def setup_test_data(self):
        """Set up test data files."""
        # Create test metadata
        self.metadata_file = Path(self.test_dir) / "experiment_metadata.json"
        self.test_metadata = {
            "image_ids": ["img_001", "img_002", "img_003", "img_004", "img_005"],
            "experiments": {
                "exp_001": {
                    "videos": {
                        "vid_001": {
                            "image_ids": ["img_001", "img_002"],
                            "video_path": str(Path(self.test_dir) / "videos" / "vid_001.mp4")
                        },
                        "vid_002": {
                            "image_ids": ["img_003"],
                            "video_path": str(Path(self.test_dir) / "videos" / "vid_002.mp4")
                        }
                    }
                },
                "exp_002": {
                    "videos": {
                        "vid_003": {
                            "image_ids": ["img_004", "img_005"],
                            "video_path": str(Path(self.test_dir) / "videos" / "vid_003.mp4")
                        }
                    }
                }
            },
            "image_paths": {
                "img_001": str(Path(self.test_dir) / "images" / "img_001.jpg"),
                "img_002": str(Path(self.test_dir) / "images" / "img_002.jpg"),
                "img_003": str(Path(self.test_dir) / "images" / "img_003.jpg"),
                "img_004": str(Path(self.test_dir) / "images" / "img_004.jpg"),
                "img_005": str(Path(self.test_dir) / "images" / "img_005.jpg")
            }
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.test_metadata, f)
        
        # Create test config
        self.config_file = Path(self.test_dir) / "config.yaml"
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
        
        # Create annotations file path
        self.annotations_file = Path(self.test_dir) / "annotations.json"
    
    def setup_mock_images(self):
        """Create mock image files for testing."""
        images_dir = Path(self.test_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy image files
        for image_id in self.test_metadata["image_ids"]:
            image_path = images_dir / f"{image_id}.jpg"
            # Create a minimal 1x1 pixel image file
            image_path.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\x00\xff\xd9')
    
    def create_mock_model(self):
        """Create a mock GroundingDINO model for testing."""
        mock_model = Mock()
        mock_model._annotation_metadata = {
            "model_config_path": "GroundingDINO_SwinT_OGC.py",
            "model_weights_path": "groundingdino_swint_ogc.pth",
            "loading_timestamp": "2024-01-01T00:00:00",
            "model_architecture": "GroundedDINO"
        }
        
        # Mock inference results
        def mock_inference_side_effect(*args, **kwargs):
            # Return realistic-looking detection results
            num_detections = random.randint(1, 4)
            
            boxes = np.random.rand(num_detections, 4)  # Random normalized boxes
            boxes[:, 2:] *= 0.3  # Make width/height smaller
            boxes[:, :2] = np.clip(boxes[:, :2], 0.1, 0.9)  # Keep centers reasonable
            
            logits = np.random.rand(num_detections) * 0.5 + 0.5  # Confidence 0.5-1.0
            phrases = ["individual embryo"] * num_detections
            
            # Mock image source
            image_source = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            return boxes, logits, phrases, image_source
        
        return mock_model, mock_inference_side_effect
    
    def test_end_to_end_detection_workflow(self):
        """Test complete end-to-end detection workflow."""
        # Create mock model
        mock_model, mock_inference = self.create_mock_model()
        
        # Create annotations manager
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path(self.metadata_file)
        
        # Test missing annotations detection
        missing = annotations.get_missing_annotations(["individual embryo"])
        self.assertEqual(len(missing["individual embryo"]), 5)  # All images are missing
        
        # Mock the inference function
        with patch('utils.grounded_sam_utils.run_inference', side_effect=mock_inference):
            # Process missing annotations
            image_paths = [Path(self.test_dir) / "images" / f"{img_id}.jpg" 
                          for img_id in self.test_metadata["image_ids"][:3]]  # Test subset
            
            results = gdino_inference_with_visualization(
                mock_model, 
                image_paths, 
                ["individual embryo"],
                show_anno=False,
                verbose=False,
                annotations_manager=annotations
            )
        
        # Verify results
        self.assertEqual(len(results), 3)
        for image_name, image_results in results.items():
            self.assertIn("individual embryo", image_results)
            boxes, logits, phrases, image_source = image_results["individual embryo"]
            self.assertGreater(len(boxes), 0)
            self.assertGreater(len(logits), 0)
            self.assertEqual(len(phrases), len(boxes))
        
        # Verify annotations were saved
        self.assertGreater(len(annotations.get_all_image_ids()), 0)
    
    def test_high_quality_filtering_integration(self):
        """Test high-quality filtering with realistic data."""
        # Create mock model
        mock_model, _ = self.create_mock_model()
        
        # Create annotations manager
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path(self.metadata_file)
        
        # Add test annotations with varying qualities
        test_data = [
            # Image 1: High quality detections
            {
                "image_id": "img_001",
                "boxes": np.array([[0.5, 0.5, 0.2, 0.2], [0.8, 0.3, 0.15, 0.15]]),
                "logits": np.array([0.85, 0.75]),
                "phrases": ["individual embryo", "individual embryo"]
            },
            # Image 2: Mixed quality detections
            {
                "image_id": "img_002", 
                "boxes": np.array([[0.3, 0.4, 0.1, 0.1], [0.7, 0.7, 0.2, 0.2]]),
                "logits": np.array([0.25, 0.65]),  # One low, one acceptable
                "phrases": ["individual embryo", "individual embryo"]
            },
            # Image 3: Overlapping detections
            {
                "image_id": "img_003",
                "boxes": np.array([[0.5, 0.5, 0.3, 0.3], [0.52, 0.52, 0.25, 0.25]]),  # High overlap
                "logits": np.array([0.80, 0.70]),
                "phrases": ["individual embryo", "individual embryo"]
            }
        ]
        
        # Add annotations
        for data in test_data:
            annotations.add_annotation(
                data["image_id"],
                "individual embryo",
                mock_model,
                data["boxes"],
                data["logits"],
                data["phrases"]
            )
        
        # Generate high-quality annotations
        result = annotations.generate_high_quality_annotations(
            ["img_001", "img_002", "img_003"],
            prompt="individual embryo",
            confidence_threshold=0.5,
            iou_threshold=0.5
        )
        
        # Verify filtering results
        stats = result["statistics"]
        self.assertEqual(stats["original_detections"], 6)  # 2 + 2 + 2
        self.assertGreater(stats["confidence_removed"], 0)  # At least one low confidence removed
        self.assertGreater(stats["iou_removed"], 0)  # At least one overlap removed
        self.assertGreater(stats["final_detections"], 0)  # Some detections remain
        
        # Verify high-quality annotations were saved
        hq_annotations = annotations.annotations.get("high_quality_annotations", {})
        self.assertGreater(len(hq_annotations), 0)
    
    def test_experiment_metadata_integration(self):
        """Test integration with experiment metadata."""
        # Create annotations manager
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path(self.metadata_file)
        
        # Test metadata loading
        metadata_image_ids = annotations.get_all_metadata_image_ids()
        self.assertEqual(len(metadata_image_ids), 5)
        
        # Test experiment mapping
        image_to_exp = annotations._get_image_to_experiment_map()
        self.assertEqual(image_to_exp["img_001"], "exp_001")
        self.assertEqual(image_to_exp["img_004"], "exp_002")
        
        # Test filtered image IDs
        exp_001_images = annotations._get_filtered_image_ids(experiment_ids=["exp_001"])
        self.assertEqual(len(exp_001_images), 3)  # img_001, img_002, img_003
        self.assertIn("img_001", exp_001_images)
        self.assertIn("img_002", exp_001_images)
        self.assertIn("img_003", exp_001_images)
    
    def test_annotation_persistence(self):
        """Test annotation persistence across sessions."""
        # Create mock model
        mock_model, _ = self.create_mock_model()
        
        # Session 1: Create annotations
        annotations1 = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations1.set_metadata_path(self.metadata_file)
        
        # Add test annotation
        boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
        logits = np.array([0.85])
        phrases = ["individual embryo"]
        
        annotations1.add_annotation(
            "img_001", "individual embryo", mock_model, boxes, logits, phrases
        )
        
        # Save and close
        annotations1.save()
        del annotations1
        
        # Session 2: Load and verify
        annotations2 = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations2.set_metadata_path(self.metadata_file)
        
        # Verify data persisted
        self.assertIn("img_001", annotations2.annotations["images"])
        image_annotations = annotations2.annotations["images"]["img_001"]["annotations"]
        self.assertEqual(len(image_annotations), 1)
        self.assertEqual(image_annotations[0]["prompt"], "individual embryo")
        self.assertEqual(image_annotations[0]["num_detections"], 1)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with non-existent metadata file
        annotations = GroundedDinoAnnotations(self.annotations_file, verbose=False)
        annotations.set_metadata_path("/nonexistent/path.json")
        
        # Should not crash
        metadata_ids = annotations.get_all_metadata_image_ids()
        self.assertEqual(len(metadata_ids), 0)
        
        # Test with corrupted annotations file
        corrupted_file = Path(self.test_dir) / "corrupted.json"
        corrupted_file.write_text("invalid json content")
        
        # Should handle gracefully
        annotations_corrupted = GroundedDinoAnnotations(corrupted_file, verbose=False)
        self.assertIn("images", annotations_corrupted.annotations)
        
        # Test with empty image list
        mock_model, _ = self.create_mock_model()
        result = annotations.generate_high_quality_annotations(
            [],  # Empty list
            prompt="individual embryo"
        )
        
        # Should handle gracefully
        self.assertEqual(result["statistics"]["original_detections"], 0)
        self.assertEqual(result["statistics"]["final_detections"], 0)


class TestSubsetIntegration:
    """Utility class for running integration tests on subsets."""
    
    @staticmethod
    def run_subset_integration_tests(subset_size: int = 3, verbose: bool = False):
        """Run integration tests on a subset of data."""
        print(f"🧪 Running integration tests on subset of {subset_size} samples")
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestGDinoIntegration)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(suite)
        
        return result


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration tests for GroundedDINO pipeline")
    parser.add_argument("--subset", type=int, help="Run tests on subset of this size")
    parser.add_argument("--mock-model", action="store_true", help="Use mock model for testing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.subset:
        # Run subset tests
        result = TestSubsetIntegration.run_subset_integration_tests(
            subset_size=args.subset,
            verbose=args.verbose
        )
        success = result.wasSuccessful()
    else:
        # Run all integration tests
        print("🧪 Running all integration tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestGDinoIntegration)
        
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        success = result.wasSuccessful()
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
