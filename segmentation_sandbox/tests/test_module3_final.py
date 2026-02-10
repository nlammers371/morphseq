#!/usr/bin/env python3
"""
Test Module 3 Implementation - Final Phase
Tests the simplified EmbryoMetadata and AnnotationBatch classes.
"""

import tempfile
import json
import os
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from annotations import EmbryoMetadata, AnnotationBatch

def create_test_sam2_data(base_dir: Path, embryo_id: str, target: str):
    """Create mock SAM2 data for testing."""
    sam2_dir = base_dir / "sam2_results" / embryo_id / target
    sam2_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock SAM2 results
    sam2_data = {
        "predictions": {
            f"{embryo_id}_{target}_snip_001": {
                "masks": {
                    "frame_000": {"mask_data": "mock_mask_1"},
                    "frame_001": {"mask_data": "mock_mask_2"},
                    "frame_002": {"mask_data": "mock_mask_3"}
                }
            },
            f"{embryo_id}_{target}_snip_002": {
                "masks": {
                    "frame_000": {"mask_data": "mock_mask_4"},
                    "frame_001": {"mask_data": "mock_mask_5"}
                }
            },
            f"{embryo_id}_{target}_snip_003": {
                "masks": {
                    "frame_000": {"mask_data": "mock_mask_6"}
                }
            }
        }
    }
    
    with open(sam2_dir / "sam2_results.json", 'w') as f:
        json.dump(sam2_data, f, indent=2)
    
    return sam2_dir / "sam2_results.json"

def test_embryo_metadata():
    """Test basic EmbryoMetadata functionality."""
    print("=== Testing EmbryoMetadata ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test initialization
        em = EmbryoMetadata(str(temp_path), "test_author")
        print("✓ EmbryoMetadata initialized successfully")
        
        # Test either/or API validation
        mode, error = em._select_mode(embryo_id="test", target="ch00")
        assert mode == "embryo" and error is None
        print("✓ Either/or API validation works for embryo mode")
        
        mode, error = em._select_mode(snip_ids=["snip1", "snip2"])
        assert mode == "snips" and error is None
        print("✓ Either/or API validation works for snips mode")
        
        mode, error = em._select_mode(embryo_id="test", snip_ids=["snip1"])
        assert mode == "error" and "both" in error
        print("✓ Either/or API correctly rejects ambiguous parameters")
        
        # Test SAM2 import
        sam2_file = create_test_sam2_data(temp_path, "embryo_001", "ch00")
        success = em.import_sam2_data("embryo_001", "ch00")
        assert success
        print("✓ SAM2 data import successful")
        
        # Test frame resolution
        frames = em.resolve_frames("all", embryo_id="embryo_001", target="ch00")
        assert len(frames) == 3
        print(f"✓ Frame resolution 'all': {len(frames)} frames")
        
        frames = em.resolve_frames("0:2", embryo_id="embryo_001", target="ch00")
        assert len(frames) == 2
        print(f"✓ Frame resolution '0:2': {len(frames)} frames")
        
        # Test phenotype annotation
        success = em.add_phenotype("normal", "test annotation", 
                                  embryo_id="embryo_001", target="ch00")
        assert success
        print("✓ Phenotype annotation successful")
        
        # Test embryo status
        status = em.get_embryo_status("embryo_001")
        assert status["exists"]
        assert status["total_annotations"] == 1
        print("✓ Embryo status retrieval successful")
        
        print("✓ All EmbryoMetadata tests passed!\n")

def test_annotation_batch():
    """Test AnnotationBatch functionality."""
    print("=== Testing AnnotationBatch ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test initialization (inherits from EmbryoMetadata)
        ab = AnnotationBatch(str(temp_path), "batch_test_author")
        print("✓ AnnotationBatch initialized successfully")
        
        # Create test data
        create_test_sam2_data(temp_path, "embryo_001", "ch00")
        create_test_sam2_data(temp_path, "embryo_002", "ch01")
        
        # Test batch SAM2 import
        import_specs = [
            {"embryo_id": "embryo_001", "target": "ch00"},
            {"embryo_id": "embryo_002", "target": "ch01"}
        ]
        result = ab.batch_import_sam2(import_specs)
        assert result["success"] == 2
        assert result["failed"] == 0
        print(f"✓ Batch SAM2 import: {result['success']}/{result['total_imports']} successful")
        
        # Test batch phenotype annotation
        operations = [
            {
                "phenotype": "normal",
                "notes": "batch test 1",
                "embryo_id": "embryo_001",
                "target": "ch00"
            },
            {
                "phenotype": "abnormal", 
                "notes": "batch test 2",
                "embryo_id": "embryo_002",
                "target": "ch01"
            }
        ]
        result = ab.batch_add_phenotype(operations)
        assert result["success"] == 2
        assert result["failed"] == 0
        print(f"✓ Batch phenotype annotation: {result['success']}/{result['total_operations']} successful")
        
        # Test batch summary
        summary = ab.get_batch_summary()
        assert summary["total_embryos"] == 2
        assert summary["total_annotations"] == 2
        print(f"✓ Batch summary: {summary['total_embryos']} embryos, {summary['total_annotations']} annotations")
        
        # Test phenotype counts
        assert "normal" in summary["phenotype_counts"]
        assert "abnormal" in summary["phenotype_counts"]
        print(f"✓ Phenotype tracking: {summary['phenotype_counts']}")
        
        print("✓ All AnnotationBatch tests passed!\n")

def test_error_handling():
    """Test error handling and edge cases."""
    print("=== Testing Error Handling ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        em = EmbryoMetadata(str(temp_path), "error_test_author")
        
        # Test invalid author
        try:
            invalid_em = EmbryoMetadata(str(temp_path), "")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "author" in str(e)
            print("✓ Invalid author properly rejected")
        
        # Test annotation without import
        success = em.add_phenotype("test", embryo_id="nonexistent", target="ch00")
        assert not success
        print("✓ Annotation rejected for non-imported data")
        
        # Test invalid frame resolution
        frames = em.resolve_frames("invalid_format", embryo_id="nonexistent", target="ch00")
        assert len(frames) == 0
        print("✓ Invalid frame specification properly handled")
        
        # Test batch operations with invalid data
        ab = AnnotationBatch(str(temp_path), "batch_error_test")
        result = ab.batch_add_phenotype([{"invalid": "operation"}])
        assert result["failed"] == 1
        print("✓ Invalid batch operations properly handled")
        
        print("✓ All error handling tests passed!\n")

def main():
    """Run all tests."""
    print("Starting Module 3 Final Implementation Tests\n")
    
    try:
        test_embryo_metadata()
        test_annotation_batch()
        test_error_handling()
        
        print("🎉 ALL TESTS PASSED! Module 3 implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)