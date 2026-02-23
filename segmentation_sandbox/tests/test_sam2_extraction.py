#!/usr/bin/env python3
"""
Test SAM2 Extraction Functionality

Tests the SAM2 data parsing and embryo structure extraction:
- Frame number extraction
- Embryo ID parsing
- Snip ID generation
- Hierarchical structure navigation
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add scripts to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annotations.embryo_metadata import EmbryoMetadata


def create_complex_sam2_data():
    """Create more complex SAM2 data to test edge cases."""
    return {
        "experiments": {
            "20240418": {
                "videos": {
                    "20240418_A01": {
                        "images": {
                            "20240418_A01_t0001": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0100": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e03": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t1000": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    },
                    "20240418_B01": {
                        "images": {
                            "20240418_B01_t0050": {
                                "embryos": {
                                    "20240418_B01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_B01_e02": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            },
            "20240419": {
                "videos": {
                    "20240419_C01": {
                        "images": {
                            "20240419_C01_t0200": {
                                "embryos": {
                                    "20240419_C01_e01": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def complex_sam2_file():
    """Create temporary complex SAM2 file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_complex_sam2_data(), f)
        f.flush()
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)
    biology_path = Path(f.name).parent / f"{Path(f.name).stem}_biology.json"
    biology_path.unlink(missing_ok=True)


def test_frame_number_extraction():
    """Test frame number extraction from various image ID formats."""
    metadata = EmbryoMetadata.__new__(EmbryoMetadata)  # Create without __init__
    
    # Test normal cases
    assert metadata._extract_frame_number("20240418_A01_t0001") == 1
    assert metadata._extract_frame_number("20240418_A01_t0100") == 100
    assert metadata._extract_frame_number("20240418_A01_t1000") == 1000
    assert metadata._extract_frame_number("experiment_video_t0050") == 50
    
    # Test edge cases
    with pytest.raises(ValueError):
        metadata._extract_frame_number("invalid_format")
    
    with pytest.raises(ValueError):
        metadata._extract_frame_number("20240418_A01_no_t_marker")


def test_embryo_structure_creation():
    """Test embryo structure creation with various embryo ID formats."""
    metadata = EmbryoMetadata.__new__(EmbryoMetadata)  # Create without __init__
    
    # Test normal embryo ID
    structure = metadata._create_embryo_structure("20240418_A01_e01")
    assert structure["embryo_id"] == "20240418_A01_e01"
    assert structure["experiment_id"] == "20240418"
    assert structure["video_id"] == "20240418_A01"
    assert structure["genotype"] is None
    assert structure["treatments"] == []
    assert structure["snips"] == {}
    
    # Test different format
    structure = metadata._create_embryo_structure("exp123_vid456_e01")
    assert structure["embryo_id"] == "exp123_vid456_e01"
    assert structure["experiment_id"] == "exp123"
    assert structure["video_id"] == "exp123_vid456"
    
    # Test edge case with insufficient parts
    structure = metadata._create_embryo_structure("e01")
    assert structure["embryo_id"] == "e01"
    assert structure["experiment_id"] == "unknown"
    assert structure["video_id"] == "unknown"


def test_snip_structure_creation():
    """Test snip structure creation."""
    metadata = EmbryoMetadata.__new__(EmbryoMetadata)  # Create without __init__
    
    structure = metadata._create_snip_structure("20240418_A01_e01_s0100", 100)
    assert structure["snip_id"] == "20240418_A01_e01_s0100"
    assert structure["frame_number"] == 100
    assert structure["phenotypes"] == []
    assert structure["flags"] == []


def test_complex_sam2_extraction(complex_sam2_file):
    """Test extraction from complex SAM2 structure."""
    metadata = EmbryoMetadata(str(complex_sam2_file))
    
    embryos = metadata.data["embryos"]
    
    # Check all expected embryos are present
    expected_embryos = [
        "20240418_A01_e01", "20240418_A01_e02", "20240418_A01_e03",
        "20240418_B01_e01", "20240418_B01_e02",
        "20240419_C01_e01"
    ]
    
    for embryo_id in expected_embryos:
        assert embryo_id in embryos, f"Missing embryo: {embryo_id}"
    
    # Check embryo with multiple frames
    e01_a01 = embryos["20240418_A01_e01"]
    expected_snips = ["20240418_A01_e01_s0001", "20240418_A01_e01_s0100", "20240418_A01_e01_s1000"]
    for snip_id in expected_snips:
        assert snip_id in e01_a01["snips"], f"Missing snip: {snip_id}"
    
    # Check snip frame numbers are correct
    assert e01_a01["snips"]["20240418_A01_e01_s0001"]["frame_number"] == 1
    assert e01_a01["snips"]["20240418_A01_e01_s0100"]["frame_number"] == 100
    assert e01_a01["snips"]["20240418_A01_e01_s1000"]["frame_number"] == 1000
    
    # Check embryo appears only at specific frames
    e02_a01 = embryos["20240418_A01_e02"]
    assert "20240418_A01_e02_s0001" in e02_a01["snips"]
    assert "20240418_A01_e02_s0100" not in e02_a01["snips"]  # Not in frame 100
    
    e03_a01 = embryos["20240418_A01_e03"]
    assert "20240418_A01_e03_s0001" not in e03_a01["snips"]  # Not in frame 1
    assert "20240418_A01_e03_s0100" in e03_a01["snips"]
    
    # Check different experiments/videos
    e01_b01 = embryos["20240418_B01_e01"]
    assert e01_b01["experiment_id"] == "20240418"
    assert e01_b01["video_id"] == "20240418_B01"
    assert "20240418_B01_e01_s0050" in e01_b01["snips"]
    
    e01_c01 = embryos["20240419_C01_e01"]
    assert e01_c01["experiment_id"] == "20240419"
    assert e01_c01["video_id"] == "20240419_C01"
    assert "20240419_C01_e01_s0200" in e01_c01["snips"]


def test_snip_id_generation_consistency():
    """Test that snip ID generation is consistent and follows format."""
    metadata = EmbryoMetadata.__new__(EmbryoMetadata)  # Create without __init__
    
    # Test various frame numbers
    test_cases = [
        ("embryo_e01", 1, "embryo_e01_s0001"),
        ("embryo_e01", 100, "embryo_e01_s0100"),
        ("embryo_e01", 1000, "embryo_e01_s1000"),
        ("20240418_A01_e02", 50, "20240418_A01_e02_s0050"),
    ]
    
    for embryo_id, frame_num, expected_snip_id in test_cases:
        # Simulate the snip ID generation logic
        snip_id = f"{embryo_id}_s{frame_num:04d}"
        assert snip_id == expected_snip_id


def test_empty_sam2_handling():
    """Test handling of empty or minimal SAM2 data."""
    empty_sam2_data = {"experiments": {}}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(empty_sam2_data, f)
        f.flush()
        temp_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(temp_file))
        assert metadata.data["embryos"] == {}
        
        # Should still have valid metadata structure
        assert "metadata" in metadata.data
        assert metadata.data["metadata"]["version"] == "simplified_v1"
        
    finally:
        temp_file.unlink(missing_ok=True)
        biology_path = temp_file.parent / f"{temp_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_malformed_sam2_handling():
    """Test handling of malformed SAM2 data."""
    # SAM2 data with invalid image IDs
    malformed_sam2_data = {
        "experiments": {
            "20240418": {
                "videos": {
                    "20240418_A01": {
                        "images": {
                            "invalid_image_id": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0100": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(malformed_sam2_data, f)
        f.flush()
        temp_file = Path(f.name)
    
    try:
        # Should handle gracefully by skipping invalid entries
        metadata = EmbryoMetadata(str(temp_file))
        
        # Should have extracted the valid embryo from valid frame
        assert "20240418_A01_e01" in metadata.data["embryos"]
        embryo = metadata.data["embryos"]["20240418_A01_e01"]
        
        # Should only have snip from valid frame
        assert "20240418_A01_e01_s0100" in embryo["snips"]
        assert len(embryo["snips"]) == 1  # Only the valid frame
        
    finally:
        temp_file.unlink(missing_ok=True)
        biology_path = temp_file.parent / f"{temp_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_extraction_performance(complex_sam2_file):
    """Test that extraction performance is reasonable."""
    import time
    
    start_time = time.time()
    metadata = EmbryoMetadata(str(complex_sam2_file))
    extraction_time = time.time() - start_time
    
    # Should extract 6 embryos with multiple snips quickly
    assert extraction_time < 1.0, f"Extraction took {extraction_time:.2f}s, expected <1.0s"
    
    # Verify correct extraction
    assert len(metadata.data["embryos"]) == 6
    total_snips = sum(len(embryo["snips"]) for embryo in metadata.data["embryos"].values())
    assert total_snips == 7  # Total snips across all embryos


if __name__ == "__main__":
    # Run tests manually for development
    import tempfile
    import traceback
    
    print("Running SAM2 extraction tests...")
    
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
            json.dump(create_complex_sam2_data(), f)
            temp_file = Path(f.name)
        
        # Run individual tests
        tests = [
            test_frame_number_extraction,
            test_embryo_structure_creation,
            test_snip_structure_creation,
            (test_complex_sam2_extraction, temp_file),
            test_snip_id_generation_consistency,
            test_empty_sam2_handling,
            test_malformed_sam2_handling,
            (test_extraction_performance, temp_file)
        ]
        
        passed = 0
        failed = 0
        
        for test_item in tests:
            if isinstance(test_item, tuple):
                test_func, *args = test_item
            else:
                test_func = test_item
                args = []
            
            try:
                print(f"Running {test_func.__name__}...", end=" ")
                test_func(*args)
                print("PASS")
                passed += 1
            except Exception as e:
                print(f"FAIL: {e}")
                traceback.print_exc()
                failed += 1
            
            # Cleanup biology file between tests
            biology_path = temp_file.parent / f"{temp_file.stem}_biology.json"
            biology_path.unlink(missing_ok=True)
        
        print(f"\nTest Results: {passed} passed, {failed} failed")
        
        # Cleanup
        temp_file.unlink(missing_ok=True)
        biology_path.unlink(missing_ok=True)
        
    except Exception as e:
        print(f"Test setup failed: {e}")
        traceback.print_exc()