#!/usr/bin/env python3
"""
Phase 2 Validation Tests

Tests enhanced API functionality:
- Parameter validation (_select_mode)
- Both API approaches: embryo vs snip
- Frame range parsing
- Clear error messages
"""

import json
import tempfile
import time
from pathlib import Path
import sys
import traceback

# Add scripts to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from annotations.embryo_metadata import EmbryoMetadata


def create_test_sam2_data():
    """Create test SAM2 data with multiple frames."""
    return {
        "experiments": {
            "20240418": {
                "videos": {
                    "20240418_A01": {
                        "images": {
                            "20240418_A01_t0010": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0020": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0030": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0040": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            },
                            "20240418_A01_t0050": {
                                "embryos": {
                                    "20240418_A01_e01": {"segmentation": {"counts": "test"}},
                                    "20240418_A01_e02": {"segmentation": {"counts": "test"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    }


def test_parameter_validation():
    """Test _select_mode parameter validation."""
    print("Testing parameter validation...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test ambiguous parameters (both approaches)
        try:
            metadata.add_phenotype("EDEMA", "test_user", 
                                 embryo_id="20240418_A01_e01", 
                                 target="all", 
                                 snip_ids=["20240418_A01_e01_s0010"])
            assert False, "Should have raised ValueError for ambiguous parameters"
        except ValueError as e:
            assert "Ambiguous parameters" in str(e)
            print("✓ Ambiguous parameter detection works")
        
        # Test missing parameters
        try:
            metadata.add_phenotype("EDEMA", "test_user")
            assert False, "Should have raised ValueError for missing parameters"
        except ValueError as e:
            assert "Missing parameters" in str(e)
            print("✓ Missing parameter detection works")
        
        # Test valid embryo approach
        result = metadata.add_phenotype("EDEMA", "test_user", embryo_id="20240418_A01_e01", target="all")
        assert result["approach"] == "embryo"
        assert result["count"] == 5  # Should apply to 5 snips
        print("✓ Valid embryo approach works")
        
        # Test valid snip approach
        snip_ids = ["20240418_A01_e02_s0010", "20240418_A01_e02_s0020"]
        result = metadata.add_phenotype("NORMAL", "test_user", snip_ids=snip_ids)
        assert result["approach"] == "snips"
        assert result["count"] == 2
        print("✓ Valid snip approach works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_frame_range_parsing():
    """Test frame range parsing functionality."""
    print("Testing frame range parsing...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test 'all' target
        result = metadata.add_phenotype("NORMAL", "test_user", embryo_id="20240418_A01_e01", target="all")
        assert result["count"] == 5
        print("✓ 'all' target works")
        
        # Test single frame number
        result = metadata.add_phenotype("EDEMA", "test_user", embryo_id="20240418_A01_e01", target="20")
        assert result["count"] == 1
        assert "20240418_A01_e01_s0020" in result["applied_to"]
        print("✓ Single frame target works")
        
        # Test closed range '20:40' (should include 20, 30, exclude 40)
        result = metadata.add_phenotype("BLUR", "test_user", embryo_id="20240418_A01_e02", target="20:40")
        assert result["count"] == 2  # frames 20, 30
        expected_snips = ["20240418_A01_e02_s0020", "20240418_A01_e02_s0030"]
        for snip in expected_snips:
            assert snip in result["applied_to"]
        print("✓ Closed range '20:40' works")
        
        # Test open-ended range '40:' (should include 40, 50)
        result = metadata.add_phenotype("CORRUPT", "test_user", embryo_id="20240418_A01_e02", target="40:")
        assert result["count"] == 2  # frames 40, 50
        expected_snips = ["20240418_A01_e02_s0040", "20240418_A01_e02_s0050"]
        for snip in expected_snips:
            assert snip in result["applied_to"]
        print("✓ Open-ended range '40:' works")
        
        # Test range from beginning ':30' (should include 10, 20) - use clean embryo with overwrite
        result = metadata.add_phenotype("DEAD", "test_user", embryo_id="20240418_A01_e01", target=":30", overwrite_dead=True)
        assert result["count"] == 2  # frames 10, 20
        expected_snips = ["20240418_A01_e01_s0010", "20240418_A01_e01_s0020"]
        for snip in expected_snips:
            assert snip in result["applied_to"]
        print("✓ Beginning range ':30' works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_snip_approach_validation():
    """Test snip approach validation."""
    print("Testing snip approach validation...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test valid snip IDs
        valid_snips = ["20240418_A01_e01_s0010", "20240418_A01_e01_s0030", "20240418_A01_e02_s0050"]
        result = metadata.add_phenotype("EDEMA", "test_user", snip_ids=valid_snips)
        assert result["count"] == 3
        assert result["approach"] == "snips"
        print("✓ Valid snip IDs work")
        
        # Test invalid snip ID
        try:
            invalid_snips = ["20240418_A01_e01_s0010", "invalid_snip_id"]
            metadata.add_phenotype("NORMAL", "test_user", snip_ids=invalid_snips)
            assert False, "Should have raised ValueError for invalid snip ID"
        except ValueError as e:
            assert "not found" in str(e)
            print("✓ Invalid snip ID detection works")
        
        # Test snip from non-existent embryo
        try:
            invalid_embryo_snips = ["nonexistent_embryo_s0010"]
            metadata.add_phenotype("NORMAL", "test_user", snip_ids=invalid_embryo_snips)
            assert False, "Should have raised ValueError for non-existent embryo"
        except ValueError as e:
            assert "not found" in str(e)
            print("✓ Non-existent embryo detection works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_error_message_clarity():
    """Test that error messages are clear and helpful."""
    print("Testing error message clarity...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test error message for invalid range format
        try:
            metadata.add_phenotype("EDEMA", "test_user", embryo_id="20240418_A01_e01", target="invalid_range")
            assert False, "Should have raised ValueError for invalid range"
        except ValueError as e:
            assert "Invalid target format" in str(e)
            assert "Use 'all'" in str(e)
            print("✓ Invalid range format error is clear")
        
        # Test error message for non-existent frame
        try:
            metadata.add_phenotype("EDEMA", "test_user", embryo_id="20240418_A01_e01", target="999")
            assert False, "Should have raised ValueError for non-existent frame"
        except ValueError as e:
            assert "not found" in str(e)
            print("✓ Non-existent frame error is clear")
        
        # Test error message for range with no matches
        try:
            metadata.add_phenotype("EDEMA", "test_user", embryo_id="20240418_A01_e01", target="100:200")
            assert False, "Should have raised ValueError for empty range"
        except ValueError as e:
            assert "No snips found in range" in str(e)
            print("✓ Empty range error is clear")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_backwards_compatibility():
    """Test that Phase 1 functionality still works."""
    print("Testing backwards compatibility...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Test old-style call (positional arguments)
        result = metadata.add_phenotype("EDEMA", "test_user", "20240418_A01_e01")
        assert result["approach"] == "embryo"
        assert result["target"] == "all"  # Should default to 'all'
        assert result["count"] == 5
        print("✓ Backwards compatibility with positional args works")
        
        # Test old-style call with explicit target='all'
        result = metadata.add_phenotype("NORMAL", "test_user", "20240418_A01_e02", "all")
        assert result["approach"] == "embryo"
        assert result["count"] == 5
        print("✓ Backwards compatibility with target='all' works")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def test_performance():
    """Test that enhanced functionality maintains good performance."""
    print("Testing enhanced API performance...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_sam2.json', delete=False) as f:
        json.dump(create_test_sam2_data(), f)
        f.flush()
        sam2_file = Path(f.name)
    
    try:
        metadata = EmbryoMetadata(str(sam2_file))
        
        # Time various operations
        start_time = time.time()
        
        # Embryo approach with ranges
        metadata.add_phenotype("NORMAL", "user1", embryo_id="20240418_A01_e01", target="all")
        metadata.add_phenotype("EDEMA", "user1", embryo_id="20240418_A01_e01", target="20:40")
        metadata.add_phenotype("BLUR", "user1", embryo_id="20240418_A01_e01", target="40:")
        
        # Snip approach
        snip_ids = ["20240418_A01_e02_s0010", "20240418_A01_e02_s0030", "20240418_A01_e02_s0050"]
        metadata.add_phenotype("CORRUPT", "user2", snip_ids=snip_ids)
        
        total_time = time.time() - start_time
        
        # Should complete quickly
        assert total_time < 0.1, f"Enhanced operations took {total_time:.3f}s, expected <0.1s"
        print(f"✓ Enhanced API operations completed in {total_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - {e}")
        traceback.print_exc()
        return False
    finally:
        sam2_file.unlink(missing_ok=True)
        biology_path = sam2_file.parent / f"{sam2_file.stem}_biology.json"
        biology_path.unlink(missing_ok=True)


def main():
    """Run all Phase 2 validation tests."""
    print("=" * 60)
    print("PHASE 2 ENHANCED API VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_parameter_validation,
        test_frame_range_parsing,
        test_snip_approach_validation,
        test_error_message_clarity,
        test_backwards_compatibility,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAIL - {test_func.__name__}: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("✅ Enhanced API is working correctly")
        print("✅ Parameter validation prevents ambiguous usage")
        print("✅ Both embryo and snip approaches work")
        print("✅ Frame range parsing is functional")
        print("✅ Error messages are clear and helpful")
        print("✅ Backwards compatibility maintained")
        print("✅ Ready to proceed to Phase 3")
    else:
        print("❌ Some tests failed - need to fix issues before Phase 3")
    
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)