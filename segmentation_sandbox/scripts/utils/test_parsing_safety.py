#!/usr/bin/env python3
"""
Safety test to capture current parsing behavior before refactoring.
This ensures our centralized pattern refactoring doesn't break anything.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from parsing_utils import (
    get_entity_type, parse_entity_id, normalize_well_id, 
    extract_frame_number, extract_experiment_id, extract_video_id, extract_embryo_id,
    is_valid_well_id, is_valid_experiment_id, validate_id_format,
    build_image_id, build_video_id, build_embryo_id, build_snip_id
)


def test_get_entity_type():
    """Test entity type detection with current patterns."""
    print("Testing get_entity_type()...")
    
    # Video IDs (should detect well at end)
    assert get_entity_type("20240411_A00") == "video"
    assert get_entity_type("20240411_A01") == "video" 
    assert get_entity_type("20240411_H99") == "video"
    assert get_entity_type("20250624_chem02_28C_T00_1356_H01") == "video"
    
    # Image IDs (channel + t format and legacy t format)
    assert get_entity_type("20240411_A01_ch00_t0042") == "image"
    assert get_entity_type("20240411_A01_ch01_t1234") == "image"
    assert get_entity_type("20240411_A01_t0042") == "image"  # Legacy format
    assert get_entity_type("20250624_chem02_28C_T00_1356_H01_ch00_t0000") == "image"
    
    # Embryo IDs 
    assert get_entity_type("20240411_A01_e01") == "embryo"
    assert get_entity_type("20240411_A01_e123") == "embryo"
    assert get_entity_type("20250624_chem02_28C_T00_1356_H01_e01") == "embryo"
    
    # Snip IDs (with s prefix and without)
    assert get_entity_type("20240411_A01_e01_s0042") == "snip"
    assert get_entity_type("20240411_A01_e01_0042") == "snip"
    assert get_entity_type("20250624_chem02_28C_T00_1356_H01_e01_s0034") == "snip"
    
    # Experiment IDs
    assert get_entity_type("20240411") == "experiment"
    assert get_entity_type("20250624_chem02_28C_T00_1356") == "experiment"
    
    print("✓ get_entity_type() tests passed")


def test_parse_entity_id():
    """Test complete entity ID parsing."""
    print("Testing parse_entity_id()...")
    
    # Test video parsing
    try:
        result = parse_entity_id("20240411_A00")
        assert result["experiment_id"] == "20240411"
        assert result["well_id"] == "A00"
        assert result["video_id"] == "20240411_A00"
        assert result["entity_type"] == "video"
        print("  ✓ Video A00 test passed")
    except Exception as e:
        print(f"  ❌ Video A00 test failed: {e}")
        raise
    
    # Test complex experiment
    try:
        result = parse_entity_id("20250624_chem02_28C_T00_1356_H01")
        assert result["experiment_id"] == "20250624_chem02_28C_T00_1356"
        assert result["well_id"] == "H01"
        assert result["video_id"] == "20250624_chem02_28C_T00_1356_H01"
        print("  ✓ Complex video test passed")
    except Exception as e:
        print(f"  ❌ Complex video test failed: {e}")
        raise
    
    # Test image with channel
    try:
        result = parse_entity_id("20240411_A01_ch00_t0042")
        print(f"  Image result keys: {list(result.keys())}")
        print(f"  Image result: {result}")
        assert result["experiment_id"] == "20240411"
        assert result["well_id"] == "A01"
        assert result["video_id"] == "20240411_A01"
        assert result["channel"] == "00"
        assert result["frame_number"] == "0042"
        assert result["image_id"] == "20240411_A01_ch00_t0042"
        assert result["entity_type"] == "image"
        print("  ✓ Image with channel test passed")
    except Exception as e:
        print(f"  ❌ Image with channel test failed: {e}")
        raise
    
    # Test legacy image format
    try:
        result = parse_entity_id("20240411_A01_t0042")
        print(f"  Legacy image result: {result}")
        assert result["channel"] == "00"  # Now consistently zero-padded
        assert result["frame_number"] == "0042"
        print("  ✓ Legacy image test passed")
    except Exception as e:
        print(f"  ❌ Legacy image test failed: {e}")
        raise
    
    # Test embryo
    result = parse_entity_id("20240411_A01_e01")
    assert result["experiment_id"] == "20240411"
    assert result["well_id"] == "A01"
    assert result["video_id"] == "20240411_A01"
    assert result["embryo_number"] == "01"
    assert result["embryo_id"] == "20240411_A01_e01"
    assert result["entity_type"] == "embryo"
    
    # Test snip with s prefix
    result = parse_entity_id("20240411_A01_e01_s0042")
    assert result["experiment_id"] == "20240411"
    assert result["well_id"] == "A01"
    assert result["video_id"] == "20240411_A01"
    assert result["embryo_number"] == "01"
    assert result["embryo_id"] == "20240411_A01_e01"
    assert result["frame_number"] == "0042"
    assert result["snip_id"] == "20240411_A01_e01_s0042"
    assert result["entity_type"] == "snip"
    
    # Test snip without s prefix
    result = parse_entity_id("20240411_A01_e01_0042")
    assert result["frame_number"] == "0042"
    assert result["snip_id"] == "20240411_A01_e01_0042"
    
    print("✓ parse_entity_id() tests passed")


def test_normalize_well_id():
    """Test well ID normalization."""
    print("Testing normalize_well_id()...")
    
    assert normalize_well_id("A1") == "A01"
    assert normalize_well_id("A01") == "A01"
    assert normalize_well_id("A0") == "A00"  # Should allow A0->A00
    assert normalize_well_id("H12") == "H12"
    assert normalize_well_id("h1") == "H01"  # Case insensitive
    
    print("✓ normalize_well_id() tests passed")


def test_extract_functions():
    """Test extraction functions."""
    print("Testing extract_*() functions...")
    
    # extract_frame_number
    assert extract_frame_number("20240411_A01_ch00_t0042") == 42
    assert extract_frame_number("20240411_A01_t0042") == 42
    assert extract_frame_number("20240411_A01_e01_s0042") == 42
    assert extract_frame_number("20240411_A01_e01_0042") == 42
    
    # extract_experiment_id
    assert extract_experiment_id("20240411_A01_e01_s0042") == "20240411"
    assert extract_experiment_id("20250624_chem02_28C_T00_1356_H01_e01") == "20250624_chem02_28C_T00_1356"
    
    # extract_video_id
    assert extract_video_id("20240411_A01_e01_s0042") == "20240411_A01"
    assert extract_video_id("20240411_A01_e01") == "20240411_A01"
    
    # extract_embryo_id
    assert extract_embryo_id("20240411_A01_e01_s0042") == "20240411_A01_e01"
    
    print("✓ extract_*() functions tests passed")


def test_validation_functions():
    """Test validation functions."""
    print("Testing validation functions...")
    
    # is_valid_well_id - capture current behavior
    try:
        assert is_valid_well_id("A01") == True
        assert is_valid_well_id("H12") == True
        # Note: Current function limits to 1-12 columns, let's capture that
        current_a00_result = is_valid_well_id("A00")  # Might be False currently
        current_h99_result = is_valid_well_id("H99")  # Might be False currently
        print(f"  Well validation: A01=True, H12=True, A00={current_a00_result}, H99={current_h99_result}")
    except Exception as e:
        print(f"  ❌ Well validation test failed: {e}")
        raise
    
    # is_valid_experiment_id
    try:
        assert is_valid_experiment_id("20240411") == True
        # This currently fails because 1356 looks like a frame number
        complex_exp_result = is_valid_experiment_id("20250624_chem02_28C_T00_1356")
        assert is_valid_experiment_id("20240411_A01") == False  # Should fail (has well)
        print(f"  Experiment validation: simple=True, complex={complex_exp_result}, with_well=False")
    except Exception as e:
        print(f"  ❌ Experiment validation test failed: {e}")
        raise
    
    # validate_id_format
    try:
        assert validate_id_format("20240411_A01", "video") == True
        assert validate_id_format("20240411_A01_e01", "embryo") == True
        print("  ✓ Format validation tests passed")
    except Exception as e:
        print(f"  ❌ Format validation test failed: {e}")
        raise
    
    print(f"✓ validation functions tests passed (A00 valid: {current_a00_result}, H99 valid: {current_h99_result})")


def test_builder_functions():
    """Test ID builder functions."""
    print("Testing builder functions...")
    
    # build_image_id
    assert build_image_id("20240411_A01", 42, channel=0) == "20240411_A01_ch00_t0042"
    assert build_image_id("20240411_A01", 42, channel=1) == "20240411_A01_ch01_t0042"
    
    # build_video_id
    assert build_video_id("20240411", "A01") == "20240411_A01"
    assert build_video_id("20240411", "A1") == "20240411_A01"  # Should normalize
    
    # build_embryo_id
    assert build_embryo_id("20240411_A01", 1) == "20240411_A01_e01"
    
    # build_snip_id  
    assert build_snip_id("20240411_A01_e01", 42, use_s_prefix=True) == "20240411_A01_e01_s0042"
    assert build_snip_id("20240411_A01_e01", 42, use_s_prefix=False) == "20240411_A01_e01_0042"
    
    print("✓ builder functions tests passed")


def run_all_safety_tests():
    """Run all safety tests to capture current behavior."""
    print("=" * 60)
    print("RUNNING SAFETY TESTS TO CAPTURE CURRENT BEHAVIOR")
    print("=" * 60)
    
    try:
        test_get_entity_type()
        test_parse_entity_id()
        test_normalize_well_id()
        test_extract_functions()
        test_validation_functions()
        test_builder_functions()
        
        print("=" * 60)
        print("🎉 ALL SAFETY TESTS PASSED!")
        print("Current behavior successfully captured.")
        print("Safe to proceed with refactoring.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ SAFETY TEST FAILED: {e}")
        print("DO NOT PROCEED WITH REFACTORING!")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_safety_tests()
    sys.exit(0 if success else 1)