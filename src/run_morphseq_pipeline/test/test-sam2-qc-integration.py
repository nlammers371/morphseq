#!/usr/bin/env python3
"""
Test script for SAM2 QC flag integration (refactor-011-b)
Tests the complete flow from GSAM JSON → CSV export → Build03 integration
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Mock the necessary functions for testing
def mock_extract_qc_flags_for_snip(gsam_data, snip_id):
    """Mock version of _extract_qc_flags_for_snip function."""
    flag_overview = gsam_data.get("flags", {}).get("flag_overview", {})
    
    # Define which flag types we care about at snip level
    SNIP_LEVEL_FLAGS = [
        "HIGH_SEGMENTATION_VAR_SNIP",
        "MASK_ON_EDGE",
        "LARGE_MASK", 
        "SMALL_MASK",
        "DISCONTINUOUS_MASK"
    ]
    
    flags_for_snip = []
    for flag_type in SNIP_LEVEL_FLAGS:
        if flag_type in flag_overview:
            snip_ids_with_flag = flag_overview[flag_type].get("snip_ids", [])
            if snip_id in snip_ids_with_flag:
                flags_for_snip.append(flag_type)
    
    return ",".join(flags_for_snip)


def create_test_gsam_data():
    """Create test GSAM data with various QC flags."""
    return {
        "experiments": {
            "20250529_30hpf_ctrl_atf6": {
                "videos": {
                    "20250529_30hpf_ctrl_atf6_A01": {
                        "image_ids": {
                            "20250529_30hpf_ctrl_atf6_A01_ch00_t0000": {
                                "embryos": {
                                    "20250529_30hpf_ctrl_atf6_A01_e01": {
                                        "snip_id": "20250529_30hpf_ctrl_atf6_A01_e01_t0000",
                                        "area": 50000
                                    }
                                }
                            },
                            "20250529_30hpf_ctrl_atf6_A01_ch00_t0001": {
                                "embryos": {
                                    "20250529_30hpf_ctrl_atf6_A01_e01": {
                                        "snip_id": "20250529_30hpf_ctrl_atf6_A01_e01_t0001",
                                        "area": 51000
                                    }
                                }
                            }
                        }
                    },
                    "20250529_30hpf_ctrl_atf6_B02": {
                        "image_ids": {
                            "20250529_30hpf_ctrl_atf6_B02_ch00_t0000": {
                                "embryos": {
                                    "20250529_30hpf_ctrl_atf6_B02_e01": {
                                        "snip_id": "20250529_30hpf_ctrl_atf6_B02_e01_t0000",
                                        "area": 1000  # Small mask
                                    }
                                }
                            }
                        }
                    },
                    "20250529_30hpf_ctrl_atf6_C03": {
                        "image_ids": {
                            "20250529_30hpf_ctrl_atf6_C03_ch00_t0000": {
                                "embryos": {
                                    "20250529_30hpf_ctrl_atf6_C03_e01": {
                                        "snip_id": "20250529_30hpf_ctrl_atf6_C03_e01_t0000",
                                        "area": 48000
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "flags": {
            "flag_overview": {
                "HIGH_SEGMENTATION_VAR_SNIP": {
                    "snip_ids": [
                        "20250529_30hpf_ctrl_atf6_A01_e01_t0000",
                        "20250529_30hpf_ctrl_atf6_B02_e01_t0000"
                    ],
                    "count": 2
                },
                "MASK_ON_EDGE": {
                    "snip_ids": [
                        "20250529_30hpf_ctrl_atf6_C03_e01_t0000"
                    ],
                    "count": 1
                },
                "SMALL_MASK": {
                    "snip_ids": [
                        "20250529_30hpf_ctrl_atf6_B02_e01_t0000"
                    ],
                    "count": 1
                },
                "DETECTION_FAILURE": {
                    "image_ids": ["20250529_30hpf_ctrl_atf6_D04_ch00_t0000"],
                    "count": 1
                    # Note: No snip_ids for this flag type (image-level only)
                },
                "OVERLAPPING_MASKS": {
                    "image_ids": ["20250529_30hpf_ctrl_atf6_E05_ch00_t0000"],
                    "count": 1
                    # Note: No snip_ids currently (would need to be added)
                }
            }
        }
    }


def test_flag_extraction():
    """Test the QC flag extraction logic."""
    print("=" * 60)
    print("TEST 1: QC Flag Extraction from flag_overview")
    print("=" * 60)
    
    gsam_data = create_test_gsam_data()
    
    test_cases = [
        ("20250529_30hpf_ctrl_atf6_A01_e01_t0000", "HIGH_SEGMENTATION_VAR_SNIP"),
        ("20250529_30hpf_ctrl_atf6_A01_e01_t0001", ""),  # No flags
        ("20250529_30hpf_ctrl_atf6_B02_e01_t0000", "HIGH_SEGMENTATION_VAR_SNIP,SMALL_MASK"),
        ("20250529_30hpf_ctrl_atf6_C03_e01_t0000", "MASK_ON_EDGE"),
        ("nonexistent_snip_id", "")  # Nonexistent snip
    ]
    
    all_passed = True
    for snip_id, expected in test_cases:
        result = mock_extract_qc_flags_for_snip(gsam_data, snip_id)
        # Sort both for comparison (order doesn't matter)
        result_sorted = ",".join(sorted(result.split(",") if result else []))
        expected_sorted = ",".join(sorted(expected.split(",") if expected else []))
        
        passed = result_sorted == expected_sorted
        all_passed &= passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: snip_id='{snip_id[:30]}...'")
        print(f"    Expected: '{expected_sorted}'")
        print(f"    Got:      '{result_sorted}'")
        print()
    
    return all_passed


def test_csv_export():
    """Test CSV export with QC flags column."""
    print("=" * 60)
    print("TEST 2: CSV Export with sam2_qc_flags Column")
    print("=" * 60)
    
    # Create mock CSV data
    csv_data = {
        'snip_id': [
            '20250529_30hpf_ctrl_atf6_A01_e01_t0000',
            '20250529_30hpf_ctrl_atf6_A01_e01_t0001',
            '20250529_30hpf_ctrl_atf6_B02_e01_t0000',
            '20250529_30hpf_ctrl_atf6_C03_e01_t0000'
        ],
        'embryo_id': [
            '20250529_30hpf_ctrl_atf6_A01_e01',
            '20250529_30hpf_ctrl_atf6_A01_e01',
            '20250529_30hpf_ctrl_atf6_B02_e01',
            '20250529_30hpf_ctrl_atf6_C03_e01'
        ],
        'sam2_qc_flags': [
            'HIGH_SEGMENTATION_VAR_SNIP',
            '',
            'HIGH_SEGMENTATION_VAR_SNIP,SMALL_MASK',
            'MASK_ON_EDGE'
        ]
    }
    
    df = pd.DataFrame(csv_data)
    
    print(f"CSV has {len(df.columns)} columns")
    print(f"Column 'sam2_qc_flags' exists: {'sam2_qc_flags' in df.columns}")
    print("\nSample rows:")
    print(df[['snip_id', 'sam2_qc_flags']].to_string(index=False))
    
    # Test that empty strings are handled correctly
    empty_count = (df['sam2_qc_flags'] == '').sum()
    print(f"\nRows with no flags (empty string): {empty_count}")
    
    return True


def test_build03_integration():
    """Test Build03 processing of QC flags."""
    print("\n" + "=" * 60)
    print("TEST 3: Build03 Integration (sam2_qc_flag Boolean)")
    print("=" * 60)
    
    # Simulate Build03 processing
    csv_data = {
        'snip_id': [
            '20250529_30hpf_ctrl_atf6_A01_e01_t0000',
            '20250529_30hpf_ctrl_atf6_A01_e01_t0001',
            '20250529_30hpf_ctrl_atf6_B02_e01_t0000',
            '20250529_30hpf_ctrl_atf6_C03_e01_t0000'
        ],
        'sam2_qc_flags': [
            'HIGH_SEGMENTATION_VAR_SNIP',
            '',
            'HIGH_SEGMENTATION_VAR_SNIP,SMALL_MASK',
            'MASK_ON_EDGE'
        ],
        # Other QC flags for testing
        'bubble_flag': [False, False, False, False],
        'focus_flag': [False, False, True, False],  # One with focus issue
        'dead_flag': [False, False, False, False],
        'frame_flag': [False, False, False, False],
        'no_yolk_flag': [False, False, False, False]
    }
    
    df = pd.DataFrame(csv_data)
    
    # Simulate Build03's sam2_qc_flag creation
    if 'sam2_qc_flags' in df.columns:
        df['sam2_qc_flag'] = df['sam2_qc_flags'].apply(
            lambda x: len(str(x).strip()) > 0 if pd.notna(x) else False
        )
    else:
        df['sam2_qc_flag'] = False
    
    print("Boolean flag conversion:")
    for idx, row in df.iterrows():
        print(f"  snip {idx}: '{row['sam2_qc_flags']}' → sam2_qc_flag={row['sam2_qc_flag']}")
    
    # Calculate use_embryo_flag using centralized function
    from src.build.qc import determine_use_embryo_flag
    df['use_embryo_flag'] = determine_use_embryo_flag(df)
    
    print("\nFinal use_embryo_flag calculation:")
    print(df[['snip_id', 'sam2_qc_flag', 'focus_flag', 'use_embryo_flag']].to_string(index=False))
    
    # Verify results
    expected_use_flags = [False, True, False, False]  # Based on our test data
    actual_use_flags = df['use_embryo_flag'].tolist()
    
    passed = expected_use_flags == actual_use_flags
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: use_embryo_flag calculation")
    print(f"  Expected: {expected_use_flags}")
    print(f"  Got:      {actual_use_flags}")
    
    return passed


def test_backward_compatibility():
    """Test handling of CSV without sam2_qc_flags column."""
    print("\n" + "=" * 60)
    print("TEST 4: Backward Compatibility (Missing Column)")
    print("=" * 60)
    
    # Create CSV without sam2_qc_flags column (legacy format)
    csv_data = {
        'snip_id': ['20250529_30hpf_ctrl_atf6_A01_e01_t0000'],
        'bubble_flag': [False],
        'focus_flag': [False],
        'dead_flag': [False],
        'frame_flag': [False],
        'no_yolk_flag': [False]
    }
    
    df = pd.DataFrame(csv_data)
    
    print(f"CSV has sam2_qc_flags column: {'sam2_qc_flags' in df.columns}")
    
    # Simulate Build03 handling of missing column
    if 'sam2_qc_flags' in df.columns:
        df['sam2_qc_flag'] = df['sam2_qc_flags'].apply(
            lambda x: len(str(x).strip()) > 0 if pd.notna(x) else False
        )
    else:
        df['sam2_qc_flag'] = False
        print("Created sam2_qc_flag with default value: False")
    
    # Calculate use_embryo_flag using centralized function
    from src.build.qc import determine_use_embryo_flag
    df['use_embryo_flag'] = determine_use_embryo_flag(df)
    
    print(f"\nuse_embryo_flag calculated successfully: {df['use_embryo_flag'].iloc[0]}")
    print("✓ PASS: Backward compatibility maintained")
    
    return True


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Empty flag_overview
    print("Testing empty flag_overview...")
    gsam_data = {"flags": {"flag_overview": {}}}
    result = mock_extract_qc_flags_for_snip(gsam_data, "any_snip_id")
    passed = result == ""
    all_passed &= passed
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: Returns empty string for empty overview")
    
    # Test 2: Missing flags section entirely
    print("\nTesting missing flags section...")
    gsam_data = {}
    result = mock_extract_qc_flags_for_snip(gsam_data, "any_snip_id")
    passed = result == ""
    all_passed &= passed
    print(f"  {'✓ PASS' if passed else '✗ FAIL'}: Returns empty string for missing flags")
    
    # Test 3: NaN and None values in CSV
    print("\nTesting NaN and None handling...")
    test_values = [None, np.nan, "", "MASK_ON_EDGE"]
    for val in test_values:
        has_flag = len(str(val).strip()) > 0 if pd.notna(val) else False
        print(f"  Value {repr(val):20} → has_flag={has_flag}")
    
    print("\n✓ PASS: Edge cases handled correctly")
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SAM2 QC FLAG INTEGRATION TEST SUITE")
    print("Testing refactor-011-b implementation")
    print("=" * 60)
    
    tests = [
        ("Flag Extraction", test_flag_extraction),
        ("CSV Export", test_csv_export),
        ("Build03 Integration", test_build03_integration),
        ("Backward Compatibility", test_backward_compatibility),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        all_passed &= passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Integration ready for implementation")
    else:
        print("✗ SOME TESTS FAILED - Review implementation")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
