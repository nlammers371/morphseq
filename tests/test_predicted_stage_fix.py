#!/usr/bin/env python3
"""
Test script to validate the predicted_stage_hpf fix for SAM2 pipeline integration.

This script tests:
1. SAM2 CSV loading with corrected well metadata
2. predicted_stage_hpf calculation 
3. Build04 compatibility (column presence)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

def test_sam2_csv_loading():
    """Test if the enhanced SAM2 CSV has required metadata."""
    print("=== Testing SAM2 CSV Loading ===")
    
    csv_path = "sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check required columns
        required_cols = ['start_age_hpf', 'temperature', 'Time Rel (s)']
        for col in required_cols:
            if col in df.columns and not df[col].isna().all():
                sample_val = df[col].dropna().iloc[0]
                print(f"✅ {col}: {sample_val}")
            else:
                print(f"❌ {col}: Missing or empty")
                return False
        return True
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return False

def test_predicted_stage_calculation():
    """Test the predicted_stage_hpf calculation in Build03A integration."""
    print("\n=== Testing predicted_stage_hpf Calculation ===")
    
    try:
        from src.build.build03A_process_images import segment_wells_sam2_csv
        
        root = '/net/trapnell/vol1/home/nlammers/projects/data/morphseq'
        exp_name = '20250612_30hpf_ctrl_atf6'
        sam2_csv_path = 'sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv'
        
        # Test the function
        df = segment_wells_sam2_csv(root, exp_name, sam2_csv_path)
        
        if 'predicted_stage_hpf' in df.columns:
            sample_values = df['predicted_stage_hpf'].head(3)
            print(f"✅ predicted_stage_hpf calculated: {sample_values.tolist()}")
            
            # Validate the calculation manually
            row = df.iloc[0]
            expected = row['start_age_hpf'] + (row['Time Rel (s)'] / 3600.0) * (0.055 * row['temperature'] - 0.57)
            actual = row['predicted_stage_hpf']
            
            if abs(expected - actual) < 0.001:
                print(f"✅ Calculation correct: {actual:.6f}")
                return True
            else:
                print(f"❌ Calculation mismatch: expected {expected:.6f}, got {actual:.6f}")
                return False
        else:
            print(f"❌ predicted_stage_hpf column missing from result")
            return False
            
    except Exception as e:
        print(f"❌ Calculation test failed: {e}")
        return False

def test_build04_compatibility():
    """Test if the generated data would work with Build04."""
    print("\n=== Testing Build04 Compatibility ===")
    
    try:
        from src.build.build03A_process_images import segment_wells_sam2_csv
        
        root = '/net/trapnell/vol1/home/nlammers/projects/data/morphseq'
        exp_name = '20250612_30hpf_ctrl_atf6'
        sam2_csv_path = 'sam2_metadata_20250612_30hpf_ctrl_atf6_enhanced.csv'
        
        df = segment_wells_sam2_csv(root, exp_name, sam2_csv_path)
        
        # Build04 requires these columns (from error analysis)
        build04_required = ['predicted_stage_hpf', 'surface_area_um', 'use_embryo_flag']
        
        missing_cols = []
        for col in build04_required:
            if col not in df.columns:
                missing_cols.append(col)
            else:
                print(f"✅ {col}: Present")
        
        if missing_cols:
            print(f"❌ Missing Build04 columns: {missing_cols}")
            return False
        else:
            print("✅ All required Build04 columns present")
            return True
            
    except Exception as e:
        print(f"❌ Build04 compatibility test failed: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("🧪 Testing SAM2 predicted_stage_hpf Fix")
    print("=" * 50)
    
    tests = [
        ("CSV Loading", test_sam2_csv_loading),
        ("Calculation", test_predicted_stage_calculation), 
        ("Build04 Compatibility", test_build04_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("🏁 TEST RESULTS:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Pipeline fix is ready.")
        print("\nNext steps:")
        print("1. Run full Build03A→Build04→Build05 pipeline")
        print("2. Verify no KeyError on 'predicted_stage_hpf'")
        print("3. Confirm training data generation")
    else:
        print("\n❌ Some tests failed. Check implementation.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())