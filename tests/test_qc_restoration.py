#!/usr/bin/env python3
"""
Test script for QC restoration on experiment 20250622_chem_28C_T00_1425

This script tests the newly implemented QC restoration functions:
- segment_wells_sam2_csv() - Load SAM2 data
- compile_embryo_stats_sam2() - Process with comprehensive QC
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

from src.build.build03A_process_images import segment_wells_sam2_csv, compile_embryo_stats_sam2

def test_qc_restoration():
    """Test the complete QC restoration pipeline."""
    
    # Experiment parameters
    root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    exp_name = "20250622_chem_28C_T00_1425"
    sam2_csv_path = root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{exp_name}.csv"
    
    print(f"🧪 Testing QC restoration on experiment: {exp_name}")
    print(f"📁 Root: {root}")
    print(f"📄 SAM2 CSV: {sam2_csv_path}")
    print()
    
    if not sam2_csv_path.exists():
        print(f"❌ SAM2 CSV not found: {sam2_csv_path}")
        return False
    
    try:
        # Step 1: Load SAM2 data using consolidated function
        print("🔄 Step 1: Loading SAM2 metadata...")
        tracked_df = segment_wells_sam2_csv(
            root=root,
            exp_name=exp_name, 
            sam2_csv_path=sam2_csv_path
        )
        print(f"✅ Loaded {len(tracked_df)} embryo records")
        print()
        
        # Step 2: Process with comprehensive QC using NEW function
        print("🔄 Step 2: Processing with comprehensive QC restoration...")
        stats_df = compile_embryo_stats_sam2(
            root=root,
            tracked_df=tracked_df
        )
        print()
        
        # Step 3: Analyze results
        print("📊 QC Analysis Results:")
        total = len(stats_df)
        usable = (stats_df["use_embryo_flag"] == "true").sum() if "use_embryo_flag" in stats_df.columns else 0
        
        # Count individual QC flags using robust logic (matching the actual QC processing)
        if "sam2_qc_flags" in stats_df.columns:
            def is_meaningful_qc_flag(val):
                import pandas as pd
                if pd.isna(val):
                    return False
                flag_str = str(val).strip().lower()
                return flag_str and flag_str not in ["", "nan", "none", "null"]
            sam2_flags = stats_df["sam2_qc_flags"].apply(is_meaningful_qc_flag).sum()
        else:
            sam2_flags = 0
        dead_flags = (stats_df["dead_flag"] == "true").sum() if "dead_flag" in stats_df.columns else 0
        frame_flags = (stats_df["frame_flag"] == "true").sum() if "frame_flag" in stats_df.columns else 0
        focus_flags = (stats_df["focus_flag"] == "true").sum() if "focus_flag" in stats_df.columns else 0
        bubble_flags = (stats_df["bubble_flag"] == "true").sum() if "bubble_flag" in stats_df.columns else 0
        no_yolk_flags = (stats_df["no_yolk_flag"] == "true").sum() if "no_yolk_flag" in stats_df.columns else 0
        
        print(f"   • Total embryos processed: {total}")
        print(f"   • Final usable embryos: {usable} ({usable/total*100:.1f}%)")
        print(f"   • SAM2 QC flags: {sam2_flags}")
        print(f"   • Dead flags: {dead_flags}")
        print(f"   • Frame flags: {frame_flags}")
        print(f"   • Focus flags: {focus_flags}")
        print(f"   • Bubble flags: {bubble_flags}")
        print(f"   • No yolk flags: {no_yolk_flags}")
        print()
        
        # Check if we have geometry data
        if "area_px" in stats_df.columns:
            geometry_computed = (stats_df["area_px"].astype(str).str.strip() != "").sum()
            print(f"   • Geometry computed: {geometry_computed}/{total} records")
        
        # Check if we have fraction_alive data
        if "fraction_alive" in stats_df.columns:
            frac_alive_computed = (stats_df["fraction_alive"].astype(str).str.strip() != "").sum()
            print(f"   • Fraction alive computed: {frac_alive_computed}/{total} records")
        
        print()
        print("✅ QC restoration test completed successfully!")
        print(f"🎯 Result: {usable} usable embryos out of {total} total ({usable/total*100:.1f}% pass rate)")
        
        # Save test results
        output_path = root / "test_qc_restoration_output.csv"
        stats_df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qc_restoration()
    sys.exit(0 if success else 1)