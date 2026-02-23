# Refactor-011-B: SAM2 QC Flag Integration into use_embryo_flag Logic

**Created**: 2025-09-06  
**Status**: In Progress  
**Depends On**: Refactor-011 Complete

## **Executive Summary**

Add snip_id-level QC flags from SAM2 analysis into the use_embryo_flag calculation pipeline. This MVP integration will extract flags from the `flag_overview` summary section which compiles all flagged snip_ids by flag type, and incorporate them into Build03's quality filtering logic.

**Key Goals:**
- Extract SAM2 QC flags at snip_id level from flag_overview compiled summary
- Add sam2_qc_flags column to SAM2 CSV export
- Integrate SAM2 QC flags into Build03's use_embryo_flag calculation
- Maintain backward compatibility with existing pipelines

## **Background & Problem**

**Current State:**
- SAM2 pipeline generates comprehensive QC analysis with snip-level flags (Step 05)
- QC flags are compiled in `flag_overview` section by flag type with snip_ids lists
- SAM2 CSV export (39 columns) doesn't include any QC flag information
- Build03 use_embryo_flag only considers Build02-based flags (bubble, focus, frame, dead, yolk)
- SAM2-specific quality issues (segmentation variability, edge masks, etc.) are ignored

**Issues:**
- Embryos with SAM2 quality issues still pass through to analysis
- Manual QC required to identify problematic SAM2 segmentations
- Inconsistent quality filtering between legacy and SAM2 pipelines

## **Scope (This Refactor)**

### **In Scope**
1. **SAM2 CSV Enhancement**: Add sam2_qc_flags column using flag_overview lookup
2. **Build03 Integration**: Parse QC flags and create boolean sam2_qc_flag column
3. **use_embryo_flag Enhancement**: Include sam2_qc_flag in final quality calculation
4. **Backward Compatibility**: Ensure existing CSVs without QC flags continue working

### **Out of Scope**
- Changes to SAM2 QC analysis logic (05_sam2_qc_analysis.py)
- Modifications to flag_overview generation
- Build02 QC flag modifications
- Performance optimizations of QC processing

## **Implementation Plan**

### **Stage 1: SAM2 CSV Export Enhancement**

#### **1.1 Add sam2_qc_flags Column**
**File**: `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`
**Changes**:
- Add 'sam2_qc_flags' to CSV_COLUMNS (making it 40 columns total)
- Implement `_extract_qc_flags_for_snip()` method using flag_overview
- Integrate QC flag extraction into CSV row generation

**Implementation**:
```python
def _extract_qc_flags_for_snip(self, snip_id: str) -> str:
    """Extract SAM2 QC flags from flag_overview summary."""
    flag_overview = self.sam2_data.get("flags", {}).get("flag_overview", {})
    
    # Find all flag types that include this snip_id
    flags_for_snip = []
    for flag_type, flag_data in flag_overview.items():
        snip_ids_with_flag = flag_data.get("snip_ids", [])
        if snip_id in snip_ids_with_flag:
            flags_for_snip.append(flag_type)
    
    return ",".join(flags_for_snip)
```

### **Stage 2: Build03 QC Flag Processing**

#### **2.1 SAM2 QC Flag Detection**
**File**: `src/build/build03A_process_images.py`
**Function**: `segment_wells_sam2_csv()`
**Changes**:
- Read sam2_qc_flags column from SAM2 CSV
- Parse comma-separated flags into boolean sam2_qc_flag column
- Pass sam2_qc_flag through to downstream processing

**Implementation**:
```python
# Add SAM2 QC flag processing in segment_wells_sam2_csv()
if 'sam2_qc_flags' in sam2_df.columns:
    sam2_df['sam2_qc_flag'] = sam2_df['sam2_qc_flags'].apply(
        lambda x: len(str(x).strip()) > 0 if pd.notna(x) else False
    )
else:
    sam2_df['sam2_qc_flag'] = False
```

#### **2.2 Enhanced use_embryo_flag Logic**
**File**: `src/build/build03A_process_images.py`
**Function**: `compile_embryo_stats()`
**Location**: Around line 1101
**Changes**:
- Include sam2_qc_flag in use_embryo_flag calculation
- Updated logic: `~(bubble_flag | focus_flag | frame_flag | dead_flag | no_yolk_flag | sam2_qc_flag)`

**Implementation**:
```python
# Enhanced master flag calculation (line ~1101)
tracked_df["use_embryo_flag"] = ~(
    tracked_df["bubble_flag"].values.astype(bool) | 
    tracked_df["focus_flag"].values.astype(bool) |
    tracked_df["frame_flag"].values.astype(bool) | 
    tracked_df["dead_flag"].values.astype(bool) |
    tracked_df["no_yolk_flag"].values.astype(bool) |
    tracked_df.get("sam2_qc_flag", False).astype(bool)  # NEW: SAM2 QC integration
)
```

## **Data Flow**

### **flag_overview Structure**
The flag_overview section compiles all QC flags by type:

```json
{
  "flags": {
    "flag_overview": {
      "HIGH_SEGMENTATION_VAR_SNIP": {
        "snip_ids": ["20250529_30hpf_ctrl_atf6_A01_e01_t0000", "20250529_30hpf_ctrl_atf6_B02_e01_t0000"],
        "count": 2
      },
      "EDGE_MASK_SNIP": {
        "snip_ids": ["20250529_30hpf_ctrl_atf6_C03_e01_t0000"],
        "count": 1
      }
    }
  }
}
```

### **Pipeline Flow**
1. **SAM2 QC Analysis**: 05_sam2_qc_analysis.py generates flags and compiles them in flag_overview
2. **CSV Export**: export_sam2_metadata_to_csv.py searches flag_overview for each snip_id
3. **CSV Output**: sam2_qc_flags column contains comma-separated flag list per snip
4. **Build03 Processing**: segment_wells_sam2_csv() creates sam2_qc_flag boolean from CSV
5. **Final Integration**: compile_embryo_stats() includes sam2_qc_flag in use_embryo_flag calculation

## **Expected SAM2 QC Flag Types**

Based on 05_sam2_qc_analysis.py:
- `HIGH_SEGMENTATION_VAR_SNIP` - High area variance between frames
- `EDGE_MASK_SNIP` - Mask touches image edges  
- `OVERLAPPING_MASK_SNIP` - Overlapping with other embryos
- `LARGE_MASK_SNIP` - Abnormally large segmentation
- `SMALL_MASK_SNIP` - Abnormally small segmentation
- `DISCONTINUOUS_MASK_SNIP` - Fragmented segmentation

## **Files Modified**

### **Enhanced Files**
1. **`segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`**
   - Add sam2_qc_flags to CSV_COLUMNS
   - Implement `_extract_qc_flags_for_snip()` method
   - Integrate QC flag extraction into `_generate_csv_rows()`

2. **`src/build/build03A_process_images.py`**
   - Add sam2_qc_flag processing in `segment_wells_sam2_csv()`
   - Update use_embryo_flag calculation in `compile_embryo_stats()`

### **New Documentation**
- `docs/refactors/refactor-011-b-sam2-qc-integration.md` (this file)

## **Testing Strategy**

### **Unit Tests**
1. **CSV Export**: Verify sam2_qc_flags column contains correct flag strings
2. **Flag Extraction**: Test `_extract_qc_flags_for_snip()` with various flag_overview scenarios
3. **Build03 Processing**: Confirm sam2_qc_flag boolean creation from CSV
4. **use_embryo_flag Integration**: Verify flagged embryos are properly excluded

### **Integration Tests**
1. **End-to-End**: Run full SAM2 pipeline with QC → CSV export → Build03 → df01
2. **Regression**: Ensure existing CSVs without sam2_qc_flags work unchanged
3. **Edge Cases**: Test with empty flag_overview, missing columns, malformed flags

### **Real Data Validation**
- Use test experiment with known SAM2 QC issues
- Verify flagged embryos are excluded from final use_embryo_flag=True dataset
- Compare before/after embryo counts to confirm QC filtering effectiveness

## **Benefits & Impact**

### **Immediate Benefits**
- **Comprehensive QC**: SAM2-specific quality issues automatically excluded
- **Automated Filtering**: No manual intervention needed for QC integration
- **Backward Compatible**: Existing pipelines continue working unchanged
- **Minimal Scope**: Only 2 files modified, focused implementation

### **Long-term Impact**
- **Higher Quality Data**: Better filtering leads to cleaner training datasets
- **Consistent QC**: SAM2 and legacy pipelines have equivalent quality standards
- **Extensible Framework**: Easy to add new SAM2 QC flag types in the future

### **Expected Results**
- More embryos filtered out due to additional SAM2 QC criteria
- Reduced manual QC workload for SAM2-processed experiments
- Better concordance between SAM2 quality analysis and final use_embryo_flag

## **Risk Assessment & Mitigation**

### **Technical Risks**
- **CSV Schema Change**: Adding column could break downstream parsers
  - *Mitigation*: Column added at end, existing columns unchanged, backward compatibility tested
- **Performance Impact**: QC flag lookup could slow CSV export
  - *Mitigation*: flag_overview lookup is O(1) per flag type, minimal overhead
- **Memory Usage**: Large flag_overview sections could impact processing
  - *Mitigation*: flag_overview is already loaded for QC analysis, no additional memory needed

### **Integration Risks**
- **Build03 Compatibility**: New column parsing could fail with old CSVs
  - *Mitigation*: Defensive programming with column existence checks and defaults
- **Flag Format Changes**: SAM2 QC flag naming could change
  - *Mitigation*: Generic flag processing, no hardcoded flag names in logic

## **Acceptance Criteria**

### **Functional Requirements**
- [ ] sam2_qc_flags column appears in SAM2 CSV export (40 columns total)
- [ ] QC flags extracted correctly from flag_overview compiled summary
- [ ] Build03 creates sam2_qc_flag boolean from CSV sam2_qc_flags column
- [ ] use_embryo_flag incorporates sam2_qc_flag in quality calculation
- [ ] Embryos with SAM2 QC flags have use_embryo_flag=False
- [ ] Existing CSVs without sam2_qc_flags column continue working

### **Quality Requirements**
- [ ] No false positives: embryos without flags not incorrectly flagged
- [ ] No false negatives: all flagged embryos properly detected and excluded
- [ ] Backward compatibility: no regression in existing pipeline functionality
- [ ] Performance: CSV export time increase <10% with QC flag processing

### **Documentation Requirements**
- [x] Complete implementation plan documented
- [ ] Code comments explain QC flag integration logic
- [ ] Testing results validate all acceptance criteria

---

## **Implementation Status**

**Current Phase**: Implementation Complete  
**Completed**: 
1. ✅ Added sam2_qc_flags to CSV_COLUMNS (40 columns total)
2. ✅ Implemented _extract_qc_flags_for_snip() method using flag_overview
3. ✅ Added QC flag extraction to CSV row generation
4. ✅ Added sam2_qc_flag processing in segment_wells_sam2_csv()
5. ✅ Enhanced use_embryo_flag calculation to include sam2_qc_flag

**Next Steps**: 
- Test end-to-end integration with real SAM2 QC data
- Validate that flagged embryos are properly excluded from analysis

This MVP establishes the foundation for comprehensive SAM2 QC integration into the main pipeline workflow.