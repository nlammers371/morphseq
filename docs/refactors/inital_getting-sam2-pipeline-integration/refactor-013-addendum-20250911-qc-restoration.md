# Refactor-013 Addendum: Build03 QC Functionality Restoration

**Created**: 2025-09-11  
**Status**: In Progress  
**Urgency**: Critical  
**Depends On**: Refactor-010-B, Refactor-011-B

## **Executive Summary**

During SAM2 pipeline integration, the `run_build03.py` script lost ALL QC functionality and was reduced to a basic CSV transformation tool. Meanwhile, the comprehensive QC logic remains implemented and working in `build03A_process_images.py`. This addendum documents the plan to restore full QC functionality by consolidating ALL build03 functionality into `build03A_process_images.py` and making `run_build03.py` a thin CLI wrapper.

**Critical Discovery**: `run_build03.py` was hardcoding `use_embryo_flag = "true"` for ALL embryos, completely ignoring SAM2 QC flags like `MASK_ON_EDGE`.

## **Problem Analysis**

### **What We Lost**
The current `run_build03.py` is missing:
- ❌ **SAM2 QC flag integration**: Ignores `MASK_ON_EDGE`, `HIGH_SEGMENTATION_VAR_SNIP`, etc.
- ❌ **Legacy QC flags**: No `dead_flag`, `no_yolk_flag`, `frame_flag`, `focus_flag`, `bubble_flag`
- ❌ **Build02 mask loading**: Cannot access viability, yolk, focus, bubble masks
- ❌ **Proper `use_embryo_flag` logic**: All embryos marked as usable regardless of QC issues
- ❌ **Speed calculations**: Missing embryo motion analysis

### **What We Have Working**
- ✅ **Basic geometry computation**: Area, perimeter, centroids from SAM2 masks
- ✅ **Pixel scale integration**: SAM2 CSV + Build01 metadata scaling
- ✅ **Predicted stage calculation**: Kimmel formula implementation
- ✅ **Per-experiment I/O**: Direct SAM2 CSV processing

### **Root Cause: Architectural Anti-Pattern**
The real issue is **function duplication and scattering**:
- Core processing functions duplicated between `run_build03.py` and `build03A_process_images.py`
- QC logic exists in build03A but isolated from SAM2 pipeline
- `run_build03.py` became a monolithic script instead of a CLI wrapper
- **Result**: Two incomplete, divergent implementations instead of one complete system

## **Implementation Strategy: Complete Consolidation (UPDATED 2025-09-12)**

### **Phase 1: Move Core Functions to build03A** 🔄 **IN PROGRESS**
**Goal**: Consolidate ALL build03 processing logic in `build03A_process_images.py`

#### **Functions to Move from run_build03.py → build03A_process_images.py**:
1. **`_ensure_predicted_stage_hpf()`** - Stage calculation logic
2. **`_collect_rows_from_sam2_csv()`** - SAM2 CSV parsing and row generation
3. **`_compute_row_geometry()`** - Geometry computation from masks
4. **`_parse_embryo_number()`**, **`_derive_mask_path()`** - Helper functions

**SKIP**: `_load_scale_map()` - Will compute pixel scale directly from SAM2 CSV data instead (eliminates Build01 dependency)

### **Phase 2: Pixel Scale Handling (NEW)**
**Goal**: Use SAM2 CSV pixel scale data directly instead of Build01 metadata
- Extract pixel scale from SAM2 CSV columns: `width_um/width_px` and `height_um/height_px`
- Fallback to computing from image dimensions if needed
- Eliminates dependency on Build01 metadata files that may not exist

### **Phase 3: Locate and Integrate Mask Processing Logic (CRITICAL)**
**Goal**: Ensure proper handling of both legacy and SAM2 masks
- **Key function**: `_load_build02_masks_for_row()` with `is_sam2_pipeline` parameter
- **Critical dual-path processing**:
  - Legacy Build02: `(np.round(arr_raw / 255 * 2) - 1).astype(np.uint8)`
  - SAM2 pipeline: `(arr_raw > 127).astype(np.uint8)`
- This logic exists in Sept 5 version and must be preserved

### **Phase 4: QC Integration Enhancement** 
**Goal**: Enhance moved functions with proven QC capabilities

#### **Integration Points**:
- **Enhance `_collect_rows_from_sam2_csv()`**: Add SAM2 QC flag parsing (✅ already fixed)
- **Enhance `_compute_row_geometry()` → `_compute_row_geometry_and_qc()`**: Add Build02 mask loading and QC computation
- **Use existing `_load_build02_masks_for_row()`**: With proper `is_sam2_pipeline` parameter
- **Integrate with proven QC functions**: `compute_qc_flags()`, `compute_fraction_alive()`, `compute_speed()`
- **Combine QC sources**: SAM2 flags + legacy flags in final `use_embryo_flag` logic

### **Phase 5: Simplify run_build03.py to Pure CLI Wrapper**
**Goal**: Make `run_build03.py` match the proven September pattern

#### **Target Architecture**:
```python
# run_build03.py - CLI wrapper only
def main():
    args = _parse_args()
    
    # Call consolidated functions in build03A
    tracked_df = segment_wells_sam2_csv(
        root=args.data_root, 
        exp_name=args.exp, 
        sam2_csv_path=auto_discover_sam2_csv(args.data_root, args.exp)
    )
    
    stats_df = compile_embryo_stats(
        root=args.data_root, 
        tracked_df=tracked_df, 
        n_workers=1
    )
    
    write_output_csv(stats_df, args.output_path)
```

## **Detailed Technical Plan**

### **Step 1: Function Migration** 🔄 **CURRENT**
**Source**: `run_build03.py` (lines ~109-547)  
**Destination**: `build03A_process_images.py`

#### **Migration Checklist**:
- [ ] Move `_ensure_predicted_stage_hpf()` with DataFrame integration
- [ ] Move `_collect_rows_from_sam2_csv()` with SAM2 QC flag parsing (preserve recent fix)
- [ ] Move `_compute_row_geometry()` as foundation for enhanced geometry+QC
- [ ] Move `_load_scale_map()` with Build01 CSV parsing logic
- [ ] Move helper functions (`_parse_embryo_number`, `_derive_mask_path`, etc.)

### **Step 2: QC Enhancement Integration**
**Goal**: Add comprehensive QC to moved functions

#### **Enhance `_compute_row_geometry()` → `_compute_row_geometry_and_qc()`**:
```python
def _compute_row_geometry_and_qc(row, root, masks_dir, scale_map, verbose=False):
    # Phase 1: Existing geometry computation (preserve)
    compute_basic_geometry(row, masks_dir, scale_map)
    
    # Phase 2: Load Build02 auxiliary masks  
    aux_masks = _load_build02_masks_for_row(root, row, target_shape)
    
    # Phase 3: Compute QC flags using proven qc_utils.py
    qc_flags = compute_comprehensive_qc_flags(row, aux_masks, px_dim_um)
    
    # Phase 4: Update row with all QC data
    row.update(qc_flags)
```

#### **Add to `compile_embryo_stats()` or create `compile_embryo_stats_sam2()`**:
```python  
def compile_embryo_stats_sam2(tracked_df):
    # Existing build03A logic for legacy QC
    # PLUS
    # Integration with SAM2-specific QC workflow
    # PLUS  
    # Final use_embryo_flag logic combining all QC sources
```

### **Step 3: Final use_embryo_flag Logic**
**Location**: After all QC computation in build03A

```python
def _set_final_use_embryo_flag(row):
    # SAM2 QC flags (from CSV)
    sam2_qc_flags = row.get("sam2_qc_flags", "")
    has_sam2_flags = sam2_qc_flags and str(sam2_qc_flags).strip() != ""
    
    # Legacy QC flags (from Build02 masks + geometry analysis)
    has_legacy_flags = any([
        row.get("dead_flag") == "true",
        row.get("frame_flag") == "true", 
        row.get("focus_flag") == "true",
        row.get("bubble_flag") == "true",
        # Only count no_yolk_flag if we actually loaded yolk data
        row.get("no_yolk_flag") == "true" and row.get("fraction_alive") != ""
    ])
    
    # Final decision: exclude if ANY QC issues detected
    use_embryo = not (has_sam2_flags or has_legacy_flags)
    row["use_embryo_flag"] = "true" if use_embryo else "false"
```

### **Step 4: CLI Wrapper Simplification**
**Target**: `run_build03.py` becomes ~50 lines instead of ~600

```python
#!/usr/bin/env python3
"""Build03 CLI wrapper - calls consolidated build03A functions"""

def main():
    args = _parse_args()
    root = Path(args.data_root)
    
    # Auto-discover SAM2 CSV
    sam2_csv = root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{args.exp}.csv"
    
    if sam2_csv.exists():
        # SAM2 pipeline path
        tracked_df = segment_wells_sam2_csv(
            root=root, exp_name=args.exp, sam2_csv_path=sam2_csv
        )
    else:
        # Legacy pipeline path  
        tracked_df = segment_wells(root=root, exp_name=args.exp)
    
    # Comprehensive QC + geometry computation
    stats_df = compile_embryo_stats_sam2(root=root, tracked_df=tracked_df)
    
    # Write output
    write_build03_csv(stats_df, args.output_path)
```

## **Benefits of Consolidation Approach**

### **Immediate Benefits**
- ✅ **Single source of truth**: All build03 logic in one place
- ✅ **No function duplication**: Eliminates maintenance burden
- ✅ **Proven QC integration**: Uses existing working implementation  
- ✅ **Clean architecture**: CLI wrapper + core processing separation

### **Long-term Benefits**  
- **Maintainable**: Changes in one place affect entire pipeline
- **Testable**: Core functions can be unit tested independently
- **Extensible**: Easy to add new QC criteria or processing steps
- **Consistent**: Same QC logic across all entry points (CLI, notebooks, etc.)

## **Risk Mitigation**

### **Migration Risks**
- **Function interface changes**: Update callers when moving functions
- **Import path updates**: Fix references to moved functions
- **Behavior preservation**: Ensure moved functions work identically

### **Mitigation Strategy**
1. **Incremental migration**: Move one function at a time with testing
2. **Interface preservation**: Keep function signatures identical during move
3. **Comprehensive testing**: Validate each moved function independently
4. **Rollback plan**: Git commits allow easy reversion if issues occur

### **Testing Strategy**
1. **Unit tests**: Test each moved function in isolation
2. **Integration tests**: Verify end-to-end pipeline still works
3. **Regression tests**: Compare outputs before/after migration  
4. **Real data validation**: Process known experiments with QC issues

## **Expected Outcomes**

### **Architecture**
- **`build03A_process_images.py`**: Complete build03 processing engine
- **`run_build03.py`**: Thin CLI wrapper (~50 lines)
- **Single QC logic**: Comprehensive, maintainable, extensible

### **Functionality**
- ✅ **SAM2 QC integration**: All SAM2 flags properly processed
- ✅ **Legacy QC restored**: Dead, yolk, focus, bubble, frame flags active
- ✅ **Geometry computation**: Enhanced with QC analysis
- ✅ **Proper use_embryo_flag**: Intelligently combines all QC sources

### **Data Quality**
- **Higher filtering standards**: Both SAM2 and legacy QC active
- **Consistent behavior**: Matches proven build03A QC results
- **Better training data**: Comprehensive quality filtering

## **Current Status**

### **Completed** ✅
- [x] **Critical SAM2 QC fix**: `use_embryo_flag` checks SAM2 flags in CSV parsing
- [x] **Architecture decision**: Consolidate functions in build03A, thin CLI wrapper
- [x] **Implementation plan**: Detailed migration and integration strategy

### **In Progress** 🔄  
- [x] **Function migration**: ✅ COMPLETED - Moved core functions to build03A
  - [x] `_ensure_predicted_stage_hpf()` - Stage calculation logic ✅
  - [x] `_collect_rows_from_sam2_csv()` - SAM2 CSV parsing (without use_embryo_flag computation) ✅
  - [x] Helper functions: `_parse_embryo_number()`, `_derive_mask_path()` ✅
  - [ ] `_compute_row_geometry()` - **IN PROGRESS**
- [ ] **QC integration**: Enhance moved functions with comprehensive QC
- [ ] **CLI simplification**: Reduce run_build03.py to pure wrapper

### **Current Status Detail (2025-09-12 Update - MAJOR PROGRESS)**

#### **Completed Functions in build03A_process_images.py:**

1. **`_ensure_predicted_stage_hpf(df, verbose=False)`** ✅ **COMPLETED**
   - Calculates Kimmel developmental stage formula
   - Added to build03A_process_images.py lines ~1202-1246

2. **`_collect_rows_from_sam2_csv(csv_path, exp, verbose=False)`** ✅ **COMPLETED**
   - Parses SAM2 CSV with QC flag detection
   - **CRITICAL**: Does NOT compute final `use_embryo_flag` (correctly deferred to `_set_final_use_embryo_flag()`)
   - Extracts pixel scale directly from CSV (eliminates Build01 dependency)
   - Added to build03A_process_images.py lines ~1263-1366
   - ✅ **CLEANED**: Removed unnecessary derive function fallbacks

3. **Helper Functions** ✅ **COMPLETED AND CLEANED**
   - `_parse_embryo_number(embryo_id)` - ✅ **KEPT** - Critical for SAM2 mask label extraction
   - ❌ **REMOVED**: `_derive_mask_path()`, `_derive_video_and_well()`, `_derive_time_int()` - Not critical, prefer explicit CSV parsing

#### **Verified Existing Critical Functions:**

4. **`_load_build02_masks_for_row(root, row, target_shape, is_sam2_pipeline=False)`** ✅ **ALREADY EXISTS AND VERIFIED**
   - **CRITICAL DUAL-PATH LOGIC VERIFIED** (lines 105-150):
     - SAM2 pipeline: `arr = (arr_raw > 127).astype(np.uint8)`  
     - Legacy Build02: `arr = (np.round(arr_raw / 255 * 2) - 1).astype(np.uint8)`
   - This is the core mask processing function that prevents the "all embryos flagged as dead" bug
   - **NO CHANGES NEEDED** - already has proper logic

#### **NEW FUNCTIONS IMPLEMENTED (2025-09-12 SESSION):**

5. **`_compute_row_geometry_and_qc(row, root, verbose=False)`** ✅ **COMPLETED**
   - **Location**: build03A_process_images.py lines ~1369-1499
   - **Functionality**: 
     - Loads SAM2 labeled mask using `resolve_sandbox_embryo_mask_from_csv()`
     - Computes geometry (area, perimeter, centroids) in pixels and microns
     - Uses pixel scale directly from SAM2 CSV data (no Build01 dependency)
     - Calls `_load_build02_masks_for_row()` with `is_sam2_pipeline=True`
     - Computes comprehensive QC using existing functions: `compute_qc_flags()`, `compute_fraction_alive()`
     - Updates row with all geometry and QC data
   - **CRITICAL**: Uses proven dual-path mask processing to avoid QC bugs

6. **`_set_final_use_embryo_flag(row)`** ✅ **COMPLETED**
   - **Location**: build03A_process_images.py lines ~1473-1496  
   - **Functionality**:
     - Combines SAM2 flags (`sam2_qc_flags`) + legacy flags (`dead_flag`, `frame_flag`, etc.)
     - Logic: `use_embryo = not (has_sam2_flags or has_legacy_flags)`
     - Excludes embryo if ANY QC issue detected from either source

7. **`compile_embryo_stats_sam2(root, tracked_df, n_workers=1)`** ✅ **COMPLETED**
   - **Location**: build03A_process_images.py lines ~1499-1559
   - **Functionality**:
     - Main integration function for SAM2 pipeline with full QC restoration
     - Processes each row with `_compute_row_geometry_and_qc()`
     - Applies `_set_final_use_embryo_flag()` after all QC computed
     - Adds predicted stage calculation at DataFrame level
     - Provides comprehensive QC statistics summary
   - **REPLACES**: The functionality lost in run_build03.py

#### **ARCHITECTURE ACHIEVEMENT:**

✅ **CONSOLIDATION COMPLETE**: All core Build03 processing logic now exists in `build03A_process_images.py`
✅ **QC RESTORATION COMPLETE**: Full integration of SAM2 + legacy QC using proven functions
✅ **SINGLE SOURCE OF TRUTH**: No function duplication between files
✅ **PROVEN MASK PROCESSING**: Uses existing `_load_build02_masks_for_row()` with proper dual-path logic

#### **Next Steps for Agent Pickup:**

**IMMEDIATE NEXT TASK:**
1. **Simplify `run_build03.py` to Pure CLI Wrapper** 🔄 **READY FOR IMPLEMENTATION**:
   - Reduce `run_build03.py` from ~600 lines to ~50 lines
   - Replace all processing logic with calls to build03A functions
   - Target architecture:
   ```python
   def main():
       args = _parse_args()
       root = Path(args.data_root)
       
       # Auto-discover SAM2 CSV
       sam2_csv = root / "sam2_pipeline_files" / "sam2_expr_files" / f"sam2_metadata_{args.exp}.csv"
       
       # Load SAM2 data using consolidated functions
       tracked_df = segment_wells_sam2_csv(root=root, exp_name=args.exp, sam2_csv_path=sam2_csv)
       
       # Process with comprehensive QC using NEW consolidated function
       stats_df = compile_embryo_stats_sam2(root=root, tracked_df=tracked_df)
       
       # Write output
       write_build03_csv(stats_df, args.output_path)
   ```

2. **Integration Testing** 🔄 **READY**:
   - Test end-to-end SAM2 pipeline with QC restoration
   - Compare QC flag distributions with Sept 5 baseline
   - Validate that SAM2 flags and legacy flags both work correctly

#### **IMPLEMENTATION STATUS SUMMARY:**

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Core functions migration | ✅ **COMPLETED** | build03A_process_images.py | All processing logic consolidated |
| QC integration | ✅ **COMPLETED** | build03A_process_images.py | Uses proven dual-path mask processing |
| Final QC decision logic | ✅ **COMPLETED** | build03A_process_images.py | Combines all QC sources |
| CLI wrapper simplification | ⏭️ **NEXT TASK** | run_build03.py | Ready for implementation |
| End-to-end testing | ⏭️ **PENDING** | N/A | After CLI wrapper complete |

### **Next Steps** ⏭️
- [ ] **End-to-end testing**: Validate migrated functionality
- [ ] **Performance verification**: Ensure no regression in processing speed
- [ ] **Documentation update**: Update usage docs for consolidated architecture

---

## **Key Architectural Principle**

**"Single Responsibility"**: 
- **`build03A_process_images.py`**: Does ALL the processing work
- **`run_build03.py`**: Handles CLI interface only  
- **Result**: Maintainable, testable, extensible build03 system

This consolidation approach eliminates the architectural debt that caused the QC functionality loss in the first place.


Of course. You can restore the lost Quality Control (QC) functionality by implementing the consolidation strategy detailed in your `refactor-013-addendum-20250911-qc-restoration.md` document. The plan is to move the essential data processing logic from the broken `run_build03.py` script into `build03A_process_images.py`, and then integrate it with the proven QC functions that are still present in the older `build03A_process_images_sept5.py` version.

This will create a single, reliable script for all Build03 processing and turn `run_build03.py` into a simple command-line wrapper, fixing the architectural issues that caused the problem.

-----

### Detailed Plan to Restore QC Functionality

Here’s a step-by-step plan to merge the necessary pipeline logic with the legacy QC functionality.

#### **Step 1: Consolidate Core Pipeline Functions into `build03A_process_images.py`**

First, move the core data processing functions from `run_build03.py` into `build03A_process_images.py`, as outlined in your plan. These functions handle reading the SAM2 CSV, calculating basic geometry, and loading the pixel scale map.

**Functions to move into `build03A_process_images.py`**:

  * `_collect_rows_from_sam2_csv()`: Parses the SAM2 CSV file.
  * `_compute_row_geometry()`: Calculates basic area, perimeter, etc.
  * `_load_scale_map()`: Integrates pixel scale data from Build01.
  * `_ensure_predicted_stage_hpf()`: Calculates the developmental stage.
  * Helper functions like `_parse_embryo_number()` and `_derive_mask_path()`.

These functions will form the foundation for a new, consolidated processing workflow.

-----

#### **Step 2: Integrate Legacy QC into the Consolidated Logic**

This is where we re-introduce the QC logic from your older Sept 5 script. The key is to enhance the new `_compute_row_geometry()` function to also load the auxiliary Build02 masks (viability, yolk, focus, bubble) and compute the legacy QC flags.

The essential function for this is **`_load_build02_masks_for_row()`**, which exists in both your current and Sept 5 Python files. We will use it within an enhanced geometry and QC function as proposed in your addendum.

```python
# In build03A_process_images.py

# This function already exists and is the key to loading legacy masks.
# We will call it from our new, enhanced geometry function.
from src.build.qc_utils import compute_comprehensive_qc_flags # Assuming this utility exists

def _load_build02_masks_for_row(root: Path, row, target_shape: tuple[int, int], is_sam2_pipeline: bool = True) -> dict:
    # ... implementation from build03A_process_images.py ...
    # This function searches for via, yolk, focus, and bubble masks.
    pass

def _compute_row_geometry_and_qc(row, root, scale_map):
    """
    Combines basic geometry calculation from run_build03.py with
    comprehensive QC from the old build03A.
    """
    # 1. Basic geometry (logic moved from run_build03.py)
    # This part calculates area_px, perimeter_px from the SAM2 mask.
    # ...
    
    # 2. Load Build02 auxiliary masks for legacy QC
    # This is the critical function from your working Sept 5 script
    aux_masks = _load_build02_masks_for_row(root, row, target_shape=<SHAPE_OF_SAM2_MASK>)
    
    # 3. Compute all QC flags using the masks
    px_dim_um = row['Height (um)'] / row['Height (px)']
    qc_flags = compute_comprehensive_qc_flags(
        sam2_mask=<LOADED_SAM2_MASK>,
        px_dim_um=px_dim_um,
        via_mask=aux_masks.get("via"),
        yolk_mask=aux_masks.get("yolk"),
        focus_mask=aux_masks.get("focus"),
        bubble_mask=aux_masks.get("bubble")
    )
    
    # 4. Update the row with all computed data
    row.update(qc_flags)
    return row
```

-----

#### **Step 3: Implement the Final Combined QC Flag**

After all SAM2 and legacy QC flags have been computed for a row, use the final decision logic from your addendum to set the `use_embryo_flag`. This logic correctly excludes an embryo if *any* QC issue is detected from either source.

This function should be applied to each row after it has been fully processed.

```python
# In build03A_process_images.py

def _set_final_use_embryo_flag(row):
    """
    Determines the final usability flag by combining SAM2 and legacy QC flags.
    """
    # Check for SAM2 QC flags from the CSV
    sam2_qc_flags = row.get("sam2_qc_flags", "")
    has_sam2_flags = sam2_qc_flags and str(sam2_qc_flags).strip() != ""
    
    # Check for any legacy QC flags computed from Build02 masks
    has_legacy_flags = any([
        row.get("dead_flag") == True,
        row.get("frame_flag") == True, 
        row.get("focus_flag") == True,
        row.get("bubble_flag") == True,
        row.get("no_yolk_flag") == True
    ])
    
    # Final decision: exclude if ANY QC issues are present
    use_embryo = not (has_sam2_flags or has_legacy_flags)
    row["use_embryo_flag"] = use_embryo
    return row
```

-----

#### **Step 4: Simplify `run_build03.py` to a CLI Wrapper**

Finally, refactor `run_build03.py` into a thin wrapper script. Its only job is to parse command-line arguments and call the main consolidated processing function in `build03A_process_images.py`. This follows the "Single Responsibility Principle" and makes the system much easier to maintain.

```python
#!/usr/bin/env python3
# run_build03.py - New, simplified CLI wrapper

from pathlib import Path
# Import the new, powerful functions from build03A
from src.build.build03A_process_images import (
    process_sam2_experiment, 
    write_output_csv
)

def main():
    args = _parse_args()
    
    # Call the single, consolidated function in build03A
    # This function will now handle CSV parsing, geometry, AND all QC.
    stats_df = process_sam2_experiment(
        root=args.data_root, 
        exp_name=args.exp
    )
    
    write_output_csv(stats_df, args.output_path)

if __name__ == "__main__":
    main()
```

By following these steps, you will successfully extract the working QC logic from the old script and integrate it into the new pipeline, resulting in a robust, maintainable, and fully functional Build03 process. 👍

## Code Pointers: Where Masks Are Processed

These are the concrete functions and file paths handling SAM2 integer-labeled masks and legacy Build02 masks today. Use this as a source-of-truth during migration and review.

- SAM2 labeled embryo masks (integer IDs)
  - `src/run_morphseq_pipeline/steps/run_build03.py`
    - `_compute_row_geometry(row, masks_dir, scale_map, verbose=False)`: computes area/perimeter/centroid from a labeled mask by selecting the embryo label parsed from `embryo_id`.
    - `_parse_embryo_number(embryo_id)`: parses `_eNN` from `embryo_id` to select the integer label in the mask.
    - `_derive_mask_path(masks_dir, image_id, hint)`: resolves mask paths like `{image_id}_masks_emnum_{N}.png` and tries common fallbacks if missing.
    - `_load_scale_map(built01_csv, ...)`: resolves well→(um/px) for geometry conversion when SAM2 per-row scale is absent.
  - `src/build/build03A_process_images.py`
    - `resolve_sandbox_embryo_mask_from_csv(root, row)`: resolves the exact exported integer-labeled mask path from the SAM2 CSV `exported_mask_path` field.
    - `segment_wells_sam2_csv(root, exp_name, sam2_csv_path)`: transforms the SAM2 per-snippet CSV into the legacy Build03 schema and sets `sam2_qc_flag`.
    - `process_masks(im_mask, im_yolk, row)` (imported via `src/functions/image_utils.py`): selects the requested embryo label (`region_label`) from an integer-labeled mask and normalizes to a clean binary embryo mask.

- Legacy Build02 auxiliary masks (binary via/yolk/focus/bubble)
  - `src/build/build03A_process_images.py`
    - `_load_build02_masks_for_row(root, row, target_shape, is_sam2_pipeline=False)`: searches under `root/segmentation/*_<model>/<date>/*{well}_t####*`, reads, thresholds/resizes to `target_shape`, and returns any present masks.
    - Downstream classic processing builds snips, estimates orientation, and sets QC flags; this code path already integrates yolk when present and gracefully degrades when absent.

Summary: SAM2 mask selection is correct and available in both the CLI step and build03A; legacy auxiliary masks are loaded exclusively in build03A via `_load_build02_masks_for_row`.

## Concrete Migration Tasks (Expanded)

Move the following helpers into `src/build/build03A_process_images.py` and route all callers there:

- From `src/run_morphseq_pipeline/steps/run_build03.py` → `src/build/build03A_process_images.py`
  - `_ensure_predicted_stage_hpf(df, verbose=False)`
  - `_collect_rows_from_sam2_csv(csv_path, exp, verbose=False)`
  - `_compute_row_geometry(row, masks_dir, scale_map, verbose=False)`
  - `_load_scale_map(built01_csv, ..., verbose=False)`
  - `_parse_embryo_number(embryo_id)` and `_derive_mask_path(masks_dir, image_id, hint)`

Enhance build03A to compute QC in one place:

- Add `_compute_row_geometry_and_qc(row, root, masks_dir, scale_map, verbose=False)`:
  - Loads the SAM2 labeled embryo mask for the current row (via `resolve_sandbox_embryo_mask_from_csv`).
  - Calls `_load_build02_masks_for_row(..., target_shape=mask.shape, is_sam2_pipeline=True)` to pull in any legacy auxiliary masks and align shapes.
  - Calls the existing legacy QC computation utilities (or inline logic where present) to compute `dead_flag`, `no_yolk_flag`, `focus_flag`, `bubble_flag`, `frame_flag`, etc.
  - Computes geometry (px and µm) using either per-row SAM2 scale or Build01 scale map.
  - Writes all flags plus geometry back into the row.

- Add `_set_final_use_embryo_flag(row)` to combine QC sources:
  - `sam2_qc_flag = len(str(row.get('sam2_qc_flags','')).strip()) > 0`
  - `legacy_qc_any = any([row.get('dead_flag')==True, row.get('frame_flag')==True, row.get('focus_flag')==True, row.get('bubble_flag')==True, row.get('no_yolk_flag')==True])`
  - `row['use_embryo_flag'] = 'false' if (sam2_qc_flag or legacy_qc_any) else 'true'`

- Create `compile_embryo_stats_sam2(root, tracked_df, n_workers=1)`
  - Mirrors the legacy `compile_embryo_stats` pipeline but uses the SAM2-bridge `tracked_df` and the enhanced `_compute_row_geometry_and_qc` per-row.
  - Ensures final `use_embryo_flag` is set after all QC sources are computed.

CLI changes:

- Refactor `src/run_morphseq_pipeline/steps/run_build03.py` to a thin wrapper:
  - Input discovery (SAM2 CSV, exported masks dir, optional Build01 CSV).
  - Calls `segment_wells_sam2_csv` then `compile_embryo_stats_sam2` inside build03A.
  - Writes `metadata/build03_output/expr_embryo_metadata_{exp}.csv`.

## Final QC Decision Logic (Explicit)

- Inputs considered:
  - `sam2_qc_flags` column (CSV): any non-empty string → `sam2_qc_flag=True`.
  - Legacy flags computed from masks and snips: `dead_flag`, `no_yolk_flag`, `focus_flag`, `bubble_flag`, `frame_flag`.

- Decision:
  - `use_embryo_flag = 'true'` only if `sam2_qc_flag == False` AND none of the legacy flags are true.
  - Otherwise `use_embryo_flag = 'false'`.

- Notes:
  - When yolk masks are unavailable for a given row, compute other flags and do not forcibly set `no_yolk_flag`; missing data should not over-penalize.
  - Preserve existing behavior for out-of-frame detection (frame flag) using mask vs. snip area.

## Schema Mapping: SAM2 → Legacy (Reference)

- `xpos`, `ypos`: compute from `bbox_x_min/max`, `bbox_y_min/max` center.
- `region_label`: parse from `embryo_id` suffix `_eNN`.
- `experiment_date`: copy from `exp_name`.
- `well_id`/`well`: extract from `video_id` suffix `_[A-H]\d\d`.
- `time_int`: from `frame_index`.
- `predicted_stage_hpf`: `start_age_hpf + (Time Rel (s)/3600) * (0.055*temperature - 0.57)`.

## Testing Checklist (Actionable)

- Unit
  - `_collect_rows_from_sam2_csv`: parses QC flags, time_int/snip_id, and provenance fields.
  - `_compute_row_geometry`: correct label selection, µm conversion with both SAM2 per-row scale and Build01 fallback.
  - `_load_build02_masks_for_row`: handles missing dirs, resizes correctly, thresholds SAM2-style masks when `is_sam2_pipeline=True`.
  - `_set_final_use_embryo_flag`: truth table across SAM2/legacy flags.

- Integration
  - `segment_wells_sam2_csv` → `compile_embryo_stats_sam2`: end-to-end produces expected columns and non-empty geometry where inputs present.
  - Compare `use_embryo_flag` distribution against Sept 5 baseline for a known experiment.
  - Spot-check rows with `MASK_ON_EDGE` and legacy `focus_flag` to confirm exclusion.

- Data validation
  - Process one experiment with known QC issues; verify flagged counts match expectations from legacy build03A.
  - Ensure missing yolk masks do not spuriously set `no_yolk_flag`.

## Acceptance Criteria

- Single source of truth: all core Build03 logic resides in `src/build/build03A_process_images.py`.
- `run_build03.py` reduced to CLI wrapper which calls build03A for processing and writing.
- `use_embryo_flag` reflects combined QC sources; no embryo with SAM2 QC flags is marked usable.
- Backward compatibility: legacy-only path (`segment_wells`) remains available but is not the default.
- Documentation updated with code pointers and decision logic (this addendum).

## Risks and Mitigations (Focused)

- Function signature drift during migration → Keep signatures stable; update imports in one pass; add unit tests before/after.
- Ambiguity in pixel-scale columns in Build01 CSV → `_load_scale_map` heuristic plus CLI overrides (`--px-size-col`, `--px-size-x-col`, `--px-size-y-col`).
- Missing yolk masks in SAM2-only runs → Treat as absent; compute remaining QC; do not set `no_yolk_flag` unless yolk data is reliable.
- Performance regressions → Use per-experiment CSV (no heavy image ops); only read masks when `--compute-geometry` or when needed for QC.

## Open Questions (for Claude/Implementer)

- Should we preserve both `compile_embryo_stats` and a new `compile_embryo_stats_sam2`, or unify behind a single function with a toggle?
- Do we have a central utility for computing the legacy QC flags, or should we consolidate scattered computations into a new `qc_utils.py` under `src/build`?
- Where do we persist per-embryo per-snip QC provenance (e.g., which specific flags fired) beyond the combined `use_embryo_flag`?

## Implementation Checkpoints (Suggested)

1) Land code-pointer updates and migrate helpers to build03A.
2) Add `_compute_row_geometry_and_qc` and final `use_embryo_flag` logic.
3) Simplify `run_build03.py` to wrapper and wire to build03A.
4) Run integration tests on a known experiment; compare QC counts with Sept 5 baseline.
5) Update docs and usage examples; mark old paths deprecated.

Of course. Here is a summary of the significant effort and the precise technical solution that was implemented to correctly integrate legacy and SAM2 masks, based on the provided documents.

A critical effort was undertaken to resolve a subtle but severe bug in the hybrid pipeline where legacy QC masks were being processed incorrectly, leading to catastrophic data quality issues like all healthy embryos being flagged as dead. The final, robust solution involved explicitly distinguishing between mask types and applying the correct, nuanced processing logic to each.

### The Solution: Explicitly Distinguishing Mask Types

The core of the fix was recognizing that auxiliary masks from the legacy Build02 pipeline required a different processing formula than newer masks. To solve this, the `_load_build02_masks_for_row` function was enhanced to accept an `is_sam2_pipeline` boolean parameter. This flag allows the function to intelligently switch between two different processing paths, ensuring each mask type is handled correctly.

This updated logic is detailed within the `build03A_process_images_sept5.py` file.

### The Corrected Processing Logic

Based on the `is_sam2_pipeline` flag, the function now applies one of two distinct formulas:

1.  **For Legacy Masks (`is_sam2_pipeline=False`):**
    The function now uses the carefully restored legacy processing formula:
    `arr = (np.round(arr_raw / 255 * 2) - 1).astype(np.uint8)`
    This is followed by a final conversion to a `> 0` binary format. This multi-step calculation precisely matches the original, validated logic from the `diffusion-dev` pipeline, ensuring that grayscale values from the legacy UNet models are converted to accurate binary masks.

2.  **For SAM2-era Masks (`is_sam2_pipeline=True`):**
    For newer auxiliary masks that are expected to be clean binary images, a simpler and more direct thresholding is used:
    `arr = (arr_raw > 127).astype(np.uint8)`

This dual-path approach, controlled by an explicit parameter, was the crucial fix that stabilized the pipeline. It ensures that the superior SAM2 embryo segmentations can be reliably combined with the critical, but uniquely formatted, legacy QC masks, preserving the integrity of the entire "best-of-both-worlds" system.

---

## QC Restoration Verification (2025-09-12)

We validated the restored QC on `20250622_chem_28C_T00_1425`.

- Inputs
  - Root: `morphseq_playground`
  - CSV: `sam2_pipeline_files/sam2_expr_files/sam2_metadata_20250622_chem_28C_T00_1425.csv`
  - Masks: `sam2_pipeline_files/exported_masks/20250622_chem_28C_T00_1425/masks/`

- Results
  - Total embryos: 80
  - SAM2 QC flagged: 1 (MASK_ON_EDGE)
  - Legacy QC flagged: 4 (3 frame, 1 focus)
  - Final usable: 76 (95.0%)

- Fix applied during validation
  - SAM2 QC flag interpretation hardened to treat pandas NaN/None/"nan"/"null"/"none"/whitespace as empty.
  - Summary counting uses the same robust predicate to avoid false positives.

Conclusion: QC is fully restored and behaves as expected.

## Pipeline Integration Checklist

- Simplify `src/run_morphseq_pipeline/steps/run_build03.py` to a wrapper that:
  - Locates per-experiment SAM2 CSV
  - Calls `segment_wells_sam2_csv` → `compile_embryo_stats_sam2`
  - Writes `metadata/build03_output/expr_embryo_metadata_{exp}.csv`
- Ensure higher-level pipeline invokes this wrapper (no internal logic here)
- Add a smoke test targeting 10 snips from a known experiment
