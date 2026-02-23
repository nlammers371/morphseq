# ND2 Position Mapping - Critical Issues Found

## Problem Summary

The ND2 position verification script successfully runs but reveals a **fundamental mismatch** between well labels and actual stage positions:

- **20250711 experiment**: Grid verification FAILS - well labels don't match clustered stage positions
- **20251106 experiment**: Grid verification FAILS - same issue

## Key Observations

### Stage Positions ARE Properly Ordered
- X coordinates form 12 distinct clusters (columns): 48694 → -50321 µm (spacing ~9000 µm)
- Y coordinates form 8 distinct clusters (rows): -31260 → 31723 µm (spacing ~9000 µm)
- **Clustering shows a perfect regular grid pattern**

### BUT Well Labels Don't Match Clusters
- The `series_number_map` from Excel is being used to map series numbers (1-96) to well names
- When applied, these well names **do NOT align** with the KMeans clusters
- This suggests the series_number_map mapping is either:
  1. **Incorrectly formatted in the Excel file**
  2. **Being parsed incorrectly by the script**
  3. **Fundamentally mismatched with how ND2 stores positions**

## Root Cause Candidates

1. **Series numbering mismatch**: The Excel series_number_map might be using a different ordering than the ND2 file
   - Excel might be column-major while ND2 is row-major (or vice versa)
   - Series numbers might have gaps (deselected wells) that aren't accounted for

2. **Row/column swap**: The ND2 file might have rows and columns swapped relative to the plate map

3. **Transpose/transpose+rotation**: The position grid might be transposed or rotated relative to the expected orientation

## Next Steps

### DO NOT use this mapping yet
The current `load_series_number_map()` implementation follows `build01B_compile_yx1_images_torch.py` exactly, but the verification shows it's not working.

### Need to investigate:
1. **What does build01B actually do with this mapping?**
   - It passes the QC check `_qc_well_assignments()` - but HOW?
   - Does build01B's _qc_well_assignments also fail, or does it pass?

2. **Verify the Excel file format**
   - Is the series_number_map actually row-major or column-major?
   - Are there deselected wells (NaN values) that cause indexing shifts?
   - Check actual Excel file format more carefully

3. **Check if ND2 file's series ordering matches expectations**
   - Extract P indices and their corresponding series numbers from ND2 metadata more carefully
   - Verify if P index truly maps 1:1 to series numbers or if there's a transformation

4. **Reverse-engineer the correct mapping**
   - Since we know the stage positions form a perfect grid
   - Use clustering to determine actual well positions
   - Build reverse mapping from stage (x,y) → well name
   - Compare with Excel's expected mapping to find the transformation

## Files Modified

- `verify_nd2_positions.py`: Implements clustering-based verification (mimics build01B's _qc_well_assignments)
  - ✓ Correctly detects that well labels don't match stage positions
  - ✓ Shows stage positions ARE properly gridded
  - ✗ Doesn't yet resolve why the mapping fails

## Critical Code Location

`build01B_compile_yx1_images_torch.py:66-84` - The `_qc_well_assignments()` function that validates well assignments

This function:
1. Clusters stage positions into row/column groups
2. Checks if well name labels match the clusters
3. **Asserts** the match (would fail if mismatch)

**Need to verify**: Does this function actually PASS in build01B when run on 20250711/20251106?
