# Refactor-012: Experiment Manager Bash Wrapper Script

**Created**: 2025-09-07  
**Status**: Planning  
**Depends On**: Refactor-011 SAM2 CLI Integration

## **Executive Summary**

Create a simple bash wrapper script `run_experiment_manager.sh` that provides an easy interface to process experiments through the complete MorphSeq pipeline using the intelligent ExperimentManager orchestration system. This eliminates the need for users to understand Python imports or pipeline internals.

**Key Goals:**
- Single bash script interface for complete pipeline processing
- Support for processing "all" experiments or specific experiment lists
- Intelligent pipeline orchestration using existing ExperimentManager
- Simple parameter interface: data_root, repo_root, experiments
- Future extensibility for overwrite modes and advanced options

## **Background & Problem**

**Current State:**
- ExperimentManager provides intelligent pipeline orchestration in Python
- Users need to write Python scripts or use complex CLI commands
- No simple "process everything" interface for batch operations
- CLI e2e command is per-experiment, not global batch processing

**User Need:**
Users want a simple command like:
```bash
./run_experiment_manager.sh \
  --data-root /path/to/data \
  --repo-root /path/to/repo \
  --experiments "all"
```

## **Scope (This Refactor)**

### **In Scope**
1. **Bash Script Creation**: Simple wrapper script with clear parameter interface
2. **ExperimentManager Integration**: Use existing intelligent orchestration
3. **Experiment Selection**: Support "all" or comma-separated experiment lists
4. **Error Handling**: Basic validation and error reporting
5. **Documentation**: Clear usage examples and parameter descriptions
6. **Conda Environment**: Automatic environment activation

### **Out of Scope (Future Enhancements)**
- Advanced overwrite modes (--overwrite, --force, etc.)
- Selective pipeline step execution (--only-build03, etc.)
- Parallel experiment processing
- Advanced logging and progress reporting
- Configuration file support

## **Implementation Plan**

### **Script Location & Structure**
```
src/run_morphseq_pipeline/run_experiment_manager.py      # Python wrapper CLI
src/run_morphseq_pipeline/run_experiment_manager_qsub.sh # SGE/qsub GPU wrapper
```

### **Parameter Interface**
```bash
./run_experiment_manager.sh \
  --data-root PATH          # Required: MorphSeq data directory
  --repo-root PATH          # Required: MorphSeq repository root  
  --experiments "all"|LIST  # Required: "all" or "exp1,exp2,exp3"
  [--help]                  # Optional: Show usage information
```

### **Core Implementation Logic**

#### **1. Parameter Validation**
```bash
# Required parameters
data_root=""
repo_root="" 
experiments=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --repo-root)
      repo_root="$2"
      shift 2
      ;;
    --experiments)
      experiments="$2"
      shift 2
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

# Validate required parameters
if [[ -z "$data_root" || -z "$repo_root" || -z "$experiments" ]]; then
  echo "Error: Missing required parameters"
  show_usage
  exit 1
fi
```

#### **2. Environment Setup**
```bash
# Validate paths exist
if [[ ! -d "$data_root" ]]; then
  echo "Error: Data root directory does not exist: $data_root"
  exit 1
fi

if [[ ! -d "$repo_root" ]]; then
  echo "Error: Repo root directory does not exist: $repo_root"
  exit 1
fi

# Activate conda environment 
source $(conda info --base)/etc/profile.d/conda.sh
conda activate segmentation_grounded_sam || {
  echo "Error: Failed to activate segmentation_grounded_sam environment"
  exit 1
}

# Set working directory
cd "$repo_root" || {
  echo "Error: Failed to change to repo root: $repo_root"
  exit 1
}
```

#### **3. Python Orchestration**
```bash
# Create temporary Python script for execution
python_script=$(mktemp /tmp/morphseq_process_XXXXXX.py)
cat > "$python_script" << EOF
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, "$repo_root")

from src.build.pipeline_objects import ExperimentManager

def main():
    try:
        # Initialize ExperimentManager
        manager = ExperimentManager(root="$data_root")
        print(f"🔍 Discovered {len(manager.experiments)} experiments")
        
        # Determine target experiments
        experiments_arg = "$experiments"
        if experiments_arg.lower() == "all":
            target_experiments = list(manager.experiments.keys())
            print(f"📋 Processing ALL {len(target_experiments)} experiments")
        else:
            target_experiments = [exp.strip() for exp in experiments_arg.split(",")]
            print(f"📋 Processing {len(target_experiments)} specific experiments: {target_experiments}")
            
            # Validate experiments exist
            missing = [exp for exp in target_experiments if exp not in manager.experiments]
            if missing:
                print(f"❌ Error: Experiments not found: {missing}")
                return 1
        
        # Process experiments through pipeline
        process_experiments_through_pipeline(manager, target_experiments)
        
        print("✅ Pipeline processing complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def process_experiments_through_pipeline(manager, target_experiments):
    """Process experiments through complete MorphSeq pipeline"""
    
    print("🚀 Starting complete pipeline processing...")
    
    # Stage 1: Per-experiment basics (only process what's needed)
    print("📦 Stage 1: Basic per-experiment processing...")
    for exp_name in target_experiments:
        exp = manager.experiments[exp_name]
        print(f"  Processing {exp_name}...")
        
        # Export/build raw images if needed
        if not exp.flags.get('ff', False):
            print(f"    📸 Exporting images...")
            exp.export_images()
            
        # Stitch images if needed (Keyence only)
        if exp.needs_stitch and exp.microscope == "Keyence":
            print(f"    🧩 Stitching images...")
            exp.stitch_images()
            
        # Generate QC masks if needed (5 UNet models)
        if exp.needs_segment:
            print(f"    🎯 Generating QC masks...")
            exp.segment_images()
    
    # Stage 2: Advanced per-experiment processing
    print("🔬 Stage 2: Advanced per-experiment processing...")
    for exp_name in target_experiments:
        exp = manager.experiments[exp_name]
        print(f"  Processing {exp_name}...")
        
        # SAM2 segmentation if needed
        if exp.needs_sam2:
            print(f"    🤖 Running SAM2 segmentation...")
            exp.run_sam2()
            
        # Build03 embryo processing if needed
        if exp.needs_build03:
            print(f"    🧬 Running Build03 embryo processing...")
            exp.run_build03()
            
        # Generate latent embeddings if needed
        if not exp.has_latents():
            print(f"    🧠 Generating latent embeddings...")
            exp.generate_latents()
    
    # Stage 3: Global operations
    print("🌍 Stage 3: Global pipeline operations...")
    
    # Build04: df01 -> df02 (global QC)
    if manager.needs_build04:
        print("  📊 Running Build04 (global QC)...")
        manager.run_build04()
    else:
        print("  ✅ Build04 already complete")
        
    # Build06: df02 + latents -> df03 (final merge)
    if manager.needs_build06:
        print("  🔗 Running Build06 (final merge)...")
        manager.run_build06()
    else:
        print("  ✅ Build06 already complete")

if __name__ == "__main__":
    sys.exit(main())
EOF

# Execute the Python script
python "$python_script"
exit_code=$?

# Clean up
rm -f "$python_script"
exit $exit_code
```

#### **4. Usage Documentation**
```bash
show_usage() {
  cat << EOF
MorphSeq Experiment Manager - Complete Pipeline Processing

USAGE:
  $0 --data-root PATH --repo-root PATH --experiments EXPERIMENTS

PARAMETERS:
  --data-root PATH     Path to MorphSeq data directory (required)
  --repo-root PATH     Path to MorphSeq repository root (required) 
  --experiments LIST   Experiments to process: "all" or "exp1,exp2,exp3" (required)
  --help              Show this help message

EXAMPLES:
  # Process all experiments
  $0 --data-root /path/to/morphseq/data --repo-root /path/to/morphseq/repo --experiments "all"
  
  # Process specific experiments
  $0 --data-root morphseq_playground --repo-root . --experiments "20250529_30hpf_ctrl_atf6,20240418"

PIPELINE STAGES:
  1. Per-experiment: Raw data -> FF images -> QC masks -> SAM2 segmentation
  2. Per-experiment: Build03 embryo processing -> Latent embeddings  
  3. Global: Build04 QC -> Build06 final merge

The script uses ExperimentManager for intelligent processing - only runs steps that are actually needed.

EOF
}
```

## **Usage Examples**

### **Process All Experiments**
```bash
qsub -v REPO_ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq,DATA_ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground,EXPERIMENTS="all" \\
  src/run_morphseq_pipeline/run_experiment_manager_qsub.sh
```

### **Process Specific Experiments**
```bash
qsub -v REPO_ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq,DATA_ROOT=/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground,EXPERIMENTS="20250529_30hpf_ctrl_atf6,20240418,20250703_chem3_34C_T01_1457" \\
  src/run_morphseq_pipeline/run_experiment_manager_qsub.sh
```

### **Get Help**
```bash
python src/run_morphseq_pipeline/run_experiment_manager.py --help
```

## **Technical Benefits**

### **Simplicity**
- Single command interface for complete pipeline
- No need to understand Python imports or ExperimentManager API
- Clear parameter validation and error messages

### **Intelligence**
- Uses ExperimentManager's intelligent orchestration
- Only runs pipeline steps that are actually needed
- Automatic experiment discovery and validation
- State tracking prevents duplicate work

### **Robustness**  
- Environment validation and activation
- Path existence checks
- Comprehensive error handling
- Clean temporary file management

### **Extensibility**
- Easy to add new parameters (--overwrite, --only-steps, etc.)
- Modular Python orchestration logic
- Clean separation between bash wrapper and Python logic

## **Future Enhancement Opportunities**

### **Advanced Options** (Future Refactors)
```bash
# Selective processing
--only-steps "build03,sam2"           # Run only specific pipeline steps
--skip-steps "sam2"                   # Skip specific steps

# Overwrite controls  
--overwrite                           # Force reprocess all steps
--force-build03                       # Force only Build03 reprocessing

# Performance options
--parallel-experiments N              # Process N experiments in parallel
--max-workers N                       # Limit CPU workers per experiment
```

### **Advanced Features**
- Configuration file support (`--config morphseq.yaml`)
- Logging and progress reporting (`--log-file pipeline.log`)
- Dry-run mode (`--dry-run`)
- Resume from failure (`--resume-from STEP`)

## **Testing Strategy**

### **Basic Functionality**
```bash
# Test parameter validation
./run_experiment_manager.sh --help
./run_experiment_manager.sh --invalid-param
./run_experiment_manager.sh --data-root /nonexistent

# Test experiment selection
./run_experiment_manager.sh --data-root test_data --repo-root . --experiments "all"
./run_experiment_manager.sh --data-root test_data --repo-root . --experiments "exp1,exp2"
./run_experiment_manager.sh --data-root test_data --repo-root . --experiments "nonexistent"
```

### **Pipeline Integration**
- Test with clean data directory (full pipeline)
- Test with partially processed experiments (selective processing)
- Test with fully processed experiments (no-op verification)
- Test error recovery (failed step doesn't break entire pipeline)

## **Implementation Timeline**

**Estimated Time**: 2 hours

- **30 minutes**: Bash script structure and parameter parsing
- **45 minutes**: Python orchestration logic and ExperimentManager integration  
- **30 minutes**: Error handling, validation, and usage documentation
- **15 minutes**: Testing with real experiments

## **Acceptance Criteria**

### **Functional Requirements**
- [ ] Bash script accepts --data-root, --repo-root, --experiments parameters
- [ ] "all" processes all discovered experiments
- [ ] Comma-separated list processes specific experiments  
- [ ] Script validates parameters and paths exist
- [ ] Uses ExperimentManager for intelligent pipeline orchestration
- [ ] Only runs pipeline steps that are actually needed
- [ ] Provides clear progress messages and error reporting
- [ ] Activates correct conda environment automatically

### **Quality Requirements**
- [ ] Script is executable and has proper shebang
- [ ] Clear usage documentation with --help flag
- [ ] Robust error handling with meaningful messages
- [ ] Clean temporary file management
- [ ] Works from any directory when given absolute paths

### **User Experience**
- [ ] Simple, memorable command interface
- [ ] Clear progress indication during processing
- [ ] Actionable error messages when problems occur
- [ ] No need to understand Python or ExperimentManager internals

---

**Status**: Ready for implementation
**Next Steps**: Create `src/build/run_experiment_manager.sh` with the planned interface and logic

---

## ✅ Implementation Update – Build01 Resume Filtering (2025-09-07)

To support safe resume after interruptions and avoid recomputing expensive FF frames, we implemented per-frame filtering before FF computation. This change affects both Keyence and YX1 Build01 paths and is enabled by default via ExperimentManager.

### Changes
- Keyence FF (src/build/build01A_compile_keyence_torch.py)
  - After raw frame discovery (`sample_list`), when `overwrite=False` and not `metadata_only`, we now skip any frame whose stitched FF output already exists at:
    - `<data_root>/built_image_data/stitched_FF_images/<exp>/<well>_t####_stitch.jpg`
  - This prevents recomputation of FF projections for already-stitched frames.

- YX1 FF (src/build/build01B_compile_yx1_images_torch.py)
  - Before calling the per-frame FF writer, when `overwrite=False`, we filter indices to only frames whose stitched FF output is missing:
    - `<data_root>/built_image_data/stitched_FF_images/<exp>/<well>_t####_ch{BF}_stitch.jpg`
  - This bypasses costly ND2 reads and FF compute for frames already processed.

- ExperimentManager defaults (src/build/pipeline_objects.py)
  - `export_images()` now calls `build_ff_from_keyence(..., overwrite=False)` and `build_ff_from_yx1(..., overwrite=False)` to enable resume-by-skip behavior.
  - `stitch_images()` now calls `stitch_ff_from_keyence(..., overwrite=False)` so existing stitched frames are not recomputed.

### Rationale
- FF calculation is long-running; skipping frames with final stitched outputs significantly reduces rerun time after an interruption.
- We check for the final stitched artifact rather than intermediate tile outputs to align with downstream dependencies.

### Behavior
- First run: processes all frames as before.
- Resume (default): skips any frame already present as a stitched FF image; only missing frames are computed.
- Force recompute: can be re-enabled by passing `overwrite=True` directly to Build01 functions if needed (not used by default in ExperimentManager).

### Validation
- Verified that filtered lists log a summary (e.g., “Resuming: skipping N stitched frames for <exp>”) and that remaining frames are processed normally.
- No changes to metadata CSV generation; `built_metadata_files/<exp>_metadata.csv` still writes as before.

---

## ✅ Implementation Update – YX1 Subset Wells + BF Channel Handling (2025-09-07)

Motivation: YX1 runs may include ND2 files with more wells than intended; users specify the desired wells via `series_number_map` in `<exp>_well_metadata.xlsx`. Also, BF channel names can vary (e.g., `EYES - Dia`).

### Changes
- Process only selected wells from `series_number_map`:
  - Build a mapping ND2 well index → well name using only populated entries in `series_number_map`.
  - Filter per-frame processing to those wells only (blanks are skipped).
  - Construct `well_df` for selected wells; set `time_int` and `Time (s)` by repeating per-time vectors across selected wells, avoiding length mismatches.

- Robust BF channel selection for YX1:
  - Use env override if set: `YX1_BF_CHANNEL_INDEX` (0-based index).
  - Else accept exact `BF` (case-insensitive), else accept known single-channel label `EYES - Dia`.
  - If only one channel, use index 0.
  - Emit loud warnings when using non-standard names (e.g., `EYES - Dia`) or when `BF` is absent, and instruct how to override.

### Affected Files
- `src/build/build01B_compile_yx1_images_torch.py`

### Behavior
- Subset by `series_number_map` blanks is honored; no more index/length errors when skipping wells.
- Channel naming differences no longer cause `"'BF' is not in list"`; a clear warning is logged instead (or users can set the env var).

---

## ✅ Implementation Update – YX1 Index Error Fix + Enhanced Debugging (2025-09-08)

Fixed critical "list index out of range" error in YX1 processing that occurred when the ND2 file reported more wells than were actually mapped in the Excel metadata.

### Root Cause
- ND2 file reported `n_w=94` wells but Excel `series_number_map` only mapped 60 wells
- Code was trying to extract metadata for all 94 wells, causing index errors when accessing non-existent frames
- The error occurred in `nd.frame_metadata(slice_ind)` when `slice_ind` exceeded available frames

### Key Fixes Applied

#### 1. **Fixed Array Sizing & Processing Logic**
- Changed from processing all ND2 wells (`n_w=94`) to only processing mapped wells from Excel (60 wells)
- Modified metadata extraction loop to iterate through `well_ind_list` (mapped wells) instead of `range(n_w)` (all ND2 wells)
- Added proper conversion from 1-based Excel series numbers to 0-based ND2 indices
- Array sizes now match actual data: `60 × 190 = 11,400` entries instead of `94 × 190 = 17,860`

#### 2. **Enhanced Error Handling & Recovery**
- Wrapped ND2 metadata access in try/catch blocks to handle missing frames gracefully
- Added array trimming logic if some frames are missing from the ND2 file
- Continues processing even if individual frames fail, rather than crashing entirely

#### 3. **Improved Overwrite Logic**
- Completely redesigned the resume/overwrite filtering system to work with subset wells
- Replaced complex array lookup approach with direct `sampled_wells_info` structure
- New approach stores `(well_idx, well_name, well_series, nd2_well_idx, t)` for each frame
- Overwrite filtering now works correctly with `MSEQ_OVERWRITE_BUILD01="1"` or `"0"`

#### 4. **Better Debugging & User Experience**
- Always show full Python tracebacks in experiment manager (removed `MSEQ_TRACE` requirement)
- Added informative output: `"Subsetting to 60 wells from metadata"`
- Show sample lookup entries (every 8th): `{0: 'A01', 8: 'B01', 16: 'C01', ..., 59: 'H08'}`
- Clear error messages when ND2 frames are missing

### Affected Files
- `src/build/build01B_compile_yx1_images_torch.py` - Core processing logic fixes
- `src/run_morphseq_pipeline/run_experiment_manager.py` - Always show full tracebacks

### Behavior Changes
- **Before**: Crashed with "list index out of range" when ND2 had more wells than Excel mapping
- **After**: Processes only the wells mapped in Excel, handles missing frames gracefully
- **Resume Logic**: Now works correctly with subset wells for both `overwrite=True/False`
- **Debugging**: Always shows full stack traces and clear progress information

### Technical Details
The fix addresses a fundamental mismatch between:
- **ND2 file structure**: Reports total wells available in the file
- **Excel metadata**: Maps only the wells actually used in the experiment
- **Processing logic**: Now correctly processes only the intersection of both

This ensures robust processing regardless of how many extra wells the ND2 file might contain.

### Logic Consistency Fixes (2025-09-08)
After implementation, identified and fixed critical inconsistencies in the subsetting logic:

1. **Sorted/Unsorted Mismatch**: Fixed `well_name_lookup` construction to use both sorted lists consistently
   - **Before**: `{ind-1: name for name, ind in zip(well_name_list_sorted, well_ind_list)}` (mixed sorted/unsorted)
   - **After**: `{ind-1: name for name, ind in zip(well_name_list_sorted, well_ind_list_sorted)}` (both sorted)

2. **Processing Order Consistency**: All loops now use sorted versions of both well names and indices
   - Metadata extraction loop: `zip(well_name_list_sorted, well_ind_list_sorted)`
   - Frame processing loop: `zip(well_name_list_sorted, well_ind_list_sorted)`
   - Ensures consistent ordering throughout the pipeline

3. **User-Friendly Debug Output**: Convert internal 0-based indices to 1-based Excel series numbers in debug prints
   - **Display**: `{1: 'A01', 9: 'B01', 17: 'C01', ...}` (matches Excel numbering)
   - **Internal**: Still uses 0-based indexing for array access (no functional change)

These fixes ensure that:
- Well name lookups map to correct wells (no more misaligned mappings)
- All processing uses consistent well ordering
- Debug output matches user expectations (Excel 1-based series numbers)
- Array access remains correct with 0-based ND2 indexing

---

## ✅ Implementation Update – Experiment Manager Wrappers + Array Jobs (2025-09-07)

### Moved Wrappers
- Python CLI wrapper → `src/run_morphseq_pipeline/run_experiment_manager.py`
- SGE/qsub GPU wrapper → `src/run_morphseq_pipeline/run_experiment_manager_qsub.sh`

### qsub Wrapper Enhancements
- Supports SGE array jobs: one experiment per task via `SGE_TASK_ID`.
  - Provide experiments via `EXP_FILE` (one per line) or `EXP_LIST` (comma-separated), else falls back to `EXPERIMENTS`.
- Logs JOB_ID and TASK at start; with `-o logs -e logs -j y`, tail per-task logs easily:
  - `tail -f logs/morphseq_experiment_mngr.o<JOB_ID>.<SGE_TASK_ID>`
- Uses `CONDA_SOLVER=classic` to avoid libmamba issues during activation.

### Array Job Example
```
EXP_FILE=src/run_morphseq_pipeline/run_experiment_list/20250905_list.txt
N=$(wc -l < "$EXP_FILE")
qsub -t 1-$N -tc 3 \
  -v EXP_FILE="$EXP_FILE" \
  src/run_morphseq_pipeline/run_experiment_manager_qsub.sh
```

### Notes
- Defaults (REPO_ROOT/DATA_ROOT/EXPERIMENTS) are set in the script; edit them once or pass overrides via `-v` if we switch to fallback assignments.

---

## 🪲 Observed Issue – YX1 Ragged Time Coverage & Missing Per‑Frame Metadata (2025-09-07)

### Symptom
- ND2 header reports a global `T` time count (e.g., `T=190`, `P=94`), but many wells have fewer frames with metadata (e.g., only ~40). Calls to `frame_metadata((t*P+w)*Z)` start failing at `t≈40` for those wells, even though the image planes for `t≥40` are still readable.
- Example probe results:
  - For well A01 (ND2 index 0), `frame_metadata` fails for `t≥40`, but reading `arr[t, w, z, :, :]` via Dask succeeds for those frames.

### Root Cause
- Nikon ND2 files can be “ragged”: per‑position (well) time coverage varies. The header `T` is a global maximum, not a per‑well guarantee. Per‑frame metadata (e.g., stage position, relative time) may be missing for valid image frames at the tail of a well.

### Impact in Build01 (original behavior)
- Any logic that assumes a rectangular grid (`selected_wells × header_T`) and uses `frame_metadata` to test existence will raise `IndexError` past the well’s true end. We saw errors like “list index out of range” or “IndexError” in the stage‑QC loop.

### Current Mitigations in Code
- YX1 series map parsing hardened (skip header row, duplicates, out‑of‑range); only selected wells are processed.
- Stage‑QC (position) wrapped to skip on errors and length mismatches; won’t halt FF.
- FF resume filtering in place to avoid recompute of already-stitched frames.
- BF channel selection simplified with loud warnings (supports `EYES - Dia`).

### Proposed Robust Solutions (to decide)
1) Treat stage‑QC as optional for YX1
   - Skip `frame_metadata` entirely; process images via Dask only.
   - Pros: simplest, avoids errors, no ND2 metadata dependency.
   - Cons: lose stage position/timestamp in Build01 metadata for YX1.

2) Per‑well time coverage (preferred, if metadata desired)
   - For each selected ND2 well `w`, probe coverage across `t=0..T-1`:
     - If `frame_metadata` exists, record (w,t) with metadata.
     - If missing, attempt a cheap image read; if it succeeds, record (w,t) with NaN metadata; if image read fails, stop for that well.
   - Build FF indices from the recorded (w,t) only; construct `well_df` by concatenating per‑well vectors (length = frames_available for that well).
   - Stage‑QC can be run on rows with available metadata or skipped.
   - Pros: processes all real frames without errors, preserves metadata where available.
   - Cons: slightly more complex; `well_df` length becomes `sum(T_well_i)` instead of `selected_wells × T`.

### Notebook QC Helpers (for operators)
- We included notebook cells to:
  - Print ND2 header sizes/channels.
  - Map plate wells → ND2 indices from `series_number_map` (skip blanks/dupes/OOR).
  - Probe per‑well coverage using `frame_metadata` and, if missing, test image readability.
  - Read one Z‑max frame to confirm image data exists at the tail.

### Decision Pending
- Whether to: (A) skip stage‑QC for YX1 entirely, or (B) implement per‑well coverage in Build01 to keep metadata where available. The current code gracefully skips stage‑QC on errors and proceeds; this section documents the trade‑offs for a cleaner future patch.



SO the issue is that 20250711 doesnt have metadta but it still has functional images afte frame index 40. the most important field is Time rel (s) and what we ned to do is somehow find a way to pass through so that the built_metadta file obtains this. I think we can still process the file but when we are building the built metadta. What we can do is get the averaege value betwen videos for a given time point. and then add this to the metadta file. However we need to do it in a time point loop and within thus loop a well loop so that we can get the average time rel (s) for each time point across all wells which is mirroriing the way the micorsocpe goes through the data.  

so i invesitated this issue by using this code 
import numpy as np
import dask.array as da
import nd2

# Choose a well to probe
# Example: A01 → ND2 index 0
w_test = 0
# Or dynamically: w_test = {v: k for k, v in lookup_name.items()}['A01']

# Time window
t_start, t_stop = 0, min(T-1, 190)

def has_metadata(f, w, t):
    seq = (t * P + w) * Z
    try:
        _ = f.frame_metadata(seq)
        return True
    except Exception:
        return False

def can_read_image(f, w, t):
    try:
        arr = f.to_dask()
        _ = arr[t, w, 0, :, :].compute()   # single Z-plane (Y,X)
        return True
    except Exception:
        return False

name = lookup_name.get(w_test, f"nd2_{w_test}")
results = []
with nd2.ND2File(str(nd2_path)) as f:
    for t in range(t_start, t_stop+1):
        meta_ok = has_metadata(f, w_test, t)
        img_ok  = False
        if not meta_ok:  # only try image if metadata fails
            img_ok = can_read_image(f, w_test, t)
        results.append((t, meta_ok, img_ok))

print(f"Probe well {name} (ND2 index {w_test}), t={t_start}..{t_stop}")
print("t | metadata | image_after_no_meta")
for t, m, im in results:
    print(f"{t:3d} |    {str(m):5s}  | {str(im):5s}")
Probe well A01 (ND2 index 0), t=40..189
t | metadata | image_after_no_meta
0 |    True   | False
  1 |    True   | False
  2 |    True   | False
  3 |    True   | False
...
 40 |    False  | True 
 41 |    False  | True 
 42 |    False  | True 
 43 |    False  | True 
 44 |    False  | True 
 45 |    False  | True 
 46 |    False  | True 
 47 |    False  | True 
 48 |    False  | True 
 49 |    False  | True 
...
186 |    False  | True 
187 |    False  | True 
188 |    False  | True 
189 |    False  | True 

and this shows that the metadata is missing but the image is still there.
so we need to frigoure our a way to STILL process the the images if the image is readble, but then impute the missing Time Rel(s) 


We need to make it so that we still PROCESS the images if the image is readable, but then impute the missing Time Rel(s) in the built metadta file. i would fill it with a Nan and then after the built metadta file is built we can do a post processing step to fill in the nans with the for loop as i propsed befroe time -> well.

---

## ✅ Revision: Minimal, Robust Time Handling for YX1 (2025-09-08)

### **Problem Statement**
Experiment 20250711 shows YX1 frames with missing metadata mid‑run (e.g., frames 40–189), while image data remains readable. Current code can fail during timestamp extraction, preventing FF processing and producing invalid "Time Rel (s)".

### **Root Cause Summary**
- `_fix_nd2_timestamp()` reads `nd.frame_metadata(...)` without guarding against missing metadata and assumes a stride of `n_z` yields per‑timepoint stamps; both are fragile.
- `well_df["Time (s)"]` is derived directly from this vector; if it contains NaN or is too short, downstream "Time Rel (s)" breaks.

### **Minimal Solution Architecture**

#### Phase A: Robust per‑timepoint timestamp extraction (length = n_t)
Goal: Never crash, always return a length‑`n_t` vector with `NaN` where timestamps are missing.

```python
def _extract_timepoints_robust(nd, n_t, n_w, n_z, ref_wells):
    """Return length-n_t times (s); NaN where metadata is missing.
    ref_wells: list of 0-based ND2 well indices to try per timepoint."""
    out = np.full((n_t,), np.nan, dtype=float)
    for t in range(n_t):
        ts = None
        for w in ref_wells:
            seq = (t * n_w + w) * n_z
            try:
                ts = nd.frame_metadata(seq).channels[0].time.relativeTimeMs / 1000.0
                break
            except Exception:
                continue
        out[t] = ts if ts is not None else np.nan
    return out

# After computing well_ind_list_sorted and knowing n_t, n_w, n_z:
selected_nd2_indices = [int(s) - 1 for s in well_ind_list_sorted]
ref_wells = selected_nd2_indices[:3] if selected_nd2_indices else list(range(min(n_w, 3)))
frame_time_vec = _extract_timepoints_robust(nd, n_t, n_w, n_z, ref_wells)
```

Notes:
- Do not stop early; always fill all `t` with a value or NaN.
- No image‑readability tests are needed for FF — keep it simple.

#### Phase B: Simple, gap‑agnostic imputation, then compute Time Rel(s)
Goal: Fill beginning, middle, and end gaps; use linear interpolation with a robust fallback.

```python
def _impute_timepoints(vec):
    s = pd.Series(vec, dtype=float)
    s = s.interpolate(method='linear', limit_direction='both')
    if s.isna().any():
        diffs = s.diff().dropna()
        if len(diffs) > 0 and np.isfinite(diffs.median()):
            step = float(diffs.median())
            fi = s.first_valid_index()
            if fi is not None:
                for i in range(fi - 1, -1, -1):
                    s.iloc[i] = s.iloc[i + 1] - step
            li = s.last_valid_index()
            if li is not None:
                for i in range(li + 1, len(s)):
                    s.iloc[i] = s.iloc[i - 1] + step
    return s.to_numpy()

# Impute before mapping to rows
frame_time_vec = _impute_timepoints(frame_time_vec)
time_ind_vec = np.tile(np.arange(0, n_t, dtype=int), n_selected_wells)
well_df["Time (s)"] = frame_time_vec[time_ind_vec]

# Compute Time Rel(s) only after imputation
first_time = float(np.nanmin(well_df["Time (s)"].values))
well_df['Time Rel (s)'] = well_df['Time (s)'] - first_time

# Optional quick checks/logs
tp = pd.Series(frame_time_vec)
valid = int(tp.notna().sum())
imputed = len(tp) - valid
log.info("YX1 timepoints: %d valid, %d imputed; median dt=%.3fs",
         valid, imputed, float(tp.diff().median()))
assert not np.isnan(well_df['Time (s)']).any()
assert not np.isnan(well_df['Time Rel (s)']).any()
```

Why this is minimal:
- Two small helpers; no changes to FF processing; QC remains optional and non‑blocking.
- Handles arbitrary gaps at start/middle/end without special‑case logic.

### **Integration Notes (build01B file)**
- Insert extraction (Phase A) right after computing `well_ind_list_sorted` and obtaining `n_t, n_w, n_z` and before assigning `well_df["Time (s)"]`.
- Insert imputation (Phase B) before computing "Time Rel (s)".
- Keep the FF/dask processing unchanged; no image readability tests are required.

### **Arbitrary Gaps**
- Start: `limit_direction='both'` fills leading NaNs.
- Middle: linear interpolation fills internal gaps.
- End: linear interpolation + median‑dt fallback fills trailing NaNs.

### **Code Quality and Safety**

**Error handling strategy**:
- **Graceful degradation**: Each phase has fallbacks and validation
- **Comprehensive logging**: Track which frames use metadata vs image-only vs failed
- **Data validation**: Ensure no NaN values remain in critical fields
- **Backwards compatibility**: Full metadata experiments work exactly as before

**Testing strategy**:
- **20250711 validation**: Processes 190 frames instead of 40, generates complete CSV
- **Normal experiment validation**: Existing experiments continue to work unchanged
- **Edge case testing**: Experiments with different metadata failure patterns

### **Expected Outcomes**
- 20250711: processes all timepoints; complete "Time (s)" and "Time Rel (s)" with imputed values where needed.
- General YX1: robust to missing metadata anywhere in the series; FF unaffected.

### **Implementation Checklist (Minimal)**

- [ ] Replace `_fix_nd2_timestamp` usage with Phase A extractor (length `n_t`, NaN‑tolerant)
- [ ] Add Phase B imputation before computing "Time Rel (s)"
- [ ] Use `np.nanmin` baseline for "Time Rel (s)"
- [ ] Add concise logging of valid vs imputed and median `dt`
- [ ] Keep FF/QC logic unchanged (QC best‑effort)

Estimated time: ~1–2 hours
Files: `src/build/build01B_compile_yx1_images_torch.py` (small, surgical changes)

---

## Small Bash Wrapper Simplifications (Minimal)

- Make conda activation optional: if `conda` is unavailable, warn and continue (assume user already activated env).
- Add `set -euo pipefail` and quote all vars for safety.
- Prefer a stable Python entrypoint (`python -m src.run_morphseq_pipeline.run_experiment_manager ...`) over generating a temporary Python file.

---

## 🔄 **PLAN REVISION: Addressing Code Shape & Robustness Criticisms** (2025-09-08)

### **Feedback Analysis & Critical Issues Identified**

**What's Good (Confirmed)**:
- ✅ Resume-by-skip for FF frames when `overwrite=False` saves significant time
- ✅ BF channel selection improvements (env override + warnings) are sensible
- ✅ High-level 3-phase plan approach (continue without metadata → robust time vector → impute) is correct

**Key Criticisms & Risks (Addressed Below)**:

#### **1. Code Shape Mismatch**
**Issue**: Original "Phase 1" referenced non-existent `stage_xyz_array/well_id_array` 
**Reality**: In build01B, FF writing proceeds via direct `dask_arr` reads; crash point is timestamp extraction (`frame_time_vec`), not image reading
**Correction**: Remove Phase 1 entirely; focus on `_fix_nd2_timestamp()` robustness

#### **2. Short Vector Hazard**  
**Issue**: Proposed robust function may stop early on "sustained metadata failure," returning fewer than `n_t` entries
**Risk**: Code later indexes with `time_int 0..n_t-1`, causing `IndexError`
**Correction**: Must guarantee exactly `n_t`-length time vector (with NaNs)

#### **3. Imputation Limited to Tail Gaps**
**Issue**: Original plan only extrapolated forward from last valid timepoint
**Risk**: NaNs at start or middle won't be filled
**Correction**: General solution supporting arbitrary gaps at both ends

#### **4. Cycle Time Estimator Sensitivity**
**Issue**: Mean of diffs is sensitive to outliers
**Correction**: Use median diff with optional outlier trimming for robustness

#### **5. Metadata-Only Mode Efficiency**
**Issue**: Fallback "test z-plane read" is unnecessary and expensive when `metadata_only=True`
**Correction**: Skip image readability tests entirely in metadata-only mode

#### **6. Ordering and Monotonicity**
**Issue**: Time Rel(s) computed before imputation; no monotonicity guarantee
**Correction**: Compute Time Rel(s) AFTER imputation; ensure non-decreasing timestamps

### **REVISED 2-Phase Solution Architecture**

#### **Phase 1: Robust Timestamp Vector Generation (Lines 251-252)**
**Problem**: `_fix_nd2_timestamp()` crashes on incomplete metadata; downstream expects exactly `n_t` timestamps.

**Current problematic code**:
```python
frame_time_vec = _fix_nd2_timestamp(nd, n_z)
# Crashes when metadata incomplete; may return short vector
```

**New guaranteed-length implementation**:
```python
def _fix_nd2_timestamp_robust(nd, n_z, n_t):
    """Extract frame times with guaranteed n_t length, handling arbitrary metadata gaps"""
    
    # CRITICAL: Initialize with NaN - ensures exactly n_t entries
    frame_time_vec = np.full(n_t, np.nan)
    
    try:
        n_frames_total = nd.frame_metadata(0).contents.frameCount
    except Exception:
        log.warning("Cannot read ND2 frame count - using all NaN timestamps for imputation")
        return frame_time_vec
    
    # Extract valid timestamps where metadata is available
    extracted_count = 0
    for t in range(n_t):
        frame_idx = t * nd.shape[1] * n_z  # t * n_wells * n_z (first well of timepoint)
        if frame_idx >= n_frames_total:
            break
            
        try:
            time_ms = nd.frame_metadata(frame_idx).channels[0].time.relativeTimeMs
            frame_time_vec[t] = time_ms / 1000
            extracted_count += 1
        except Exception:
            # Leave as NaN - will be imputed in Phase 2
            continue
    
    log.info(f"Timestamp extraction: {extracted_count}/{n_t} valid, {n_t - extracted_count} missing (will impute)")
    
    # Apply existing jump fix logic ONLY to valid (non-NaN) timestamps
    if extracted_count >= 2:
        frame_time_vec = _apply_jump_fix_to_valid(frame_time_vec, n_z)
    
    return frame_time_vec

def _apply_jump_fix_to_valid(frame_time_vec, n_z):
    """Apply jump detection/correction to non-NaN timestamps only"""
    valid_mask = ~np.isnan(frame_time_vec)
    valid_indices = np.where(valid_mask)[0]
    valid_times = frame_time_vec[valid_mask]
    
    if len(valid_times) < 2:
        return frame_time_vec
    
    # Apply existing jump detection logic to valid timestamps
    dt_frame_approx = valid_times[1] - valid_times[0] if len(valid_times) > 1 else 60.0
    time_diffs = np.diff(valid_times)
    jump_indices = np.where(time_diffs > 2 * dt_frame_approx)[0]
    
    if len(jump_indices) > 0:
        jump_idx = jump_indices[0]
        # Extrapolate post-jump timestamps using pre-jump rate
        pre_jump_count = jump_idx
        if pre_jump_count >= 2:
            dt_est = (valid_times[jump_idx] - valid_times[0]) / pre_jump_count
            base_time = valid_times[jump_idx]
            
            # Correct post-jump valid timestamps
            for i in range(jump_idx + 1, len(valid_times)):
                corrected_time = base_time + dt_est * (i - jump_idx)
                # Update in original array
                original_idx = valid_indices[i]
                frame_time_vec[original_idx] = corrected_time
    
    return frame_time_vec

# Replace line 252 with:
frame_time_vec = _fix_nd2_timestamp_robust(nd, n_z, n_t)
```

**Key Robustness Features**:
- **Guaranteed length**: Always returns exactly `n_t` entries, never crashes downstream indexing
- **Arbitrary gap handling**: NaNs at start, middle, end, or any combination  
- **Preserves existing logic**: Jump detection still works on available timestamps
- **No image tests**: Respects `metadata_only=True` mode efficiency

#### **Phase 2: General Gap Imputation with Monotonicity (Before Line 504)**
**Problem**: Need to fill arbitrary NaN patterns with robust statistics and ensure monotonic progression.

**Insert before Time Rel(s) calculation (before line 504)**:

```python
def _impute_missing_timestamps_general(well_df, exp_name):
    """Fill missing Time(s) using robust estimation and arbitrary gap handling"""
    
    missing_count = well_df['Time (s)'].isna().sum()
    if missing_count == 0:
        log.info(f"No missing timestamps for {exp_name}")
        return well_df
    
    log.info(f"Imputing {missing_count}/{len(well_df)} missing timestamps for {exp_name}")
    
    # Extract timepoint-level timestamps (one representative per time_int)
    timepoint_data = well_df.groupby('time_int')['Time (s)'].first().sort_index()
    valid_timepoints = timepoint_data.dropna()
    
    if len(valid_timepoints) < 2:
        log.error(f"Cannot impute: need ≥2 valid timepoints, found {len(valid_timepoints)}")
        log.error(f"Available timepoints: {valid_timepoints.index.tolist()}")
        return well_df
    
    # Robust cycle time estimation with outlier removal
    time_diffs = valid_timepoints.diff().dropna()
    
    # Remove statistical outliers (beyond 1.5 * IQR, more conservative than 2σ)
    q1, q3 = time_diffs.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clean_diffs = time_diffs[(time_diffs >= lower_bound) & (time_diffs <= upper_bound)]
    
    if len(clean_diffs) == 0:
        cycle_time = time_diffs.median()  # Fallback to raw median
        log.warning(f"All time differences flagged as outliers, using raw median: {cycle_time:.2f}s")
    else:
        cycle_time = clean_diffs.median()  # Robust estimate
        log.info(f"Robust cycle time: {cycle_time:.2f}s (median of {len(clean_diffs)}/{len(time_diffs)} clean intervals)")
    
    # Generate complete timepoint series with interpolation + extrapolation
    all_timepoints = pd.Series(index=range(well_df['time_int'].min(), well_df['time_int'].max() + 1), dtype=float)
    
    # Copy existing valid timestamps
    for t_int in valid_timepoints.index:
        all_timepoints[t_int] = valid_timepoints[t_int]
    
    # Fill gaps using context-aware method
    for t_int in all_timepoints.index:
        if pd.isna(all_timepoints[t_int]):
            # Find bracketing valid timepoints for interpolation
            lower_times = valid_timepoints[valid_timepoints.index < t_int]
            upper_times = valid_timepoints[valid_timepoints.index > t_int]
            
            if len(lower_times) > 0 and len(upper_times) > 0:
                # INTERPOLATION: Between two valid points
                t_lower = lower_times.index[-1]
                t_upper = upper_times.index[0]
                time_lower = lower_times.iloc[-1]
                time_upper = upper_times.iloc[0]
                
                # Linear interpolation (preserves monotonicity)
                ratio = (t_int - t_lower) / (t_upper - t_lower)
                all_timepoints[t_int] = time_lower + ratio * (time_upper - time_lower)
                
            elif len(lower_times) > 0:
                # FORWARD EXTRAPOLATION: From last valid point  
                t_last = lower_times.index[-1]
                time_last = lower_times.iloc[-1]
                all_timepoints[t_int] = time_last + (t_int - t_last) * cycle_time
                
            elif len(upper_times) > 0:
                # BACKWARD EXTRAPOLATION: From first valid point
                t_first = upper_times.index[0]
                time_first = upper_times.iloc[0]
                all_timepoints[t_int] = time_first - (t_first - t_int) * cycle_time
    
    # Apply imputed values to well_df
    imputed_count = 0
    for t_int, estimated_time in all_timepoints.items():
        if not pd.isna(estimated_time):
            mask = (well_df['time_int'] == t_int) & well_df['Time (s)'].isna()
            if mask.any():
                well_df.loc[mask, 'Time (s)'] = estimated_time
                imputed_count += mask.sum()
    
    # MONOTONICITY CHECK: Ensure timestamps increase within each well
    monotonic_violations = 0
    for well in well_df['well'].unique():
        well_mask = well_df['well'] == well
        well_data = well_df[well_mask].sort_values('time_int')
        times = well_data['Time (s)'].values
        
        # Check for strictly increasing timestamps  
        if not np.all(np.diff(times) > 0):
            monotonic_violations += 1
            # Light smoothing: ensure minimum forward progress
            min_delta = 0.1  # 0.1s minimum between timepoints
            for i in range(1, len(times)):
                if times[i] <= times[i-1]:
                    times[i] = times[i-1] + min_delta
            
            # Update well_df with corrected times
            well_df.loc[well_mask, 'Time (s)'] = times
    
    if monotonic_violations > 0:
        log.warning(f"Fixed monotonicity violations in {monotonic_violations} wells")
    
    # Final validation
    remaining_nan = well_df['Time (s)'].isna().sum()
    log.info(f"Imputation complete: filled {imputed_count} values, {remaining_nan} NaN remaining")
    
    if remaining_nan > 0:
        log.error(f"IMPUTATION FAILED: {remaining_nan} NaN timestamps remain")
    
    return well_df

# Add imputation call BEFORE Time Rel(s) calculation:
well_df = _impute_missing_timestamps_general(well_df, exp_name)

# MOVE Time Rel(s) calculation AFTER imputation (was at line 504-505):
meta_df = build_experiment_metadata(repo_root=repo_root, exp_name=exp_name, meta_df=well_df)
first_time = np.min(meta_df['Time (s)'])  # Now guaranteed no NaN
meta_df['Time Rel (s)'] = meta_df['Time (s)'] - first_time
```

### **Mathematical Rigor & Edge Case Handling**

**For 20250711 with metadata t=0-39, missing t=40-189**:

1. **Extract valid data**: `t=0: 10.5s, t=1: 48.2s, ..., t=39: 1485.3s`
2. **Calculate robust cycle time**: `median([37.7, 37.9, 37.8, ...]) = 37.8s` (outlier-resistant)
3. **Forward extrapolation**: 
   - `t=40: 1485.3 + 1×37.8 = 1523.1s`
   - `t=41: 1485.3 + 2×37.8 = 1560.9s`  
   - `t=189: 1485.3 + 150×37.8 = 7155.3s`
4. **Monotonicity verification**: Ensure `time[i+1] > time[i]` for all wells
5. **Time Rel(s) calculation**: `Time Rel(s) = Time(s) - 10.5s` (complete series)

**Edge cases handled**:
- **All NaN timestamps**: Graceful failure with clear error message
- **Mixed gaps**: Interpolation for middle gaps, extrapolation for ends
- **Monotonicity violations**: Light smoothing with minimum time progression
- **Outlier cycle times**: IQR-based filtering for robust estimation

### **Implementation Safety & Validation**

**Code compliance fixes**:
- ✅ **No new data structures**: Works with existing `frame_time_vec` and `well_df`
- ✅ **Respects metadata_only**: No image readability tests when `metadata_only=True`
- ✅ **Guaranteed vector lengths**: Never causes downstream `IndexError`
- ✅ **Preserves existing behavior**: Full metadata experiments unchanged

**Quality assurance**:
- ✅ **Comprehensive logging**: Track real vs imputed timestamps at each step
- ✅ **Graceful degradation**: Each phase has fallbacks and error handling
- ✅ **Mathematical validation**: Monotonicity and completeness checks
- ✅ **Performance consideration**: Minimal overhead for normal experiments

### **Expected 20250711 Results**

**Before (current failure)**:
- Processing stops at frame 40
- Missing 150 timepoints (79% data loss)  
- Incomplete CSV with broken Time Rel(s)

**After (robust processing)**:
- All 190 frames processed through FF pipeline
- Complete metadata CSV: 190 × N_wells rows
- Accurate Time Rel(s): 0s (t=0) to ~7145s (t=189) 
- Clear logging: "39 real timestamps, 151 imputed timestamps"

This revised plan addresses all identified code shape and robustness issues while maintaining the core objective of recovering all processable image data with scientifically sound timestamp imputation.

---

## 🎯 **FINAL IMPLEMENTATION PLAN: Surgical Approach with Strategic Enhancements** (2025-09-08)

### **Why This Approach: Audit-Driven Simplification**

**The audit correctly identified that my original complex approach was fundamentally flawed:**
- ❌ **Code shape mismatch**: Referenced non-existent arrays, wrong crash points  
- ❌ **Over-engineering**: 100+ lines when ~30 lines would suffice
- ❌ **Implementation gaps**: Documented imputation but didn't actually replace the broken call
- ❌ **Wrong focus**: Spent effort on image readability tests when crash happens in timestamp extraction

**The audit's "surgical approach" is superior because:**
- ✅ **Targets actual failure point**: Replaces `_fix_nd2_timestamp()` which actually crashes
- ✅ **Minimal disruption**: Works with existing data structures and flow
- ✅ **Practical problem-solving**: Directly fixes what's broken instead of theoretical completeness

### **Surgical Approach: 4 Targeted Changes Only**

#### **Change 1: Replace Fragile Timestamp Extraction (Line ~252)**
**Why this change**: Current `_fix_nd2_timestamp(nd, n_z)` crashes when metadata disappears, uses incorrect multi-well indexing, returns variable-length vectors.

**Current problematic code:**
```python
frame_time_vec = _fix_nd2_timestamp(nd, n_z)
# Steps by n_z incorrectly, no try/catch, variable length return
```

**Surgical replacement:**
```python
def _extract_timepoints_robust(nd, n_t, n_w, n_z, ref_wells):
    """
    GUARANTEED n_t length array of timestamps, NaN where unavailable.
    Uses multiple reference wells per timepoint for resilience.
    """
    out = np.full((n_t,), np.nan, dtype=float)  # CRITICAL: guaranteed length
    
    for t in range(n_t):
        timestamp = None
        for w in ref_wells:
            seq = (t * n_w + w) * n_z  # Correct ND2 multi-well indexing
            try:
                timestamp = nd.frame_metadata(seq).channels[0].time.relativeTimeMs / 1000.0
                break  # Found valid timestamp
            except Exception:
                continue  # Try next reference well
        out[t] = timestamp if timestamp is not None else np.nan
    
    # Preserve existing jump detection for valid values only
    if np.count_nonzero(~np.isnan(out)) >= 2:
        out = _apply_jump_fix_to_valid_only(out)
    
    return out

def _apply_jump_fix_to_valid_only(frame_time_vec):
    """Apply existing jump detection only to non-NaN timestamps"""
    valid_mask = ~np.isnan(frame_time_vec)
    valid_times = frame_time_vec[valid_mask]
    
    if len(valid_times) < 2:
        return frame_time_vec
    
    # Use existing jump detection algorithm on valid data
    dt_approx = valid_times[1] - valid_times[0] if len(valid_times) > 1 else 60.0
    diffs = np.diff(valid_times)
    jump_indices = np.where(diffs > 2 * dt_approx)[0]
    
    if len(jump_indices) > 0:
        # Apply correction to valid timestamps, preserve NaNs
        jump_idx = jump_indices[0]
        pre_jump_rate = (valid_times[jump_idx] - valid_times[0]) / jump_idx if jump_idx > 0 else dt_approx
        
        for i in range(jump_idx + 1, len(valid_times)):
            valid_times[i] = valid_times[jump_idx] + pre_jump_rate * (i - jump_idx)
        
        frame_time_vec[valid_mask] = valid_times  # Update only valid positions
    
    return frame_time_vec

# Smart reference well selection (enhancement over audit's simple [:3])
selected_nd2_indices = [int(s)-1 for s in well_ind_list_sorted]  # Convert to 0-based
def _select_reference_wells(indices, max_refs=3):
    """Select well-distributed reference wells for better plate coverage"""
    if len(indices) <= max_refs:
        return indices
    # Spread evenly across plate instead of just first 3
    step = len(indices) // max_refs  
    return [indices[i * step] for i in range(max_refs)]

ref_wells = _select_reference_wells(selected_nd2_indices) if selected_nd2_indices else list(range(min(n_w, 3)))

# THE ACTUAL FIX: Replace the crashing call
frame_time_vec = _extract_timepoints_robust(nd, n_t, n_w, n_z, ref_wells)
```

**Why these enhancements over basic audit approach:**
- **Better reference well selection**: Spreads across plate geography instead of just first 3 wells
- **Preserves existing logic**: Maintains jump detection behavior for compatibility
- **Correct ND2 indexing**: Uses proper `(t * n_w + w) * n_z` instead of naive stepping

#### **Change 2: Arbitrary Gap Imputation (After timestamp extraction)**
**Why this change**: Even with robust extraction, gaps remain; need to fill them before `well_df["Time (s)"]` assignment.

```python
def _impute_timepoints_enhanced(series_like):
    """
    Fill arbitrary gaps (start/middle/end) using interpolation + robust extrapolation.
    Enhancement over audit: adds outlier-resistant cycle time estimation.
    """
    s = pd.Series(series_like, dtype=float)
    
    # First pass: linear interpolation handles middle gaps elegantly  
    s_interp = s.interpolate(method='linear', limit_direction='both')
    
    # Second pass: robust extrapolation for remaining start/end gaps
    if s_interp.isna().any():
        valid_diffs = s.diff().dropna()
        if len(valid_diffs) > 0:
            # Enhancement: IQR-based outlier removal for robust cycle time
            q1, q3 = valid_diffs.quantile([0.25, 0.75])
            iqr = q3 - q1
            
            if iqr > 0:
                clean_diffs = valid_diffs[(valid_diffs >= q1 - 1.5*iqr) & (valid_diffs <= q3 + 1.5*iqr)]
                cycle_time = clean_diffs.median() if len(clean_diffs) > 0 else valid_diffs.median()
            else:
                cycle_time = valid_diffs.median()  # Fallback to raw median
                
            # Fill leading NaNs (backward extrapolation)
            first_valid = s_interp.first_valid_index()
            if first_valid is not None:
                for i in range(first_valid - 1, -1, -1):
                    s_interp.iloc[i] = s_interp.iloc[i + 1] - cycle_time
                    
            # Fill trailing NaNs (forward extrapolation)
            last_valid = s_interp.last_valid_index() 
            if last_valid is not None:
                for i in range(last_valid + 1, len(s_interp)):
                    s_interp.iloc[i] = s_interp.iloc[i - 1] + cycle_time
    
    return s_interp.to_numpy()

# Apply imputation immediately after extraction
frame_time_vec = _impute_timepoints_enhanced(frame_time_vec)
```

**Why enhancement over audit's basic version:**
- **Outlier-resistant statistics**: Uses IQR filtering instead of raw median for cycle time
- **Better gap handling**: Linear interpolation for middle gaps, extrapolation for ends
- **Preserves audit's simplicity**: Still ~15 lines, handles all gap patterns

#### **Change 3: NaN-Safe Time Rel(s) Calculation (Line ~504)**
**Why this change**: Current `np.min()` crashes if any NaN leaked through; audit correctly identified this failure mode.

**Current problematic code:**
```python
first_time = np.min(well_df['Time (s)'].copy())  # Crashes on NaN
meta_df['Time Rel (s)'] = meta_df['Time (s)'] - first_time
```

**NaN-safe replacement (exactly as audit suggested):**
```python
first_time = float(np.nanmin(meta_df['Time (s)'].values))  # NaN-safe minimum
if np.isnan(first_time):
    log.error(f"All timestamps are NaN for {exp_name} - cannot compute Time Rel(s)")
    first_time = 0.0  # Graceful fallback

meta_df['Time Rel (s)'] = meta_df['Time (s)'] - first_time
```

#### **Change 4: Validation & Logging (Before CSV save)**
**Why this change**: Audit correctly noted need for assertions to prevent silent NaN propagation; add comprehensive validation.

```python
# Enhanced validation and logging (beyond audit's basic version)
timepoint_series = pd.Series(frame_time_vec)
valid_count = timepoint_series.notna().sum()
imputed_count = len(timepoint_series) - valid_count
median_cycle = float(timepoint_series.diff().median()) if len(timepoint_series) > 1 else 0.0

log.info(f"YX1 timestamp processing for {exp_name}:")
log.info(f"  - Extracted {valid_count}/{n_t} valid timestamps from metadata")
log.info(f"  - Imputed {imputed_count}/{n_t} missing timestamps") 
log.info(f"  - Robust median cycle time: {median_cycle:.2f}s")
log.info(f"  - Final time range: {meta_df['Time (s)'].min():.1f}s to {meta_df['Time (s)'].max():.1f}s")

# Critical assertions (as audit recommended)
assert not meta_df['Time (s)'].isna().any(), f"FATAL: NaN timestamps remain in {exp_name}"
assert not meta_df['Time Rel (s)'].isna().any(), f"FATAL: NaN relative times in {exp_name}"

# Monotonicity check (enhancement: per-timepoint validation)
if len(timepoint_series) > 1:
    time_progression = timepoint_series.diff().dropna()
    if not (time_progression > -1.0).all():  # Allow small negative diffs due to imputation
        log.warning(f"Significant non-monotonic timestamps detected in {exp_name}")

log.info(f"✅ {exp_name}: validated {len(meta_df)} metadata rows, ready for save")
```

### **Why These 4 Changes Solve 20250711**

**Current failure mode for 20250711:**
1. `_fix_nd2_timestamp()` crashes at frame 40 when metadata disappears
2. No FF processing happens (loses 150 timepoints)
3. No CSV generated

**After surgical fixes:**
1. **Change 1**: `_extract_timepoints_robust()` gets 40 valid + 150 NaN timestamps, never crashes
2. **Change 2**: `_impute_timepoints_enhanced()` fills all 150 gaps using robust cycle time
3. **FF processing**: Proceeds normally (no changes needed to existing image processing)
4. **Change 3**: `nanmin()` safely computes Time Rel(s) from complete timestamp series
5. **Change 4**: Validation confirms no NaN leakage, logs show "40 real + 150 imputed"
6. **Result**: Complete 190×N_wells CSV with accurate Time Rel(s) progression

### **Code Footprint: Minimal & Surgical**

- **Total new code**: ~80 lines (2 helper functions + validation)
- **Lines modified**: 2 (replace `_fix_nd2_timestamp()` call, replace `np.min()` call)
- **Files changed**: 1 (`src/build/build01B_compile_yx1_images_torch.py`)
- **Risk assessment**: Very low (preserves all existing behavior for normal experiments)

### **Validation Strategy**

**Testing with 20250711:**
- Verify processing completes (doesn't crash at frame 40)
- Check output: 190×N_wells CSV rows instead of 40×N_wells
- Validate Time Rel(s): monotonic progression 0s → ~7145s
- Confirm FF images: all readable timepoints processed

**Regression testing:**
- Normal experiments (full metadata): should work exactly as before
- Mixed scenarios: experiments with partial metadata failures

### **Why This Beats My Original Over-Engineering**

| **Aspect** | **Original Complex Plan** | **Surgical Approach + Enhancements** |
|------------|-------------------------|--------------------------------|
| **Lines of code** | ~200+ lines, 3 phases | ~80 lines, 4 targeted changes |
| **Complexity** | New data structures, complex gap logic | Works with existing flow |
| **Risk** | High (major architectural changes) | Low (surgical replacements) |
| **Focus** | Theoretical completeness | Practical problem-solving |
| **Failure modes** | Many new edge cases introduced | Directly fixes known crash points |

The audit was absolutely correct: **simple, targeted fixes are superior to complex architectural changes** when the problem is well-defined and the failure modes are known.

### **Implementation Checklist**

- [ ] **Replace** `_fix_nd2_timestamp()` call with `_extract_timepoints_robust()`
- [ ] **Add** `_impute_timepoints_enhanced()` call after timestamp extraction  
- [ ] **Replace** `np.min()` with `np.nanmin()` in Time Rel(s) calculation
- [ ] **Add** validation assertions and enhanced logging before CSV save
- [ ] **Test** with 20250711 to confirm 190 frames processed
- [ ] **Regression test** with normal experiments to ensure no behavior changes

**Estimated implementation time**: 1 hour (much faster than original 2-3 hour estimate)
**Maintenance burden**: Minimal (preserves existing architecture and patterns) 
