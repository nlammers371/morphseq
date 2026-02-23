# Refactor-013: Per-Experiment File Structure Implementation

**Created**: 2025-09-10  
**Status**: Implementation  
**Depends On**: Refactor-011 SAM2 CLI Integration, Refactor-012 Experiment Manager

## **CRITICAL REQUIREMENTS FOR FUTURE REFERENCE**
⚠️ **This approach will be audited and critiqued - ensure thorough explanation**

1. **Tracking Changes**: Document exactly how ExperimentManager tracking logic changes
2. **Code Changes**: Specify precise modifications to each pipeline component  
3. **Integration Changes**: Explain how per-experiment files integrate into pipeline flow
4. **Parallel Processing**: Detail how this enables true experiment-level parallelization
5. **Migration (No Back-Compat)**: Document explicit drop of monolithic fallback

---

## **Executive Summary**

Restructure MorphSeq pipeline from monolithic JSON/CSV files to per-experiment outputs, enabling true parallel processing and granular experiment tracking. This refactor splits large aggregated files (GDINO detections, SAM2 segmentations, Build03 embryo metadata) into experiment-scoped files.

**Key Deliverables:**
- `gdino_detections_{exp}.json` (per-experiment GDINO output)
- `grounded_sam_segmentations_{exp}.json` (per-experiment SAM2 output)
- Build03 per-experiment: `metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv`
- Build04 per-experiment: `metadata/build04/per_experiment/qc_staged_{exp}.csv`
- Build06 per-experiment: `metadata/build06/per_experiment/embryo_latents_final_{exp}.csv`
- ExperimentManager tracking + needs logic for per-experiment files
- Optional cohort aggregation only at the very end (concat per-experiment finals)

---

## **Current State Analysis**

### **Monolithic File Problems:**
1. **GDINO**: Single `gdino_detections.json` for all experiments (run_sam2.py:168)
2. **SAM2**: Single `grounded_sam_segmentations.json` for all experiments (run_sam2.py:196)
3. **Build03**: Single `embryo_metadata_df01.csv` for all experiments (pipeline_objects.py:354)
4. **Tracking**: Per-experiment tracking exists but files are aggregated

### **Existing Per-Experiment Infrastructure:**
✅ **SAM2 CSV**: Already outputs `sam2_metadata_{exp}.csv` (pipeline_objects.py:291)
✅ **ExperimentManager**: Has per-experiment status tracking (run_experiment_manager.py:188-294)
✅ **Experiment Class**: Individual experiment lifecycle management (pipeline_objects.py:101-850)

---

## **Implementation Plan**

### **Phase 1: SAM2 Pipeline Per-Experiment JSON Files**

#### **1.1 GDINO Detection Output (Stage 2)**

**Primary File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/steps/run_sam2.py`
- **Exact Location**: Line 169 
- **Current Code**: `annotations_path = sam2_root / "detections" / "gdino_detections.json"`
- **New Code**: `annotations_path = sam2_root / "detections" / f"gdino_detections_{exp}.json"`

**Secondary Files to Modify:**
- `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/pipelines/03_gdino_detection.py`
  - **Modification**: Add `--experiment-filter` parameter to limit processing to single experiment
  - **Output Logic**: Write to per-experiment JSON instead of appending to monolithic file

**ExperimentManager Integration**:
- **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/build/pipeline_objects.py`
- **Add Property** (after line 294):
```python
@property
def gdino_detections_path(self) -> Path:
    return self.data_root / "sam2_pipeline_files" / "detections" / f"gdino_detections_{self.date}.json"
```

#### **1.2 SAM2 Segmentation Output (Stage 3)**  

**Primary File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/steps/run_sam2.py`
- **Exact Location**: Line 196
- **Current Code**: `sam2_output_path = sam2_root / "segmentation" / "grounded_sam_segmentations.json"`
- **New Code**: `sam2_output_path = sam2_root / "segmentation" / f"grounded_sam_segmentations_{exp}.json"`

**Secondary Files to Modify:**
- `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py`
  - **Modification**: Accept per-experiment detection input, output per-experiment segmentation JSON
- Line 244 in `run_sam2.py`: Update `--sam2-annotations` parameter to use per-experiment file
- Line 202 in `run_sam2.py`: Update `--annotations` parameter to use per-experiment detection file

**ExperimentManager Integration**:
- **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/build/pipeline_objects.py`
- **Add Property** (after gdino_detections_path):
```python
@property  
def sam2_segmentations_path(self) -> Path:
    return self.data_root / "sam2_pipeline_files" / "segmentation" / f"grounded_sam_segmentations_{self.date}.json"
```

#### **1.2.1 Mask Export Optimization (Stage 5)**

**Primary File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/steps/run_sam2.py`
- **Exact Location**: Line 244-247
- **Current Code**: 
```python
export_args = [
    "--sam2-annotations", str(sam2_output_path.absolute()),
    "--output", str(masks_output_dir.absolute()),
    "--entities-to-process", exp
]
```
- **New Code**:
```python
export_args = [
    "--sam2-annotations", str(sam2_output_path.absolute()),  # Now per-experiment file
    "--output", str(masks_output_dir.absolute()),
    # Remove --entities-to-process since input is already experiment-scoped
]
```

**Secondary Files to Modify:**
- `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/pipelines/06_export_masks.py`
  - **Modification**: Remove experiment filtering logic since input is already per-experiment
  - **Performance**: Faster processing without needing to filter large JSON

**Benefits**:
- Faster mask export (no filtering of large JSON)
- Cleaner processing logic (no experiment filtering needed)
- Memory efficiency (load only relevant experiment's segmentations)

#### **1.3 Experiment Metadata Per-Experiment**
**File**: `src/run_morphseq_pipeline/steps/run_sam2.py:168`

**Current Code:**
```python
metadata_path = sam2_root / "raw_data_organized" / "experiment_metadata.json"
```

**New Code:**
```python
# Per-experiment metadata files  
metadata_path = sam2_root / "raw_data_organized" / f"experiment_metadata_{exp}.json"
```

### **Phase 2: Build03 Per-Experiment Embryo Metadata**

#### **2.1 Build03 Output (Per-Experiment Only)**
**File**: `src/build/build03A_process_images.py`

**Current Behavior**: Appends to single `embryo_metadata_df01.csv`

**New Behavior**:
- Write only per-experiment output:
  - `metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv`
- Maintain experiment-scoped embryo data only
- Remove any direct df01 writes from Build03

**Integration Changes:**
- ExperimentManager property: `build03_path` → `{data_root}/metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv`
- `needs_build03`: check per-experiment file existence and freshness vs upstream (SAM2 CSV or QC masks)
- `run_build03()`: output per-experiment CSV only

#### **2.2 Build04 Per-Experiment QC + Stage**
**File**: `src/build/build04_perform_embryo_qc.py`

**Current Behavior**: Reads single `embryo_metadata_df01.csv`

**New Behavior**:
- Read one experiment at a time:
  - Input: `metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv`
  - Output: `metadata/build04/per_experiment/qc_staged_{exp}.csv`
- Use shared reference params for stage inference so results are cohort-consistent
- No aggregation in Build04; keep everything per-experiment

### **Phase 3: ExperimentManager Tracking Updates**

#### **3.1 New Tracking Properties**
**File**: `src/build/pipeline_objects.py` (Experiment class)

**Add Properties:**
```python
@property
def gdino_detections_path(self) -> Path:
    return self.data_root / "sam2_pipeline_files" / "detections" / f"gdino_detections_{self.date}.json"

@property
def sam2_segmentations_path(self) -> Path:
    return self.data_root / "sam2_pipeline_files" / "segmentation" / f"grounded_sam_segmentations_{self.date}.json"

@property
def build03_path(self) -> Path:
    return self.data_root / "metadata" / "build03" / "per_experiment" / f"expr_embryo_metadata_{self.date}.csv"

@property
def build04_path(self) -> Path:
    return self.data_root / "metadata" / "build04" / "per_experiment" / f"qc_staged_{self.date}.csv"

@property
def build06_final_path(self) -> Path:
    return self.data_root / "metadata" / "build06" / "per_experiment" / f"embryo_latents_final_{self.date}.csv"
```

#### **3.2 Updated Status Logic (Simplified)**
**File**: `src/build/pipeline_objects.py:515-528, 584-610`

**Current Code (baseline):**
```python
def needs_sam2(self) -> bool:
    return not self.sam2_csv_path.exists()
```

**New Code (per-experiment stages, keep simple freshness):**
```python
def needs_sam2(self) -> bool:
    # Primary signal is the per-experiment SAM2 CSV
    if not self.sam2_csv_path.exists():
        return True
    # Optional freshness: if upstream per-exp JSONs exist and are newer than CSV, re-run
    try:
        csv_m = self.sam2_csv_path.stat().st_mtime
        for p in (self.gdino_detections_path, self.sam2_segmentations_path):
            if p.exists() and p.stat().st_mtime > csv_m:
                return True
        return False
    except Exception:
        return False

def needs_build03(self) -> bool:
    if not self.build03_path.exists():
        return True
    try:
        return self.sam2_csv_path.exists() and self.sam2_csv_path.stat().st_mtime > self.build03_path.stat().st_mtime
    except Exception:
        return False

def needs_build04(self) -> bool:
    if not self.build04_path.exists():
        return True
    try:
        return self.build03_path.exists() and self.build03_path.stat().st_mtime > self.build04_path.stat().st_mtime
    except Exception:
        return False

def needs_build06(self) -> bool:
    if not self.build06_final_path.exists():
        return True
    try:
        newer = self.build04_path.exists() and self.build04_path.stat().st_mtime > self.build06_final_path.stat().st_mtime
        # Also rerun if latents for this experiment are newer than the final
        if self.has_latents() and self.get_latent_path("20241107_ds_sweep01_optimum").stat().st_mtime > self.build06_final_path.stat().st_mtime:
            return True
        return newer
    except Exception:
        return False
```

#### **3.3 Status Display Update**
**File**: `src/run_morphseq_pipeline/run_experiment_manager.py:188-210`

**Add Tracking (per-experiment focus):**
```python
gdino_ok   = exp.gdino_detections_path.exists()
sam2_seg_ok= exp.sam2_segmentations_path.exists()
sam2_ok    = exp.sam2_csv_path.exists()
b03_ok     = exp.build03_path.exists()
b04_ok     = exp.build04_path.exists()
b06_ok     = exp.build06_final_path.exists()

parts = [
    f"RAW {'✅' if raw_ok else '❌'}",
    f"FF {'✅' if ff_ok else '❌'}",
    f"GDINO {'✅' if gdino_ok else '❌'}",
    f"SAM2-SEG {'✅' if sam2_seg_ok else '❌'}",
    f"SAM2-CSV {'✅' if sam2_ok else '❌'}",
    f"B03-EXP {'✅' if b03_ok else '❌'}",
    f"B04-EXP {'✅' if b04_ok else '❌'}",
    f"B06-EXP {'✅' if b06_ok else '❌'}",
    f"LATENTS {'✅' if lat_ok else '❌'}",
]
```

### **Phase 4: Pipeline Script Modifications**

#### **4.1 SAM2 Pipeline Scripts**
**Files**: `segmentation_sandbox/scripts/pipelines/03_gdino_detection.py`, `04_sam2_video_processing.py`

**Changes:**
- Reuse existing `--entities_to_process` flag to pass a single experiment
- Output per-experiment JSON files instead of aggregated files
- Maintain experiment-scoped processing logic

#### **4.2 Build04/06 Per-Experiment I/O**
**Files**: `src/build/build04_perform_embryo_qc.py`, `src/run_morphseq_pipeline/steps/run_build06.py`

**Changes:**
- Build04: Read Build03 per-exp CSV → write per-exp QC/staged CSV
  - Input: `metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv`
  - Output: `metadata/build04/per_experiment/qc_staged_{exp}.csv`
- Build06: Read Build04 per-exp QC/staged + per-exp latents → write per-exp final
  - Input: `metadata/build04/per_experiment/qc_staged_{exp}.csv`
  - Output: `metadata/build06/per_experiment/embryo_latents_final_{exp}.csv`
- Optional (when a single cohort file is needed):
  - Concat all `embryo_latents_final_*.csv` → `metadata/combined_metadata_files/embryo_metadata_final_with_latents.csv`

---

## **Directory Structure Changes**

### **Before (Monolithic):**
```
data_root/
├── sam2_pipeline_files/
│   ├── detections/
│   │   └── gdino_detections.json                 # ALL experiments
│   ├── segmentation/  
│   │   └── grounded_sam_segmentations.json      # ALL experiments
│   └── sam2_expr_files/
│       ├── sam2_metadata_20250520.csv           # Per-experiment ✅
│       └── sam2_metadata_20250512.csv           # Per-experiment ✅
└── metadata/
    └── combined_metadata_files/
        └── embryo_metadata_df01.csv             # ALL experiments
```

### **After (Per-Experiment, Experiment = Fundamental Unit):**
```
data_root/
├── sam2_pipeline_files/
│   ├── detections/
│   │   ├── gdino_detections_20250520.json
│   │   └── gdino_detections_20250512.json
│   ├── segmentation/
│   │   ├── grounded_sam_segmentations_20250520.json  
│   │   └── grounded_sam_segmentations_20250512.json
│   └── sam2_expr_files/
│       ├── sam2_metadata_20250520.csv
│       └── sam2_metadata_20250512.csv
└── metadata/
    ├── build03/
    │   └── per_experiment/
    │       ├── expr_embryo_metadata_20250520.csv
    │       └── expr_embryo_metadata_20250512.csv
    ├── build04/
    │   └── per_experiment/
    │       ├── qc_staged_20250520.csv
    │       └── qc_staged_20250512.csv
    ├── build06/
    │   └── per_experiment/
    │       ├── embryo_latents_final_20250520.csv
    │       └── embryo_latents_final_20250512.csv
    └── combined_metadata_files/
        └── embryo_metadata_final_with_latents.csv   # optional cohort concat of per-exp finals
```

---

## **Parallel Processing Benefits**

### **Current Bottlenecks:**
- GDINO processes all experiments → single JSON (memory intensive)
- SAM2 reads entire detection JSON for one experiment  
- Build03 locks entire df01.csv for single experiment append

### **After Per-Experiment:**
- **GDINO**: Process experiments independently → separate JSON files
- **SAM2**: Read only relevant experiment's detection JSON
- **Build03**: Write independent CSV files under build03/per_experiment
- **Build04**: Process each experiment independently → build04/per_experiment outputs
- **Build06**: Merge latents per experiment → build06/per_experiment outputs
- **ExperimentManager**: True per-experiment parallelization via qsub/slurm

### **Tracking Logic Improvements:**
- Granular progress tracking per experiment per pipeline stage
- Independent experiment retry without affecting others
- Clear experiment-level dependency resolution

---

## **I/O Safety, Schema, and Concurrency**

### Atomic Writes (All Writers)
- Always write to a temporary path, then atomically replace the target.
- Strategy:
  - `tmp = target.with_suffix(target.suffix + '.tmp')` (optionally include PID)
  - Write → flush → fsync → `os.replace(tmp, target)`
- Applies to: per-experiment JSON (GDINO/SAM2), Build03/04/06 per-experiment CSVs, optional cohort final.
- Readers must ignore `*.tmp` files.

### Optional Cohort Concat Locking (When Used)
- If producing a single cohort final, acquire a `.lock` file via O_EXCL, and atomically replace the final CSV on completion.

### Directory Creation
- Writers must ensure directories exist:
  - Build03 → `metadata/build03/per_experiment/`
  - Build04 → `metadata/build04/per_experiment/`
  - Build06 → `metadata/build06/per_experiment/`
  - Optional cohort → `metadata/combined_metadata_files/`

### Flag Consistency
- Use only `--entities_to_process` to specify experiment(s) in SAM2 scripts.

### Required Schemas/Contracts
- GDINO detections (per-experiment JSON):
  - Minimal structure:
    {
      "images": {
        "<image_id>": {"detections": [{"bbox_xyxy_norm": [x1,y1,x2,y2], "score": 0.0, "prompt": "individual embryo"}]}
      }
    }
- SAM2 segmentations (per-experiment JSON):
  - Minimal structure:
    {
      "experiments": {"<exp>": {"videos": {"<video_id>": {"image_ids": {"<image_id>": {"embryos": {"<embryo_id>": {"segmentation": {"format": "rle"|"polygon", "size": [H,W], "data": ...}, "bbox": [x1,y1,x2,y2]}}}}}}}}
    }
- Build03 per-experiment CSV (input to Build04): required columns
  - Identity/time: `snip_id, experiment_id|experiment_date, video_id, image_id, embryo_id, well, time_int, Time Rel (s)`
  - Morphology/QC: `surface_area_um, length_um, width_um, use_embryo_flag, bubble_flag, focus_flag, frame_flag, dead_flag, no_yolk_flag`
  - Stage/meta: `predicted_stage_hpf, temperature, medium, phenotype, control_flag, short_pert_name`
  - Build04 must assert these exist and fill safe defaults for optional fields.

### Idempotency & Re-run Rules (mtime-based)
- Build03: rerun if `sam2_metadata_{exp}.csv` is newer than `expr_embryo_metadata_{exp}.csv`.
- Build04: rerun if Build03 per-exp output is newer than `qc_staged_{exp}.csv`.
- Build06: rerun if Build04 per-exp output or per-exp latent file is newer than `embryo_latents_final_{exp}.csv`.

### Storage & Retention Guidance
- Keep Build06 per-experiment finals as canonical; prune intermediate JSONs or keep N latest runs as desired.
- Optional cohort final is generated on demand; avoid treating it as a constantly-updated artifact.

### Line Reference Policy
- Prefer function/section references over line numbers. Update doc when function names change.

### Minimal Testing Checklist
- Atomic writes produce `*.tmp` then final; readers ignore `*.tmp`.
- Build04 asserts Required Columns; emits clear errors if missing.
- Per-experiment e2e: Build03/04/06 per-exp outputs exist for a single experiment.
- Optional: run two experiments concurrently; verify independent outputs.

---

## **Migration Strategy**

### **No Backwards Compatibility (Explicit)**
- The pipeline now requires per-experiment artifacts exclusively.
- Build03 writes only per-experiment CSVs; Build04 aggregates those into df01 and proceeds.
- We remove runtime fallbacks to monolithic files across stages.
- Optional: provide a one-time converter to split legacy monolithic JSONs into per-experiment files for archival or reprocessing, but it is not required to operate the new pipeline.

### **Testing Strategy:**
1. **Unit Tests**: Per-experiment file creation/reading
2. **Integration Tests**: End-to-end with per-experiment files
3. **Parallel Tests**: Multiple experiments processed simultaneously
4. (Optional) **Converter Tests**: If a one-time monolithic splitter is authored honeslty only these need to be done at the 

---

## **Success Criteria**

1. ✅ Per-experiment files for all stages:
   - Build03 → `metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv`
   - Build04 → `metadata/build04/per_experiment/qc_staged_{exp}.csv`
   - Build06 → `metadata/build06/per_experiment/embryo_latents_final_{exp}.csv`
2. ✅ Parallel processing: Experiments run independently end-to-end
3. ✅ ExperimentManager tracking: Simple existence/freshness checks per experiment
4. ✅ Optional aggregation: Only needed to produce a cohort final (concat of per-exp Build06 outputs)
5. ✅ Performance: No shared-writer bottlenecks; reduced memory footprint per step
6. ✅ Monolithic removed: No runtime fallbacks to monolithic files

---

## **Implementation Status**

- [ ] **Phase 1**: SAM2 Pipeline Per-Experiment JSON Files
  - [ ] 1.1: GDINO Detection Output 
  - [ ] 1.2: SAM2 Segmentation Output
  - [ ] 1.3: Experiment Metadata Per-Experiment
- [x] **Phase 2**: Build03 Per-Experiment Embryo Metadata ✅ **COMPLETE**
  - [x] 2.1: Build03 Output Modification ✅ **COMPLETE** 
    - **Status**: `run_build03_pipeline()` function implemented and tested
    - **QC Improvements**: SAM2 QC integration, surgical frame_flag modifications
    - **Test Results**: 80 embryos → 78 usable (97.5% pass rate) for experiment 20250622_chem_28C_T00_1425
    - **Integration**: Ready for ExperimentManager per-experiment file structure
  - [ ] 2.2: Build04 Aggregation Update ⚠️ **NEXT PRIORITY**
- [ ] **Phase 3**: ExperimentManager Tracking Updates
  - [ ] 3.1: New Tracking Properties
  - [ ] 3.2: Updated Status Logic
  - [ ] 3.3: Status Display Update
- [ ] **Phase 4**: Pipeline Script Modifications
  - [ ] 4.1: SAM2 Pipeline Scripts
  - [ ] 4.2: Build04/06 Aggregation Logic

---

## **Notes for Future Reference**

This refactor addresses the fundamental architectural limitation of monolithic file aggregation that prevents true parallel processing. The implementation explicitly removes monolithic fallbacks and makes the experiment the fundamental unit across all builds, enabling granular tracking and independent processing.

The key insight is that the ExperimentManager already has the infrastructure for per-experiment tracking - we're extending this to cover the intermediate JSON files that were previously aggregated, enabling each experiment to be processed completely independently.

---

## **Implementation Flow and File Modification Summary**

### **Exact Files to Modify (In Order):**

1. **Phase 1A: ExperimentManager Properties**
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/build/pipeline_objects.py`
   - **Lines**: Add after line 294 (after sam2_csv_path property)
   - **Action**: Add gdino_detections_path, sam2_segmentations_path, embryo_metadata_path properties

2. **Phase 1B: SAM2 Pipeline JSON Output**
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/steps/run_sam2.py`
   - **Line 169**: Change gdino detections path to per-experiment
   - **Line 196**: Change SAM2 segmentations path to per-experiment  
   - **Lines 244-247**: Remove entities-to-process filter for mask export

3. **Phase 1C: SAM2 Pipeline Scripts**
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/pipelines/03_gdino_detection.py`
   - **Action**: Add experiment filtering, output per-experiment JSON
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py`
   - **Action**: Read per-experiment detection, output per-experiment segmentation
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/pipelines/06_export_masks.py`
   - **Action**: Remove experiment filtering logic

4. **Phase 2A: Build03 Per-Experiment Output**
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/build/build03A_process_images.py`
   - **Function**: segment_wells, compile_embryo_stats
   - **Action**: Output to per-experiment CSV instead of appending to df01

5. **Phase 2B: Build04 Aggregation Logic**
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/build/build04_perform_embryo_qc.py`
   - **Action**: Read multiple per-experiment files, aggregate to df01/df02

6. **Phase 3: Status Logic Updates**
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/build/pipeline_objects.py`
   - **Lines 515-528**: Update needs_sam2() logic
   - **Lines 584-610**: Update needs_build03() logic
   - **File**: `/net/trapnell/vol1/home/mdcolon/proj/morphseq/src/run_morphseq_pipeline/run_experiment_manager.py`
   - **Lines 188-210**: Update status display to show new file types

### **Implementation Flow:**

```
┌─ Phase 1: SAM2 Per-Experiment Files ─────────────────────────┐
│  1. Add tracking properties to ExperimentManager            │
│  2. Modify run_sam2.py to use per-experiment paths          │
│  3. Update SAM2 pipeline scripts for per-experiment I/O     │
│  ✓ Result: gdino_detections_{exp}.json + grounded_sam_      │
│    segmentations_{exp}.json + optimized mask export         │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ Phase 2: Build03 Per-Experiment Files ─────────────────────┐
│  4. Modify Build03 to output per-experiment embryo CSV      │
│  5. Update Build04 to aggregate from per-experiment files   │
│  ✓ Result: expr_embryo_metadata_{exp}.csv →                 │
│    embryo_metadata_df01.csv (aggregated)                    │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ Phase 3: Tracking Logic Updates ───────────────────────────┐
│  6. Update needs_sam2()/needs_build03() status methods      │
│  7. Update ExperimentManager status display                 │
│  ✓ Result: Granular per-experiment tracking for all files  │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─ Result: True Per-Experiment Parallelization ──────────────┐
│  • Each experiment processes independently                  │
│  • No monolithic file bottlenecks                          │
│  • Parallel qsub/slurm job submission possible             │
│  • Memory efficient (experiment-scoped data loading)       │
└──────────────────────────────────────────────────────────────┘
```

### **Testing Flow (Minimal Smoke Checks):**

1. Per-experiment SAM2: After running a single experiment, verify files exist:
   - `gdino_detections_{exp}.json` contains images for only `{exp}`
   - `grounded_sam_segmentations_{exp}.json` contains videos/images for only `{exp}`
   - `sam2_metadata_{exp}.csv` has expected columns: `[image_id, video_id, embryo_id, bbox_*, exported_mask_path, ...]`
2. Build03 per-experiment: Verify `metadata/build03/per_experiment/expr_embryo_metadata_{exp}.csv` exists with core columns (`snip_id, embryo_id, time_int, surface_area_um, use_embryo_flag, predicted_stage_hpf`).
3. Build04 per-experiment: Verify `metadata/build04/per_experiment/qc_staged_{exp}.csv` exists; spot-check stage/QC fields.
4. Build06 per-experiment: Verify `metadata/build06/per_experiment/embryo_latents_final_{exp}.csv` exists and includes staged columns plus embeddings.
5. Optional parallel smoke: Run two experiments concurrently and confirm their per-experiment files are created independently without conflicts.
