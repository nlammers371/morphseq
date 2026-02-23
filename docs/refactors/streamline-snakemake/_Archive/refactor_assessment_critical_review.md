# Critical Review: MorphSeq Pipeline Refactor Plan

**Author:** Claude Code Analysis
**Date:** 2025-10-06
**Status:** DRAFT FOR REVIEW

---

## Executive Summary

**VERDICT: The proposed refactor plan is over-engineered and replicates the problems it's trying to solve.**

The original plan focuses on:
- "Containerization" into service objects
- Dependency injection patterns
- Entity tracking systems
- Backup rotation systems
- Complex class hierarchies

**What we actually need:**
- **Simple, pure functions** in clearly-named files
- **Minimal abstractions** - functions over classes
- **Let Snakemake do its job** - it handles dependencies, file tracking, backups
- **Flatten the hell out of everything**

---

## Core Philosophy: Simplicity Over Architecture

### ❌ What NOT to Do (from current plan)

```python
# DON'T DO THIS - Overengineered
class SegmentationProcessor:
    def __init__(self, loader: AnnotationStore, propagator: PropagationRunner,
                 encoder: MaskEncoder, config: PipelineContext):
        self.loader = loader
        self.propagator = propagator
        ...

    def process(self, job_spec: SegmentationJobSpec) -> SegmentationResult:
        ...
```

**Problems:**
- Multiple layers of abstraction
- Dependency injection complexity
- Typed dataclasses everywhere
- More code to maintain
- Harder to debug
- Can't easily call from Snakemake

### ✅ What TO Do (Simple & Flat)

```python
# DO THIS - Simple and direct
def run_sam2_on_video(
    video_frames: list[np.ndarray],
    seed_annotations: dict,
    model_checkpoint: Path,
    output_path: Path
) -> dict:
    """Run SAM2 propagation on a video.

    Args:
        video_frames: List of frame arrays
        seed_annotations: Detection boxes {frame_idx: [(x,y,w,h), ...]}
        model_checkpoint: Path to SAM2 model weights
        output_path: Where to save results JSON

    Returns:
        Dictionary of masks per frame
    """
    model = load_sam2_model(model_checkpoint)
    masks = propagate_masks(model, video_frames, seed_annotations)
    save_masks_json(masks, output_path)
    return masks
```

**Benefits:**
- One function does one thing
- Clear inputs/outputs
- Easy to test
- Easy to call from Snakemake rule
- No hidden state
- Traceable execution

---

## The Real Problems with Current Codebase

After reading the actual code, here are the **real** issues:

### 1. Massive Monolithic Scripts
- `build03A_process_images.py`: **1753 lines**
- `build04_perform_embryo_qc.py`: **1344 lines**
- `pipeline_objects.py`: **1593 lines**

**Solution:** Extract logical functions, not create wrapper classes.

### 2. Hardcoded Path Discovery
```python
# Current bad pattern in build03A
base_override = os.environ.get("MORPHSEQ_SANDBOX_MASKS_DIR")
base = Path(base_override) if base_override else (root / "segmentation_sandbox" / "data" / "exported_masks")
mask_dir = base / str(date) / "masks"
stub = f"{date}_{well}_ch00_t{int(time_int):04}_masks_emnum_1.png"
candidates = sorted(mask_dir.glob(stub))
```

**Solution:** Snakemake rules define paths via wildcards. Functions just take `Path` arguments.

```python
# Snakemake rule handles paths
rule process_embryo:
    input: "data/masks/{experiment}/{well}_t{time:04d}.png"
    output: "results/snips/{experiment}/{well}_{embryo}_t{time:04d}.png"
    run:
        process_embryo_mask(input[0], output[0])  # Simple!
```

### 3. Duplicated ID Parsing
- In `build03A_process_images.py`
- In `build04_perform_embryo_qc.py`
- In `segmentation_sandbox/scripts/utils/parsing_utils.py` (good version!)
- In multiple other places

**Solution:** One canonical `src/data_pipeline/identifiers/parsing.py` module. Import it everywhere.

### 4. Unnecessary Object Hierarchies
- `GroundedSamAnnotations` class with 20+ methods
- `GroundedDinoAnnotations` class
- `BaseFileHandler` with backup rotation
- `EntityIDTracker` validation

**Solution:** Delete all of these. Use simple functions + Snakemake + Git.

---

## Proposed Simple Structure

### Directory Layout (Flat & Focused)

```
src/data_pipeline/
├── io/
│   ├── load_masks.py          # load_sam2_mask(), load_unet_mask()
│   ├── load_metadata.py        # load_experiment_csv(), load_stage_ref()
│   └── save_results.py         # save_snips(), save_features_csv()
│
├── segmentation/
│   ├── sam2_inference.py       # load_sam2_model(), propagate_masks()
│   ├── unet_inference.py       # load_unet_model(), segment_embryo()
│   ├── mask_formats.py         # encode_rle(), decode_rle(), mask_to_polygon()
│   └── mask_postprocess.py     # clean_mask(), remove_small_objects()
│
├── qc/
│   ├── death_qc.py             # compute_dead_flag2_persistence() [ALREADY EXISTS!]
│   ├── spatial_qc.py           # compute_qc_flags(), compute_fraction_alive() [FROM qc_utils.py]
│   └── tracking_qc.py          # compute_speed(), validate_trajectory()
│
├── features/
│   ├── _morphology_metrics_.py           # compute_area(), compute_perimeter(), compute_shape_metrics()
│   └── stage_inference.py      # infer_stage_from_size(), predict_hpf()
│
├── identifiers/
│   └── parsing.py              # parse_entity_id(), build_snip_id(), extract_frame_number()
│                               # [FROM parsing_utils.py - already well designed!]
│
└── transforms/
    ├── cropping.py             # crop_embryo(), rotate_to_axis(), extract_snip()
    └── alignment.py            # align_to_reference(), compute_angle()
```

### Key Principles

1. **One file = One responsibility**
2. **Functions > Classes** (use classes only for stateful models)
3. **No BaseFileHandler** - Use pandas/json/pathlib directly
4. **No EntityIDTracker** - Snakemake tracks files
5. **No AnnotationStore** - Snakemake tracks dependencies
6. **No dependency injection** - Functions take arguments

---

## File-by-File Migration Plan

### Priority 1: Core Infrastructure (Week 1)

| Current File | Lines | Action | Target Location | Notes |
|-------------|-------|--------|-----------------|-------|
| `segmentation_sandbox/scripts/utils/parsing_utils.py` | ~800 | **Move as-is** | `src/data_pipeline/identifiers/parsing.py` | Already well-designed! Just move it. |
| `src/build/qc_utils.py` | 135 | **Move as-is** | `src/data_pipeline/qc/spatial_qc.py` | Clean pure functions. |
| `src/data_pipeline/quality_control/death_detection.py` | 317 | **Rename** | `src/data_pipeline/qc/death_qc.py` | Already exists and works! |
| `segmentation_sandbox/scripts/utils/mask_utils.py` | ~200 | **Move as-is** | `src/data_pipeline/segmentation/mask_formats.py` | Pure conversion functions. |

**Outcome:** After Week 1, we have working ID parsing, QC flags, death detection, and mask format conversions.

---

### Priority 2: Segmentation Functions (Week 2)

| Current File | Lines | Extract What | Target Location | Kill What |
|-------------|-------|--------------|-----------------|-----------|
| `segmentation_sandbox/scripts/detection_segmentation/sam2_utils.py` | ~600 | `load_sam2_model()`, `run_sam2_propagation()`, `run_bidirectional_propagation()` | `src/data_pipeline/segmentation/sam2_inference.py` | GroundedSamAnnotations class, all entity tracking, annotation store logic |
| `segmentation_sandbox/scripts/detection_segmentation/grounded_dino_utils.py` | ~500 | `load_groundingdino_model()`, `run_inference()` | `src/data_pipeline/segmentation/gdino_inference.py` | GroundedDinoAnnotations class, JSON tracking |
| `src/segmentation/ml_preprocessing/apply_unet.py` | ~200 | `load_unet_model()`, model inference loop | `src/data_pipeline/segmentation/unet_inference.py` | Hardcoded paths, main() CLI logic |

**Strategy:**
- Extract the actual ML inference code (model loading, forward pass)
- Delete all the annotation management classes
- Let Snakemake track which videos need processing

**Example Migration:**

```python
# OLD (from sam2_utils.py) - 600 lines with GroundedSamAnnotations class
class GroundedSamAnnotations:
    def __init__(self, ...):
        self._load_seed_annotations()
        self._load_experiment_metadata()
        ...

    def process_missing_annotations(self):
        for video in self.get_missing_videos():
            self.process_video(video)

    # ... 20 more methods ...

# NEW (in sam2_inference.py) - ~100 lines of pure functions
def load_sam2_model(checkpoint_path: Path, config_path: Path, device: str = "cuda") -> torch.nn.Module:
    """Load SAM2 model from checkpoint."""
    ...

def propagate_masks(
    model: torch.nn.Module,
    frames: list[np.ndarray],
    seed_boxes: dict[int, list[tuple]],
    device: str = "cuda"
) -> dict[int, list[np.ndarray]]:
    """Run SAM2 mask propagation on video frames.

    Args:
        model: Loaded SAM2 model
        frames: List of video frames as numpy arrays
        seed_boxes: {frame_idx: [(x,y,w,h), ...]} bounding boxes
        device: "cuda" or "cpu"

    Returns:
        {frame_idx: [mask1, mask2, ...]} predicted masks per frame
    """
    ...
```

Then in Snakemake:
```python
rule run_sam2:
    input:
        frames="data/videos/{exp}/{video}/frames/",
        seeds="results/gdino/{exp}/{video}/detections.json",
        model="models/sam2_checkpoint.pth"
    output:
        "results/sam2/{exp}/{video}/masks.json"
    run:
        from data_pipeline.segmentation.sam2_inference import load_sam2_model, propagate_masks

        model = load_sam2_model(input.model, config="configs/sam2.yaml")
        frames = load_video_frames(input.frames)
        seeds = json.load(open(input.seeds))

        masks = propagate_masks(model, frames, seeds)
        save_masks_json(masks, output[0])
```

---

### Priority 3: Feature Extraction (Week 3)

| Current File | Lines | Extract What | Target Location | Kill What |
|-------------|-------|--------------|-----------------|-----------|
| `src/build/build03A_process_images.py` | 1753 | Snip cropping logic, feature extraction, morphology calculations | `src/data_pipeline/transforms/cropping.py`, `src/data_pipeline/features/morphology.py` | Everything else - the script becomes a Snakemake rule |
| `src/functions/image_utils.py` | ~300 | `crop_embryo_image()`, `get_embryo_angle()`, `process_masks()` | `src/data_pipeline/transforms/cropping.py` | Duplicated ID parsing |
| `src/functions/spline_morph_spline_metrics.py` | ~400 | `add_pca_components()`, `compute_spline_distances()`, spline fitting | `src/data_pipeline/features/splines.py` | None - looks clean |
| `src/build/infer_developmental_age.py` | ~200 | Stage inference models | `src/data_pipeline/features/stage_inference.py` | Hardcoded paths |

**Strategy:**
- Extract pure feature computation functions
- Keep transform operations (crop, rotate) separate from measurements
- Stage inference functions should just take morphology features as input

---

### Priority 4: Build04 QC Pipeline (Week 4)

| Current File | Lines | Extract What | Target Location | Kill What |
|-------------|-------|--------------|-----------------|-----------|
| `src/build/build04_perform_embryo_qc.py` | 1344 | Stage inference orchestration logic | Becomes Snakemake rules | The entire script - functions already extracted in Week 1-3 |

**Strategy:**
The entire build04 script becomes a Snakemake workflow that calls:
- `src/data_pipeline/qc/death_qc.py::compute_dead_flag2_persistence()` ✓ Already done!
- `src/data_pipeline/qc/spatial_qc.py::compute_qc_flags()` ✓ Already done!
- `src/data_pipeline/features/stage_inference.py::predict_stage()` (extracted in Week 3)
- Simple pandas operations for combining results

Example Snakefile:
```python
rule build04_qc:
    input:
        "results/build03/{exp}/embryo_metadata.csv",
        "metadata/stage_ref_df.csv"
    output:
        "results/build04/{exp}/qc_staged.csv"
    run:
        from data_pipeline.qc.death_qc import compute_dead_flag2_persistence
        from data_pipeline.qc.spatial_qc import compute_qc_flags
        from data_pipeline.features.stage_inference import predict_stage

        df = pd.read_csv(input[0])
        df = predict_stage(df, stage_ref=input[1])
        df = compute_dead_flag2_persistence(df, dead_lead_time=2.0)
        # Add other QC flags...
        df.to_csv(output[0], index=False)
```

---

### Priority 5: Delete Overengineering (Week 5)

**Files to DELETE entirely:**

1. **`src/build/pipeline_objects.py`** (1593 lines)
   - Reason: ExperimentManager is exactly the overengineering we're removing
   - Replacement: Snakemake handles orchestration

2. **`segmentation_sandbox/scripts/utils/base_file_handler.py`**
   - Reason: Backup rotation not needed with Snakemake + Git
   - Replacement: Direct pandas/json I/O

3. **`segmentation_sandbox/scripts/utils/entity_id_tracker.py`**
   - Reason: Snakemake tracks file dependencies
   - Replacement: Snakemake's DAG

4. **`segmentation_sandbox/scripts/utils/experiment_metadata_utils.py`**
   - Reason: Metadata loading should be simple pandas operations
   - Replacement: `src/data_pipeline/io/load_metadata.py` with basic CSV readers

**Classes to DELETE from extracted files:**
- `GroundedSamAnnotations` - keep only the inference functions
- `GroundedDinoAnnotations` - keep only the model loading functions
- `SAM2MetadataExporter` - replace with simple function
- `SimpleMaskExporter` - replace with simple function

---

## What Already Exists and Works!

**Good news:** Some refactoring already started!

### ✅ Already Done Right

1. **`src/data_pipeline/quality_control/death_detection.py`** (317 lines)
   - Clean implementation of death persistence validation
   - Pure functions with clear inputs/outputs
   - Well-documented
   - **Action:** Just rename to `src/data_pipeline/qc/death_qc.py`

2. **`src/build/qc_utils.py`** (135 lines)
   - `compute_fraction_alive()`, `compute_qc_flags()`, `compute_speed()`
   - All pure functions with no side effects
   - **Action:** Move to `src/data_pipeline/qc/spatial_qc.py` as-is

3. **`segmentation_sandbox/scripts/utils/parsing_utils.py`** (~800 lines)
   - Comprehensive ID parsing with backward compatibility
   - Well-documented with examples
   - Already follows single-responsibility principle
   - **Action:** Move to `src/data_pipeline/identifiers/parsing.py` as-is

4. **`segmentation_sandbox/scripts/utils/mask_utils.py`** (~200 lines)
   - Pure mask format conversion functions
   - RLE, polygon, bbox utilities
   - **Action:** Move to `src/data_pipeline/segmentation/mask_formats.py` as-is

**Lesson:** When the code is already good, just move it. Don't "refactor" working code.

---

## Response to Original Plan's Phases

### Original Phase 1: "Package Skeleton & Configuration"
**Proposed:** Create empty modules with docstrings, environment-variable overrides for paths

**My Take:** ❌ Skip the "skeleton" approach. Just move working code.
- Don't create empty files with docstrings
- Don't add environment variable systems
- Snakemake config handles paths via `configfile: "config.yaml"`

### Original Phase 2: "Shared Utilities and IO"
**Proposed:** Consolidate helpers into `data_pipeline/io` and `data_pipeline/utils`, build metadata schema definitions

**My Take:** ⚠️ Partially agree, but simpler
- Move `parsing_utils.py` → done
- Move `qc_utils.py` → done
- **Skip** metadata schema definitions (overengineering)
- **Skip** BaseFileHandler consolidation (delete it instead)

### Original Phase 3: "Segmentation Module"
**Proposed:** Port SAM2 orchestration, define `SAM2SegmentationPipeline` interface class

**My Take:** ❌ No interface classes
- Extract pure inference functions (model loading, propagation)
- Delete GroundedSamAnnotations class hierarchy
- Snakemake orchestrates, not Python classes

### Original Phase 4: "Snip Generation & Feature Engineering"
**Proposed:** Move cropping/rotation to `transforms/snip_generation.py` with pure functions and dataclasses

**My Take:** ✅ YES, but no dataclasses
- Extract cropping/rotation as pure functions
- Extract stage inference as pure functions
- Don't wrap in dataclass input/output objects

### Original Phase 5: "QC Package"
**Proposed:** Collect QC thresholds into composable pipelines with typed dataclasses

**My Take:** ❌ No "composable pipelines" or dataclasses
- QC functions already exist and work (`qc_utils.py`, `death_detection.py`)
- Just move them and call from Snakemake
- Don't create `SegmentationQC`, `EmbryoMetadataQC` classes

### Original Phase 6: "Orchestration & CLI"
**Proposed:** Replace build scripts with thin CLI, Hydra/Lightning integration

**My Take:** ❌ No CLI, no Hydra, no Lightning
- Snakemake **IS** the orchestrator
- No need for Click/argparse CLI wrappers
- No need for Hydra config system (use Snakemake config)
- No need for Lightning integration (what?!)

### Original Phase 7: "Deprecation & Cleanup"
**Proposed:** Convert build scripts to compatibility shims with deprecation warnings

**My Take:** ✅ YES, but simpler
- Just delete `build03A_process_images.py`, `build04_perform_embryo_qc.py`, `pipeline_objects.py`
- Replace with Snakemake rules
- No "compatibility shims" - clean break

---

## Proposed Timeline (Realistic)

### Week 1: Core Infrastructure (No Code Changes)
- Move 4 files as-is to new locations
- Add `__init__.py` files
- Update imports in 2-3 places to verify it works
- **Deliverable:** Core utilities importable from new locations

### Week 2: Extract Segmentation Functions
- Extract SAM2/GDINO inference from classes
- Extract UNET inference
- Delete annotation management classes
- Write simple Snakemake rule for one experiment
- **Deliverable:** Can run SAM2 via Snakemake on test data

### Week 3: Extract Features & Transforms
- Extract snip cropping from build03A
- Extract morphology features
- Extract spline fitting
- Extract stage inference
- **Deliverable:** Can generate features via Snakemake

### Week 4: Build Snakemake Workflow
- Replace build03A with Snakemake rules
- Replace build04 with Snakemake rules
- Test on full experiment
- **Deliverable:** One complete experiment runs end-to-end

### Week 5: Cleanup & Delete
- Delete pipeline_objects.py
- Delete BaseFileHandler, EntityIDTracker
- Delete old build scripts
- Update documentation
- **Deliverable:** Clean, flat codebase

---

## Specific File Assessments Needed

To proceed, I recommend creating detailed analysis `.md` files for these **critical extraction targets:**

### Must Analyze (Complex Extraction)
1. **`build03A_process_images.py`** - 1753 lines, needs careful extraction
2. **`build04_perform_embryo_qc.py`** - 1344 lines, partially done
3. **`sam2_utils.py`** - Extract inference from GroundedSamAnnotations class
4. **`grounded_dino_utils.py`** - Extract inference from GroundedDinoAnnotations class
5. **`apply_unet.py`** - Extract model loading/inference

### Can Move As-Is (Simple)
6. **`parsing_utils.py`** - Already good, just move
7. **`qc_utils.py`** - Already good, just move
8. **`mask_utils.py`** - Already good, just move
9. **`death_detection.py`** - Already good, just rename
10. **`spline_morph_spline_metrics.py`** - Already good, just move

### To Delete (Document Why)
11. **`pipeline_objects.py`** - Why we're deleting ExperimentManager
12. **`base_file_handler.py`** - Why we're deleting backup systems
13. **`entity_id_tracker.py`** - Why we're deleting entity tracking

---

## Key Questions for User

1. **Do you agree with "simple functions > classes" approach?**
   - Or do you want some class structures retained?

2. **Are you comfortable deleting `pipeline_objects.py` entirely?**
   - This is 1593 lines of orchestration code
   - Snakemake replaces it, but it's a big change

3. **Should we keep ANY of the annotation management systems?**
   - Or trust Snakemake to track which files need processing?

4. **UNet models - how are they used today?**
   - `apply_unet.py` seems like old code with hardcoded Windows paths
   - Are UNet models still in production pipeline?
   - Or has SAM2 replaced them?

5. **What about the sandbox pipeline scripts 01-07?**
   - Do these become Snakemake rules?
   - Or are they just for interactive development?

---

## Recommendation

**Start with Week 1 (file moves) to validate the approach:**

1. Move `parsing_utils.py` → `src/data_pipeline/identifiers/parsing.py`
2. Move `qc_utils.py` → `src/data_pipeline/qc/spatial_qc.py`
3. Rename `death_detection.py` → `src/data_pipeline/qc/death_qc.py`
4. Move `mask_utils.py` → `src/data_pipeline/segmentation/mask_formats.py`

Then update imports in 2-3 existing scripts to verify everything works.

**If Week 1 goes well, proceed to Week 2 (extraction).**

**If Week 1 reveals issues, revise the plan.**

---

## Bottom Line

The original plan is **too abstract and over-engineered**. It replicates the same "service object + dependency injection" patterns that made the current codebase hard to maintain.

**What we actually need:**
- **Flat file structure** with clearly-named modules
- **Pure functions** that do one thing well
- **No classes** unless absolutely necessary (ML models only)
- **Let Snakemake orchestrate** - it's literally designed for this
- **Delete the overengineering** - BaseFileHandler, EntityIDTracker, pipeline_objects.py

**Start simple. Move working code. Delete complexity. Write Snakemake rules.**

The goal is to make the codebase **boring and predictable**, not "well-architected."
