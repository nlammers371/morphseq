# MorphSeq Pipeline Stages & Refactor Plan (UPDATED)

**Author:** Claude Code Analysis
**Date:** 2025-10-06
**Status:** READY FOR REVIEW
**Updates:** Corrected based on pipeline_objects.py analysis

---

## Executive Summary

**Pipeline Flow from `pipeline_objects.py`:**
```
Per-Experiment Steps:
  Raw Data → Build01 (Stitch) → Build02 (UNet QC masks) → SAM2 (Optional)
    → Build03 (Snips + Features) → Embeddings (Latents)

Global Steps:
  All Build03 outputs → df01 (combined)
  df01 → Build04 (QC + Staging) → df02 (combined)
  df02 + All Embeddings → Build06 (Final Merge) → df03 (combined)
```

**Key Insight:** UNet (Build02) is **STILL IN PRODUCTION** - used for yolk/bubble/focus/viability masks when SAM2 is not available.

---

## Pipeline Stages (Corrected Based on Actual Code)

### **Stage 1: Image Stitching & Organization** (Build01)
**Experiment-Level Step**

**Input:** Raw microscope files (Keyence or YX1)
**Output:** Stitched FF images + experiment metadata CSV

**Current Implementation:**
- `build01A_compile_keyence_torch.py` - Keyence stitching
- `build01B_compile_yx1_images_torch.py` - YX1 processing
- `build01AB_stitch_keyence_z_slices.py` - Z-slice stitching
- Called by: `Experiment.run_build01()`

**Outputs:**
- `built_image_data/stitched_FF_images/{experiment}/`
- `metadata/built_metadata_files/{experiment}.csv`

**Functions to Extract:**
```python
# src/data_pipeline/stage1_stitching/
- keyence_stitching.py
  - stitch_keyence_tiles()
  - focus_stack_z_slices()
  - align_with_qc()

- yx1_stitching.py
  - build_ff_from_yx1()

- metadata_generation.py
  - scrape_keyence_metadata()
  - build_experiment_metadata()
```

**Snakemake Rule:**
```python
rule build01_stitch:
    input:
        raw="raw_image_data/{microscope}/{experiment}/"
    output:
        images=directory("built_image_data/stitched_FF_images/{experiment}/"),
        metadata="metadata/built_metadata_files/{experiment}.csv"
    run:
        from data_pipeline.stage1_stitching.keyence_stitching import stitch_keyence_tiles
        stitch_keyence_tiles(input.raw, output.images, output.metadata)
```

---

### **Stage 2: UNet Segmentation** (Build02) - **STILL IN PRODUCTION**
**Experiment-Level Step**

**Input:** Stitched FF images
**Output:** UNet-predicted masks (embryo, yolk, viability, focus, bubble)

**Current Implementation:**
- `build02B_segment_bf_main.py::apply_unet()`
- Called by: `Experiment.generate_qc_masks()`
- **Models used:** `mask_v0_0100`, `via_v1_0100`, `yolk_v1_0050`, `focus_v0_0100`, `bubble_v0_0100`

**Outputs:**
- `segmentation/{model_name}_predictions/{experiment}/*.png`

**Functions to Extract:**
```python
# src/data_pipeline/stage2_unet_segmentation/
- unet_inference.py
  - load_unet_model()
  - apply_unet_to_experiment()
  - batch_predict_masks()
```

**Snakemake Rule:**
```python
rule build02_unet:
    input:
        images="built_image_data/stitched_FF_images/{experiment}/"
    output:
        masks=directory("segmentation/unet_predictions/{experiment}/")
    params:
        models=["mask_v0_0100", "via_v1_0100", "yolk_v1_0050", "focus_v0_0100", "bubble_v0_0100"]
    run:
        from data_pipeline.stage2_unet_segmentation.unet_inference import apply_unet_to_experiment

        for model_name in params.models:
            apply_unet_to_experiment(input.images, output.masks, model_name)
```

**Critical Note:** UNet is used **alongside** SAM2, not replaced by it. SAM2 does embryo tracking, UNet does auxiliary masks (yolk, bubbles, focus, viability).

---

### **Stage 3: SAM2 Detection & Tracking Pipeline** (Optional)
**Experiment-Level Step (6 sub-stages)**

**Input:** Stitched FF images
**Output:** SAM2 embryo masks + tracking metadata

This is the sandbox pipeline (scripts 01-07), called by `Experiment.run_sam2()`

#### **Stage 3A: Video Preparation**
**Current:** `segmentation_sandbox/scripts/pipelines/01_prepare_videos.py`
**Extract to:** `src/data_pipeline/stage3_sam2/video_preparation.py`

#### **Stage 3B: Grounded DINO Detection**
**Current:** `segmentation_sandbox/scripts/pipelines/03_gdino_detection.py`
**Extract to:** `src/data_pipeline/stage3_sam2/gdino_detection.py`

#### **Stage 3C: SAM2 Mask Propagation**
**Current:** `segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py`
**Extract to:** `src/data_pipeline/stage3_sam2/sam2_propagation.py`

#### **Stage 3D: SAM2 QC Analysis**
**Current:** `segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py`
**Extract to:** `src/data_pipeline/stage3_sam2/sam2_qc.py`

#### **Stage 3E: Export Masks to PNG**
**Current:** `segmentation_sandbox/scripts/pipelines/06_export_masks.py`
**Extract to:** `src/data_pipeline/stage3_sam2/export_masks.py`

#### **Stage 3F: Flatten JSON to CSV**
**Current:** `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`
**Extract to:** `src/data_pipeline/stage3_sam2/format_csv.py`

**Outputs:**
- `sam2_pipeline_files/exported_masks/{experiment}/masks/*.png`
- `metadata/sam2_output/{experiment}_sam2_metadata.csv`

---

### **Stage 4: Snip Creation & Morphology** (Build03)
**Experiment-Level Step**

**Input:**
- Option A: SAM2 CSV + exported masks (preferred)
- Option B: Build02 UNet masks (legacy fallback)

**Output:** Cropped embryo snips + morphology features CSV → **contributes to df01**

**Current Implementation:**
- `build03A_process_images.py` (1753 lines - MASSIVE)
- Called by: `Experiment.run_build03()`
- Logic: If SAM2 CSV exists, use it; else fall back to Build02 masks

**Outputs:**
- `built_image_data/snips/{experiment}/*.png`
- `metadata/build03_output/expr_embryo_metadata_{experiment}.csv` → contributes to **df01**

**Functions to Extract:**
```python
# src/data_pipeline/stage4_snip_creation/
- cropping.py
  - compute_embryo_angle()  # PCA-based rotation
  - crop_embryo_region()
  - rotate_to_standard_orientation()
  - add_synthetic_noise()  # Training augmentation

- morphology_features.py
  - compute_area()
  - compute_perimeter()
  - compute_centroid()
  - compute_aspect_ratio()
  - extract_contour()

- build03_orchestration.py
  - run_build03_per_experiment()  # Main orchestrator
  - load_masks_from_sam2_or_unet()  # Branching logic
```

**Snakemake Rule:**
```python
rule build03_snips:
    input:
        # Option A: SAM2 path
        sam2_csv="metadata/sam2_output/{experiment}_sam2_metadata.csv",
        sam2_masks="sam2_pipeline_files/exported_masks/{experiment}/",
        # Option B: UNet fallback
        unet_masks="segmentation/unet_predictions/{experiment}/",
        images="built_image_data/stitched_FF_images/{experiment}/"
    output:
        snips=directory("built_image_data/snips/{experiment}/"),
        csv="metadata/build03_output/expr_embryo_metadata_{experiment}.csv"
    run:
        from data_pipeline.stage4_snip_creation.build03_orchestration import run_build03_per_experiment

        # Automatically selects SAM2 if available, else UNet
        run_build03_per_experiment(
            experiment=wildcards.experiment,
            sam2_csv=input.sam2_csv if Path(input.sam2_csv).exists() else None,
            output_dir=output.snips,
            output_csv=output.csv
        )
```

**Critical Notes:**
- Build03 outputs are **per-experiment CSVs**
- These get concatenated into **df01** (global file) by ExperimentManager
- df01 = combined embryo data from ALL experiments

---

### **Stage 5: Embedding Generation** (Latents)
**Experiment-Level Step** - **FOUND IN `src/analyze/gen_embeddings/`**

**Input:** Snip images from Build03
**Output:** Latent embeddings CSV per experiment

**Current Implementation:**
- `src/analyze/gen_embeddings/pipeline_integration.py::ensure_embeddings_for_experiments()`
- Called by: `Experiment.generate_latents(model_name="20241107_ds_sweep01_optimum")`
- Uses Python 3.9 subprocess for legacy model compatibility

**Outputs:**
- `models/legacy/{model_name}/latents/{experiment}_latents.csv`

**Current Code Structure:**
```
src/analyze/gen_embeddings/
├── __init__.py
├── pipeline_integration.py   # Main embedding orchestration
├── subprocess_runner.py       # Python 3.9 subprocess wrapper
├── file_utils.py             # Check existing embeddings
└── cli.py                    # Command-line interface
```

**Action:** This is **ALREADY WELL-ORGANIZED**! Just move to new structure.

**New Location:**
```python
# src/data_pipeline/stage5_embedding_generation/
- embedding_inference.py      # From pipeline_integration.py
- subprocess_wrapper.py        # From subprocess_runner.py
- file_validation.py          # From file_utils.py
- cli.py                      # Keep as-is
```

**Snakemake Rule:**
```python
rule generate_embeddings:
    input:
        snips="built_image_data/snips/{experiment}/",
        build03_csv="metadata/build03_output/expr_embryo_metadata_{experiment}.csv"
    output:
        embeddings="models/legacy/{model_name}/latents/{experiment}_latents.csv"
    params:
        model_name=config["embedding_model"]
    run:
        from data_pipeline.stage5_embedding_generation.embedding_inference import ensure_embeddings_for_experiments

        ensure_embeddings_for_experiments(
            data_root=config["data_root"],
            experiments=[wildcards.experiment],
            model_name=params.model_name
        )
```

**Critical Notes:**
- Embeddings are **per-experiment** (not global)
- Complex logic: subprocess orchestration for Python 3.9 compatibility
- **Keep in separate folder** as requested (complicated logic deserves its own space)

---

### **Stage 6: Global QC & Staging** (Build04)
**GLOBAL Step** - Operates on combined df01

**Input:** df01 (combined embryo data from all Build03 runs)
**Output:** df02 (QC'd + staged embryo data)

**Current Implementation:**
- `build04_perform_embryo_qc.py` (1344 lines)
- Called by: `ExperimentManager.run_build04()`
- Can also run per-experiment via `Experiment.run_build04_per_experiment()`

**Outputs:**
- Per-experiment: `metadata/build04_output/qc_staged_{experiment}.csv`
- **Global:** `metadata/qc_staged_df.csv` (df02)

**Functions to Extract:**
```python
# src/data_pipeline/stage6_embryo_qc/
- stage_inference.py
  - infer_stage_from_morphology()
  - predict_hpf_from_area()
  - apply_stage_reference()

- spatial_qc.py  [ALREADY EXISTS in qc_utils.py]
  - compute_qc_flags()
  - compute_fraction_alive()

- death_detection.py  [ALREADY EXISTS!]
  - compute_dead_flag2_persistence()

- trajectory_smoothing.py
  - compute_speed()
  - smooth_trajectory()  # Savitzky-Golay
  - detect_tracking_errors()

- build04_orchestration.py
  - run_build04_per_experiment()  # Per-experiment version
  - run_build04_global()  # Global df01 → df02
```

**Snakemake Rule (Per-Experiment):**
```python
rule build04_qc_per_experiment:
    input:
        csv="metadata/build03_output/expr_embryo_metadata_{experiment}.csv",
        stage_ref="metadata/stage_ref_df.csv"
    output:
        qc_csv="metadata/build04_output/qc_staged_{experiment}.csv"
    params:
        dead_lead_time=2.0
    run:
        from data_pipeline.stage6_embryo_qc.build04_orchestration import run_build04_per_experiment

        run_build04_per_experiment(
            input_csv=input.csv,
            stage_ref=input.stage_ref,
            output_csv=output.qc_csv,
            dead_lead_time=params.dead_lead_time
        )
```

**Snakemake Rule (Global):**
```python
rule build04_qc_global:
    input:
        df01="metadata/expr_embryo_df.csv",  # Combined from all Build03
        stage_ref="metadata/stage_ref_df.csv"
    output:
        df02="metadata/qc_staged_df.csv"
    run:
        from data_pipeline.stage6_embryo_qc.build04_orchestration import run_build04_global

        run_build04_global(
            df01=input.df01,
            stage_ref=input.stage_ref,
            output_df02=output.df02
        )
```

---

### **Stage 7: Final Dataset Merge** (Build06)
**GLOBAL Step** - Merges df02 + embeddings

**Input:**
- df02 (QC'd embryo data)
- Embeddings from all experiments

**Output:** df03 (final analysis-ready dataset)

**Current Implementation:**
- `src/run_morphseq_pipeline/steps/run_build06_per_exp.py::build06_merge_per_experiment()`
- Called by:
  - Per-experiment: `Experiment.run_build06_per_experiment()`
  - Global: `ExperimentManager.run_build06()`

**Outputs:**
- Per-experiment: `metadata/build06_output/{experiment}_df03.csv`
- **Global:** `metadata/qc_staged_embryo_df_w_emb.csv` (df03)

**Functions to Extract:**
```python
# src/data_pipeline/stage7_final_merge/
- merge_embeddings.py
  - load_embeddings_for_experiments()
  - merge_embeddings_with_qc()
  - validate_final_dataset()

- build06_orchestration.py
  - run_build06_per_experiment()
  - run_build06_global()
```

**Snakemake Rule (Global):**
```python
rule build06_final_merge:
    input:
        df02="metadata/qc_staged_df.csv",
        embeddings=expand("models/legacy/{model}/latents/{exp}_latents.csv",
                         model=config["embedding_model"],
                         exp=EXPERIMENTS)
    output:
        df03="metadata/qc_staged_embryo_df_w_emb.csv"
    run:
        from data_pipeline.stage7_final_merge.merge_embeddings import merge_embeddings_with_qc

        df03 = merge_embeddings_with_qc(
            df02=input.df02,
            embedding_files=input.embeddings
        )
        df03.to_csv(output.df03, index=False)
```

---

## Revised Directory Structure (Stage-Based)

```
src/data_pipeline/

├── stage1_stitching/              # Build01: Raw → Stitched FF images
│   ├── keyence_stitching.py
│   ├── yx1_stitching.py
│   └── metadata_generation.py
│
├── stage2_unet_segmentation/      # Build02: UNet auxiliary masks (STILL IN PRODUCTION)
│   └── unet_inference.py
│
├── stage3_sam2/                   # SAM2 Pipeline (Optional, 6 sub-steps)
│   ├── video_preparation.py      # 01: Organize videos
│   ├── gdino_detection.py         # 03: Grounded DINO
│   ├── sam2_propagation.py        # 04: SAM2 tracking
│   ├── sam2_qc.py                 # 05: Detection QC
│   ├── export_masks.py            # 06: Export to PNG
│   ├── format_csv.py              # Flatten JSON to CSV
│   └── mask_formats.py            # RLE/polygon utils [from mask_utils.py]
│
├── stage4_snip_creation/          # Build03: Masks → Snips + Morphology → df01
│   ├── cropping.py
│   ├── morphology_features.py
│   └── build03_orchestration.py
│
├── stage5_embedding_generation/   # Latents: Snips → Embeddings
│   ├── embedding_inference.py     # IMPORTANT: Separate folder for complex logic
│   ├── subprocess_wrapper.py
│   ├── file_validation.py
│   └── cli.py
│
├── stage6_embryo_qc/              # Build04: df01 → QC + Staging → df02
│   ├── stage_inference.py
│   ├── spatial_qc.py              # [from qc_utils.py]
│   ├── death_detection.py         # [ALREADY EXISTS!]
│   ├── trajectory_smoothing.py
│   └── build04_orchestration.py
│
├── stage7_final_merge/            # Build06: df02 + Embeddings → df03
│   ├── merge_embeddings.py
│   └── build06_orchestration.py
│
├── identifiers/                   # Shared across all stages
│   └── parsing.py                 # [from parsing_utils.py]
│
└── metadata_handler/              # Metadata operations (NOT "metadata")
    └── enrich_metadata.py         # Perturbation merging, etc.
```

**Key Changes Based on Feedback:**
1. ✅ **Stage-based folders** - Clear pipeline progression visible in structure
2. ✅ **`stage5_embedding_generation/` separate** - Complex subprocess logic deserves its own folder
3. ✅ **Renamed `metadata/` → `metadata_handler/`** - Avoid confusion with actual metadata files
4. ✅ **Renamed `features/` → stages 4, 5, 6** - Too vague; split by actual pipeline stages
5. ✅ **No `embryo_features.py`** - That's just morphology in stage4
6. ✅ **Spline fitting NOT in pipeline** - Confirmed; only in analysis notebooks

---

## Updated Migration Plan

### ✅ Week 1: Move Core Utilities (No Extraction)

| Source | Target | Lines | Status |
|--------|--------|-------|--------|
| `segmentation_sandbox/scripts/utils/parsing_utils.py` | `src/data_pipeline/identifiers/parsing.py` | ~800 | Already good |
| `src/build/qc_utils.py` | `src/data_pipeline/stage6_embryo_qc/spatial_qc.py` | 135 | Already good |
| `src/data_pipeline/quality_control/death_detection.py` | `src/data_pipeline/stage6_embryo_qc/death_detection.py` | 317 | Already good |
| `segmentation_sandbox/scripts/utils/mask_utils.py` | `src/data_pipeline/stage3_sam2/mask_formats.py` | ~200 | Already good |
| `src/analyze/gen_embeddings/*.py` | `src/data_pipeline/stage5_embedding_generation/` | ~300 | Already organized! |

**Deliverable:** Core utilities in new locations, all imports working

---

### 🔨 Week 2: Extract Stage 1-2 Functions

**Stage 1 (Stitching):**
- Extract from `build01A_compile_keyence_torch.py`
- Extract from `build01B_compile_yx1_images_torch.py`
- Target: `stage1_stitching/*.py`

**Stage 2 (UNet):**
- Extract from `build02B_segment_bf_main.py`
- Target: `stage2_unet_segmentation/unet_inference.py`
- **Keep in production** - not deprecated!

**Create Snakemake rules for Stages 1-2**

**Deliverable:** Can stitch images and run UNet via Snakemake

---

### 🔨 Week 3: Extract Stage 3 (SAM2 Pipeline)

**Extract from sandbox pipeline scripts:**
- `01_prepare_videos.py` → `stage3_sam2/video_preparation.py`
- `03_gdino_detection.py` → `stage3_sam2/gdino_detection.py`
- `04_sam2_video_processing.py` → `stage3_sam2/sam2_propagation.py`
- `05_sam2_qc_analysis.py` → `stage3_sam2/sam2_qc.py`
- `06_export_masks.py` → `stage3_sam2/export_masks.py`
- `export_sam2_metadata_to_csv.py` → `stage3_sam2/format_csv.py`

**Delete:**
- `GroundedSamAnnotations` class
- `GroundedDinoAnnotations` class
- `SimpleMaskExporter` class
- `SAM2MetadataExporter` class
- `BaseFileHandler` usage
- Entity tracking systems

**Create Snakemake rules for Stage 3 (6 sub-rules)**

**Deliverable:** SAM2 pipeline runs via Snakemake

---

### 🔨 Week 4: Extract Stage 4-6 Functions

**Stage 4 (Build03):**
- Extract from `build03A_process_images.py` (1753 lines → ~200 lines in 3 files)
- Target: `stage4_snip_creation/*.py`

**Stage 6 (Build04):**
- Extract from `build04_perform_embryo_qc.py` (1344 lines → ~150 lines in 5 files)
- Target: `stage6_embryo_qc/*.py`

**Create Snakemake rules for Stages 4 & 6**

**Deliverable:** Build03 and Build04 run via Snakemake

---

### 🔨 Week 5: Extract Stage 7 & Cleanup

**Stage 7 (Build06):**
- Extract from `run_build06_per_exp.py`
- Target: `stage7_final_merge/*.py`

**Delete Overengineering:**
- `pipeline_objects.py` (1593 lines) → Replaced by Snakemake
- `base_file_handler.py` → Not needed
- `entity_id_tracker.py` → Not needed
- All sandbox scripts 01-07 → Become Snakemake rules
- `build03A_process_images.py` → After extraction
- `build04_perform_embryo_qc.py` → After extraction

**Create full Snakemake workflow (Stages 1-7)**

**Deliverable:** Complete end-to-end Snakemake pipeline, clean codebase

---

## Snakemake Workflow (Full Pipeline)

```python
# Snakefile

configfile: "config.yaml"

EXPERIMENTS = config["experiments"]
MODEL_NAME = config["embedding_model"]

rule all:
    input:
        # Global final output
        "metadata/qc_staged_embryo_df_w_emb.csv"

# ========== STAGE 1: STITCHING ==========
rule stage1_stitch_keyence:
    input: "raw_image_data/Keyence/{experiment}/"
    output:
        images=directory("built_image_data/stitched_FF_images/{experiment}/"),
        meta="metadata/built_metadata_files/{experiment}.csv"
    run:
        from data_pipeline.stage1_stitching.keyence_stitching import stitch_keyence_tiles
        stitch_keyence_tiles(input[0], output.images, output.meta)

# ========== STAGE 2: UNET SEGMENTATION ==========
rule stage2_unet_masks:
    input: "built_image_data/stitched_FF_images/{experiment}/"
    output: directory("segmentation/unet_predictions/{experiment}/")
    params:
        models=["mask_v0_0100", "via_v1_0100", "yolk_v1_0050", "focus_v0_0100", "bubble_v0_0100"]
    run:
        from data_pipeline.stage2_unet_segmentation.unet_inference import apply_unet_to_experiment
        for model in params.models:
            apply_unet_to_experiment(input[0], output[0], model)

# ========== STAGE 3: SAM2 PIPELINE (OPTIONAL) ==========
rule stage3a_prepare_videos:
    input: "built_image_data/stitched_FF_images/{experiment}/"
    output:
        videos=directory("sam2_pipeline_files/videos/{experiment}/"),
        meta="sam2_pipeline_files/{experiment}/video_metadata.json"
    run:
        from data_pipeline.stage3_sam2.video_preparation import organize_images_to_videos
        organize_images_to_videos(input[0], output.videos, output.meta)

rule stage3b_gdino_detection:
    input:
        videos="sam2_pipeline_files/videos/{experiment}/",
        model=config["gdino_model"]
    output: "sam2_pipeline_files/{experiment}/gdino_detections.json"
    run:
        from data_pipeline.stage3_sam2.gdino_detection import detect_embryos
        detect_embryos(input.videos, input.model, output[0])

rule stage3c_sam2_propagation:
    input:
        videos="sam2_pipeline_files/videos/{experiment}/",
        detections="sam2_pipeline_files/{experiment}/gdino_detections.json",
        model=config["sam2_model"]
    output: "sam2_pipeline_files/{experiment}/sam2_results.json"
    run:
        from data_pipeline.stage3_sam2.sam2_propagation import propagate_masks
        propagate_masks(input.videos, input.detections, input.model, output[0])

rule stage3d_sam2_qc:
    input: "sam2_pipeline_files/{experiment}/sam2_results.json"
    output: "sam2_pipeline_files/{experiment}/sam2_results_qc.json"
    run:
        from data_pipeline.stage3_sam2.sam2_qc import add_detection_qc_flags
        add_detection_qc_flags(input[0], output[0])

rule stage3e_export_masks:
    input: "sam2_pipeline_files/{experiment}/sam2_results_qc.json"
    output:
        masks=directory("sam2_pipeline_files/exported_masks/{experiment}/"),
        manifest="sam2_pipeline_files/{experiment}/export_manifest.csv"
    run:
        from data_pipeline.stage3_sam2.export_masks import export_labeled_masks
        export_labeled_masks(input[0], output.masks, output.manifest)

rule stage3f_flatten_csv:
    input:
        masks="sam2_pipeline_files/{experiment}/sam2_results_qc.json",
        manifest="sam2_pipeline_files/{experiment}/export_manifest.csv"
    output: "metadata/sam2_output/{experiment}_sam2_metadata.csv"
    run:
        from data_pipeline.stage3_sam2.format_csv import flatten_sam2_to_csv
        flatten_sam2_to_csv(input.masks, input.manifest, output[0])

# ========== STAGE 4: SNIP CREATION ==========
rule stage4_build03_snips:
    input:
        sam2_csv="metadata/sam2_output/{experiment}_sam2_metadata.csv",
        sam2_masks="sam2_pipeline_files/exported_masks/{experiment}/",
        unet_masks="segmentation/unet_predictions/{experiment}/",
        images="built_image_data/stitched_FF_images/{experiment}/"
    output:
        snips=directory("built_image_data/snips/{experiment}/"),
        csv="metadata/build03_output/expr_embryo_metadata_{experiment}.csv"
    run:
        from data_pipeline.stage4_snip_creation.build03_orchestration import run_build03_per_experiment

        # Use SAM2 if available, else UNet
        sam2_path = input.sam2_csv if Path(input.sam2_csv).exists() else None

        run_build03_per_experiment(
            experiment=wildcards.experiment,
            sam2_csv=sam2_path,
            unet_masks=input.unet_masks,
            images=input.images,
            output_snips=output.snips,
            output_csv=output.csv
        )

# ========== STAGE 5: EMBEDDING GENERATION ==========
rule stage5_generate_embeddings:
    input:
        snips="built_image_data/snips/{experiment}/",
        csv="metadata/build03_output/expr_embryo_metadata_{experiment}.csv"
    output:
        embeddings=f"models/legacy/{MODEL_NAME}/latents/{{experiment}}_latents.csv"
    run:
        from data_pipeline.stage5_embedding_generation.embedding_inference import ensure_embeddings_for_experiments

        ensure_embeddings_for_experiments(
            data_root=config["data_root"],
            experiments=[wildcards.experiment],
            model_name=MODEL_NAME
        )

# ========== STAGE 6: EMBRYO QC (PER-EXPERIMENT) ==========
rule stage6_build04_per_experiment:
    input:
        csv="metadata/build03_output/expr_embryo_metadata_{experiment}.csv",
        stage_ref="metadata/stage_ref_df.csv"
    output: "metadata/build04_output/qc_staged_{experiment}.csv"
    params:
        dead_lead_time=2.0
    run:
        from data_pipeline.stage6_embryo_qc.build04_orchestration import run_build04_per_experiment

        run_build04_per_experiment(
            input_csv=input.csv,
            stage_ref=input.stage_ref,
            output_csv=output[0],
            dead_lead_time=params.dead_lead_time
        )

# ========== STAGE 6: EMBRYO QC (GLOBAL) ==========
rule stage6_build04_global:
    input:
        per_exp_csvs=expand("metadata/build03_output/expr_embryo_metadata_{exp}.csv", exp=EXPERIMENTS),
        stage_ref="metadata/stage_ref_df.csv"
    output: "metadata/qc_staged_df.csv"
    run:
        from data_pipeline.stage6_embryo_qc.build04_orchestration import run_build04_global

        # Concatenate all Build03 outputs into df01
        df01 = pd.concat([pd.read_csv(f) for f in input.per_exp_csvs])

        run_build04_global(df01, input.stage_ref, output[0])

# ========== STAGE 7: FINAL MERGE (GLOBAL) ==========
rule stage7_build06_final:
    input:
        df02="metadata/qc_staged_df.csv",
        embeddings=expand(f"models/legacy/{MODEL_NAME}/latents/{{exp}}_latents.csv", exp=EXPERIMENTS)
    output: "metadata/qc_staged_embryo_df_w_emb.csv"
    run:
        from data_pipeline.stage7_final_merge.merge_embeddings import merge_embeddings_with_qc

        df03 = merge_embeddings_with_qc(
            df02=input.df02,
            embedding_files=input.embeddings
        )
        df03.to_csv(output[0], index=False)
```

---

## Config File

```yaml
# config.yaml

experiments:
  - "20240418"
  - "20250612_30hpf_ctrl_atf6"

data_root: "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"

models:
  gdino_model: "models/groundingdino_checkpoint.pth"
  sam2_model: "models/sam2_checkpoint.pth"
  embedding_model: "20241107_ds_sweep01_optimum"

params:
  gdino_confidence: 0.3
  crop_radius_um: 150
  dead_lead_time: 2.0
```

---

## Summary of Changes Based on Feedback

### ✅ Corrected Misunderstandings

1. **UNet IS still in production** - Used for yolk/bubble/focus/viability masks
2. **Embedding generation EXISTS** - In `src/analyze/gen_embeddings/`
3. **Splines NOT in pipeline** - Only in analysis notebooks (confirmed)

### ✅ Structural Improvements

1. **Stage-based folder naming** - Clear pipeline progression
2. **`stage5_embedding_generation/` separate** - Complex logic deserves own space
3. **Renamed `metadata/` → `metadata_handler/`** - Avoid confusion
4. **Split vague "features"** - Now split across stages 4, 5, 6

### ✅ Kept Simple

1. **No overengineering** - Functions over classes
2. **Let Snakemake orchestrate** - Delete pipeline_objects.py
3. **Move working code first** - Don't refactor what works (death_qc, parsing, etc.)

---

## Next Steps

**If this plan is approved:**

Create detailed extraction guides for the 10 critical files:
1. `build01A_compile_keyence_torch.py` (Stage 1)
2. `build02B_segment_bf_main.py` (Stage 2)
3. `01_prepare_videos.py` through `export_sam2_metadata_to_csv.py` (Stage 3, 6 files)
4. `build03A_process_images.py` (Stage 4)
5. `build04_perform_embryo_qc.py` (Stage 6)
6. `run_build06_per_exp.py` (Stage 7)

Each guide will show:
- Exact functions to extract
- Where they go
- What to delete
- Example refactored code

**Ready to proceed?**
