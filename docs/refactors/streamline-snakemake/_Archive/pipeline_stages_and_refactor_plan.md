# MorphSeq Pipeline Stages & Refactor Plan

**Author:** Claude Code Analysis
**Date:** 2025-10-06
**Status:** DRAFT FOR REVIEW

---

## Pipeline Overview: Experiment-Level Processing

**Key Insight:** Snakemake operates at **experiment level**, not individual embryo/frame level. Each stage processes an entire experiment and outputs metadata CSVs or file collections.

---

## Current Pipeline Stages (As Implemented)

### **Stage 0: Raw Data Acquisition**
**Input:** Keyence/YX1 microscope raw files (`.nd2`, `.tif`, etc.)
**Output:** Unprocessed image stacks

**Current Implementation:**
- Manual data transfer to shared storage
- No automated processing

**Refactor Action:**
- ✅ Keep as-is (manual step)
- Document expected input structure

---

### **Stage 1: Image Stitching & Organization**
**Input:** Raw microscope files per experiment
**Output:** Stitched FF images organized by experiment/well/timepoint

**Current Implementation:**
- `build01A_compile_keyence_torch.py` - Keyence stitching with z-stack focus
- `build01B_compile_yx1_images_torch.py` - YX1 processing
- `build01AB_stitch_keyence_z_slices.py` - Z-slice stitching

**Outputs:**
- `built_image_data/stitched_FF_images/{experiment_id}/{well}_t{time}.png`
- `metadata/experiment_metadata/{experiment_id}.csv` (basic metadata)

**Functions to Extract:**
```python
# src/data_pipeline/preprocessing/
- stitching.py
  - stitch_keyence_tiles()
  - stitch_yx1_tiles()
  - focus_stack_z_slices()

- metadata_init.py
  - scrape_keyence_metadata()
  - build_experiment_metadata()
  - generate_image_ids()
```

**Snakemake Rule:**
```python
rule stitch_experiment:
    input:
        raw_dir="raw_data/{experiment}/"
    output:
        images="built_image_data/stitched_FF_images/{experiment}/",
        metadata="metadata/experiment_metadata/{experiment}.csv"
    run:
        from data_pipeline.preprocessing.stitching import stitch_keyence_tiles
        from data_pipeline.preprocessing.metadata_init import build_experiment_metadata

        stitch_keyence_tiles(input.raw_dir, output.images)
        build_experiment_metadata(output.images, output.metadata)
```

**Critical Notes:**
- Already has good functional separation
- Keep focus stacking logic (LoG_focus_stacker)
- Extract metadata scraping cleanly

---

### **Stage 2: Legacy UNet Segmentation (Optional/Deprecated?)**
**Input:** Stitched FF images
**Output:** UNet-predicted embryo/yolk/viability masks

**Current Implementation:**
- `build02B_segment_bf_main.py` - Applies pre-trained UNet models
- `src/segmentation/ml_preprocessing/apply_unet.py` - UNet inference

**Status:** ⚠️ **UNCLEAR IF STILL USED**
- Code has Windows hardcoded paths (`E:\\Nick\\Dropbox...`)
- May be replaced by SAM2 pipeline
- Only used for yolk masks in build03A?

**Question for User:**
- Is UNet still in production pipeline?
- Or has SAM2 completely replaced it?
- If still used, only for yolk detection?

**Proposed Action:**
- If deprecated → Delete entirely
- If still used → Extract to `src/data_pipeline/segmentation/unet_inference.py`

---

### **Stage 3: SAM2 Pipeline (Detection → Tracking → Mask Generation)**

This is actually **3 sub-stages** in the sandbox pipeline:

#### **Stage 3A: Video Preparation & Metadata Bootstrap**
**Input:** Stitched FF images
**Output:** Organized video frames + experiment metadata JSON

**Current Implementation:**
- `segmentation_sandbox/scripts/pipelines/01_prepare_videos.py`

**Outputs:**
- `data/raw_data_organized/{experiment}/videos/{video_id}/frames/*.jpg`
- `data/experiments_metadata.json` (entity tracking)

**Functions to Extract:**
```python
# src/data_pipeline/segmentation/
- video_prep.py
  - organize_images_to_videos()
  - create_video_frames()
```

**Snakemake Rule:**
```python
rule prepare_sam2_videos:
    input:
        images="built_image_data/stitched_FF_images/{experiment}/"
    output:
        videos=directory("sam2_data/{experiment}/videos/"),
        metadata="sam2_data/{experiment}/video_metadata.json"
    run:
        from data_pipeline.segmentation.video_prep import organize_images_to_videos
        organize_images_to_videos(input.images, output.videos, output.metadata)
```

#### **Stage 3B: Grounded DINO Detection**
**Input:** Video frames
**Output:** Bounding box detections per frame (seed annotations)

**Current Implementation:**
- `segmentation_sandbox/scripts/pipelines/03_gdino_detection.py`
- Uses `GroundedDinoAnnotations` class (TO BE DELETED)

**Outputs:**
- `data/gdino_annotations/{experiment}.json` (detection boxes)

**Functions to Extract:**
```python
# src/data_pipeline/segmentation/
- gdino_inference.py
  - load_gdino_model()
  - detect_embryos()
  - filter_detections()  # confidence thresholds, NMS
```

**Snakemake Rule:**
```python
rule detect_embryos_gdino:
    input:
        videos="sam2_data/{experiment}/videos/",
        model="models/groundingdino_checkpoint.pth"
    output:
        detections="sam2_data/{experiment}/gdino_detections.json"
    params:
        confidence=0.3,
        box_threshold=0.25
    run:
        from data_pipeline.segmentation.gdino_inference import load_gdino_model, detect_embryos

        model = load_gdino_model(input.model)
        detections = detect_embryos(input.videos, model,
                                   confidence=params.confidence,
                                   box_threshold=params.box_threshold)
        save_json(detections, output.detections)
```

#### **Stage 3C: SAM2 Mask Propagation**
**Input:** Video frames + GDINO seed boxes
**Output:** Segmentation masks per embryo per frame

**Current Implementation:**
- `segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py`
- Uses `GroundedSamAnnotations` class (TO BE DELETED)

**Outputs:**
- `data/gsam_results/{experiment}.json` (masks in RLE format + metadata)

**Functions to Extract:**
```python
# src/data_pipeline/segmentation/
- sam2_inference.py
  - load_sam2_model()
  - propagate_masks_forward()
  - propagate_masks_bidirectional()
  - assign_embryo_ids()  # Track embryos across frames
```

**Snakemake Rule:**
```python
rule sam2_propagation:
    input:
        videos="sam2_data/{experiment}/videos/",
        detections="sam2_data/{experiment}/gdino_detections.json",
        model="models/sam2_checkpoint.pth"
    output:
        masks="sam2_data/{experiment}/sam2_results.json"
    run:
        from data_pipeline.segmentation.sam2_inference import load_sam2_model, propagate_masks_forward

        model = load_sam2_model(input.model)
        frames = load_video_frames(input.videos)
        seeds = load_json(input.detections)

        masks = propagate_masks_forward(model, frames, seeds)
        save_json(masks, output.masks)
```

#### **Stage 3D: SAM2 QC Analysis**
**Input:** SAM2 masks JSON
**Output:** QC flags added to masks (detection quality metrics)

**Current Implementation:**
- `segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py` (76KB!)
- Adds detection-level QC: mask area, bbox consistency, tracking quality

**Outputs:**
- Updated `data/gsam_results/{experiment}.json` with QC flags

**Functions to Extract:**
```python
# src/data_pipeline/qc/
- detection_qc.py
  - check_mask_area()
  - check_bbox_consistency()
  - check_tracking_continuity()
  - flag_low_confidence_detections()
```

**Snakemake Rule:**
```python
rule sam2_detection_qc:
    input:
        masks="sam2_data/{experiment}/sam2_results.json"
    output:
        qc_masks="sam2_data/{experiment}/sam2_results_qc.json"
    run:
        from data_pipeline.qc.detection_qc import add_detection_qc_flags

        masks = load_json(input.masks)
        qc_masks = add_detection_qc_flags(masks)
        save_json(qc_masks, output.qc_masks)
```

#### **Stage 3E: Export Masks to PNG**
**Input:** SAM2 masks JSON
**Output:** Integer-labeled PNG masks per frame

**Current Implementation:**
- `segmentation_sandbox/scripts/pipelines/06_export_masks.py`
- Uses `SimpleMaskExporter` class (TO BE DELETED)

**Outputs:**
- `data/exported_masks/{experiment}/masks/{well}_t{time}_masks_emnum_{N}.png`
- Metadata manifest tracking exports

**Functions to Extract:**
```python
# src/data_pipeline/io/
- export_masks.py
  - decode_rle_to_array()
  - create_labeled_mask_png()  # Integer labels per embryo
  - save_mask_png()
```

**Snakemake Rule:**
```python
rule export_sam2_masks:
    input:
        masks="sam2_data/{experiment}/sam2_results_qc.json"
    output:
        mask_dir=directory("sam2_data/{experiment}/exported_masks/"),
        manifest="sam2_data/{experiment}/export_manifest.csv"
    run:
        from data_pipeline.io.export_masks import export_labeled_masks

        export_labeled_masks(input.masks, output.mask_dir, output.manifest)
```

#### **Stage 3F: Flatten SAM2 JSON to CSV**
**Input:** SAM2 masks JSON
**Output:** Row-per-embryo-per-frame CSV with mask paths

**Current Implementation:**
- `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`
- Flattens nested JSON to tabular format

**Outputs:**
- `metadata/sam2_output/{experiment}_sam2_metadata.csv`
- Columns: `experiment_id, video_id, embryo_id, image_id, time_int, exported_mask_path, bbox, area, ...`

**Functions to Extract:**
```python
# src/data_pipeline/io/
- format_sam2_csv.py
  - flatten_sam2_json_to_rows()
  - validate_csv_schema()
```

**Snakemake Rule:**
```python
rule flatten_sam2_to_csv:
    input:
        masks="sam2_data/{experiment}/sam2_results_qc.json",
        manifest="sam2_data/{experiment}/export_manifest.csv"
    output:
        csv="metadata/sam2_output/{experiment}_sam2_metadata.csv"
    run:
        from data_pipeline.io.format_sam2_csv import flatten_sam2_json_to_rows

        masks = load_json(input.masks)
        manifest = pd.read_csv(input.manifest)

        df = flatten_sam2_json_to_rows(masks, manifest)
        df.to_csv(output.csv, index=False)
```

---

### **Stage 4: Metadata Enrichment**
**Input:** SAM2 CSV + experiment metadata
**Output:** Enriched CSV with perturbation info, genotype, etc.

**Current Implementation:**
- `segmentation_sandbox/scripts/pipelines/07_embryo_metadata_update.py`
- Merges master perturbation metadata

**Functions to Extract:**
```python
# src/data_pipeline/metadata/
- enrich_metadata.py
  - merge_perturbation_metadata()
  - add_genotype_phenotype_mapping()
  - standardize_column_names()
```

**Snakemake Rule:**
```python
rule enrich_metadata:
    input:
        sam2_csv="metadata/sam2_output/{experiment}_sam2_metadata.csv",
        master_meta="metadata/master_perturbation.csv"
    output:
        enriched="metadata/sam2_enriched/{experiment}_metadata.csv"
    run:
        from data_pipeline.metadata.enrich_metadata import merge_perturbation_metadata

        df = pd.read_csv(input.sam2_csv)
        master = pd.read_csv(input.master_meta)

        enriched = merge_perturbation_metadata(df, master)
        enriched.to_csv(output.enriched, index=False)
```

---

### **Stage 5: Snip Creation & Morphology Feature Extraction**
**Input:** SAM2 enriched CSV + exported masks + stitched images
**Output:** Cropped/rotated snip images + morphology features CSV

**Current Implementation:**
- `build03A_process_images.py` (1753 lines - MASSIVE)

**What it does:**
1. Loads SAM2 masks (integer-labeled PNGs)
2. Loads stitched FF images
3. Computes embryo angle (PCA-based rotation)
4. Crops embryo region with padding
5. Rotates to standard orientation
6. Adds synthetic noise for augmentation (training data)
7. Computes morphology: area, perimeter, centroid, etc.
8. Saves snips as images
9. Outputs CSV with morphology + snip paths

**Outputs:**
- `built_image_data/snips/{experiment}/{embryo_id}_t{time}.png`
- `metadata/build03_output/expr_embryo_metadata_{experiment}.csv`

**Functions to Extract:**
```python
# src/data_pipeline/transforms/
- cropping.py
  - compute_embryo_angle()  # PCA-based
  - crop_embryo_region()
  - rotate_to_axis()
  - add_synthetic_noise()  # For training augmentation

# src/data_pipeline/features/
- morphology.py
  - compute_area()
  - compute_perimeter()
  - compute_centroid()
  - compute_aspect_ratio()
  - extract_contour()
```

**Snakemake Rule:**
```python
rule create_snips_and_features:
    input:
        csv="metadata/sam2_enriched/{experiment}_metadata.csv",
        masks="sam2_data/{experiment}/exported_masks/",
        images="built_image_data/stitched_FF_images/{experiment}/"
    output:
        snips=directory("built_image_data/snips/{experiment}/"),
        features="metadata/build03_output/expr_embryo_metadata_{experiment}.csv"
    params:
        crop_radius_um=150,
        output_size=(320, 576)
    run:
        from data_pipeline.transforms.cropping import crop_and_rotate_embryo
        from data_pipeline.features.morphology import compute_morphology_features

        df = pd.read_csv(input.csv)

        # Process each row: crop snip + compute features
        results = []
        for idx, row in df.iterrows():
            mask = load_mask(row, input.masks)
            image = load_image(row, input.images)

            snip, rotation_angle = crop_and_rotate_embryo(
                image, mask,
                radius_um=params.crop_radius_um,
                output_size=params.output_size
            )

            features = compute_morphology_features(mask)

            save_snip(snip, output.snips, row)
            results.append({**row, **features, 'rotation_angle': rotation_angle})

        pd.DataFrame(results).to_csv(output.features, index=False)
```

**Critical Notes:**
- This is the MOST BLOATED stage (1753 lines!)
- Extract pure transform functions (crop, rotate)
- Extract pure feature computation (area, perimeter)
- Kill all the hardcoded path discovery logic
- Kill duplicate ID parsing

---

### **Stage 6: Embryo-Level QC & Death Detection**
**Input:** Build03 morphology CSV
**Output:** QC-flagged CSV with death detection and stage inference

**Current Implementation:**
- `build04_perform_embryo_qc.py` (1344 lines)
- Already uses `src/data_pipeline/quality_control/death_detection.py` ✓

**What it does:**
1. Load morphology features from Build03
2. Infer developmental stage (HPF) from size/time
3. Compute spatial QC flags (frame, yolk, bubble, focus)
4. Compute death flags (dead_flag, dead_flag2 with persistence)
5. Compute speed/trajectory metrics
6. Apply Savitzky-Golay smoothing to trajectories
7. Output final QC CSV

**Outputs:**
- `metadata/build04_output/qc_staged_{experiment}.csv`

**Functions to Extract:**
```python
# src/data_pipeline/features/
- stage_inference.py
  - infer_stage_from_morphology()
  - predict_hpf_from_area()
  - apply_stage_reference()

# src/data_pipeline/qc/
- spatial_qc.py  [ALREADY EXISTS in qc_utils.py]
  - compute_qc_flags()
  - compute_fraction_alive()

- death_qc.py  [ALREADY EXISTS!]
  - compute_dead_flag2_persistence()

- tracking_qc.py
  - compute_speed()
  - smooth_trajectory()  # Savitzky-Golay
  - detect_tracking_errors()
```

**Snakemake Rule:**
```python
rule embryo_qc:
    input:
        features="metadata/build03_output/expr_embryo_metadata_{experiment}.csv",
        stage_ref="metadata/stage_ref_df.csv"
    output:
        qc_csv="metadata/build04_output/qc_staged_{experiment}.csv"
    params:
        dead_lead_time=2.0,
        sg_window=5
    run:
        from data_pipeline.features.stage_inference import infer_stage_from_morphology
        from data_pipeline.qc.spatial_qc import compute_qc_flags
        from data_pipeline.qc.death_qc import compute_dead_flag2_persistence
        from data_pipeline.qc.tracking_qc import compute_speed, smooth_trajectory

        df = pd.read_csv(input.features)

        # Stage inference
        df = infer_stage_from_morphology(df, input.stage_ref)

        # Spatial QC flags
        df = compute_qc_flags(df)

        # Death detection
        df = compute_dead_flag2_persistence(df, dead_lead_time=params.dead_lead_time)

        # Trajectory QC
        df = compute_speed(df)
        df = smooth_trajectory(df, window=params.sg_window)

        df.to_csv(output.qc_csv, index=False)
```

**Critical Notes:**
- Already partially refactored (death_detection.py exists!)
- Stage inference is scattered - needs extraction
- QC flags already clean (qc_utils.py)

---

### **Stage 7: Embryo Embeddings Generation**
**Input:** QC'd snip images
**Output:** Learned embeddings (VAE/autoencoder latent space)

**Current Implementation:**
- ⚠️ **NOT FOUND IN CURRENT CODEBASE**
- May be in separate training/inference scripts
- Possibly in `results/` notebooks?

**Question for User:**
- Where is embedding generation implemented?
- Is this a separate ML training pipeline?
- Do embeddings get merged back into metadata CSV?

**Proposed Action:**
```python
# src/data_pipeline/features/
- embeddings.py
  - load_embedding_model()
  - generate_embeddings()
  - add_embeddings_to_csv()
```

**Snakemake Rule:**
```python
rule generate_embeddings:
    input:
        snips="built_image_data/snips/{experiment}/",
        qc_csv="metadata/build04_output/qc_staged_{experiment}.csv",
        model="models/vae_embeddings.pth"
    output:
        embeddings="metadata/embeddings/{experiment}_embeddings.csv"
    run:
        from data_pipeline.features.embeddings import generate_embeddings

        df = pd.read_csv(input.qc_csv)
        embeddings = generate_embeddings(input.snips, input.model)

        df_with_embeddings = pd.merge(df, embeddings, on='snip_id')
        df_with_embeddings.to_csv(output.embeddings, index=False)
```

---

## Critical Observations

### ✅ What's Already Good

1. **Death detection is clean!**
   - `src/data_pipeline/quality_control/death_detection.py` (317 lines)
   - Well-documented, pure functions
   - Just needs to move to `qc/death_qc.py`

2. **Spatial QC is clean!**
   - `src/build/qc_utils.py` (135 lines)
   - Pure functions: `compute_qc_flags()`, `compute_fraction_alive()`, `compute_speed()`
   - Just needs to move to `qc/spatial_qc.py`

3. **ID parsing is well-designed!**
   - `segmentation_sandbox/scripts/utils/parsing_utils.py` (~800 lines)
   - Comprehensive, backward-compatible
   - Just needs to move to `identifiers/parsing.py`

4. **Mask utilities are pure functions!**
   - `segmentation_sandbox/scripts/utils/mask_utils.py` (~200 lines)
   - RLE encoding/decoding, format conversion
   - Just needs to move to `segmentation/mask_formats.py`

### ❌ What Needs Major Refactoring

1. **build03A_process_images.py** (1753 lines)
   - Monolithic snip generation + feature extraction
   - Hardcoded path discovery
   - Duplicate ID parsing
   - **Action:** Extract ~10 pure functions to `transforms/` and `features/`

2. **build04_perform_embryo_qc.py** (1344 lines)
   - Already using extracted death_qc.py ✓
   - Still has embedded stage inference logic
   - **Action:** Extract stage inference to `features/stage_inference.py`

3. **SAM2 annotation classes** (GroundedSamAnnotations, GroundedDinoAnnotations)
   - Over-engineered class hierarchies
   - Mix inference logic with state management
   - **Action:** Extract pure inference functions, delete classes

4. **Unnecessary systems:**
   - `BaseFileHandler` - backup rotation not needed
   - `EntityIDTracker` - Snakemake tracks dependencies
   - `pipeline_objects.py` - ExperimentManager replaced by Snakemake
   - **Action:** DELETE

---

## Proposed Simplified Directory Structure

```
src/data_pipeline/

├── preprocessing/           # Stage 1: Raw → Stitched
│   ├── stitching.py        # Tile stitching, focus stacking
│   └── metadata_init.py    # Scrape microscope metadata
│
├── segmentation/            # Stages 3A-3F: Detection → Masks
│   ├── video_prep.py       # Organize images into video structure
│   ├── gdino_inference.py  # Grounded DINO detection
│   ├── sam2_inference.py   # SAM2 mask propagation
│   ├── unet_inference.py   # [Optional] Legacy UNet segmentation
│   └── mask_formats.py     # RLE/polygon conversion [from mask_utils.py]
│
├── transforms/              # Stage 5: Cropping & Alignment
│   ├── cropping.py         # Crop embryo regions
│   └── alignment.py        # PCA rotation, angle computation
│
├── features/                # Stages 5 & 7: Feature Extraction
│   ├── morphology.py       # Area, perimeter, shape metrics
│   ├── stage_inference.py  # HPF prediction from size
│   ├── splines.py          # Spline fitting [from spline_morph_spline_metrics.py]
│   └── embeddings.py       # [Future] VAE embeddings
│
├── qc/                      # Stage 6: Quality Control
│   ├── spatial_qc.py       # Frame/bubble/yolk/focus flags [from qc_utils.py]
│   ├── death_qc.py         # Death persistence [ALREADY EXISTS!]
│   ├── detection_qc.py     # SAM2 detection quality
│   └── tracking_qc.py      # Speed, trajectory smoothing
│
├── metadata/                # Stage 4: Metadata Operations
│   └── enrich_metadata.py  # Merge perturbation data
│
├── identifiers/             # Used across all stages
│   └── parsing.py          # ID parsing [from parsing_utils.py]
│
└── io/                      # File I/O across all stages
    ├── load_masks.py       # Load SAM2/UNet masks
    ├── load_metadata.py    # CSV/JSON readers
    ├── export_masks.py     # Export to PNG
    ├── format_sam2_csv.py  # Flatten JSON to CSV
    └── save_results.py     # Write outputs
```

---

## File Migration Checklist

### ✅ Move As-Is (Week 1)
| Source | Target | Lines | Status |
|--------|--------|-------|--------|
| `segmentation_sandbox/scripts/utils/parsing_utils.py` | `src/data_pipeline/identifiers/parsing.py` | ~800 | Already good |
| `src/build/qc_utils.py` | `src/data_pipeline/qc/spatial_qc.py` | 135 | Already good |
| `src/data_pipeline/quality_control/death_detection.py` | `src/data_pipeline/qc/death_qc.py` | 317 | Already good |
| `segmentation_sandbox/scripts/utils/mask_utils.py` | `src/data_pipeline/segmentation/mask_formats.py` | ~200 | Already good |

### 🔨 Extract Functions (Weeks 2-3)
| Source | Extract What | Target | Delete What |
|--------|--------------|--------|-------------|
| `build01A_compile_keyence_torch.py` | Stitching, focus stacking | `preprocessing/stitching.py` | Hardcoded paths |
| `build03A_process_images.py` | Crop, rotate, morphology | `transforms/cropping.py`, `features/morphology.py` | Everything else |
| `build04_perform_embryo_qc.py` | Stage inference | `features/stage_inference.py` | Becomes Snakemake rule |
| `sam2_utils.py` | Model loading, propagation | `segmentation/sam2_inference.py` | GroundedSamAnnotations class |
| `grounded_dino_utils.py` | Model loading, detection | `segmentation/gdino_inference.py` | GroundedDinoAnnotations class |
| `01_prepare_videos.py` | Video organization | `segmentation/video_prep.py` | CLI wrapper |
| `05_sam2_qc_analysis.py` | Detection QC | `qc/detection_qc.py` | BaseFileHandler usage |
| `06_export_masks.py` | PNG export | `io/export_masks.py` | SimpleMaskExporter class |
| `export_sam2_metadata_to_csv.py` | CSV flattening | `io/format_sam2_csv.py` | SAM2MetadataExporter class |
| `07_embryo_metadata_update.py` | Metadata merging | `metadata/enrich_metadata.py` | Entity tracking |

### 🗑️ Delete Entirely (Week 4)
| File | Reason | Replacement |
|------|--------|-------------|
| `pipeline_objects.py` (1593 lines) | Orchestration overengineering | Snakemake |
| `base_file_handler.py` | Backup system not needed | Direct pandas/json |
| `entity_id_tracker.py` | File tracking not needed | Snakemake DAG |
| All sandbox pipeline `01-07` scripts | Become Snakemake rules | Snakefile |
| `build03A_process_images.py` | After extraction | Snakemake rule |
| `build04_perform_embryo_qc.py` | After extraction | Snakemake rule |

---

## Snakemake Workflow Structure

```python
# Snakefile

configfile: "config.yaml"

EXPERIMENTS = config["experiments"]

rule all:
    input:
        expand("metadata/build04_output/qc_staged_{exp}.csv", exp=EXPERIMENTS)

# Stage 1: Stitching
rule stitch_experiment:
    input: "raw_data/{experiment}/"
    output:
        images=directory("built_image_data/stitched_FF_images/{experiment}/"),
        meta="metadata/experiment_metadata/{experiment}.csv"
    run:
        from data_pipeline.preprocessing.stitching import stitch_keyence_tiles
        stitch_keyence_tiles(input[0], output.images, output.meta)

# Stage 3A: Prepare SAM2 videos
rule prepare_sam2_videos:
    input: "built_image_data/stitched_FF_images/{experiment}/"
    output:
        videos=directory("sam2_data/{experiment}/videos/"),
        meta="sam2_data/{experiment}/video_metadata.json"
    run:
        from data_pipeline.segmentation.video_prep import organize_images_to_videos
        organize_images_to_videos(input[0], output.videos, output.meta)

# Stage 3B: GDINO detection
rule detect_embryos:
    input:
        videos="sam2_data/{experiment}/videos/",
        model=config["gdino_model"]
    output: "sam2_data/{experiment}/gdino_detections.json"
    params:
        confidence=config["gdino_confidence"]
    run:
        from data_pipeline.segmentation.gdino_inference import detect_embryos
        detect_embryos(input.videos, input.model, output[0],
                      confidence=params.confidence)

# Stage 3C: SAM2 propagation
rule sam2_propagation:
    input:
        videos="sam2_data/{experiment}/videos/",
        detections="sam2_data/{experiment}/gdino_detections.json",
        model=config["sam2_model"]
    output: "sam2_data/{experiment}/sam2_results.json"
    run:
        from data_pipeline.segmentation.sam2_inference import propagate_masks
        propagate_masks(input.videos, input.detections, input.model, output[0])

# Stage 3D: SAM2 QC
rule sam2_qc:
    input: "sam2_data/{experiment}/sam2_results.json"
    output: "sam2_data/{experiment}/sam2_results_qc.json"
    run:
        from data_pipeline.qc.detection_qc import add_qc_flags
        add_qc_flags(input[0], output[0])

# Stage 3E: Export masks
rule export_masks:
    input: "sam2_data/{experiment}/sam2_results_qc.json"
    output:
        masks=directory("sam2_data/{experiment}/exported_masks/"),
        manifest="sam2_data/{experiment}/export_manifest.csv"
    run:
        from data_pipeline.io.export_masks import export_labeled_masks
        export_labeled_masks(input[0], output.masks, output.manifest)

# Stage 3F: Flatten to CSV
rule flatten_sam2:
    input:
        masks="sam2_data/{experiment}/sam2_results_qc.json",
        manifest="sam2_data/{experiment}/export_manifest.csv"
    output: "metadata/sam2_output/{experiment}_sam2_metadata.csv"
    run:
        from data_pipeline.io.format_sam2_csv import flatten_to_csv
        flatten_to_csv(input.masks, input.manifest, output[0])

# Stage 4: Enrich metadata
rule enrich_metadata:
    input:
        sam2="metadata/sam2_output/{experiment}_sam2_metadata.csv",
        master=config["master_perturbation_csv"]
    output: "metadata/sam2_enriched/{experiment}_metadata.csv"
    run:
        from data_pipeline.metadata.enrich_metadata import merge_perturbation
        merge_perturbation(input.sam2, input.master, output[0])

# Stage 5: Create snips + morphology
rule create_snips:
    input:
        csv="metadata/sam2_enriched/{experiment}_metadata.csv",
        masks="sam2_data/{experiment}/exported_masks/",
        images="built_image_data/stitched_FF_images/{experiment}/"
    output:
        snips=directory("built_image_data/snips/{experiment}/"),
        features="metadata/build03_output/expr_embryo_metadata_{experiment}.csv"
    params:
        crop_radius=config["crop_radius_um"]
    run:
        from data_pipeline.transforms.cropping import process_snips
        from data_pipeline.features.morphology import compute_features
        process_snips(input.csv, input.masks, input.images,
                     output.snips, output.features,
                     crop_radius=params.crop_radius)

# Stage 6: Embryo QC
rule embryo_qc:
    input:
        features="metadata/build03_output/expr_embryo_metadata_{experiment}.csv",
        stage_ref=config["stage_reference_csv"]
    output: "metadata/build04_output/qc_staged_{experiment}.csv"
    params:
        dead_lead_time=config["dead_lead_time"]
    run:
        from data_pipeline.features.stage_inference import infer_stage
        from data_pipeline.qc.spatial_qc import compute_qc_flags
        from data_pipeline.qc.death_qc import compute_dead_flag2_persistence

        df = pd.read_csv(input.features)
        df = infer_stage(df, input.stage_ref)
        df = compute_qc_flags(df)
        df = compute_dead_flag2_persistence(df, params.dead_lead_time)
        df.to_csv(output[0], index=False)
```

---

## Key Decisions & Questions

### 1. UNet Segmentation Status
**Question:** Is UNet (build02B) still used in production?
- Code has Windows paths (`E:\\Nick\\Dropbox...`)
- May only be used for yolk masks?
- Or fully replaced by SAM2?

**Action:**
- If deprecated → Delete `build02B` and `apply_unet.py`
- If still used → Extract to `segmentation/unet_inference.py`

### 2. Embedding Generation
**Question:** Where is Stage 7 (embryo embeddings) implemented?
- Not found in `src/build/` or sandbox pipelines
- May be in separate training scripts?

**Action:**
- Identify embedding generation code
- Create `features/embeddings.py` module
- Add Snakemake rule for embedding stage

### 3. Snakemake vs. Sandbox Pipeline Scripts
**Current:** Sandbox has 7 sequential scripts (`01-07`)
**Proposed:** Replace with Snakemake rules

**Benefit:**
- Automatic dependency tracking
- Parallel execution where possible
- Checkpointing/resume on failure
- No need for GroundedSamAnnotations state management

**Question:** Keep sandbox scripts for interactive dev?
- Option A: Delete after migration (clean break)
- Option B: Keep as "dev/" utilities for debugging

### 4. Config Management
**Current:** Hardcoded paths, environment variables, scattered configs
**Proposed:** Single `config.yaml` for Snakemake

```yaml
# config.yaml
experiments:
  - "20240418"
  - "20250612_30hpf_ctrl_atf6"

paths:
  raw_data: "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/"
  models: "models/"

models:
  gdino_model: "models/groundingdino_checkpoint.pth"
  sam2_model: "models/sam2_checkpoint.pth"

params:
  gdino_confidence: 0.3
  crop_radius_um: 150
  dead_lead_time: 2.0
  stage_reference_csv: "metadata/stage_ref_df.csv"
  master_perturbation_csv: "metadata/master_perturbation.csv"
```

---

## Implementation Timeline

### Week 1: Move Working Code (No Extraction)
**Goal:** Validate approach with minimal changes

- Move 4 files as-is:
  - `parsing_utils.py` → `identifiers/parsing.py`
  - `qc_utils.py` → `qc/spatial_qc.py`
  - `death_detection.py` → `qc/death_qc.py`
  - `mask_utils.py` → `segmentation/mask_formats.py`
- Update imports in 2-3 scripts to verify
- Run one experiment end-to-end with new imports

**Deliverable:** Core utilities importable from new structure

---

### Week 2: Extract SAM2 Pipeline Functions
**Goal:** Replace sandbox classes with pure functions

**Extract from:**
- `sam2_utils.py` → `segmentation/sam2_inference.py`
  - Functions: `load_sam2_model()`, `propagate_masks_forward()`, `assign_embryo_ids()`
  - Delete: `GroundedSamAnnotations` class

- `grounded_dino_utils.py` → `segmentation/gdino_inference.py`
  - Functions: `load_gdino_model()`, `detect_embryos()`, `filter_detections()`
  - Delete: `GroundedDinoAnnotations` class

- `01_prepare_videos.py` → `segmentation/video_prep.py`
  - Functions: `organize_images_to_videos()`

- `05_sam2_qc_analysis.py` → `qc/detection_qc.py`
  - Functions: `check_mask_area()`, `check_tracking_continuity()`

- `06_export_masks.py` → `io/export_masks.py`
  - Functions: `export_labeled_masks()`
  - Delete: `SimpleMaskExporter` class

- `export_sam2_metadata_to_csv.py` → `io/format_sam2_csv.py`
  - Functions: `flatten_sam2_to_csv()`
  - Delete: `SAM2MetadataExporter` class

**Create basic Snakemake rules for SAM2 pipeline (stages 3A-3F)**

**Test:** Run SAM2 pipeline on one experiment via Snakemake

**Deliverable:** SAM2 pipeline runs via Snakemake

---

### Week 3: Extract Build03A Functions
**Goal:** Replace monolithic build03A with extracted functions

**Extract from build03A_process_images.py:**
- `transforms/cropping.py`
  - `compute_embryo_angle()` (PCA-based)
  - `crop_embryo_region()`
  - `rotate_to_axis()`
  - `add_synthetic_noise()` (training augmentation)

- `features/morphology.py`
  - `compute_area()`
  - `compute_perimeter()`
  - `compute_centroid()`
  - `compute_aspect_ratio()`

**Create Snakemake rule for Stage 5 (snip creation)**

**Test:** Generate snips + features for one experiment

**Deliverable:** Stage 5 runs via Snakemake

---

### Week 4: Extract Build04 Functions & Complete Pipeline
**Goal:** Full end-to-end Snakemake workflow

**Extract from build04_perform_embryo_qc.py:**
- `features/stage_inference.py`
  - `infer_stage_from_morphology()`
  - `predict_hpf_from_area()`

- `qc/tracking_qc.py`
  - `compute_speed()` (may already be in spatial_qc.py)
  - `smooth_trajectory()` (Savitzky-Golay)

**Create Snakemake rule for Stage 6 (embryo QC)**

**Test:** Run full pipeline on 3 experiments

**Deliverable:** Complete Snakemake workflow (Stages 1-6)

---

### Week 5: Cleanup & Delete
**Goal:** Remove all overengineering

**Delete:**
- `pipeline_objects.py` (1593 lines)
- `base_file_handler.py`
- `entity_id_tracker.py`
- `build03A_process_images.py` (after extraction)
- `build04_perform_embryo_qc.py` (after extraction)
- All sandbox pipeline scripts `01-07` (after extraction)

**Create:**
- `config.yaml` for Snakemake
- `README.md` documenting new structure
- Migration guide for updating notebooks

**Deliverable:** Clean, flat codebase with working Snakemake pipeline

---

## Summary: Simple is Better

**Core Principles:**
1. ✅ **Extract pure functions** - No classes unless absolutely necessary
2. ✅ **Let Snakemake orchestrate** - Delete state management systems
3. ✅ **One file = One job** - Clear module boundaries
4. ✅ **Move working code first** - Don't refactor what already works
5. ✅ **Delete complexity** - No BaseFileHandler, EntityIDTracker, pipeline_objects

**Expected Outcome:**
- ~4000 lines of bloat deleted (pipeline_objects, annotation classes, file handlers)
- ~2000 lines of clean, reusable functions extracted
- Snakemake handles all orchestration (dependency tracking, parallelization, checkpointing)
- Easy to extend for new organisms/experiments (just update config.yaml)
- Simple enough that a new team member can understand the full pipeline in a day

**The goal is boring, predictable code that works.**
