# Input

## -Per Experiment 




# Legacy Segmentation & Build Pipeline Overview

## 1. Introduction

This document provides a detailed description of the legacy `morphseq` image processing pipeline as implemented outside of the `segmentation_sandbox`. Its purpose is to take raw 2D or 3D brightfield image data, identify and segment key components like embryos and yolks, track embryos over time, perform quality control, and export processed image "snips" for downstream analysis.

Understanding this existing workflow is crucial for integrating new segmentation pipelines and for general project maintenance.

## 2. Core Dependencies

- **Environment:** The pipeline runs in the `torch-env` conda environment.
- **Key Libraries:**
    - `pytorch`: The core deep learning framework.
    - `pytorch-lightning`: Used as a high-level wrapper for the model definition.
    - `segmentation-models-pytorch`: Provides the Feature Pyramid Network (FPN) architecture and ResNet encoders. This is a critical dependency.
    - `scikit-image`: Used extensively for image manipulation (resizing, labeling, region properties).
    - `pandas`: Used for all metadata manipulation.
    - `nd2`: Used for reading raw `.nd2` z-stack files from the YX1 microscope.

## 3. Pipeline Flow

The pipeline is a multi-stage process, orchestrated by a series of `build` scripts. Each stage produces outputs that are consumed by the next.

---

### **Stage 0: Raw Image Metadata Compilation**

This initial stage is responsible for compiling fundamental metadata directly from the raw image files, establishing the foundational data for all subsequent pipeline stages.

- **Scripts:**
    - `src/build/build01A_compile_keyence_torch.py` (for Keyence microscope data)
    - `src/build/build01B_compile_yx1_images_torch.py` (for YX1 microscope data)

#### **Process:**

These scripts process raw image data (e.g., `.nd2` files for YX1, individual image tiles for Keyence) and extract critical acquisition parameters. Crucially, they generate metadata on a **per-well-time-point basis**. This means that for each unique combination of well and time point, a dedicated set of metadata is recorded, including:

-   Physical dimensions (`Height (um)`, `Width (um)`)
-   Pixel dimensions (`Height (px)`, `Width (px)`)
-   Microscope settings (e.g., `Objective`, `BF Channel`)
-   Time information (`Time (s)`, `Time Rel (s)`)

This granular, per-image metadata is essential for accurate physical scaling and tracking throughout the pipeline.

#### **Outputs:**

-   **Raw Image Metadata CSV:** A CSV file located at `metadata/built_metadata_files/{experiment_date}_metadata.csv`. This file serves as the primary source of raw image acquisition parameters for downstream stages.

---

### **Stage 1: Sequential Mask Generation**

This stage generates all the necessary segmentation masks for downstream processing. It does **not** use a single multi-target model, but rather a series of individual models run sequentially.

- **Script:** `src/build/build02B_segment_bf_main.py`
- **Entry Point:** The `apply_unet` function is the core worker. The `if __name__ == "__main__"` block calls this function four times in a row.

#### **Process:**

1.  **Model Definition:** The script uses the `FishModel` class defined in `src/functions/core_utils_segmentation.py`. This class implements a **Feature Pyramid Network (FPN)** with a **ResNet34** encoder backbone, provided by the `segmentation-models-pytorch` library.

2.  **Sequential Execution:** `apply_unet` is called once for each required mask, loading a different set of pre-trained weights each time. The typical execution order is:
    1.  **Embryo/Viability Segmentation** (`unet_emb_v4_0050`): A 2-class model to distinguish living vs. dead embryo regions.
    2.  **Bubble Segmentation** (`unet_bubble_v0_0050`): A 1-class model to identify air bubbles.
    3.  **Yolk Segmentation** (`unet_yolk_v0_0050`): A 1-class model to identify the embryo yolk.
    4.  **Focus Segmentation** (`unet_focus_v2_0050`): A 1-class model to identify out-of-focus regions.

#### **Inputs:**

-   **Images:** Raw 2D brightfield images (JPG, PNG, TIF) located in `built_image_data/stitched_FF_images/<experiment_date>/`.
-   **Model Weights:** Pre-trained `.pth` model files located in `segmentation/segmentation_models/`.

#### **Outputs:**

-   **Crucial Point:** Each model saves its predictions to a **separate directory**.
-   **Location:** `segmentation/<model_name>_predictions/<experiment_date>/`
-   **Format:** The output masks are single-channel 8-bit **JPG files**. They are **not** simple binary (0/255) masks. The pixel values are multi-level integers (e.g., 85, 170, 255) that encode class information based on the model's output probabilities. Downstream scripts must interpret these values correctly.

---

### **Stage 2: 2D Embryo Tracking and QC Analysis**

This stage takes the 2D masks from Stage 1, identifies and tracks individual embryos over time, and performs quality control checks.

- **Script:** `src/build/build03A_process_images.py`
- **Entry Points:** `segment_wells` followed by `compile_embryo_stats`.

#### **Process:**

1.  **Embryo Detection (`count_embryo_regions`):**
    -   Loads the embryo mask generated by `unet_emb_v4_0050`.
    -   **Implicit Binarization:** It converts the multi-level JPG mask into a binary mask using a mathematical operation that acts as a threshold (e.g., `(np.round(im / 255 * 2) - 1)`).
    -   Uses `skimage.measure.label` on the binarized mask to find distinct embryo objects.
    -   Extracts the centroid and a temporary `region_label` for each detected embryo.
    -   Uses the viability mask to calculate the `frac_alive` for each embryo.

2.  **Temporal Tracking (`do_embryo_tracking`):**
    -   Takes the per-frame embryo detections (centroids).
    -   Uses a linear sum assignment algorithm based on proximity to track each unique embryo over time, assigning it a stable `embryo_id`.

3.  **Quality Control (`get_embryo_stats`):**
    -   For each tracked embryo, this function loads the full set of masks from Stage 1 (embryo, yolk, bubble, focus).
    -   It uses the stored `region_label` to isolate the specific embryo of interest from the embryo mask.
    -   It performs a series of QC checks by comparing the embryo's location to the other masks, setting flags like `no_yolk_flag`, `focus_flag`, and `bubble_flag`.
    -   It calculates morphological statistics like `length_um`, `width_um`, and `surface_area_um`.

#### **Inputs:**

-   The various segmentation masks generated in Stage 1.

#### **Outputs:**

-   **Master Metadata File:** A rich CSV file (`embryo_metadata_df.csv`) containing a row for every detected embryo at every timepoint, including its ID, position, QC flags, and morphological stats. This file is the primary input for the final stage.

#### **Critical System Limitations:**

The legacy tracking system has several fundamental problems that make it fragile and computationally expensive:

1. **Fragile `region_label` System:**
   - Relies on `skimage.measure.label` to assign temporary region numbers to embryo blobs
   - Region labels are frame-specific and have no temporal consistency
   - Vulnerable to segmentation artifacts and touching objects

2. **Complex Tracking Algorithm:**
   - `do_embryo_tracking` uses linear sum assignment based on centroid proximity
   - Computationally intensive Hungarian algorithm for every frame
   - Prone to ID switching when embryos move close together or temporarily disappear

3. **Redundant Calculations:**
   - `get_embryo_stats` runs `skimage.measure.regionprops` to recalculate area, centroids, bounding boxes
   - These same properties could be pre-computed during segmentation
   - Results in duplicated computational effort and potential inconsistencies

4. **Error-Prone Pipeline:**
   - Multiple failure points: detection → labeling → tracking → statistics calculation
   - Silent failures possible at each stage (e.g., missed embryos, incorrect region isolation)
   - Difficult to debug when tracking fails partway through long time series

5. **Memory and Performance Issues:**
   - Must load and process multiple mask files for each embryo
   - Scales poorly with number of embryos and timepoints
   - No built-in validation or error recovery mechanisms

#### **Refactoring Implications for SAM2 Integration:**

The limitations above make Stage 2 the primary target for SAM2 integration. The refactoring strategy completely eliminates the problematic components:

**Functions to be DELETED entirely:**
- `count_embryo_regions()` - Replaced by SAM2's instance-aware masks
- `do_embryo_tracking()` - Replaced by SAM2's inherent temporal consistency  

**Function to be SIMPLIFIED:**
- `get_embryo_stats()` - Refactored from full property calculation to QC-only focus:
  - **REMOVE:** All regionprops calculations (area, centroid, bbox)
  - **REMOVE:** Region label isolation logic  
  - **KEEP:** QC flag calculations against U-Net masks (yolk, bubble, focus)
  - **KEEP:** Morphological measurements (length, width, surface area)
  - **NEW:** Accept CSV row as input instead of region_label

**Architecture Changes:**
- **Input Method:** Replace `glob.glob()` image discovery with `pd.read_csv()` as primary data source
- **Processing Logic:** Iterate over CSV rows instead of detected regions
- **Data Flow:** Pre-computed metadata → QC validation instead of detection → calculation
- **Error Handling:** CSV validation instead of mask parsing error handling

**Expected Outcomes:**
- **Code Reduction:** ~50-70% fewer lines in `build03A_process_images.py`
- **Performance Improvement:** 50-80% faster execution due to eliminated calculations
- **Reliability Improvement:** No more tracking failures or ID switching
- **Maintainability:** Clear separation of segmentation (SAM2) vs QC logic (legacy)

---

### **Stage 3: 3D Z-Stack Processing & Snippet Export**

The final stage uses the 2D tracking data to find the best focal plane in the original 3D image stacks and export high-quality 2D image snips.

- **Script:** `src/build/build03B_export_z_snips.py`
- **Entry Point:** `extract_embryo_z_snips`

#### **Process:**

1.  **Data Loading:**
    -   Loads the master metadata CSV produced by Stage 2.
    -   For a given embryo, it loads the original raw 3D image stack (either a `.nd2` file or a `.tif` stack).
    -   It also re-loads the 2D embryo and yolk masks from Stage 1.

2.  **Focus Calculation (`LoG_focus_stacker`):**
    -   **Critical Dependency:** To find the most in-focus z-slice, the script calculates focus scores (using a Laplacian of Gaussian filter) **only on the embryo "body"**.
    -   The body is defined by **subtracting the yolk mask from the embryo mask**. This means the yolk mask is essential for this calculation to work as intended.

3.  **Image Processing & Export (`export_embryo_snips_z`):**
    -   Once the optimal z-slice is found, the script orients the image using the embryo and yolk masks (`get_embryo_angle`).
    -   It rotates and crops a standardized "snip" from the selected z-slice.
    *   It saves the final processed 2D snip to `training_data/bf_embryo_snips_z<N>/`.

#### **Inputs:**

-   The metadata CSV from Stage 2.
-   The raw 3D image stacks (e.g., `.nd2` files).
-   The 2D embryo and yolk masks from Stage 1.

#### **Outputs:**

-   Final, processed, 2D brightfield image snips of each embryo, taken from the optimal focal plane.
---

## 4. SAM2 Segmentation Pipeline Advantages

## Quick Reference: 2D Snips vs Z‑Snips

This project exports standardized 2D “snips” that feed the embedding models. There are two ways to produce snips:

- 2D Snips (default path)
  - Script: `src/build/build03A_process_images.py`
  - Inputs:
    - Sandbox‑organized 2D frames: `segmentation_sandbox/data/raw_data_organized/<date>/images/<video_id>/<image_id>*.jpg`
    - SAM2 embryo masks (integer‑labeled): `segmentation_sandbox/data/exported_masks/<date>/masks/<well>_t####*.tif` (override base with `MORPHSEQ_SANDBOX_MASKS_DIR`)
    - Optional legacy yolk mask: `built_image_data/segmentation/yolk_*/<date>/<well>_t####*.tif`
  - Core steps: normalize/select mask (`process_masks`), compute orientation (`get_embryo_angle`), rotate (`rotate_image`), crop (`crop_embryo_image`).
  - Outputs:
    - Cropped: `training_data/bf_embryo_snips/<date>/<snip_id>.jpg`
    - Uncropped: `training_data/bf_embryo_snips_uncropped/<date>/<snip_id>.jpg`
    - Masks: `training_data/bf_embryo_masks/emb_<snip_id>.jpg`, `training_data/bf_embryo_masks/yolk_<snip_id>.jpg`

- Z‑Snips (3D‑aware, optional)
  - Script: `src/build/build03B_export_z_snips.py`
  - Inputs:
    - Same embryo/yolk masks as 2D Snips
    - Z‑stacks per timepoint:
      - YX1: `raw_image_data/YX1/<date>/*.nd2` (channel “BF”, uses `nd2_series_num`, `time_int`)
      - Keyence stitched Z stacks: `built_image_data/keyence_stitched_z/<date>/<well>_t####_stack.tif` (produced by `build01AB_stitch_keyence_z_slices.py` or `build01A_compile_keyence_torch.py`)
    - Z‑resolution: from ND2 metadata (YX1) or `metadata/experiment_metadata.csv` (Keyence)
  - Core steps: LoG focus scores on embryo body (`LoG_focus_stacker`), pick best slice(s), orient + crop like 2D path.
  - Outputs:
    - Cropped: `training_data/bf_embryo_snips_zNN/<date>/<snip_id>_zXX.jpg`
    - Uncropped: `training_data/bf_embryo_snips_zNN_uncropped/<date>/<snip_id>_zXX.jpg`
    - Per‑snip z‑metadata: `metadata/metadata_files_temp_zNN/<snip_id>.csv` (`z_res_um`, `z_ind`, `z_pos_rel`)

Downstream Embeddings
- Both snip types are encoded identically by the VAE:
  - Lightning eval: `src/analyze/assess_vae_results.py` ➜ writes `models/<class>/<name>/embryo_stats_df.csv` with `z_mu_*`.
  - Ad‑hoc (single/many images): `src/vae/auxiliary_scripts/assess_image_set.py` ➜ `AutoModel.load_from_folder(.../final_model)` then `encoder(x).embedding`.


The SAM2 (Segment Anything Model 2) segmentation pipeline, implemented in the `segmentation_sandbox`, provides a superior alternative to the legacy Stage 2 tracking system by addressing all major limitations.

### **Key Architectural Advantages:**

#### **1. Instance-Aware Masks**
- **Integer-Labeled Masks:** SAM2 outputs masks where pixel value directly corresponds to stable `embryo_id`
- **No Region Labeling:** Eliminates the fragile `skimage.measure.label` step entirely  
- **Consistent IDs:** Each embryo maintains the same pixel value across all frames in a video
- **Example:** All pixels for embryo `20240411_A01_e01` have value `1`, embryo `20240411_A01_e02` has value `2`

#### **2. Inherent Temporal Tracking**
- **Built-in Consistency:** SAM2 handles temporal tracking automatically during segmentation
- **No Hungarian Algorithm:** Eliminates complex linear sum assignment tracking
- **Robust to Movement:** Handles embryos moving close together or temporary occlusion
- **ID Stability:** No risk of ID switching mid-video

#### **3. Pre-Computed Metadata**
- **Rich Annotations:** SAM2 outputs include area, bounding box, mask confidence for each embryo
- **No Regionprops:** Eliminates redundant area/centroid calculations in build scripts
- **Consistent Measurements:** All metrics calculated from same source masks
- **Performance Gain:** ~50-80% reduction in computational time for metadata extraction

#### **4. Comprehensive Output Format**
SAM2 produces `GroundedSam2Annotations.json` with complete embryo metadata:
```json
{
    "embryos": {
        "20240411_A01_e01": {
            "snip_id": "20240411_A01_e01_s0042",
            "area": 5000.0,
            "bbox": [0.1, 0.2, 0.3, 0.4], 
            "mask_confidence": 0.95,
            "segmentation": {"counts": "RLE_string", "size": [1024, 1024]}
        }
    }
}
```

### **ID Format Compatibility:**

#### **Native SAM2 Format:**
- **snip_id:** Uses `_s` prefix (e.g., `20240411_A01_e01_s0042`) 
- **Frame Indexing:** `frame_index` matches snip numbering for parsing consistency
- **No Conversion Needed:** Analysis confirmed frame_index == time_int, so `_s` format works directly

#### **Parsing Integration:**
- **Existing Utilities:** `parsing_utils.py` handles both `_s` and `_t` formats
- **ID Extraction:** All existing ID parsing functions work with SAM2 native format
- **Well Mapping:** SAM2 embryo_ids trace back to well IDs for metadata integration

## 5. SAM2 Integration Strategy

The integration of SAM2 into the legacy pipeline follows a **metadata bridge architecture** that transforms build scripts from data-processors into data-consumers.

### **Core Integration Principle:**

**From Producer to Consumer:** Instead of having build scripts calculate embryo properties from raw masks, they consume pre-computed metadata from SAM2's rich output format.

### **Two-Phase Implementation:**

#### **Phase 1: Metadata Bridge Script**
- **Script:** `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`
- **Purpose:** Flatten nested `GroundedSam2Annotations.json` into build-script-friendly CSV
- **Input:** SAM2 JSON with nested experiment → video → image → embryo structure
- **Output:** Flat CSV with one row per embryo per frame

**CSV Schema:**
```
image_id, embryo_id, snip_id, frame_index, area_px, 
bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, 
mask_confidence, exported_mask_path, experiment_id, 
video_id, is_seed_frame
```

#### **Phase 2: Build Script Refactoring** 
- **Target:** `src/build/build03A_process_images.py`
- **Architecture Change:** CSV-driven processing instead of glob-based image discovery
- **Function Elimination:** Remove `count_embryo_regions`, `do_embryo_tracking` entirely
- **Function Simplification:** Refactor `get_embryo_stats` to focus only on QC checks

### **Data Flow Transformation:**

**Legacy Flow:**
```
Raw Images → Mask Generation → Region Labeling → 
Tracking Algorithm → Property Calculation → QC Analysis
```

**SAM2 Integration Flow:**
```
GroundedSam2Annotations.json → CSV Bridge → 
Pre-computed Metadata Loading → QC Analysis Only
```

### **Risk Mitigation Strategies:**

#### **1. Schema Validation**
- JSON structure validation with version checks
- CSV schema enforcement with type checking
- Graceful handling of missing or malformed fields

#### **2. File System Robustness**
- Exported mask file existence validation
- Clear error messages for missing dependencies
- Fallback mechanisms for partial data

#### **3. Performance Optimization**
- DataFrame operations optimization for large datasets
- Memory-efficient processing for multi-experiment batches
- Benchmark targets: Bridge <30s, Build <2x legacy time

#### **4. Backward Compatibility**
- Maintain existing output schema for downstream tools
- Preserve QC flag calculations and thresholds
- Keep well metadata integration unchanged

### **Integration Benefits:**

1. **Complexity Reduction:** ~50% code reduction in build scripts
2. **Reliability Improvement:** Eliminates fragile tracking algorithm 
3. **Performance Gains:** Pre-computed metadata eliminates redundant calculations
4. **Maintainability:** Clear separation between segmentation and QC logic
5. **Scalability:** Better handling of large multi-experiment datasets

---

### **Well Metadata Integration System**

The legacy pipeline includes a sophisticated well metadata system that provides biological context (genotype, treatments, phenotypes) separate from the image processing pipeline.

#### **File Structure:**

- **Location:** `metadata/plate_metadata/{experiment_date}_well_metadata.xlsx`
- **Format:** Multi-sheet Excel files with standardized 96-well plate layout

#### **Excel Sheet Structure:**

Each Excel file contains multiple sheets defining experimental conditions:

1. **`medium`** - Culture medium specifications per well
2. **`genotype`** - Genetic background and mutations per well  
3. **`chem_perturbation`** - Chemical treatments and dosages
4. **`start_age_hpf`** - Starting age in hours post fertilization
5. **`embryos_per_well`** - Number of embryos per well
6. **`temperature`** - Incubation temperature per well
7. **`qc`** (optional) - Well-level quality control flags

#### **Processing Pipeline:**

- **Primary Function:** `load_experiment_plate_metadata()` in `src/build/export_utils.py`
- **Process:**
  1. Loads Excel file using `pd.ExcelFile()`
  2. Parses each sheet into 8×12 well layout (A01-H12)
  3. Flattens 2D plate data into long-format DataFrame
  4. Creates well-level metadata with one row per well per experiment
  5. Merges with embryo tracking data based on well ID matching

#### **Integration Points:**

- **Early Stage:** Used in `build01B_compile_yx1_images_torch.py` during initial image compilation
- **Export Stage:** Primary integration happens during final data export via `export_utils.py`
- **QC Analysis:** Referenced in `build04_perform_embryo_qc.py` for phenotype-based quality control

#### **Key Design Principles:**

- **Separation of Concerns:** Well metadata (biological context) is completely separate from image processing pipeline
- **Well-Based Matching:** Integration occurs via well ID (e.g., "A01") rather than individual embryo IDs
- **Plate-Level Organization:** Each Excel file represents one experimental plate with standardized layout
- **Flexible Schema:** Different experiments can have different combinations of treatments and conditions

#### **Implications for SAM2 Integration:**

- **No Direct Interaction:** SAM2 bridge script should focus purely on segmentation metadata
- **Existing Integration Path:** Well metadata merging is handled by established functions in `export_utils.py`
- **ID Compatibility:** SAM2 embryo IDs must be traceable back to well IDs for proper metadata integration
- **Parallel Processing:** Well metadata and segmentation metadata can be processed independently and merged downstream

---

## 5. Build Scripts Summary (Legacy Pipeline Tasks)

This section maps each legacy build script to its role, inputs, and outputs for quick reference.

### Orchestration

- `src/build/pipeline_objects.py`: High-level manager for per-date runs.
  - Purpose: Discovers experiments, determines “needs_*” via filesystem checks, and runs steps: export → stitch → segment → stats/snips.
  - Key methods: `export_images`, `stitch_images`, `stitch_z_images`, `segment_images`, `process_image_masks`.

### 01A — Keyence Full‑Focus (FF) Compile

- `src/build/build01A_compile_keyence_torch.py`
  - Purpose: Focus‑stack Keyence tile Z‑stacks into 2D FF tiles; write per‑well/time metadata; stitch tiles into FF montages.
  - Inputs: `raw_image_data/Keyence/<date>/XY*/[P*/]T*/...CH*.jpg`; plate metadata Excel.
  - Outputs: `built_image_data/Keyence/FF_images/<date>/ff_*/*.jpg`; `built_image_data/stitched_FF_images/<date>/*_stitch.jpg`; `metadata/built_metadata_files/<date>_metadata.csv`.
  - Notes: Uses LoG focus stacking and stitch2d with per‑date orientation and master alignment params (`master_params.json`).

### 01AB — Keyence Z‑Slice Stitch (per timepoint)

- `src/build/build01AB_stitch_keyence_z_slices.py`
  - Purpose: Stitch multi‑tile Keyence Z‑stacks into a canonical canvas per timepoint.
  - Inputs: Raw Keyence per‑well folders; FF tile dir + `master_params.json` from 01A; built metadata for sizing.
  - Outputs: `built_image_data/Keyence_stitched_z/<date>/<well>_t####_stack.tif`.

### 01B — YX1 Full‑Focus (FF) Compile

- `src/build/build01B_compile_yx1_images_torch.py`
  - Purpose: Read YX1 `.nd2` (T×W×Z×C×Y×X), fix timestamps, validate well mapping, focus‑stack BF to 2D FF, and build metadata.
  - Inputs: `raw_image_data/YX1/<date>/*.nd2`; plate metadata Excel.
  - Outputs: `built_image_data/stitched_FF_images/<date>/*_stitch.jpg`; `metadata/built_metadata_files/<date>_metadata.csv`.
  - Notes: Verifies well layout via stage positions (KMeans). Handles ND2 time stamp jumps.

### 02B — Segment BF Images

- `src/build/build02B_segment_bf_main.py`
  - Purpose: Run trained UNet/FPN models (mask/via/yolk/focus/bubble) on stitched FF images; write per‑model predictions.
  - Inputs: `built_image_data/stitched_FF_images/<date>/*`; model weights under `segmentation/segmentation_models`.
  - Outputs: `segmentation/<model>_predictions/<date>/*.jpg` (class‑coded grayscale outputs).

### 03A — Process Images (Masks → Metadata → 2D Snips)

- `src/build/build03A_process_images.py`
  - Purpose: Legacy path to collect embryo detections and QC; exports 2D snips. SAM2 bridge now preferred for IDs/metrics.
  - Inputs: Segmentation outputs; built metadata; SAM2 mask paths via `segmentation_sandbox` when available.
  - Outputs: Combined frame/embryo metadata CSVs; `training_data/bf_embryo_snips/<date>/<snip_id>.jpg` (+ uncropped and masks).
  - Notes: Includes helpers to sample background, compute orientation, rotate/crop, and optional legacy tracking (deprecated).

### 03B — Export Z‑Snips (focus‑aware)

- `src/build/build03B_export_z_snips.py`
  - Purpose: Choose in‑focus Z‑planes using LoG scores in embryo body and export cropped z‑slice snips (1/3/5 slices).
  - Inputs: Keyence stitched Z‑stacks or YX1 ND2; SAM2 embryo masks and legacy yolk masks; combined metadata.
  - Outputs: `training_data/bf_embryo_snips_zNN/<date>/<snip_id>_zXX.jpg` (+ uncropped); per‑snip z‑meta CSVs.

### 04 — Embryo QC + Stage Inference

- `src/build/build04_perform_embryo_qc.py`
  - Purpose: Apply rule‑based QC (size outliers, proximity to death), infer standardized stage (hpf), and build curation tables.
  - Inputs: `metadata/combined_metadata_files/embryo_metadata_df01.csv`; stage reference (`metadata/stage_ref_df.csv`); perturbation keys.
  - Outputs: `embryo_metadata_df02.csv`; `curation/curation_df.csv`; `curation/embryo_curation_df.csv`.
  - 
#### Statsmodels Usage in Build04

  Purpose: Statsmodels is used for developmental stage inference in Build04 - a critical biological
  analysis step.

  Specific Function: infer_embryo_stage_orig() uses Ordinary Least Squares (OLS) regression to:

  1. Calibrate stage predictions using reference embryos (wild-type controls)
  2. Build regression model with predictors:
    - stage (predicted developmental stage)
    - stage² (quadratic term)
    - cohort_id (experimental batch effects)
    - interaction (stage × cohort interaction)
  3. Predict developmental stages for all embryos based on:
    - Surface area measurements (μm²)
    - Temporal progression through development
    - Experimental batch corrections

  Biological Context:
  - Zebrafish embryos develop through predictable stages (24hpf, 30hpf, 36hpf, etc.)
  - Surface area correlates with developmental stage progression
  - The regression corrects for batch-to-batch variations in culture conditions

  Alternative: We could potentially replace with sklearn's LinearRegression, but statsmodels provides:
  - More detailed statistical diagnostics
  - Better handling of interaction terms
  - Compatibility with existing analysis pipeline

  Impact on Testing: This means Build04 is doing scientific analysis, not just data processing - it's
  inferring biological stages from morphological measurements.


### 05 — Make Training Snips

- `src/build/build05_make_training_snips.py`
  - Purpose: Copy/export curated snips into training folder structure, optionally grouped by labels and rescaled.
  - Inputs: Curated metadata (`embryo_metadata_df02.csv`, curation CSVs); 2D snips.
  - Outputs: `training_data/<train_name>/images/<label>/*.jpg` and a copy of the training metadata.

### Shared Utilities

- `src/build/export_utils.py`: Patterns for file checks; GPU/CPU memory heuristics; LoG focus stackers; Keyence orientation helper; plate metadata merge.
- `src/build/data_classes.py`: `MultiTileZStackDataset` to load Keyence tile Z‑stacks for batched FF projection/stitching.

### Typical Microscope Flows

- Keyence: 01A → 01AB → 02B → 03A → 03B → 04 → 05
- YX1: 01B → 02B → 03A → 03B → 04 → 05

---

## 6. Mask Usage

This section summarizes what each mask is used for across the legacy build pipeline and where they are produced.

### Mask Types

- Embryo mask (`mask_*`): main object mask per frame (required for snips, geometry, orientation, QC coverage).
- Viability mask (`via_*`): alive vs. dead signal (used for background sampling and dead flags).
- Yolk mask (`yolk_*`): yolk region (used for orientation and to exclude yolk from focus scoring).
- Focus mask (`focus_*`): out-of-focus regions (QC flag; not used in geometry).
- Bubble mask (`bubble_*`): air bubble regions (QC flag; not used in geometry).

### Where Masks Are Produced

- `src/build/build02B_segment_bf_main.py`: Runs models and writes per‑model predictions under `segmentation/<model>_predictions/<date>/*.jpg`.
- Orchestrated by `src/build/pipeline_objects.py` via `segment_images()`.

### How Masks Are Used

- 2D snips (geometry, orientation, cropping)
  - Embryo mask required; yolk helps orientation.
  - Files: `src/build/build03A_process_images.py`
  - Steps: normalize masks (`process_masks`), compute angle (`get_embryo_angle`), rotate and crop (`rotate_image`, `crop_embryo_image`).
  - Also saves cropped mask artifacts under `training_data/bf_embryo_masks/`.

- Background/noise estimation (alive region only)
  - Uses embryo + viability to exclude embryo and dead areas when sampling background pixels.
  - File: `src/build/build03A_process_images.py` (`estimate_image_background`).

- Z‑snips (focus‑aware selection)
  - Uses embryo minus yolk (“body”) region to compute LoG focus scores and select best Z‑planes.
  - Files: `src/build/build03B_export_z_snips.py` (LoG scoring, orientation, crop, export).
  - Note: For 3D Z‑snips, a yolk mask is expected; code errors if legacy yolk mask is missing.

- QC flags and downstream filtering
  - Viability contributes to `fraction_alive` and `dead_flag` (legacy). In the SAM2 bridge path, `fraction_alive` is placeholder and `dead_flag` derived accordingly.
  - Focus/bubble flags exist and are carried through, but are disabled (set False) in the current SAM2 bridge path.
  - Files: `src/build/build03A_process_images.py` (flag assembly) and `src/build/build04_perform_embryo_qc.py` (QC usage and curation tables).

### SAM2 Bridge Notes

- Embryo mask comes from `segmentation_sandbox/data/exported_masks/<date>/masks/<well>_t####*` as integer‑labeled instances; the pipeline selects the `region_label` for a given snip.
- 2D snips path tolerates missing legacy yolk by substituting an empty yolk mask; Z‑snips path requires a yolk mask to exclude yolk from focus scoring.
- Focus/bubble masks are not produced by SAM2; flags remain False in the bridge path unless legacy masks are available.

---

## 7. Build04 Perturbation Key Management System

The `perturbation_name_key.csv` file is a critical curation artifact that enforces consistent biological labeling throughout the pipeline.

### **Purpose and Role**

**Build04 Dependency**: Build04 performs QC and stage inference by merging embryo metadata with curated perturbation information:
- **Location**: `<root>/metadata/perturbation_name_key.csv` 
- **Code**: `src/build/build04_perform_embryo_qc.py:432`
- **Legacy Root**: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/`

**Core Function**: Maps `master_perturbation` values to standardized biological annotations:
```python
pert_name_key = pd.read_csv(os.path.join(root, 'metadata', "perturbation_name_key.csv"))
embryo_metadata_df = embryo_metadata_df.merge(pert_name_key, how="left", on="master_perturbation", indicator=True)
```

**Enforcement**: Build04 **raises an exception** if any `master_perturbation` values are missing from the key:
```python
if np.any(embryo_metadata_df["_merge"] != "both"):
    problem_perts = np.unique(embryo_metadata_df.loc[embryo_metadata_df["_merge"] != "both", "master_perturbation"])
    raise Exception("Some perturbations were not found in key: " + ', '.join(problem_perts.tolist()))
```

### **Required Schema**

**Essential Columns**:
- `master_perturbation`: Join key (unique identifier)
- `short_pert_name`: Standardized display name
- `phenotype`: Biological classification (wt, mutant names)
- `control_flag`: Boolean indicating control conditions 
- `pert_type`: Type classification (control, crispant, fluor, CRISPR)
- `background`: Genetic background (wik, ab, specific lines)

**Example Structure**:
```csv
master_perturbation,short_pert_name,phenotype,control_flag,pert_type,background
atf6,atf6,unknown,False,CRISPR,wik
inj-ctrl,inj-ctrl,wt,True,control,wik
EM,EM,wt,True,medium,wik
wik,wik,wt,True,control,wik
ab,ab,wt,True,control,ab
```

### **Data Flow and Origins**

**Source Pipeline**: `master_perturbation` values originate from well metadata Excel sheets:
1. **Excel Input**: `metadata/plate_metadata/{exp}_well_metadata.xlsx`
2. **Assembly Logic**: Combined from `chem_perturbation` and `genotype` fields (`build04_perform_embryo_qc.py:322-338`)
3. **Legacy Processing**: Historically processed by `src/_Archive/build_orig/build03A_process_embryos_main_par.py`

**Downstream Usage**: The enriched metadata drives multiple Build04 outputs:
- **Curation Tables**: `metadata/combined_metadata_files/curation/curation_df.csv`
- **Training Keys**: `metadata/combined_metadata_files/curation/perturbation_train_key.csv`
- **Metric Keys**: `metadata/combined_metadata_files/curation/perturbation_metric_key.csv`

### **Current Management Limitations**

**Manual Curation**: The key file is **not auto-generated** - it's maintained manually:
- **No Generator Scripts**: Repository searches show only reads, never writes to `perturbation_name_key.csv`
- **Manual Editing**: Values added/maintained by hand, often informed by well Excel sheets
- **No Version Control**: File lives in production directories, not under version control
- **Coverage Gaps**: New experiments can fail if novel `master_perturbation` values aren't pre-added to key

**Discovery Commands**:
```bash
# Search for existing key files
find /net/trapnell/vol1/home/nlammers/projects/data/morphseq -name "*perturbation*key*" -o -name "*pert*key*"

# Check coverage against current data
rg -n 'short_pert_name|phenotype|control_flag|pert_type|background' /net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/*.csv
```

### **Recommended Management Improvements**

**1. Source Control Integration**:
- Move `perturbation_name_key.csv` into repository under `metadata/` or `configs/`
- Version control all changes with clear commit messages
- Keep per-root overrides optional, default to versioned copy

**2. Schema Validation**:
- Define allowed values for `phenotype`, `pert_type`, `background`
- Enforce uniqueness of `master_perturbation`
- Add validation to `src/run_morphseq_pipeline/validation.py`

**3. Template Generation**:
```python
# Proposed utility script
def generate_perturbation_key_template(df01_path):
    """Scan df01.csv and output template with unique master_perturbation values"""
    df = pd.read_csv(df01_path)
    unique_perts = df['master_perturbation'].unique()
    
    template = []
    for pert in unique_perts:
        template.append({
            'master_perturbation': pert,
            'short_pert_name': pert,  # Default to same name
            'phenotype': 'unknown',   # Requires manual curation
            'control_flag': 'inj-ctrl' in pert.lower(),  # Heuristic
            'pert_type': 'unknown',   # Requires manual curation  
            'background': 'wik'       # Default background
        })
    
    return pd.DataFrame(template)
```

**4. Configurable Paths**:
- Add `--pert-key` argument to Build04 CLI for custom key file paths
- Maintain backward compatibility with default location
- Support environment-specific keys (development vs production)

**5. Coverage Validation**:
```python
# Proposed validation checks
def validate_perturbation_coverage(df01_path, key_path):
    """Verify all master_perturbation values in df01 exist in key"""
    df01 = pd.read_csv(df01_path)
    key = pd.read_csv(key_path)
    
    df01_perts = set(df01['master_perturbation'].unique())
    key_perts = set(key['master_perturbation'].unique())
    
    missing = df01_perts - key_perts
    if missing:
        raise ValueError(f"Missing perturbations in key: {missing}")
    
    return True
```

This systematic approach would prevent Build04 crashes due to missing perturbation mappings and provide clear curation workflows for new experiments.
