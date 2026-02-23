# Refactor PRD 005: Fundamental Metadata Integration for Legacy Compatibility


## 1. Objective

This document outlines a critical architectural refactoring aimed at resolving a fundamental data incompatibility discovered during the integration of the SAM2 segmentation pipeline with the legacy `morphseq` processing. The objective is to ensure that essential raw image metadata, specifically physical dimensions (`Height (um)`) and pixel dimensions (`Height (px)`), are accurately and directly propagated through the SAM2 pipeline's metadata system, thereby enabling seamless compatibility with downstream legacy analysis.

## 2. Background: The Fundamental Incompatibility

During the refactoring efforts detailed in previous PRDs (001-004), a core incompatibility was identified:

*   **Legacy Pipeline:** The original `morphseq` pipeline relied on direct access to `Height (um)` and `Height (px)` of the raw image files. These values were extracted early (e.g., by `build01A_compile_keyence_torch.py` and `build01B_compile_yx1_images_torch.py`) and propagated through its metadata system. This allowed for precise physical scaling of image snips.
*   **SAM2 Pipeline (Current Integration):** The `segmentation_sandbox`'s SAM2 pipeline, as currently integrated, does not carry forward these specific raw image dimensions. Its `experiment_metadata.json` and `GroundedSam2Annotations.json` primarily store pixel dimensions of *processed images* that the SAM2 models operate on, not the original raw image physical dimensions. This necessitated the use of empirically derived formulas (e.g., `(rs_factor / 3.648) * 6.5` for `px_dim_raw`) to bridge this information gap, introducing a "magic number" and a less direct data flow.

This gap represents a breakdown in information lineage, where crucial raw image properties are lost or abstracted away, leading to complex workarounds.

## 3. Proposed Solution: Direct Metadata Integration

The proposed solution is to directly integrate the raw image `Height (um)` and `Height (px)` into the `segmentation_sandbox`'s central `experiment_metadata.json` object at the earliest possible stage. This will ensure this critical information is available and directly accessible throughout the pipeline.

### 3.1. Strategy Overview

The strategy involves modifying the initial data organization stage of the SAM2 pipeline to extract and store the raw image dimensions, and then ensuring this information is correctly aggregated for downstream use.

### 3.2. Implementation Steps

#### **Phase 1: Data Organizer Enhancement (Primary Focus)**

1.  **Enhance `scripts/data_organization/data_organizer.py` - Core Schema Transformation:**

    **Key Method Updates:**
    - **`scan_video_directory()`** - Primary target for metadata integration
    - **`scan_organized_experiments()`** - Ensure compatibility with enhanced schema
    
    **Specific Implementation Changes:**
    *   **CSV Loading Logic**: For each video being processed, parse the `video_id` to extract `experiment_date` and `well_id`. Load the corresponding `metadata/built_metadata_files/{experiment_date}_metadata.csv` file.
    *   **Well-Level Metadata Integration**: Match CSV rows by `well_id` and extract well-constant metadata (`medium`, `genotype`, `chem_perturbation`, `start_age_hpf`, `embryos_per_well`, `temperature`, `well_qc_flag`). Store this directly at the video object level.
    *   **Image Dictionary Transformation**: Convert the existing `image_ids` list to a dictionary structure keyed by `image_id`. For each image, create the nested `raw_image_data_info` object.
    *   **Frame-Specific Metadata Population**: Match each `image_id` with CSV data by `well_id` and `time_int` (extracted from frame number). Populate the `raw_image_data_info` with frame-specific values including physical dimensions (`raw_height_um`, `raw_height_px`, etc.), acquisition parameters, and time information.
    *   **Error Handling**: Implement graceful fallbacks when CSV files are missing or data is incomplete, with appropriate warning messages.

#### **Phase 2: Bridge Script Integration (Secondary Priority)**

2.  **Update `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`:**
    *   Modify the bridge script to read the enhanced `experiment_metadata.json` structure (dictionary-based `image_ids`)
    *   Extract physical dimensions from `raw_image_data_info` and well metadata from video level
    *   Add new columns to output CSV: `raw_height_um`, `raw_width_um`, `raw_height_px`, `raw_width_px`, plus well metadata columns

#### **Phase 3: Legacy Build Script Simplification (Tertiary Priority)**

3.  **Simplify `src/build/build03A_process_images.py`:**
    *   Replace the empirical formula `(rs_factor / 3.648) * 6.5` with direct calculation from CSV data
    *   Update to use new CSV columns: `row['px_dim_raw'] = row['raw_height_um'] / row['raw_height_px']`

### 3.3. Detailed Metadata Schema and Integration Plan

Based on the analysis of `test_data/sample_built_metadata.csv` (a representative file from `metadata/built_metadata_files/`), the following insights are crucial for the integration:

**Content of `built_metadata_files/{experiment_date}_metadata.csv`:**

This CSV contains per-well-time-point metadata, including:
*   `well_id`, `well`, `nd2_series_num`, `microscope`, `time_int`, `Height (um)`, `Width (um)`, `Height (px)`, `Width (px)`, `BF Channel`, `Objective`, `Time (s)`, `experiment_date`, `medium`, `genotype`, `chem_perturbation`, `start_age_hpf`, `embryos_per_well`, `temperature`, `well_qc_flag`, `Time Rel (s)`.

**Enhanced `experiment_metadata.json` Schema Modification:**

To effectively integrate this information without redundancy and maintain logical structure, both the `image_ids` field and the video-level structure within `experiment_metadata.json` must undergo a significant schema enhancement.

### **Complete Enhanced Schema Structure:**

{
  "file_info": { ... },
  "experiments": {
    "<experiment_id>": {
      "experiment_id": "<experiment_id>",
      "created_time": "ISO-8601 or null",
      "metadata": { ... },               // optional experiment-level info
      "videos": {
        "<video_id>": {
          "video_id": "<video_id>",
          "well_id": "A01",
          "mp4_path": "...",             // from old doc (if present)
          "processed_jpg_images_dir": "...",
          "total_frames": 123,
          "image_size": [H, W],          // from old doc (if present)
          "source_csv": "...",           // from new doc (well-level fields)
          "medium": "E3",
          "genotype": "wildtype",
          "chem_perturbation": "none",
          "start_age_hpf": 24,
          "embryos_per_well": 1,
          ... and any other video level info from orignal gsam pipline and the build scripts amasses at this stage
          "image_ids": {
            "<image_id>": {
              "frame_index": 0,
              "raw_image_data_info": {
                "Height (um)": 7080.86,
                "Height (px)": 2189,
                "Width (um)": 7080.86,
                "Width (px)": 2189,
                "microscope": "YX1",
                "objective": "Plan Apo λ 4x",
                "channel": 0,
                "microscope": "YX1",
                "bf_channel": 0,
                "nd2_series_num": 1,
                "raw_time_s": 0.977,
                "relative_time_s": 0.0,
                "stitched_image_path": "/path/to/stitched.jpg" //This is the the original stitched (not the renamed one after processing)
              }
            }
          }
        }
      }
    }
  },
  "entity_tracking": {
    "experiments": [],
    "videos": [],
    "images": [],
    "embryos": [],
    "snips": []
  }
}



### **Schema Population Strategy:**

**1. Video-Level Metadata Population:**
- Parse `video_id` (e.g., "20240418_A01") to extract `experiment_date` ("20240418") and `well_id` ("A01")
- Load corresponding CSV: `metadata/built_metadata_files/{experiment_date}_metadata.csv`
- Match CSV rows by `well_id` to retrieve well-constant metadata
- Store well-level metadata directly at video object level: `medium`, `genotype`, `chem_perturbation`, `start_age_hpf`, `embryos_per_well`, `temperature`, `well_qc_flag`

**2. Image-Level Metadata Population:**
- Convert `image_ids` from list format to dictionary format keyed by `image_id`
- For each `image_id`, match with CSV by `well_id` and `time_int` (frame number)
- Populate `raw_image_data_info` with frame-specific data:
  - **Physical dimensions**: `raw_height_um`, `raw_width_um`, `raw_height_px`, `raw_width_px`
  - **Acquisition parameters**: `microscope`, `objective`, `bf_channel`, `nd2_series_num`
  - **Time information**: `raw_time_s`, `relative_time_s`
  - **File references**: `stitched_image_path`

**3. Data Source Integration:**
- Primary source: `metadata/built_metadata_files/{experiment_date}_metadata.csv` generated by legacy build scripts
- Fallback handling: Graceful degradation when CSV files are missing
- Data validation: Ensure physical dimensions are reasonable (>0) and required fields are present

**Impact on Downstream Scripts:**

This change to the `experiment_metadata.json` schema is **significant**. All scripts that read or write to `experiment_metadata.json` (e.g., `data_organizer.py`, `gdino_detection.py`, `sam2_segmentation.py`, `export_sam2_metadata_to_csv.py`) will need to be updated to expect `image_ids` as a dictionary of objects, not a list of strings. This represents a major refactoring effort across the `segmentation_sandbox` pipeline.

### 3.4. Consistency Across Microscope Data Sources
 Key Similarities:

   * Modular Design: Both scripts are structured with functions for distinct tasks (e.g., initial processing, stitching).
   * Metadata-Driven: Both collect and manage metadata using Pandas DataFrames.
   * Image-Level Granularity: Crucially, both process images at a granular level (well/time-point/position) and extract metadata for each. This directly maps to the image_id concept in
     the SAM2 pipeline.
   * Common Output: Both save their primary metadata output to metadata/built_metadata_files/{experiment_date}_metadata.csv. This is the consistent source we can leverage.
   * Stitched FF Image Output: Both produce stitched full-focus (FF) images, which are the inputs for later stages.

  Key Differences (Keyence Specifics):

   * Metadata Sourcing: Keyence uses scrape_keyence_metadata (which parses embedded XML metadata from raw Keyence files) to extract Height (um), Width (um), Height (px), Width (px),
     Objective, Time (s), and Channel. YX1 uses the nd2 library for similar information.
   * Focus Stacking: Keyence uses doLap for Laplacian focus stacking, while YX1 uses LoG_focus_stacker.
   * Stitching: Keyence explicitly performs stitching of individual image tiles using stitch2d.StructuredMosaic within stitch_experiment. YX1's build_ff_from_yx1 processes already
     organized ND2 files, implying stitching for YX1 would be a separate upstream step if multiple fields of view per well were acquired.

  Confirmation of Image-Level Data Creation:

  Yes, the metadata is indeed generated at the image level (or well/time-point/position level), which directly corresponds to the image_id in the SAM2 pipeline's experiment_metadata.json.
   For each unique image (well/time-point/position), a row of metadata is generated, including the physical and pixel dimensions.

  Reinforcing `image_id` Storage for Simplicity:

  Given that both Keyence and YX1 build scripts consistently produce this rich metadata at the image level, storing all relevant information directly within the image_id object in
  experiment_metadata.json is absolutely the most straightforward and simple approach for implementation. This aligns perfectly with the schema change proposed in
  refactor-005-metadata-integration-prd.md. It avoids complex lookups, redundant storage, and maintains a clear, self-contained record for each image.
  
## 4. Benefits

*   **Elimination of "Magic Numbers":** Removes the reliance on empirically derived constants like `3.648`, leading to a more transparent and robust scaling mechanism.
*   **Improved Data Lineage:** Ensures that critical raw image properties are explicitly carried forward from the earliest stage of the SAM2 pipeline.
*   **Enhanced Accuracy:** Direct sourcing of physical dimensions reduces potential for errors or inconsistencies introduced by estimation.
*   **Cleaner Codebase:** Simplifies downstream calculations in `build03A_process_images.py` by providing direct access to necessary metadata.
*   **Architectural Consistency:** Aligns the SAM2 pipeline's metadata handling more closely with the principles of comprehensive data tracking.

## 5. Reference to Previous Refactoring PRDs

This refactoring effort builds upon the insights and challenges encountered in previous stages. For context on the fundamental incompatibility and the evolution of the integration strategy, please refer to:

*   `docs/refactors/refactor-001-segmentation-pipeline-integration-prd.md`
*   `docs/refactors/refactor-002-segmentation-pipeline-integration-prd.md`
*   `docs/refactors/refactor-003-segmentation-pipeline-integration-prd.md`
*   `docs/refactors/refactor-004-debugging-and-stabilization.md` (if applicable, assuming this exists or is planned)
