# MorphSeq Pipeline Data Formats and QC Flags

## Overview

This document describes all the data formats, QC flags, and intermediate outputs used throughout the MorphSeq embryo segmentation pipeline.

## Expected Input Data Structure

The pipeline expects stitched images to be organized in the following directory structure:

```
stitched_images_dir/
├── 20241215/          # Experiment date directory (experiment_id)
│   ├── well_A01_t001.tif
│   ├── well_A01_t002.png
│   ├── well_A02_t001.jpg
│   └── ...
├── 20241216/          # Another experiment
│   ├── well_A01_t001.tif
│   └── ...
└── 20241220/
    └── ...
```

**PIPELNE DATA STRCUTURE FORMAT (IMPORTANT):**
- Top-level directories are experiment IDs (typically dates like `YYYYMMDD`)
-  Each experiment directory contains only image files (no subdirectories)
  -  Image filenames in these directories should follow a pattern that allows extraction of:
  - Well identifier (e.g., `A01`, `B12`) - "well"
  - Timepoint number (e.g., `t001`, `t002`) " the sequence that images are taken in that well (video)" 

- from this there are video_ids, image_ids, and embryo_ids are formated 
- experiment id is `YYYYMMDD`
- video id is `{experiment_id}_{well_id}` (e.g., `20241215_A01`)
- image_id id `{experiment_id}_{well_id}_{time_point}` (e.g., `20241215_A01_t001`)
- embryo_id `{experiment_id}_{well_id}_{time_point}_{embryo_number}` (e.g., `20241215_A01_t001_e01`) (1 indexed)

- Supported image formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`
- this is how we will keep track of data in morphseq pipeline processing this is very important 
## ID System

The pipeline uses a simple, consistent ID system:

- **experiment_id**: Directory name (e.g., `20241215`)
- **video_id**: `{experiment_id}_{well_id}` (e.g., `20241215_A01`)
- **image_id**: Assigned sequentially for frames within COCO format
- **annotation_id**: Added during detection/tracking stages (embryo_ids)

## Pipeline Stages and Outputs

### Stage 0:
installing correcpt packages as done in morphseq/segmentation_sandbox/scripts/model_installation_utils 

### Stage 1: Organize Experiments and Format Videos and Images

The script `scripts/01_prepare_videos.py` incrementally processes raw stitched images to:
- Discover and group images by experiment (date) and well (video_id).
- Convert source images into downscaled JPEGs.
- Generate summary videos with frame overlays.
- Maintain a cumulative JSON metadata file (`raw_data_organized/experiment_metadata.json`) that tracks all processed
  experiments, videos, and images, skipping already-processed items unless `--overwrite` is specified.

**Structure of `experiment_metadata.json`:**
```json
{
  "script_version": "01_prepare_videos.py",
  "creation_time": "YYYY-MM-DDThh:mm:ss",
  "experiment_ids": ["20240411", ...],
  "video_ids": ["20240411_A01", ...],
  "image_ids": ["20240411_A01_0000", ...],
  "experiments": {
    "20240411": {
      "experiment_id": "20240411",
      "first_processed_time": "YYYY-MM-DDThh:mm:ss",
      "last_processed_time": "YYYY-MM-DDThh:mm:ss",
      "videos": {
        "20240411_A01": {
          "video_id": "20240411_A01",
          "well_id": "A01",
          "mp4_path": "/path/to/20240411/vids/20240411_A01.mp4",
          "processed_jpg_images_dir": "/path/to/20240411/images/20240411_A01",
          "image_ids": ["20240411_A01_0000", ...],
          "total_source_images": 100,
          "valid_frames": 100,
          "video_resolution": [512, 512],
          "last_processed_time": "YYYY-MM-DDThh:mm:ss"
        }
      }
    }
  }
}
```
Note: downstream scripts should reference this metadata structure rather than re-reading raw JSON directories.

### Stage 2: Image Quality Control

This stage is for assessing the quality of the images before they are used for segmentation. It involves both automated and manual processes to flag images that are blurry, dark, bright, or have other issues. The QC flags are stored in a single CSV file, `image_quality_qc.csv`, located in the `data/quality_control` directory. This file is the single source of truth for image quality.

**Key components:**

- **`image_quality_qc.csv`**: A CSV file that stores all QC flags. Each row corresponds to an image and includes the `experiment_id`, `video_id`, `image_id`, the `qc_flag` itself, any `notes`, and the `annotator` (which can be a person's name or 'automatic').
- **`utils/image_qc_utils.py`**: A Python script containing the core functions for managing the QC data. This includes functions for loading, saving, flagging, and removing QC entries.
- **`scripts/manual_image_quality_qc.py`**: A command-line tool for manually adding, removing, or checking QC flags. This allows for human-in-the-loop quality control.
- **`scripts/02_image_quality_qc.py`**: A script for running automated QC checks on images. It calculates metrics like blurriness and brightness and assigns QC flags automatically.

### Stage 3 (gdino): 
Apply Grounded Dino to all images (recording annotations in gdino annotations (recording prompt and what model configs were used) 
saving this output soemwhere in data
Flags examples:

### Stage 4 SAM2: 
- choose 
Apply grounded sam to track these 
### Stage 5:





1. **Video-Centric COCO**: Traditional COCO extended with video and experient metadata
2. **Temporal Tracking**: Trajectories link annotations across time
3. **Robust QC**: Comprehensive flagging at all levels (global, video, frame, trajectory)
4. **Intermediate Outputs**: All stages save intermediate results for debugging
5. **Analysis-Ready**: Final outputs in both COCO (for ML) and CSV (for analysis)

## Usage in Analysis

- **COCO JSON**: Use for training/evaluation of detection/tracking models
- **Trajectory CSV**: Use for statistical analysis of embryo development
- **QC Reports**: Use for data quality assessment and filtering
- **Video Metadata**: Use for understanding data provenance and processing parameters
