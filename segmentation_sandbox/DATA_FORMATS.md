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

**PIPELINE DATA STRUCTURE FORMAT (IMPORTANT):**
- Top-level directories are experiment IDs (typically dates like `YYYYMMDD`)
- Each experiment directory contains only image files (no subdirectories)
- Image filenames in these directories should follow a pattern that allows extraction of:
  - Well ID (e.g., `A01`, `B02`)
  - Timepoint number (e.g., `t001`, `t002`) - the sequence that images are taken in that well (video)

## ID System

The pipeline uses a hierarchical, consistent ID system:

- **experiment_id**: `YYYYMMDD` (e.g., `20241215`)
- **video_id**: `{experiment_id}_{well_id}` (e.g., `20241215_A01`)
- **image_id**: `{experiment_id}_{well_id}_{time_point}` (e.g., `20241215_A01_t001`)
- **embryo_id**: `{experiment_id}_{well_id}_{time_point}_{embryo_number}` (e.g., `20241215_A01_t001_e01`) (1-indexed)

## Quality Control (QC) System

### Hierarchical JSON Structure

The QC system uses a hierarchical JSON format (`data/quality_control/experiment_data_qc.json`) inspired by COCO and embryo_metadata.json:

```json
{
  "valid_qc_flag_categories": {
    "experiment_level": {
      "POOR_IMAGING_CONDITIONS": "Overall poor imaging conditions for entire experiment",
      "CONTAMINATION": "Bacterial or other contamination affecting experiment",
      "EQUIPMENT_FAILURE": "Microscope or other equipment issues",
      "PROTOCOL_DEVIATION": "Deviation from standard imaging protocol"
    },
    "video_level": {
      "DRY_WELL": "Well dried out during imaging",
      "FOCUS_DRIFT": "Focus drifted significantly during video",
      "STAGE_DRIFT": "Stage movement during video acquisition",
      "SHORT_SEQUENCE": "Video sequence too short for analysis",
      "BUBBLE_FORMATION": "Bubbles formed during imaging"
    },
    "image_level": {
      "BLUR": "Image is blurry (low variance of Laplacian)",
      "DARK": "Image is too dark (low mean brightness)",
      "BRIGHT": "Image is overexposed (high mean brightness)",
      "CORRUPT": "Cannot read/process image",
      "OUT_OF_FOCUS": "Image is out of focus",
      "DEBRIS": "Contains debris affecting analysis",
      "LOW_CONTRAST": "Poor contrast (low standard deviation)",
      "ARTIFACT": "Contains artifacts or technical issues"
    },
    "embryo_level": {
      "DEAD_EMBRYO": "Embryo appears dead",
      "EMBRYO_NOT_DETECTED": "No embryo detected in expected location",
      "MALFORMED": "Embryo shows developmental abnormalities",
      "MULTIPLE_EMBRYOS": "Multiple embryos detected when expecting one",
      "PARTIAL_EMBRYO": "Only part of embryo visible in frame"
    }
  },
  "experiments": {
    "20241215": {
      "flags": ["POOR_IMAGING_CONDITIONS"],
      "authors": ["mcolon"],
      "notes": ["Manual review - lighting issues throughout"],
      "videos": {
        "20241215_A01": {
          "flags": ["DRY_WELL"],
          "authors": ["mcolon"],
          "notes": ["Well dried out after frame 50"],
          "images": {
            "20241215_A01_t001": {
              "flags": ["BLUR"],
              "authors": ["automatic"],
              "notes": ["Automatic: blur_score=45 < threshold=100"]
            },
            "20241215_A01_t002": {
              "flags": [],
              "authors": [],
              "notes": []
            }
          },
          "embryos": {
            "20241215_A01_t001_e01": {
              "flags": ["DEAD_EMBRYO"],
              "authors": ["expert_reviewer"],
              "notes": ["Manual inspection - no movement detected"]
            }
          }
        }
      }
    }
  }
}
```

### QC Workflow

The proper QC workflow follows this sequence:

1. **INITIALIZE**: `initialize_qc_structure_from_metadata()` populates QC JSON with ALL experiments/videos/images from metadata
2. **MANUAL**: Human experts review and flag problematic entities at any level
3. **AUTOMATIC**: `02_image_quality_qc.py` performs algorithmic QC on remaining unflagged images
4. **DONE**: Complete hierarchical QC dataset ready for pipeline

### QC Philosophy

- **Default state**: Entities with no flags are assumed to be good quality
- **Only flag problems**: QC flags indicate actual detected issues
- **Manual precedence**: Manual annotations always take precedence over automatic
- **Multi-level**: QC can be applied at experiment, video, image, or embryo levels
- **Author tracking**: Every flag tracks who added it (manual vs automatic)

### Key QC Functions

```python
# Initialize QC structure from metadata
qc_data = initialize_qc_structure_from_metadata(
    data_dir=data_dir,
    experiment_metadata_path=metadata_path
)

# Flag entities at different levels
flag_experiment(data_dir, "20241215", "POOR_IMAGING_CONDITIONS", "mcolon", "Bad lighting")
flag_video(data_dir, "20241215_A01", "DRY_WELL", "mcolon", "Well dried out")
flag_image(data_dir, "20241215_A01_t001", "BLUR", "automatic", "blur_score=45")
flag_embryo(data_dir, "20241215_A01_t001_e01", "DEAD_EMBRYO", "expert", "No movement")

# Get QC flags for entities
flags = get_qc_flags(data_dir, "image", "20241215_A01_t001", 
                     {"experiment_id": "20241215", "video_id": "20241215_A01"})

# Get overall QC summary
summary = get_qc_summary(data_dir)
```

## Pipeline Stages and Outputs

### Stage 0: Model Installation
Installing correct packages as done in `morphseq/segmentation_sandbox/scripts/model_installation_utils`

### Stage 1: Organize Experiments and Format Videos and Images

The script `scripts/01_prepare_videos.py` incrementally processes raw stitched images to:
- Discover and group images by experiment (date) and well (video_id)
- Convert source images into downscaled JPEGs
- Generate summary videos with frame overlays
- Maintain a cumulative JSON metadata file (`raw_data_organized/experiment_metadata.json`) that tracks all processed experiments, videos, and images

### Stage 2: Experiment Data Quality Control

The script `scripts/02_image_quality_qc.py` performs hierarchical quality control:
- Initialize QC structure from experiment metadata
- Automatically flag problematic images using algorithmic detection
- Support manual QC annotations at all levels
- Generate QC summaries and reports

The QC system maintains a hierarchical JSON structure that mirrors the experiment metadata organization, making it easy to track quality issues at the appropriate level (experiment, video, image, or embryo).

## Benefits of New QC System

1. **Hierarchical organization**: Mirror experiment metadata structure
2. **Multi-level QC**: Flag issues at the appropriate level (experiment/video/image/embryo)
3. **Author tracking**: Know who flagged what (manual vs automatic)
4. **Flexible categories**: Easy to add new QC flag types
5. **Better maintenance**: Single JSON file with clear structure
6. **Legacy compatibility**: Maintains compatibility with existing workflows
7. **Scalable**: Easy to process multiple experiments and maintain QC state