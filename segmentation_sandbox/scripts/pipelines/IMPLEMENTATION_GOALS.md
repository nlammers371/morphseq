# Pipeline Scripts Implementation Goals

## Overview
The pipeline scripts provide a user-friendly interface to the MorphSeq segmentation pipeline. Each script corresponds to a major processing stage and can be run independently or as part of the full pipeline.

## Design Principles

### 1. Consistent Interface
All pipeline scripts follow the same design patterns:
- **Dry-run capability**: `--dry-run` flag shows what would be processed without executing
- **Verbose output**: `--verbose` flag provides detailed processing information
- **Experiment filtering**: `--experiments` parameter to process specific experiments
- **Progress indicators**: Clear status messages with emojis for easy scanning
- **Error handling**: Graceful error handling with helpful error messages

### 2. Argument Patterns
Based on the original `run_01_03_04_05.sh` script, the following arguments should be consistent:

#### Common Arguments (all scripts)
- `--verbose`: Verbose output (action="store_true")
- `--dry-run`: Show processing plan without execution (action="store_true")
- `--experiments`: Comma-separated experiment IDs to process (optional, default: all)

#### Script-Specific Arguments
- `--workers`: Number of parallel workers (default: 8 for compute-intensive tasks)
- `--config`: Pipeline config YAML file path (for model configurations)
- `--save-interval`: Auto-save frequency for long-running processes

### 3. Autosave Requirements
**All annotation and metadata classes must implement autosave functionality:**
- Constructor parameter: `auto_save_interval: Optional[int] = None`
- Auto-save triggers after N operations (experiments, annotations, etc.)
- Manual save: `save()` method resets operation counter
- Disable autosave: Set interval to `None` or `0`
- Examples:
  - `ExperimentMetadata`: Auto-save after N experiments processed
  - `GroundedDinoAnnotations`: Auto-save after N images processed  
  - `EmbryoMetadata`: Auto-save after N annotation operations
  - Pipeline scripts: `--save-interval` parameter passed to underlying classes

### 4. EntityIDTracker Integration
**EntityIDTracker Design Philosophy:**
- **Pure Container**: Serves as a simple container for entity data validation and tracking
- **Embedded Approach**: Entity trackers embedded directly in pipeline JSON files (not separate files)
- **Implicit Context**: Pipeline step is implicit from file context (Module 0 files = Module 0 tracking)
- **Static Methods**: Format-specific operations (JSON, CSV) handled by static helper methods
- **Clean Interface**: `add_entity_tracker()`, `update_entity_tracker()`, `compare_files()`, `validate_consistency()`
- **Integration**: All save() methods should call `EntityIDTracker.update_entity_tracker()` before writing

### 5. File Path Conventions
Following the original script structure:
- **Input directory**: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images/`
- **Output base**: `$ROOT/data/`
- **Config file**: `$ROOT/configs/pipeline_config.yaml`
- **Metadata file**: `$ROOT/data/raw_data_organized/experiment_metadata.json`

### 6. Output Structure
Each script creates structured output following this pattern:
```
data/
├── raw_data_organized/
│   └── experiment_metadata.json
├── annotation_and_masks/
│   ├── gdino_annotations/
│   ├── sam2_annotations/
│   └── jpg_masks/
├── embryo_metadata/
└── videos/
```

## Implementation Status

### ✅ Completed Scripts

#### 01_prepare_videos.py
- **Purpose**: Organize raw stitched images and create videos
- **Status**: Template complete, needs Module 0 integration
- **Key Features**: 
  - Dry-run capability ✅
  - Smart incremental processing (skips already processed experiments) ✅
  - Experiment filtering ✅
  - Worker parallelization ✅
  - Directory validation ✅
  - Processing status analysis ✅
  - Auto-save intervals ✅

#### 03_gdino_detection.py
- **Purpose**: Run GroundedDINO detection with filtering
- **Status**: Template complete, needs Module 2 integration
- **Key Features**:
  - Dry-run capability ✅
  - Detection parameter tuning ✅
  - High-quality filtering ✅
  - Auto-save intervals ✅

#### 04_sam2_segmentation.py
- **Purpose**: SAM2 video segmentation using detection results
- **Status**: Template complete, needs Module 3 integration
- **Key Features**:
  - Dry-run capability ✅
  - Temporal processing parameters ✅
  - Object tracking limits ✅
  - Segmentation validation ✅

#### 05_analysis_export.py
- **Purpose**: Export masks and perform analysis
- **Status**: Template complete, needs Module 4 integration
- **Key Features**:
  - Dry-run capability ✅
  - Multiple export formats ✅
  - Quality filtering ✅
  - Analysis reporting ✅

### 🚧 Missing Scripts

#### 02_image_quality_qc.py
- **Purpose**: Perform quality control on images
- **Status**: Not yet implemented
- **Requirements**: Should follow same pattern as other scripts

### 📋 Default Parameters

Based on `run_01_03_04_05.sh`, these defaults should be used:

```python
# Detection parameters
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Processing parameters
WORKERS = 8
SAVE_INTERVAL = 250  # for SAM2
AUTO_SAVE_EXPERIMENTS = 1  # for metadata classes
PROPAGATION_FRAMES = 10
MAX_OBJECTS_PER_FRAME = 20

# Quality parameters
MIN_MASK_AREA = 100
QUALITY_THRESHOLD = 0.5
```

## Example Usage

### Full Pipeline Execution
```bash
# Step 1: Prepare data
python scripts/pipelines/01_prepare_videos.py \
  --input /net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images/ \
  --output data/ \
  --workers 8 \
  --verbose

# Step 3: Detection
python scripts/pipelines/03_gdino_detection.py \
  --metadata data/raw_data_organized/experiment_metadata.json \
  --output data/annotation_and_masks/gdino_annotations/gdino_annotations.json \
  --config configs/pipeline_config.yaml \
  --confidence-threshold 0.45 \
  --verbose

# Step 4: Segmentation
python scripts/pipelines/04_sam2_segmentation.py \
  --metadata data/raw_data_organized/experiment_metadata.json \
  --annotations data/annotation_and_masks/gdino_annotations/gdino_annotations.json \
  --output data/annotation_and_masks/sam2_annotations/sam2_annotations.json \
  --config configs/pipeline_config.yaml \
  --save-interval 250 \
  --verbose

# Step 5: Export
python scripts/pipelines/05_analysis_export.py \
  --metadata data/raw_data_organized/experiment_metadata.json \
  --segmentations data/annotation_and_masks/sam2_annotations/sam2_annotations.json \
  --output data/annotation_and_masks/jpg_masks/ \
  --workers 8 \
  --verbose
```

### Testing and Development
```bash
# Dry-run to check processing plan
python scripts/pipelines/01_prepare_videos.py \
  --input /path/to/input \
  --output /path/to/output \
  --dry-run \
  --verbose

# Process specific experiments only
python scripts/pipelines/03_gdino_detection.py \
  --metadata experiment_metadata.json \
  --output detections.json \
  --experiments "20240506,20250703_chem3_28C_T00_1325" \
  --dry-run
```

## Implementation Priorities

1. **High Priority**: 
   - Complete Module 0 integration in `01_prepare_videos.py`
   - Add dry-run capability to all scripts ✅
   - Ensure consistent argument patterns ✅

2. **Medium Priority**:
   - Implement `02_image_quality_qc.py`
   - Add configuration file support
   - Enhanced error reporting

3. **Low Priority**:
   - Progress bars for long operations
   - Resume capability for interrupted processing
   - Parallel experiment processing

## Testing Strategy

Each script should be tested with:
1. **Dry-run mode**: Verify processing plan is correct
2. **Small dataset**: Test with 1-2 experiments
3. **Parameter validation**: Test edge cases and invalid inputs
4. **Integration**: Test script chaining with real data

## Notes

- All scripts create placeholder outputs when modules aren't implemented
- **Smart incremental processing**: Scripts detect already processed experiments and skip them
- **Dry-run analysis**: Shows exactly what's new vs already processed
- Error messages include helpful next steps
- Status indicators use emojis for easy visual scanning
- File paths are validated before processing
- Progress is logged to both stdout and log files when appropriate
