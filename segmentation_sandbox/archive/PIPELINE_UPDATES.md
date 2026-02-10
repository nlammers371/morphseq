# Summary of Pipeline Updates

## Key Changes Made

### 1. **Fixed Directory Structure Logic**
- **Removed impossible QC flags**: `MISSING_INPUT_DIR` and `NO_INPUT_IMAGES` now raise exceptions instead of flags
- **Added nested directory support**: Pipeline now expects `stitched_dir/experiment_date/images` structure
- **Experiment ID extraction**: Automatically extracts experiment IDs from directory names (typically dates)

### 2. **Enhanced Video/Experiment ID System**
- **Full video IDs**: Videos now named as `experiment_id_video_id` (e.g., `20241215_well_A01.mp4`)
- **Metadata structure**: All outputs include `experiment_id`, `video_id`, and `full_video_id` fields
- **Improved filename parsing**: Robust extraction of well IDs and timepoints from various filename patterns

### 3. **Updated Detection Data Format**
- **Raw vs filtered detections**: Separate arrays for raw GroundingDINO outputs and filtered results
- **Detection statistics**: Track rejection reasons and filtering statistics
- **Updated QC flags**: Distinguish between raw detection issues and post-filtering problems

### 4. **Removed Subjective QC Flags**
- **Removed length-based flags**: No more `VERY_SHORT_SEQUENCE` or `SHORT_SEQUENCE` - let analysis decide
- **Removed size-based flags**: No more `MASK_TOO_SMALL`/`MASK_TOO_LARGE` - embryo size varies naturally
- **Kept technical flags**: Focus on actual corruption, processing failures, and dimension inconsistencies

### 5. **Consistent Data Structure**
All pipeline stages now include experiment and video identifiers:
```json
{
  "20241215_well_A01": {
    "experiment_id": "20241215",
    "video_id": "well_A01", 
    "full_video_id": "20241215_well_A01",
    // ... stage-specific data
  }
}
```

### 6. **Experiment Data Quality Control (QC) System Refactor**
- **Hierarchical JSON Structure**: QC system now uses `experiment_data_qc.json` with hierarchical organization mirroring experiment metadata
- **Multi-level QC**: Support for QC flags at experiment, video, image, and embryo levels
- **Unified Core Utilities**: All QC-related logic moved to `scripts/experiment_data_qc_utils.py` (moved from utils to scripts)
- **Author Tracking**: Every QC flag tracks who added it (manual vs automatic) with timestamps and notes
- **Flexible Flag Categories**: Predefined QC flag categories for each level with descriptions
- **COCO-inspired Structure**: JSON format inspired by COCO annotations for better organization
- **Legacy Compatibility**: Backward compatibility functions for existing CSV-based workflows
- **Better Maintenance**: Single JSON file eliminates data fragmentation and improves QC tracking

## Expected Input Structure

```
stitched_images_dir/
├── 20241215/              # Experiment date (experiment_id)
│   ├── well_A01_t001.tif  # Images with well and timepoint info
│   ├── well_A01_t002.tif
│   ├── well_B01_t001.tif
│   └── ...
├── 20241216/              # Another experiment
│   └── ...
└── 20241220/
    └── ...
```

## Updated QC Philosophy

### **Technical Issues → Flags**
- File corruption
- Dimension inconsistencies  
- Processing failures
- Single-frame sequences (can't make video)

### **Analysis Decisions → No Flags**
- Sequence length (2 frames vs 200 frames)
- Mask/embryo size (natural variation)
- Detection counts (before filtering)

### **Critical Issues → Exceptions**
- Missing directories
- No data to process
- Configuration errors

## Benefits

1. **Clearer data tracking**: Easy to trace any result back to specific experiment and well
2. **Better QC focus**: Flags indicate real technical problems, not analysis preferences
3. **Flexible analysis**: Raw and filtered data available for different analysis needs
4. **Robust processing**: Proper exception handling for unrecoverable errors
5. **Scalable structure**: Easy to process multiple experiments in batch

## Migration Notes

- Update any existing scripts that expect simple video IDs to use the new `experiment_id_video_id` format
- QC filtering logic should now check for specific technical flags rather than length/size flags
- Detection analysis can choose between raw and filtered detection arrays as needed

## New Features Added

### JPEG Frame Generation
- **Added individual JPEG frame creation** during video preparation stage
- Each video now generates corresponding JPEG frames stored in `data/intermediate/jpeg_frames/{video_id}/`
- JPEG frames use standard naming: `frame_0000.jpg`, `frame_0001.jpg`, etc.
- JPEG paths are included in video metadata as `jpeg_images` field
- Quality controlled by `jpeg_quality` parameter (default: 95)
- Frames are only created if they don't exist or if `overwrite_existing` is True
- Added logging for JPEG frame creation process

### Video Metadata Enhancement
- Added `jpeg_images` field containing paths to individual JPEG frames
- Updated config to include `paths.intermediate.jpeg_frames` directory
- JPEG creation process is logged for debugging

### Configuration Updates
- Added `paths.intermediate.jpeg_frames: "data/intermediate/jpeg_frames"` to config
- This supports downstream processing stages that need individual frame access
# Summary of Pipeline Updates

## Key Changes Made

### 1. **Fixed Directory Structure Logic**
- **Removed impossible QC flags**: `MISSING_INPUT_DIR` and `NO_INPUT_IMAGES` now raise exceptions instead of flags
- **Added nested directory support**: Pipeline now expects `stitched_dir/experiment_date/images` structure
- **Experiment ID extraction**: Automatically extracts experiment IDs from directory names (typically dates)

### 2. **Enhanced Video/Experiment ID System**
- **Full video IDs**: Videos now named as `experiment_id_video_id` (e.g., `20241215_well_A01.mp4`)
- **Metadata structure**: All outputs include `experiment_id`, `video_id`, and `full_video_id` fields
- **Improved filename parsing**: Robust extraction of well IDs and timepoints from various filename patterns

### 3. **Updated Detection Data Format**
- **Raw vs filtered detections**: Separate arrays for raw GroundingDINO outputs and filtered results
- **Detection statistics**: Track rejection reasons and filtering statistics
- **Updated QC flags**: Distinguish between raw detection issues and post-filtering problems

### 4. **Removed Subjective QC Flags**
- **Removed length-based flags**: No more `VERY_SHORT_SEQUENCE` or `SHORT_SEQUENCE` - let analysis decide
- **Removed size-based flags**: No more `MASK_TOO_SMALL`/`MASK_TOO_LARGE` - embryo size varies naturally
- **Kept technical flags**: Focus on actual corruption, processing failures, and dimension inconsistencies

### 5. **Consistent Data Structure**
All pipeline stages now include experiment and video identifiers:
```json
{
  "20241215_well_A01": {
    "experiment_id": "20241215",
    "video_id": "well_A01", 
    "full_video_id": "20241215_well_A01",
    // ... stage-specific data
  }
}
```

### 6. **Image Quality Control (QC) System Refactor**
- **Unified Core Utilities**: All QC-related logic (flagging, removing, loading, saving) is now centralized in `utils/image_quality_qc_utils/image_quality_qc_utils.py`.
- **Simplified CLI**: A single command-line tool, `utils/image_quality_qc_utils/image_quality_qc.py`, now handles all manual and automated QC operations.
- **Centralized QC Data**: All QC flags are stored in a single CSV file at `data/quality_control/image_quality_qc.csv`, eliminating data fragmentation.
- **Streamlined Automated QC**: The automated QC script (`scripts/02_image_quality_qc.py`) now uses only the blur metric and supports parallel processing for better performance.
- **Updated Jupyter Demo**: The notebook `utils/image_quality_qc_utils/image_quality_qc_demo.ipynb` has been revamped to provide a clear guide to the new system.
- **Code Cleanup**: Redundant scripts (e.g., `manual_image_quality_qc.py`) have been removed to create a cleaner codebase.

## Expected Input Structure

```
stitched_images_dir/
├── 20241215/              # Experiment date (experiment_id)
│   ├── well_A01_t001.tif  # Images with well and timepoint info
│   ├── well_A01_t002.tif
│   ├── well_B01_t001.tif
│   └── ...
├── 20241216/              # Another experiment
│   └── ...
└── 20241220/
    └── ...
```

## Updated QC Philosophy

### **Technical Issues → Flags**
- File corruption
- Dimension inconsistencies  
- Processing failures
- Single-frame sequences (can't make video)

### **Analysis Decisions → No Flags**
- Sequence length (2 frames vs 200 frames)
- Mask/embryo size (natural variation)
- Detection counts (before filtering)

### **Critical Issues → Exceptions**
- Missing directories
- No data to process
- Configuration errors

## Benefits

1. **Clearer data tracking**: Easy to trace any result back to specific experiment and well
2. **Better QC focus**: Flags indicate real technical problems, not analysis preferences
3. **Flexible analysis**: Raw and filtered data available for different analysis needs
4. **Robust processing**: Proper exception handling for unrecoverable errors
5. **Scalable structure**: Easy to process multiple experiments in batch

## Migration Notes

- Update any existing scripts that expect simple video IDs to use the new `experiment_id_video_id` format
- QC filtering logic should now check for specific technical flags rather than length/size flags
- Detection analysis can choose between raw and filtered detection arrays as needed

## New Features Added

### JPEG Frame Generation
- **Added individual JPEG frame creation** during video preparation stage
- Each video now generates corresponding JPEG frames stored in `data/intermediate/jpeg_frames/{video_id}/`
- JPEG frames use standard naming: `frame_0000.jpg`, `frame_0001.jpg`, etc.
- JPEG paths are included in video metadata as `jpeg_images` field
- Quality controlled by `jpeg_quality` parameter (default: 95)
- Frames are only created if they don't exist or if `overwrite_existing` is True
- Added logging for JPEG frame creation process

### Video Metadata Enhancement
- Added `jpeg_images` field containing paths to individual JPEG frames
- Updated config to include `paths.intermediate.jpeg_frames` directory
- JPEG creation process is logged for debugging

### Configuration Updates
- Added `paths.intermediate.jpeg_frames: "data/intermediate/jpeg_frames"` to config
- This supports downstream processing stages that need individual frame access
