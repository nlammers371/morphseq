# Save image_id with Channel Information Throughout Pipeline

## Overview
This document outlines the changes needed to save `image_id` values WITH channel information to disk throughout the morphseq pipeline. Currently, the pipeline stores frame numbers and ignores channel information from stitch files, losing both the full `image_id` context and channel traceability needed for proper entity tracking and future multi-channel fluorescence support.

## Current State Analysis

### Key Files Using frame_number:
1. **scripts/utils/parsing_utils.py** (lines 32, 42, 90-91, 122, 174-191, 220, 228, 236-238, 252, 255, 260-262, 284-289, 292-297, 305-310)
2. **scripts/data_organization/data_organizer.py** (line 755)
3. **scripts/utils/video_generation/** (multiple files)
4. **scripts/detection_segmentation/sam2_utils.py**
5. **scripts/annotations/** (multiple files)

### Current Behavior:
- **Input Files**: `A01_t0000_ch00_stitch.png` (contains well_id, frame, AND channel)
- **Current Parsing**: Extracts well_id and frame, **IGNORES channel information**
- **Disk Storage**: Images saved as `NNNN.jpg` (frame number only, channel lost)
- **JSON Metadata**: Contains `image_ids` like `"20240411_A01_t0000"` (no channel info)
- **Pipeline Processing**: Uses `frame_number` for internal processing, channel information discarded

### Target Behavior:
- **Input Files**: `A01_t0000_ch00_stitch.png` (same)
- **New Parsing**: Extract well_id, frame, AND channel
- **Disk Storage**: Images saved as `{image_id}.jpg` → `20240411_A01_ch00_t0000.jpg`
- **JSON Metadata**: Contains `image_ids` like `"20240411_A01_ch00_t0000"` (with channel)
- **Pipeline Processing**: Full `image_id` with channel preserved throughout

## Required Changes

### 1. Core Utilities (scripts/utils/parsing_utils.py)
**Impact**: HIGH - Core parsing functions used throughout pipeline

**Changes Needed**:
- **Update ID Format Documentation**: Change examples from `"20250624_..._H01_t0042"` to `"20250624_..._H01_ch00_t0042"`
- **Update `get_entity_type()`**: Add regex to recognize `_ch\d+_t\d{3,4}$` pattern while maintaining `_t\d{3,4}$` backward compatibility
- **Update `_parse_backwards_image()`**: Extract both channel and frame from image_id, add `"channel"` field to returned dictionary
- **Update `build_image_id()`**: Add channel parameter: `build_image_id(video_id, frame_number, channel=0)`
- **Update reverse-mapping functions**: Handle both `_ch00_t0000` and `_t0000` formats in filename conversion

**Example Changes**:
```python
# NEW: get_entity_type() regex update
elif re.search(r'_ch\d+_t\d{3,4}$', entity_id) or re.search(r'_t\d{3,4}$', entity_id):
    return "image"

# NEW: build_image_id() with channel support  
def build_image_id(video_id: str, frame_number: int, channel: int = 0) -> str:
    return f"{video_id}_ch{channel:02d}_t{frame_number:04d}"

# NEW: _parse_backwards_image() with channel extraction
match = re.search(r'_ch(\d+)_t(\d{3,4})$|_t(\d{3,4})$', image_id)
# Extract both channel and frame, default channel=0 for legacy format
```

### 2. Data Organization (scripts/data_organization/data_organizer.py)
**Impact**: HIGH - Core data organization module

**Changes Needed**:
- **Update `parse_stitch_filename()`**: Extract channel from filenames like `A01_t0000_ch00_stitch.png`, return `(well_id, frame, channel)` tuple
- **Update `organize_experiment()`**: Handle new 3-tuple return from `parse_stitch_filename()`, pass channel to `process_well()`
- **Update `process_well()`**: Use `build_image_id(video_id, int(frame), channel)` for JPEG filename generation
- **Update `scan_video_directory()`**: Parse channel from existing image_id filenames, handle both old and new formats
- **Update `get_image_path_from_id()`**: Work with new channel-inclusive image_id format

**Key Implementation**:
```python
# NEW: parse_stitch_filename() with channel extraction
def parse_stitch_filename(filename):
    well_match = re.search(r'([A-H]\d{2})', filename)
    frame_match = re.search(r't?(\d{3,4})', filename) 
    channel_match = re.search(r'ch(\d{2})', filename)  # NEW
    
    if well_match and frame_match:
        well_id = well_match.group(1)
        frame = frame_match.group(1)
        channel = int(channel_match.group(1)) if channel_match else 0  # Default ch00
        return well_id, frame, channel
```

### 3. Pipeline Scripts
**Impact**: MEDIUM - Scripts already use image_id for processing

**Current State**: Most pipeline scripts (03_gdino_detection.py, 04_sam2_segmentation.py, 06_export_masks.py) already work with `image_id` values and use parsing utilities to resolve to file paths.

**Changes Needed**:
- Update any direct file path construction to use updated utilities
- Verify compatibility with new filename format
- Test end-to-end pipeline with new format

### 4. Video Generation (scripts/utils/video_generation/)
**Impact**: MEDIUM - Video overlay functionality

**Changes Needed**:
- Update image ID overlay positioning and formatting
- Ensure video generation works with both filename formats
- Consider displaying full `image_id` in video overlays instead of just frame numbers

### 5. Mask Export (scripts/utils/simple_mask_exporter.py)
**Impact**: LOW-MEDIUM - Export utilities

**Changes Needed**:
- Update mask filename generation to match image filename format
- Ensure exported masks maintain `image_id` association
- Consider naming exported masks with full `image_id` for better traceability

## Implementation Strategy

### **Backward Compatibility Approach**
1. **Default Channel**: Use `ch00` as default for legacy data without channel information
2. **Dual Format Support**: All parsing functions handle both old (`_t0000`) and new (`_ch00_t0000`) formats
3. **Graceful Degradation**: If channel info missing from stitch files, default to channel 0
4. **Mixed Dataset Support**: Pipeline works with combination of old and new format files

### **Channel Extraction Logic**
```regex
# Primary pattern (new format with channel)
_ch(\d+)_t(\d{3,4})$

# Fallback pattern (legacy format without channel) 
_t(\d{3,4})$

# Stitch filename channel extraction
ch(\d{2})  # Extract channel from input filenames
```

### **Phase 1: Core Utilities (Channel-Aware Parsing)**
1. Update `parsing_utils.py` with channel support and backward compatibility
2. Maintain support for existing `_t0000` format image_ids
3. Test utilities with both old and new image_id formats

### **Phase 2: Data Organization (Channel Extraction)**
1. Update `data_organizer.py` to extract channel from stitch filenames
2. Generate image_ids with channel information: `_ch00_t0000`
3. Test with sample multi-channel stitch files

### **Phase 3: Pipeline Integration (Channel Preservation)**
1. Update pipeline scripts to work with channel-inclusive image_ids
2. Run integration tests with real multi-channel data
3. Verify all downstream processes preserve channel information

### **Phase 4: Future Multi-Channel Support**
1. Test with actual fluorescence channel data (ch01, ch02, etc.)
2. Validate channel-specific processing capabilities
3. Document multi-channel workflow patterns

## Considerations

### Storage Impact
- **Filename Length**: Channel-inclusive `image_id` filenames will be longer (e.g., `20240411_A01_ch00_t0000.jpg` vs `0000.jpg`)
- **Directory Scanning**: Slightly slower due to longer filenames, but improved traceability
- **Disk Usage**: Negligible increase (filename length in directory entries)

### Backward Compatibility  
- **Existing Data**: Must handle both filename formats during transition
- **Mixed Datasets**: Pipeline should work with both old and new formats
- **Migration Path**: Provide utility to convert existing datasets if needed

### Performance Impact
- **File Operations**: Minimal impact on I/O performance
- **Memory Usage**: Slightly higher memory for filename strings
- **Processing Speed**: No significant impact on core pipeline performance

## Benefits

### Improved Traceability
- Full context available in filename without metadata lookup
- Easier debugging and data exploration
- Better integration with external tools

### Enhanced Entity Tracking
- Consistent entity identification across all pipeline stages
- **Channel information preserved**: Critical for multi-channel fluorescence analysis
- Reduced dependency on metadata for file identification
- Simplified data export and sharing with full context

### Future-Proofing
- **Multi-Channel Ready**: Support for fluorescence channels (brightfield, GFP, RFP, etc.)
- Easier integration with expanded entity tracking
- Better support for parallel processing and distributed systems
- Improved data lineage tracking with channel context

## Recommended Timeline

**Week 1**: Implement core parsing utilities with channel support and backward compatibility
**Week 2**: Update data_organizer.py to extract channel from stitch files and generate channel-inclusive image_ids  
**Week 3**: Update pipeline scripts and run integration tests with multi-channel data
**Week 4**: Documentation, validation with real fluorescence data, and migration planning

## Testing Requirements

1. **Unit Tests**: Update parsing utility tests for channel extraction and dual format support
2. **Integration Tests**: Full pipeline run with channel-inclusive image_ids
3. **Backward Compatibility**: Mixed format dataset processing (old `_t0000` + new `_ch00_t0000`)
4. **Channel Extraction**: Validate proper channel parsing from stitch filenames like `A01_t0000_ch00_stitch.png`
5. **Performance Tests**: Ensure no significant performance regression
6. **Multi-Channel Support**: Test with different channel numbers (ch00, ch01, ch02)
7. **Data Integrity**: Verify all entity associations and channel information are preserved


## Journal (progress log)

2025-08-18: Implemented the simplest behavior in the core parsing utilities:
images are saved on disk using the full `image_id` as the filename
(`{image_id}.jpg`). All helper functions were updated to match this
convention and to still accept legacy numeric filenames when reading.

Steps completed in this session:
- Updated `scripts/utils/parsing_utils.py` to always use the full
    `image_id` when building filenames and paths (no feature flags):
    - `get_image_filename_from_id(image_id)` now returns `{image_id}.jpg`.
    - `get_relative_image_path(image_id)` / `build_image_path_from_base(...)`
        were updated to use the full-id filenames.
    - `image_id_to_disk_filename(image_id)` returns `{image_id}.jpg`.
    - `get_image_id_from_filename(video_id, filename)` added to reverse-map
        on-disk filenames back to `image_id`, with a numeric fallback for old
        `NNNN.jpg` names.

What I validated quickly:
- Confirmed the module imports and the primary helpers return the expected
    full-id filenames and perform reverse mapping for both full and numeric
    filenames.

Next recommended steps (manual):
1. Update `scripts/data_organization/data_organizer.py` to write images with
     the new `{image_id}.jpg` filenames by switching calls to
     `build_image_path_from_base` / `get_relative_image_path` to the new
     convention.
2. Update all pipeline callers (SAM2 scripts, mask exporters, video
     generation) to read/write `{image_id}.jpg` and use
     `get_image_id_from_filename` when converting filenames back to IDs.
3. Add unit tests for the new helpers and an integration test to run the
     pipeline on a small sample dataset after switching naming.

If you'd like, I can continue and patch `data_organizer.py` next to wire in
the config flag and add a small unit test; say the word and I'll proceed.

Additional progress (2025-08-18):
- Updated `scripts/data_organization/data_organizer.py` to write JPEGs using
    the full `image_id` as the filename (`{video_id}_tNNNN.jpg`) and to
    update scanning/path helpers (`scan_video_directory`, `get_image_path_from_id`)
    so metadata now records full `image_id`s and `processed_jpg_images_dir` still
    points to the images dir for each video.
- Performed quick smoke tests in the project environment to import the
    modified modules and exercise the primary helpers (`build_image_id`,
    `get_image_filename_from_id`, `get_image_path_from_id`) to ensure no
    import-time errors and expected return values.

**IMPORTANT UPDATE (2025-08-19): Channel Support Requirements**
- Discovered that input stitch files contain channel information: `A01_t0000_ch00_stitch.png`
- Current pipeline IGNORES channel info, losing critical metadata for future multi-channel fluorescence
- **NEW TARGET**: Include channel in image_ids: `20240411_A01_ch00_t0000.jpg` instead of `20240411_A01_t0000.jpg`
- This requires comprehensive updates to both `parsing_utils.py` AND `data_organizer.py`
- Updated implementation plan to include channel extraction and backward compatibility strategy

Next steps (updated plan):
1. Update `parsing_utils.py` with channel-aware regex patterns and parsing functions
2. Update `data_organizer.py` to extract channel from stitch filenames
3. Maintain backward compatibility for existing channel-less data
4. Test with multi-channel stitch files to validate fluorescence workflow support

## Concrete Examples

### Input to Output Transformation
```
# Input stitch file
A01_t0000_ch00_stitch.png

# Current behavior (channel ignored)
Parsed: well_id="A01", frame="0000", channel=IGNORED
Output: 20240411_A01_t0000.jpg

# NEW behavior (channel preserved)
Parsed: well_id="A01", frame="0000", channel="00"
Output: 20240411_A01_ch00_t0000.jpg
```

### Multi-Channel Support
```
# Brightfield channel
A01_t0000_ch00_stitch.png → 20240411_A01_ch00_t0000.jpg

# Fluorescence channels  
A01_t0000_ch01_stitch.png → 20240411_A01_ch01_t0000.jpg
A01_t0000_ch02_stitch.png → 20240411_A01_ch02_t0000.jpg
```

### Backward Compatibility
```
# Legacy data (no channel info)
Existing: 20240411_A01_t0000.jpg → Still works, treated as ch00

# New data (with channel info)  
New: 20240411_A01_ch00_t0000.jpg → Full channel support

# Mixed datasets supported during transition
```