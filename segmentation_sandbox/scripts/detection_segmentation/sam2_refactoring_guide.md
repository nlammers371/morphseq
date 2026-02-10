# SAM2 Output Structure Refactoring Guide

## Overview

This guide details the changes needed to modify your SAM2 processing script to produce the desired output structure with enhanced metadata and reorganized data hierarchy.

## Current vs Desired Structure

### Current Structure
```json
{
  "experiments": {
    "experiment_id": {
      "videos": {
        "video_id": {
          "video_id": "video_id",
          "images": {
            "image_id": {
              "image_id": "image_id",
              "frame_index": 0,
              "is_seed_frame": true,
              "embryos": {
                "embryo_id": {
                  "embryo_id": "embryo_id",
                  "snip_id": "snip_id",
                  "segmentation": {...},
                  "bbox": [...],
                  "area": 12345,
                  "mask_confidence": 0.85
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### Desired Structure
```json
{
  "experiments": {
    "experiment_id": {
      "videos": {
        "video_id": {
          "seed_frame_info": {
            "seed_frame": "image_id",
            "seed_frame_index": 0,
            "bbox_format": "xyxy",
            "bbox_units": "pixels",
            "detections": [
              {
                "embryo_idx": 0,
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": 0.91,
                "source": "GroundedDINO"
              }
            ]
          },
          "embryo_ids": ["embryo_id1", "embryo_id2"],
          "image_ids": {
            "image_id": {
              "frame_index": 0,
              "is_seed_frame": true,
              "embryos": {
                "embryo_id": {
                  "embryo_id": "embryo_id",
                  "snip_id": "snip_id",
                  "segmentation_format": "rle",
                  "segmentation": {...},
                  "bbox": [...],
                  "area": 12345,
                  "mask_confidence": 0.85
                }
              }
            }
          }
        }
      }
    }
  }
}
```

## Key Changes Required

### 1. Data Structure Reorganization

**Change**: `videos.video_id.images` → `videos.video_id.image_ids`

**Rationale**: The new structure separates metadata (seed_frame_info, embryo_ids) from frame data (image_ids)

### 2. Add Seed Frame Information

**New field**: `seed_frame_info` containing:
- `seed_frame`: Image ID of the seed frame
- `seed_frame_index`: Frame index of seed frame
- `bbox_format`: Format specification ("xyxy")
- `bbox_units`: Units specification ("pixels") 
- `detections`: Array of original GroundedDINO detections with metadata

### 3. Add Embryo ID List

**New field**: `embryo_ids` - Array of all embryo IDs for this video

### 4. Enhanced Embryo Data

**New field**: `segmentation_format` - Explicitly specify the segmentation format

## Implementation Steps

### Step 1: Modify the `process_video` method

Update the method that creates the video structure in your `GroundedSamAnnotations` class:

```python
def process_video(self, video_id: str) -> Dict:
    """Process a single video with SAM2 segmentation - REFACTORED for new structure."""
    if self.verbose:
        print(f"🎬 Processing video: {video_id}")
    
    # Get video annotations and process as before...
    video_groups = self.group_annotations_by_video()
    if video_id not in video_groups:
        raise ValueError(f"Video {video_id} not found in annotations")
    
    video_annotations = video_groups[video_id]
    processing_start_time = datetime.now().isoformat()
    
    try:
        sam2_results, video_metadata, seed_frame_info = process_single_video_from_annotations(
            video_id, video_annotations, self, self.predictor, 
            processing_stats, self.segmentation_format, self.verbose
        )
        
        # Extract experiment ID
        exp_id = extract_experiment_id(video_id)
        well_id = video_id.replace(f"{exp_id}_", "")
        
        # Initialize experiment structure if needed
        if exp_id not in self.results["experiments"]:
            self.results["experiments"][exp_id] = {
                "experiment_id": exp_id,
                "first_processed_time": processing_start_time,
                "last_processed_time": processing_start_time,
                "videos": {}
            }
        
        # CREATE NEW STRUCTURE - this is the key change
        video_structure = {
            "video_id": video_id,
            "well_id": well_id,
            "seed_frame_info": {
                "seed_frame": seed_frame_info["seed_frame"],
                "seed_frame_index": seed_frame_info["seed_frame_index"],
                "bbox_format": "xyxy",
                "bbox_units": "pixels",
                "detections": self._format_seed_detections(seed_frame_info["detections"])
            },
            "embryo_ids": seed_frame_info["embryo_ids"],
            "image_ids": self._convert_sam2_results_to_image_ids_format(sam2_results),
            "sam2_success": True,
            "processing_timestamp": processing_start_time
        }
        
        # Store video structure
        self.results["experiments"][exp_id]["videos"][video_id] = video_structure
        
        # Update snip_ids list as before...
        
        return sam2_results
        
    except Exception as e:
        # Handle errors as before...
        raise
```

### Step 2: Add Helper Methods

Add these new methods to your `GroundedSamAnnotations` class:

```python
def _format_seed_detections(self, seed_detections: List[Dict]) -> List[Dict]:
    """Format seed detections for the new structure."""
    formatted_detections = []
    
    for idx, detection in enumerate(seed_detections):
        # Extract bbox (handle different field names from GroundedDINO)
        bbox = detection.get('box_xyxy') or detection.get('bbox_xyxy') or detection.get('bbox')
        confidence = detection.get('confidence', 0.0)
        
        formatted_detection = {
            "embryo_idx": idx,
            "bbox_xyxy": bbox,
            "confidence": confidence,
            "source": "GroundedDINO"
        }
        formatted_detections.append(formatted_detection)
    
    return formatted_detections

def _convert_sam2_results_to_image_ids_format(self, sam2_results: Dict) -> Dict:
    """Convert SAM2 results from old 'images' format to new 'image_ids' format."""
    image_ids_structure = {}
    
    for image_id, image_data in sam2_results.items():
        # Extract the core data, removing the image_id redundancy
        converted_image_data = {
            "frame_index": image_data["frame_index"],
            "is_seed_frame": image_data["is_seed_frame"],
            "embryos": {}
        }
        
        # Process embryos with enhanced format
        for embryo_id, embryo_data in image_data.get("embryos", {}).items():
            enhanced_embryo_data = {
                "embryo_id": embryo_data["embryo_id"],
                "snip_id": embryo_data["snip_id"],
                "segmentation_format": embryo_data.get("segmentation_format", self.segmentation_format),
                "segmentation": embryo_data["segmentation"],
                "bbox": embryo_data["bbox"],
                "area": embryo_data["area"],
                "mask_confidence": embryo_data["mask_confidence"]
            }
            converted_image_data["embryos"][embryo_id] = enhanced_embryo_data
        
        image_ids_structure[image_id] = converted_image_data
    
    return image_ids_structure
```

### Step 3: Update the `process_single_video_from_annotations` function

Modify this function to ensure it returns the enhanced seed_frame_info:

```python
def process_single_video_from_annotations(video_id: str, video_annotations: Dict, grounded_sam_instance,
                                         predictor, processing_stats: Dict, segmentation_format: str = 'rle',
                                         verbose: bool = True) -> Tuple[Dict, Dict, Dict]:
    """Process a single video - ENHANCED to return detailed seed frame info."""
    
    # ... existing code for video processing ...
    
    # Find seed frame and detections
    seed_image_id, seed_detections_dict = find_seed_frame_from_video_annotations(
        video_annotations, video_id
    )
    
    seed_frame_idx = image_ids_ordered.index(seed_image_id)
    seed_detections = seed_detections_dict['detections']
    
    # Assign embryo IDs
    embryo_ids = assign_embryo_ids(video_id, len(seed_detections))
    
    # ... existing SAM2 propagation code ...
    
    # ENHANCED: Create comprehensive seed frame info
    seed_frame_info = {
        "video_id": video_id,
        "seed_frame": seed_image_id,
        "seed_frame_index": seed_frame_idx,
        "num_embryos": len(seed_detections),
        "detections": seed_detections,  # Original GroundedDINO detections
        "embryo_ids": embryo_ids,
        "is_first_frame": (seed_frame_idx == 0),
        "all_frames": image_ids_ordered,
        "requires_bidirectional_propagation": (seed_frame_idx > 0)
    }
    
    return sam2_results, video_info, seed_frame_info
```

### Step 4: Update Embryo Data Processing

Ensure that each embryo data structure includes the `segmentation_format` field:

```python
# In your mask processing code, add the segmentation_format field:
embryo_data = {
    "embryo_id": embryo_id,
    "snip_id": snip_id,
    "segmentation_format": segmentation_format,  # ADD THIS LINE
    "segmentation": segmentation,
    "bbox": bbox,
    "area": area,
    "mask_confidence": 0.85
}
```

## Testing the Changes

### Verification Steps

1. **Structure Check**: Verify the output has `image_ids` instead of `images`
2. **Seed Frame Info**: Confirm `seed_frame_info` contains all required fields
3. **Embryo IDs**: Check that `embryo_ids` list is present at video level
4. **Enhanced Fields**: Verify `segmentation_format` is in each embryo data

### Sample Test Code

```python
# Test the new structure
gsam = GroundedSamAnnotations(
    filepath="test_output.json",
    seed_annotations_path="your_gdino_annotations.json",
    experiment_metadata_path="your_experiment_metadata.json"
)

# Process a single video
gsam.process_video("20250612_30hpf_ctrl_atf6_A01")

# Verify structure
results = gsam.results
exp_data = results["experiments"]["20250612_30hpf_ctrl_atf6"]
video_data = exp_data["videos"]["20250612_30hpf_ctrl_atf6_A01"]

# Check required fields exist
assert "seed_frame_info" in video_data
assert "embryo_ids" in video_data  
assert "image_ids" in video_data
assert "images" not in video_data  # Should be removed

# Check seed_frame_info structure
seed_info = video_data["seed_frame_info"]
assert "bbox_format" in seed_info
assert "bbox_units" in seed_info
assert "detections" in seed_info

print("✅ Structure verification passed!")
```

## Migration Notes

### Performance Considerations

- The new structure is slightly more nested but provides better organization
- Memory usage should be similar since we're reorganizing, not duplicating data
- Processing time should remain the same

## Enhanced File Documentation

### Documentation Header Template

Add this comprehensive documentation block at the top of your refactored file to help developers understand the complete data flow:

```python
#!/usr/bin/env python3
"""
SAM2 Video Processing Pipeline for Embryo Segmentation
=====================================================

OVERVIEW
--------
This module processes time-lapse embryo videos using SAM2 (Segment Anything Model 2) 
for automated segmentation and tracking. It integrates with GroundedDINO annotations 
to provide seed frames and propagates segmentation masks across entire video sequences.

PIPELINE ARCHITECTURE
---------------------
The processing pipeline follows a four-stage architecture:

1. SEED DISCOVERY: Analyzes GroundedDINO annotations to identify high-quality 
   detection frames that can serve as segmentation seeds
   
2. VIDEO GROUPING: Organizes image sequences by video_id and determines 
   optimal seed frames to minimize bidirectional propagation
   
3. SAM2 PROPAGATION: Uses SAM2's video predictor to propagate segmentation 
   masks from seed frames across entire video sequences
   
4. STRUCTURED OUTPUT: Transforms raw SAM2 results into organized JSON structure 
   with enhanced metadata for downstream analysis

DATA FLOW DIAGRAM
-----------------
GroundedDINO Annotations → Video Grouping → Seed Frame Selection → SAM2 Processing → Structured Output

Input:  high_quality_annotations.json (from GroundedDINO)
        experiment_metadata.json (video structure info)
        
Process: SAM2 video segmentation with temporal propagation
         
Output: GroundedSam2Annotations.json (structured segmentation results)

KEY CLASSES AND RESPONSIBILITIES  
--------------------------------
- GroundedSamAnnotations: Main orchestrator class that manages the entire pipeline
  * Loads and validates input data (annotations + metadata)
  * Groups annotations by video and selects optimal seed frames
  * Coordinates SAM2 model loading and video processing
  * Generates structured output with entity validation

IMPORTED DEPENDENCIES
---------------------
- EntityIDTracker (from scripts.utils.entity_id_tracker): Cross-pipeline entity validation
  * Ensures consistent ID formats across experiments
  * Validates entity hierarchies (experiment → video → image → embryo → snip)
  * Tracks processing completeness across pipeline stages
  * Provides cross-referencing between GroundedDINO and SAM2 results

- ExperimentMetadata (from scripts.metadata.experiment_metadata): Metadata management
  * Provides video directory paths and temporal frame ordering
  * Handles experiment organization and metadata lookup
  * Manages base data path configuration for file access

- BaseFileHandler (from scripts.utils.base_file_handler): Atomic file operations
  * Provides safe JSON reading/writing with atomic operations
  * Handles file locking and error recovery for concurrent access
  * Ensures data integrity during save operations

- parsing_utils (from scripts.utils.parsing_utils): ID parsing and extraction
  * Standardized entity ID parsing across pipeline modules
  * Frame number and video ID extraction utilities
  * Consistent entity type detection and validation

INPUT REQUIREMENTS
------------------
1. GroundedDINO Annotations (JSON):
   - Must contain 'high_quality_annotations' section
   - Filtered annotations for target prompt (e.g., 'individual embryo')
   - Bounding box coordinates in consistent format

2. Experiment Metadata (JSON):
   - Video-to-directory mapping
   - Temporal frame ordering (image_ids list)
   - Experiment organization structure

3. SAM2 Model Files:
   - Model configuration file (.yaml)
   - Model checkpoint file (.pth)
   - Compatible with sam2.build_sam import structure

OUTPUT STRUCTURE SPECIFICATION
------------------------------
The output follows a hierarchical structure optimized for downstream analysis:

experiments/{experiment_id}/videos/{video_id}/
├── seed_frame_info/          # Metadata about segmentation seed
│   ├── seed_frame           # Image ID of seed frame
│   ├── seed_frame_index     # Temporal position of seed
│   ├── bbox_format          # Coordinate format specification
│   ├── bbox_units           # Measurement units specification  
│   └── detections[]         # Original GroundedDINO detections
├── embryo_ids[]             # List of all embryo IDs in video
└── image_ids/               # Frame-by-frame segmentation data
    └── {image_id}/
        ├── frame_index      # Temporal position
        ├── is_seed_frame    # Boolean flag
        └── embryos/         # Per-embryo segmentation results
            └── {embryo_id}/
                ├── snip_id           # Unique segmentation identifier
                ├── segmentation      # Mask data (RLE or polygon)
                ├── segmentation_format # Format specification
                ├── bbox             # Bounding box coordinates
                ├── area             # Mask area in pixels
                └── mask_confidence  # SAM2 confidence score

PROCESSING STRATEGIES
--------------------
- FORWARD PROPAGATION: Used when seed frame is the first frame (t0000)
  * Minimizes computational overhead
  * Maintains temporal consistency
  * Preferred strategy when high-quality detections exist in early frames

- BIDIRECTIONAL PROPAGATION: Used when seed frame is mid-sequence
  * Propagates forward from seed to end, backward from seed to beginning
  * Combines results maintaining temporal order
  * Used when best detections occur later in video sequence

ENTITY ID STANDARDIZATION
-------------------------
All entity IDs follow hierarchical naming conventions:
- Experiment: YYYYMMDD_condition_treatment (e.g., "20250612_30hpf_ctrl_atf6")
- Video: {experiment_id}_{well_id} (e.g., "20250612_30hpf_ctrl_atf6_A01")  
- Image: {video_id}_t{frame:04d} (e.g., "20250612_30hpf_ctrl_atf6_A01_t0042")
- Embryo: {video_id}_e{embryo:02d} (e.g., "20250612_30hpf_ctrl_atf6_A01_e01")
- Snip: {embryo_id}_s{frame:04d} (e.g., "20250612_30hpf_ctrl_atf6_A01_e01_s0042")

ERROR HANDLING AND VALIDATION
-----------------------------
- Input validation ensures all required files exist before processing
- Entity validation confirms ID consistency across processing stages  
- Video processing includes error recovery with detailed failure logging
- Output validation verifies structural integrity before save operations

PERFORMANCE CONSIDERATIONS
--------------------------
- SAM2 model supports both GPU and CPU processing (GPU strongly recommended)
- Video processing supports batch operation with configurable auto-save intervals
- Memory usage scales with video length and number of objects per frame
- Processing time approximately 1-3 seconds per frame for typical embryo videos

USAGE PATTERNS
--------------
# Import the main class and dependencies:
from sam2_utils import GroundedSamAnnotations
# Dependencies are auto-imported from:
# - scripts.utils.entity_id_tracker import EntityIDTracker  
# - scripts.metadata.experiment_metadata import ExperimentMetadata
# - scripts.utils.base_file_handler import BaseFileHandler
# - scripts.utils.parsing_utils import parse_entity_id, extract_frame_number, etc.

# Basic usage for new processing:
gsam = GroundedSamAnnotations(
    filepath="output/sam2_results.json",
    seed_annotations_path="input/gdino_annotations.json", 
    experiment_metadata_path="metadata/experiment_metadata.json"
)
gsam.set_sam2_model_paths(config_path, checkpoint_path)
gsam.process_missing_annotations(max_videos=5)

# Resuming interrupted processing:
gsam = GroundedSamAnnotations(filepath="existing_results.json", ...)
missing_videos = gsam.get_missing_videos()
gsam.process_missing_annotations(video_ids=missing_videos)

# Quality control and validation:
gsam.print_summary()
gsam.print_entity_comparison()
comparison = gsam.get_entity_comparison()

INTEGRATION POINTS
-----------------
- Input: Integrates with GroundedDINO pipeline (Module 1)
- Output: Provides structured data for analysis pipeline (Module 3)
- Dependencies: 
  * Module 0 utilities (parsing_utils, entity_id_tracker, base_file_handler)
  * Experiment metadata management (scripts.metadata.experiment_metadata)
  * SAM2 model framework (external sam2 package)
- Validation: Uses imported EntityIDTracker for cross-stage validation

VERSION HISTORY
---------------
- v1.0: Initial implementation with basic SAM2 integration
- v2.0: REFACTORED - Enhanced structure with metadata separation
        * Added seed_frame_info with detection provenance
        * Reorganized data hierarchy (images → image_ids) 
        * Embedded entity validation and cross-referencing
        * Improved bidirectional propagation handling
"""
```

### Why This Documentation Matters

This comprehensive header serves as a **mental map** for anyone working with your code. Think of it like having a detailed blueprint before building a house - it shows not just what each room does, but how the rooms connect and why they're arranged that way.

The documentation addresses three critical developer needs:

**Cognitive Load Reduction**: Complex pipelines like SAM2 processing involve many interconnected steps. By clearly laying out the architecture and data flow, developers can focus on understanding one piece at a time rather than trying to reverse-engineer the entire system.

**Onboarding Acceleration**: New team members can understand the system's purpose and design philosophy before diving into implementation details. The data flow diagram and key classes section provide the essential mental framework for code comprehension.

**Maintenance Guidance**: The entity ID standardization and error handling sections help developers understand the assumptions and contracts that the code relies on. This prevents accidental breaking changes and helps with debugging when things go wrong.

### Documentation as System Design

Notice how this documentation doesn't just describe what the code does - it explains **why design decisions were made**. For example, explaining why bidirectional propagation exists (when seed frames aren't optimal at the beginning) helps developers understand when to use different processing strategies.

The hierarchical structure specification serves as both documentation and validation tool. Future modifications can be checked against this structure to ensure compatibility with downstream analysis tools.

This type of comprehensive documentation transforms your code from a working implementation into a **maintainable system component** that other developers can confidently modify and extend.