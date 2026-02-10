# Rebase Roadmapping: Modules, Core Functions & Integration Flow

## Module Context & Progression Flow

Rebase and refactor the segmentation pipeline by progressing sequentially through each module. For each module:
- Define its **core function**.
- Ensure it **consumes validated outputs** from upstream modules and **produces validated outputs** for downstream modules.
- Implement key functions and utilities.
- Write and run **unit tests** to verify correctness before moving on.

1. **utils/** — Foundation Utilities ✅ COMPLETE 
   Core function: parsing, ID tracking, and atomic JSON I/O.  
   Connects to: *all modules*.  
   Key functions: `parse_entity_id()`, `EntityIDTracker`, `atomic_json_write()`.

2. **data_organization/** — Raw Data Processing ✅ COMPLETE
   Core function: organize raw experiment data into a structured folder layout.  
   Input: raw data directory.  
   Output: structured data and initial metadata for **metadata/**.  
   Key function: `DataOrganizer.process_experiments()`.

3. **metadata/** — All Metadata Management 🚧 NEXT
   Core function: load, validate, and manage both experiment and embryo metadata against schemas.  
   Input: outputs from **data_organization/** and **detection_segmentation/**.  
   Output: `experiment_metadata.json` and `embryo_metadata.json` for **detection_segmentation/** and **pipelines/**.  
   Key classes: `ExperimentMetadata`, `EmbryoMetadata` (in embryo_metadata/ submodule), `SchemaManager`.

4. **detection_segmentation/** — Detection & Segmentation 🔄 PENDING
   Core function: run object detection and segmentation on organized images.  
   Input: `experiment_metadata.json` and structured image folders.  
   Output: `detections.json`, `grounded_sam_annotations.json`.  
   Key classes: `GroundedDinoAnnotations`, `GroundedSamAnnotations`, `GSAMQualityControl`, `SimpleMaskExporter`.

5. **annotations/** — General Annotation Utilities 🔄 PENDING
   Core function: general annotation processing utilities and shared functionality.  
   Input: metadata and segmentation outputs.  
   Output: enhanced annotations and shared utilities.  
   Key classes: General annotation helpers.

6. **pipelines/** — Integration Scripts 🔄 PENDING
   Core function: orchestrate the end-to-end pipeline, invoking each module in order.  
   Input: none (acts as entry point).  
   Output: final artifacts and reports.  
   Key function: `run_morphseq_pipeline()` in `full_pipeline.py`.

---  
## Complete Directory Structure

```
segmentation_sandbox/
└── scripts/
    ├── utils/                           # Shared foundation utilities
    │   ├── __init__.py
    │   ├── parsing_utils.py            # ID parsing (backwards from end)
    │   ├── entity_id_tracker.py        # Entity validation & tracking
    │   ├── base_file_handler.py        # JSON I/O with atomic writes
    │   └── video_generation/            # ✅ NEW: Modular video system
    │       ├── __init__.py              # Progressive enhancement support
    │       ├── video_config.py         # Colorblind palette & settings
    │       ├── video_generator.py      # Foundation + enhanced videos
    │       └── overlay_manager.py      # Smart overlay positioning
    │
    ├── data_organization/               # ✅ COMPLETE: Raw data processing
    │   ├── __init__.py
    │   └── data_organizer.py           # Stitch → organized structure + foundation videos
    │
    ├── metadata/                        # All metadata management
    │   ├── __init__.py
    │   ├── experiment_metadata.py      # Experiment metadata class
    │   ├── schema_manager.py           # Permitted values & validation
    │   └── embryo_metadata/            # Embryo-specific metadata
    │       ├── __init__.py
    │       ├── embryo_metadata.py      # Main embryo metadata class
    │       ├── unified_managers.py     # Business logic utilities
    │       ├── annotation_batch.py     # Batch operations
    │       └── embryo_metadata_tutorial.ipynb
    │
    ├── detection_segmentation/          # Technical pipeline
    │   ├── __init__.py
    │   ├── grounded_dino_utils.py     # Detection annotations
    │   ├── sam2_utils.py               # Video segmentation
    │   ├── gsam_quality_control.py     # Technical QC
    │   └── mask_exporter.py            # Export masks as images
    │
    ├── annotations/                     # General annotation utilities
    │   └── __init__.py
    │
    ├── pipelines/                       # Integration scripts
    │   ├── __init__.py
    │   ├── full_pipeline.py            # Complete workflow
    │   └── pipeline_config.yaml        # Configuration
    │
    └── tests/                          # Unit tests
        ├── test_parsing.py
        ├── test_entity_tracking.py
        ├── test_data_organizer.py     # ✅ Module 0 testing
        ├── test_video_generation.py   # ✅ Video utilities testing
        ├── test_detection.py
        └── test_annotations.py
```

## Key Design Principles

1. **Backwards ID Parsing**: All IDs parsed from end due to complex experiment names
2. **Entity Hierarchy**: experiment → video → image/embryo → snip
3. **Image Naming Convention**:
   - Disk: `0000.jpg` (no 't' prefix)
   - JSON: `exp_A01_t0000` (with 't' prefix)
4. **Snip ID Format**: `{embryo_id}_s{frame}` (always '_s' prefix)

## 🎬 Video Generation Strategy (Progressive Enhancement)

### Foundation Video Approach
- **Stage 0 (Module 0)**: Basic image_id overlay (generated once and stored)
  - Created: Foundation MP4 videos with image_id positioned 10% down from top-right
  - Stored: `raw_data_organized/{experiment}/vids/{video_id}.mp4`
  - Purpose: Efficient base for all future visualization enhancements

### On-Demand Overlay Enhancement  
*Note: Only basic image_id videos are generated permanently. All other visualizations are created on-the-fly.*

- **Stage 1 (Module 2)**: + GDINO detection bounding boxes
  - Enhanced videos with colorblind-friendly detection boxes
  - Generated on-demand using `VideoGenerator.create_enhanced_video()`
  - Stored temporarily in `visualization_output/detections/`

- **Stage 2 (Module 3)**: + SAM2 segmentation masks
  - Foundation + detection boxes + semi-transparent masks
  - Generated on-demand with smart overlay positioning
  - Stored temporarily in `visualization_output/segmentation/`

- **Stage 3 (Module 4)**: + Embryo metadata and QC flags
  - Complete visualization: foundation + detections + masks + metadata + QC
  - Generated on-demand for final analysis videos
  - Stored temporarily in `visualization_output/complete/`

### Key Benefits:
1. **Storage Efficiency**: Only foundation videos stored permanently
2. **Flexibility**: Mix and match overlay types without recreating base videos  
3. **Performance**: Fast overlay generation using dictionary mapping `{image_id: overlay_data}`
4. **Accessibility**: Colorblind-friendly pastel palette for all overlays

## Module Dependencies

```
utils/ (no dependencies)
    ↓
data_organization/ (uses utils)
    ↓
metadata/ (uses utils) — handles experiment metadata + embryo_metadata/ submodule
    ↓
detection_segmentation/ (uses utils, metadata)
    ↓
annotations/ (uses utils, metadata) — general annotation utilities
    ↓
pipelines/ (uses all) but each module can be run independently in a s.sh file 
```

## Complete Pipeline Flow

```python
# pipelines/full_pipeline.py

from pathlib import Path
from data_organization import DataOrganizer
from metadata import ExperimentMetadata
from metadata.embryo_metadata import EmbryoMetadata
from detection_segmentation import (
    GroundedDinoAnnotations,
    GroundedSamAnnotations,
    GSAMQualityControl,
    SimpleMaskExporter
)

def run_morphseq_pipeline(
    raw_data_dir: Path,
    output_dir: Path,
    experiment_names: List[str],
    gdino_model,
    sam2_config: str,
    sam2_checkpoint: str
):
    """Complete MorphSeq pipeline from raw data to annotations."""
    
    # Step 1: Organize raw data
    print("Step 1: Organizing raw data...")
    organizer = DataOrganizer()
    organizer.process_experiments(
        source_dir=raw_data_dir,
        output_dir=output_dir,
        experiment_names=experiment_names
    )
    
    # Step 2: Load experiment metadata
    print("Step 2: Loading experiment metadata...")
    metadata_path = output_dir / "raw_data_organized" / "experiment_metadata.json"
    exp_meta = ExperimentMetadata(metadata_path)
    
    # Step 3: Run detection
    print("Step 3: Running GroundedDINO detection...")
    detections_path = output_dir / "detections.json"
    gdino = GroundedDinoAnnotations(detections_path)
    gdino.set_metadata_path(metadata_path)
    
    gdino.process_missing_annotations(
        model=gdino_model,
        prompts="individual embryo",
        experiment_ids=experiment_names,
        box_threshold=0.35,
        text_threshold=0.25
    )
    
    # Step 4: Run segmentation
    print("Step 4: Running SAM2 segmentation...")
    segmentations_path = output_dir / "grounded_sam_annotations.json"
    gsam = GroundedSamAnnotations(
        filepath=segmentations_path,
        seed_annotations_path=detections_path,
        experiment_metadata_path=metadata_path,
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint
    )
    
    gsam.process_missing_annotations(
        auto_save_interval=5,
        max_videos=None  # Process all
    )
    
    # Step 5: Run quality control
    print("Step 5: Running quality control...")
    qc = GSAMQualityControl(segmentations_path)
    qc.run_all_checks(
        author="pipeline_auto_qc",
        process_all=False  # Only new entities
    )
    
    # Step 6: Export masks (optional)
    print("Step 6: Exporting masks...")
    mask_dir = output_dir / "embryo_masks"
    exporter = SimpleMaskExporter(
        sam2_path=segmentations_path,
        output_dir=mask_dir,
        format="png"
    )
    exporter.process_missing_masks(experiment_ids=experiment_names)
    
    # Step 7: Initialize embryo metadata
    print("Step 7: Setting up embryo metadata...")
    embryo_meta_path = output_dir / "embryo_metadata.json"
    em = EmbryoMetadata(
        sam_annotation_path=segmentations_path,
        embryo_metadata_path=embryo_meta_path,
        gen_if_no_file=True
    )
    
    print("✅ Pipeline complete!")
    return {
        "experiment_metadata": exp_meta,
        "detections": gdino,
        "segmentations": gsam,
        "quality_control": qc,
        "embryo_metadata": em
    }
```

## Data Flow Examples

### 1. ID Parsing Flow
```python
# Complex experiment ID
snip_id = "20250624_chem02_28C_T00_1356_H01_e01_s0034"

# utils.parsing_utils handles this:
parsed = parse_entity_id(snip_id)
# Returns:
# {
#   'experiment_id': '20250624_chem02_28C_T00_1356',
#   'video_id': '20250624_chem02_28C_T00_1356_H01',
#   'embryo_id': '20250624_chem02_28C_T00_1356_H01_e01',
#   'frame_number': '0034'
# }
```

## ✅ Module 0 Completion Status (July 31, 2025)

### **Completed Components:**
- ✅ **Foundation Utilities**: `parsing_utils.py`, `entity_id_tracker.py`, `base_file_handler.py`
- ✅ **DataOrganizer**: Complete rewrite with autosave, skip logic, metadata generation
- ✅ **Video Generation System**: Modular utilities for foundation + enhanced videos
- ✅ **Testing**: Comprehensive tests for data organization and video generation
- ✅ **Progressive Enhancement**: Architecture ready for Modules 2-4 overlays

### **Key Features Delivered:**
- **Foundation Videos**: Basic MP4s with image_id overlay (10% down from top-right)
- **Autosave Functionality**: Incremental metadata saves after each experiment  
- **Smart Skip Logic**: Efficient processing with `overwrite` parameter
- **Colorblind-Friendly**: Accessible pastel palette for future overlays
- **Backwards Compatibility**: Matches original script behavior

### **Test Results:**
- **56 Videos**: Successfully processed experiment `20250703_chem3_28C_T00_1325`
- **Autosave**: Correctly skipped existing data on second run
- **Overwrite**: Successfully reprocessed when `overwrite=True`
- **Metadata**: Valid JSON structure with complete hierarchy

**Status**: Module 0 is production-ready. Foundation videos are being generated correctly with proper image_id positioning and the video enhancement system is ready for downstream modules.

### 2. Entity Validation Flow
```python
# After any save operation
entities = EntityIDTracker.extract_entities(data)
EntityIDTracker.validate_hierarchy(entities)
# Raises if orphaned entities exist
```

### 3. Image Path Resolution
```python
# In metadata
image_id = "20240411_A01_t0042"  # Has 't' prefix

# On disk
image_path = "raw_data_organized/20240411/images/20240411_A01/0042.jpg"  # No 't'
```

## Migration Checklist

- [x] ✅ Move parsing functions to utils/parsing_utils.py
- [x] ✅ Update all imports to use shared utilities
- [x] ✅ Replace regex parsing with parsing_utils functions
- [x] ✅ Add entity validation to all save methods
- [x] ✅ Ensure snip_ids use '_s' prefix format
- [x] ✅ Verify image naming convention (disk vs JSON)
- [x] ✅ Test complete pipeline flow (Module 0)
- [x] ✅ Update documentation
- [ ] 🚧 Proceed to Module 1 (Metadata Management)

## Common Issues & Solutions

1. **Import errors**: Ensure `__init__.py` files exist in all directories
2. **ID parsing failures**: Check for backwards parsing from end
3. **Entity validation errors**: Look for orphaned snips/embryos
4. **Image not found**: Verify 't' prefix handling (JSON has it, disk doesn't)
5. **Snip format**: Should be `embryo_id_s0000` not `embryo_id_0000`